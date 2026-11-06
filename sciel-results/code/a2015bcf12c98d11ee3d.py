"""K4-free, no independent 6-set graph maximizer.
Uses deterministic constructions (Paley, Clebsch, complement of known graphs)
and a randomized local search to find larger valid graphs within time budget.
"""
import time
import random

DEADLINE = 140  # safety margin before 150s kill

def _build_adj(n, edge_set):
    adj = [0] * n
    for (u, v) in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _has_k4(adj, n):
    # Check for K4: for each triple, check if all three pairs are edges
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                # Check if i,j,k form a triangle; now check 4th vertex
                common = adj[i] & adj[j] & adj[k]
                # common should contain i,j,k themselves; check for a 4th
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                if mask:
                    return True
    return False

def _has_independent_6(adj, n):
    # Independent set of 6: 6 vertices with no edges between any pair
    # Check via complement adjacency
    comp = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
    # Look for K6 in complement
    for i in range(n):
        for j in range(i + 1, n):
            if not (comp[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                    continue
                for l in range(k + 1, n):
                    if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                        continue
                    # i,j,k,l form a K4 in complement; check 5th and 6th
                    common_ijk = comp[i] & comp[j] & comp[k] & comp[l]
                    mask = common_ijk & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                    # Need two more vertices from mask that are connected in complement
                    bits = []
                    m = mask
                    while m:
                        b = m & (-m)
                        bits.append(b.bit_length() - 1)
                        m ^= b
                    # Check if any two in bits are adjacent in complement
                    for a in range(len(bits)):
                        for b in range(a + 1, len(bits)):
                            va, vb = bits[a], bits[b]
                            if (comp[va] >> vb) & 1:
                                return True
    return False

def _count_k4_and_ind6(adj, n):
    """Return (num_K4, num_ind6) quickly for small checks."""
    k4_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] >> j) & 1:
                continue
            for k in range(j + 1, n):
                if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                    continue
                common = adj[i] & adj[j] & adj[k]
                mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                k4_count += mask.bit_count()
    # For independent 6, just check existence
    ind6 = _has_independent_6(adj, n)
    return k4_count, 1 if ind6 else 0

def _paley_graph(p):
    """Paley graph on p vertices where p ≡ 1 mod 4."""
    if p % 4 != 1:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    edge_set = set()
    for i in range(p):
        for j in range(i + 1, p):
            if (j - i) % p in squares:
                edge_set.add((i, j))
    return {'n': p, 'edges': sorted(edge_set)}

def _clebsch_graph():
    """Clebsch graph on 16 vertices."""
    n = 16
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            # Clebsch: distance 1 in 4-cube (Hamming weight 1) OR weight 4
            xor = i ^ j
            w = bin(xor).count('1')
            if w == 1 or w == 4:
                edge_set.add((i, j))
    return {'n': n, 'edges': sorted(edge_set)}

def _complement_of(graph):
    n = graph['n']
    existing = set(tuple(e) for e in graph['edges'])
    edge_set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in existing and (j, i) not in existing:
                edge_set.add((i, j))
    return {'n': n, 'edges': sorted(edge_set)}

def _validate(graph):
    """Check if graph is valid (no K4, no independent 6)."""
    n = graph['n']
    if n < 1 or n > 40:
        return False
    adj = _build_adj(n, graph['edges'])
    if _has_k4(adj, n):
        return False
    if _has_independent_6(adj, n):
        return False
    return True

def _local_search(graph, time_limit):
    """Randomized local search to fix violations and grow the graph."""
    t0 = time.time()
    n = graph['n']
    edge_set = set(tuple(e) for e in graph['edges'])
    
    for attempt in range(200):
        if time.time() - t0 > time_limit:
            break
        
        # Start from current graph or random perturbation
        adj = _build_adj(n, edge_set)
        
        # Check current validity
        valid = not _has_k4(adj, n) and not _has_independent_6(adj, n)
        if valid:
            return graph
        
        # Try to fix by flipping edges
        flips_made = 0
        max_flips = 50
        random.seed(attempt * 7919 + 13)
        
        for _ in range(max_flips):
            if time.time() - t0 > time_limit:
                break
            if flips_made >= max_flips:
                break
                
            # Find a violation to fix
            # Check K4 first
            found_fix = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj[i] >> j) & 1:
                        continue
                    for k in range(j + 1, n):
                        if not ((adj[i] >> k) & 1 and (adj[j] >> k) & 1):
                            continue
                        common = adj[i] & adj[j] & adj[k]
                        mask = common & ~(1 << i) & ~(1 << j) & ~(1 << k)
                        if mask:
                            # Found K4: remove one edge from it
                            # Pick a random edge in the K4
                            candidates = [(i,j), (i,k), (j,k)]
                            # Also consider edges to the 4th vertex
                            m = mask
                            while m:
                                b = m & (-m)
                                v4 = b.bit_length() - 1
                                candidates.extend([(i,v4), (j,v4), (k,v4)])
                                m ^= b
                            u, v = random.choice(candidates)
                            if u > v:
                                u, v = v, u
                            edge_set.discard((u, v))
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                            flips_made += 1
                            found_fix = True
                            break
                        break
                    if found_fix:
                        break
                if found_fix:
                    break
            
            if not found_fix:
                # Try to fix independent 6
                comp = [((1 << n) - 1) ^ adj[x] ^ (1 << x) for x in range(n)]
                found_ind = False
                for i in range(n):
                    for j in range(i + 1, n):
                        if not (comp[i] >> j) & 1:
                            continue
                        for k in range(j + 1, n):
                            if not ((comp[i] >> k) & 1 and (comp[j] >> k) & 1):
                                continue
                            for l in range(k + 1, n):
                                if not ((comp[i] >> l) & 1 and (comp[j] >> l) & 1 and (comp[k] >> l) & 1):
                                    continue
                                common_c = comp[i] & comp[j] & comp[k] & comp[l]
                                mask = common_c & ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                                if mask:
                                    # Add an edge between two non-adjacent vertices in the ind set
                                    # This breaks the independent set
                                    m = mask
                                    verts = []
                                    while m:
                                        b = m & (-m)
                                        verts.append(b.bit_length() - 1)
                                        m ^= b
                                    if len(verts) >= 1:
                                        a = random.choice(verts)
                                        b = random.choice(verts)
                                        if a == b:
                                            continue
                                        if a > b:
                                            a, b = b, a
                                        if (a, b) not in edge_set:
                                            edge_set.add((a, b))
                                            adj[a] |= (1 << b)
                                            adj[b] |= (1 << a)
                                            flips_made += 1
                                            found_ind = True
                                break
                            if found_ind:
                                break
                        if found_ind:
                            break
                    if found_ind:
                        break
                
                if not found_ind:
                    # Random flip
                    a = random.randint(0, n - 1)
                    b = random.randint(0, n - 1)
                    if a == b:
                        continue
                    if a > b:
                        a, b = b, a
                    if (a, b) in edge_set:
                        edge_set.discard((a, b))
                        adj[a] &= ~(1 << b)
                        adj[b] &= ~(1 << a)
                    else:
                        edge_set.add((a, b))
                        adj[a] |= (1 << b)
                        adj[b] |= (1 << a)
                    flips_made += 1
        
        # After flipping, check if valid
        if not _has_k4(adj, n) and not _has_independent_6(adj, n):
            graph['edges'] = sorted(edge_set)
            return graph

    # Return best effort
    graph['edges'] = sorted(edge_set)
    return graph

def construct():
    t0 = time.time()
    best_graph = None
    best_n = 0

    # Try deterministic constructions
    candidates = []
    
    # Paley graphs for primes p ≡ 1 mod 4 up to 37
    for p in [17, 29, 37]:
        g = _paley_graph(p)
        if g is not None:
            candidates.append(g)
    
    # Clebsch graph
    candidates.append(_clebsch_graph())
    
    # Complements of the above
    for g in list(candidates):
        cg = _complement_of(g)
        candidates.append(cg)

    # Evaluate all candidates
    for g in candidates:
        if time.time() - t0 > DEADLINE - 10:
            break
        if _validate(g):
            if g['n'] > best_n:
                best_n = g['n']
                best_graph = g

    # If we have a good graph, try to grow it via local search
    if best_graph is not None and time.time() - t0 < DEADLINE - 20:
        # Try to add vertices
        n = best_n
        # Start from best graph and try to add vertices
        edge_set = set(tuple(e) for e in best_graph['edges'])
        
        for new_n in range(n + 1, min(n + 6, 41)):
            if time.time() - t0 > DEADLINE - 10:
                break
            # Try to add a vertex with optimal connections
            adj = _build_adj(n, edge_set)
            for trial in range(100):
                # Randomly connect new vertex
                new_adj = [0] * new_n
                for i in range(n):
                    new_adj[i] = adj[i]
                for j in range(n):
                    # Randomly decide to connect
                    if random.random() < 0.5:
                        new_adj[new_n - 1] |= (1 << j)
                        new_adj[j] |= (1 << (new_n - 1))
                
                if not _has_k4(new_adj, new_n) and not _has_independent_6(new_adj, new_n):
                    # Valid! Add edge to set
                    for j in range(n):
                        if (new_adj[new_n - 1] >> j) & 1:
                            edge_set.add((j, new_n - 1))
                    n = new_n
                    best_n = new_n
                    break
            if n == new_n:
                best_graph = {'n': n, 'edges': sorted(edge_set)}

        # Run local search on current best
        remaining = DEADLINE - (time.time() - t0)
        if remaining > 5:
            best_graph = _local_search(best_graph, remaining - 5)
            if best_graph['n'] > best_n:
                best_n = best_graph['n']

    if best_graph is None:
        # Fallback: return a trivial valid graph
        return {'n': 1, 'edges': []}

    return best_graph
