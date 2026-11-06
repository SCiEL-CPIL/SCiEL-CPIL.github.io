import time
import sys
import random

DEADLINE_SECONDS = 320.0
START_TIME = time.time()

def _time_left():
    return DEADLINE_SECONDS - (time.time() - START_TIME)

def _paley_graph(q):
    """Return edges of the Paley graph on q vertices (q a prime congruent to 1 mod 4)."""
    residues = {(x * x) % q for x in range(1, q)}
    return [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def _build_adj(edges, n):
    """Build adjacency as list of int bitsets."""
    adj = [0] * n
    for i, j in edges:
        adj[i] |= (1 << j)
        adj[j] |= (1 << i)
    return adj

def _has_k5(adj, n):
    """Check if graph has a clique of size 5 using bitsets."""
    for a in range(n):
        for b in range(a + 1, n):
            if not (adj[a] & (1 << b)):
                continue
            common_ab = adj[a] & adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_abc = common_ab & adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    # Found 4-clique {a,b,c,d}, check if any vertex connects to all four
                    common_abcd = common_abc & adj[d]
                    if common_abcd:
                        return True
    return False

def _has_indep5(adj, n):
    """Check if graph has an independent set of size 5 using bitsets."""
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if adj[a] & (1 << b):
                continue
            common_non_ab = non_adj[a] & non_adj[b] & ~((1 << (b + 1)) - 1)
            c_mask = common_non_ab
            while c_mask:
                c_bit = c_mask & (-c_mask)
                c = c_bit.bit_length() - 1
                c_mask ^= c_bit
                common_non_abc = common_non_ab & non_adj[c] & ~((1 << (c + 1)) - 1)
                d_mask = common_non_abc
                while d_mask:
                    d_bit = d_mask & (-d_mask)
                    d = d_bit.bit_length() - 1
                    d_mask ^= d_bit
                    common_non_abcd = common_non_abc & non_adj[d]
                    if common_non_abcd:
                        return True
    return False

def _is_55_free(adj, n):
    """Check if graph is (5,5)-free."""
    return not _has_k5(adj, n) and not _has_indep5(adj, n)

def _count_4cliques_with_vertex(adj, n, v):
    """Count number of 4-cliques that include vertex v."""
    count = 0
    for a in range(n):
        if a == v:
            continue
        if not (adj[v] & (1 << a)):
            continue
        for b in range(n):
            if b == v or b == a:
                continue
            if not (adj[v] & (1 << b)) or not (adj[a] & (1 << b)):
                continue
            for c in range(n):
                if c == v or c == a or c == b:
                    continue
                if not (adj[v] & (1 << c)) or not (adj[a] & (1 << c)) or not (adj[b] & (1 << c)):
                    continue
                count += 1
    return count

def _count_4indep_with_vertex(adj, n, v):
    """Count number of 4-independent sets that include vertex v."""
    all_mask = (1 << n) - 1
    non_adj = [all_mask ^ adj[i] ^ (1 << i) for i in range(n)]
    count = 0
    for a in range(n):
        if a == v:
            continue
        if adj[v] & (1 << a):
            continue
        for b in range(n):
            if b == v or b == a:
                continue
            if adj[v] & (1 << b) or adj[a] & (1 << b):
                continue
            for c in range(n):
                if c == v or c == a or c == b:
                    continue
                if adj[v] & (1 << c) or adj[a] & (1 << c) or adj[b] & (1 << c):
                    continue
                count += 1
    return count

def _try_add_vertex(adj, n):
    """
    Try to add a new vertex (index n) to the graph.
    Returns (success, new_adj) where new_adj is the adjacency for n+1 vertices.
    Uses a greedy approach: try each possible edge pattern, check validity.
    """
    # For each vertex i in 0..n-1, decide if edge (n, i) exists.
    # We need to avoid creating K5 or independent 5-set.
    # A K5 containing the new vertex requires 4 existing vertices that form a K4 and are all connected to new vertex.
    # An independent 5-set containing new vertex requires 4 existing vertices that form an independent 4-set and are all NOT connected to new vertex.
    
    # Strategy: try different numbers of edges from new vertex, starting with balanced.
    # For a given set S of vertices connected to new vertex, the constraint is:
    # - No 4-clique in S (otherwise K5 with new vertex)
    # - No 4-independent set in complement of S (otherwise indep 5 with new vertex)
    
    # This is hard to satisfy directly. Instead, use local search.
    # Start with random assignment, then flip edges to reduce violations.
    
    best_adj = None
    best_violations = float('inf')
    
    for trial in range(20):
        if _time_left() < 10:
            break
            
        # Random initial assignment
        new_adj = adj[:] + [0]
        for i in range(n):
            if random.random() < 0.5:
                new_adj[n] |= (1 << i)
                new_adj[i] |= (1 << n)
        
        # Local search: flip edges to reduce violations
        violations = 0
        for _ in range(100):
            if _time_left() < 5:
                break
            
            # Count violations (K5 and indep5 containing new vertex)
            k5_count = 0
            indep5_count = 0
            
            # Check for K5 containing new vertex: need 4-clique among neighbors of new vertex
            neigh_mask = new_adj[n]
            # Find 4-cliques in subgraph induced by neighbors
            neigh_list = []
            m = neigh_mask
            while m:
                bit = m & (-m)
                v = bit.bit_length() - 1
                neigh_list.append(v)
                m ^= bit
            
            # Check 4-cliques in neighbor subgraph
            for a in range(len(neigh_list)):
                va = neigh_list[a]
                for b in range(a + 1, len(neigh_list)):
                    vb = neigh_list[b]
                    if not (new_adj[va] & (1 << vb)):
                        continue
                    common = new_adj[va] & new_adj[vb] & neigh_mask & ~((1 << (vb + 1)) - 1)
                    # Actually need to be careful with ordering
                    common = new_adj[va] & new_adj[vb] & neigh_mask
                    # Remove va, vb
                    common &= ~(1 << va)
                    common &= ~(1 << vb)
                    c_mask = common
                    while c_mask:
                        c_bit = c_mask & (-c_mask)
                        vc = c_bit.bit_length() - 1
                        c_mask ^= c_bit
                        if not (new_adj[va] & (1 << vc)) or not (new_adj[vb] & (1 << vc)):
                            continue
                        common2 = common & new_adj[vc]
                        common2 &= ~(1 << vc)
                        d_mask = common2
                        while d_mask:
                            d_bit = d_mask & (-d_mask)
                            vd = d_bit.bit_length() - 1
                            d_mask ^= d_bit
                            if (new_adj[va] & (1 << vd)) and (new_adj[vb] & (1 << vd)) and (new_adj[vc] & (1 << vd)):
                                k5_count += 1
            
            # Check for indep5 containing new vertex: need 4-independent set among non-neighbors
            non_neigh_mask = ((1 << n) - 1) ^ neigh_mask
            non_neigh_list = []
            m = non_neigh_mask
            while m:
                bit = m & (-m)
                v = bit.bit_length() - 1
                non_neigh_list.append(v)
                m ^= bit
            
            # Check 4-independent sets in non-neighbor subgraph
            for a in range(len(non_neigh_list)):
                va = non_neigh_list[a]
                for b in range(a + 1, len(non_neigh_list)):
                    vb = non_neigh_list[b]
                    if new_adj[va] & (1 << vb):
                        continue
                    common_non = non_neigh_mask & ~((1 << va) - 1)
                    common_non &= ~(1 << va)
                    common_non &= ~(1 << vb)
                    # Both va and vb not connected to each other, and both not connected to new vertex
                    # Need common non-neighbors that are also not connected to va and vb
                    c_mask = common_non
                    while c_mask:
                        c_bit = c_mask & (-c_mask)
                        vc = c_bit.bit_length() - 1
                        c_mask ^= c_bit
                        if new_adj[va] & (1 << vc) or new_adj[vb] & (1 << vc):
                            continue
                        common2 = common_non & ~((1 << vc) - 1)
                        common2 &= ~(1 << vc)
                        d_mask = common2
                        while d_mask:
                            d_bit = d_mask & (-d_mask)
                            vd = d_bit.bit_length() - 1
                            d_mask ^= d_bit
                            if not (new_adj[va] & (1 << vd)) and not (new_adj[vb] & (1 << vd)) and not (new_adj[vc] & (1 << vd)):
                                indep5_count += 1
            
            total_violations = k5_count + indep5_count
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_adj = new_adj[:]
                if total_violations == 0:
                    return True, best_adj
            
            # Try flipping a random edge from new vertex
            if random.random() < 0.3:
                # Try flipping an edge to a random vertex
                i = random.randint(0, n - 1)
                if new_adj[n] & (1 << i):
                    new_adj[n] &= ~(1 << i)
                    new_adj[i] &= ~(1 << n)
                else:
                    new_adj[n] |= (1 << i)
                    new_adj[i] |= (1 << n)
        
        if best_violations == 0:
            return True, best_adj
    
    if best_violations == 0:
        return True, best_adj
    return False, None

def construct():
    # Start with Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    n = q
    
    adj = _build_adj(edges, n)
    
    best_n = n
    best_edges = edges[:]
    
    # Try to add vertices one by one
    max_attempts = 20
    for attempt in range(max_attempts):
        if _time_left() < 30:
            break
        
        success, new_adj = _try_add_vertex(adj, n)
        
        if not success:
            break
        
        n = n + 1
        adj = new_adj
        
        # Verify validity
        if not _is_55_free(adj, n):
            # Should not happen, but break if invalid
            n = n - 1
            break
        
        # Update edges
        for i in range(n - 1):
            if adj[n - 1] & (1 << i):
                edges.append((i, n - 1))
        
        if n > best_n:
            best_n = n
            best_edges = edges[:]
    
    # Local search polish: try to flip edges in the current graph to maintain validity
    # and potentially allow more vertex additions
    if _time_left() > 60:
        current_n = n
        for _ in range(100):
            if _time_left() < 30:
                break
            
            # Try flipping a random edge
            if len(edges) < 10:
                break
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            
            # Remove edge
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.pop(idx)
            
            if _is_55_free(adj, current_n):
                # Try to add a vertex
                success, new_adj = _try_add_vertex(adj, current_n)
                if success:
                    current_n += 1
                    adj = new_adj
                    for i in range(current_n - 1):
                        if adj[current_n - 1] & (1 << i):
                            edges.append((i, current_n - 1))
                    if current_n > best_n:
                        best_n = current_n
                        best_edges = edges[:]
                    n = current_n
                else:
                    # Put edge back
                    edges.append((u, v))
                    edges.sort()
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            else:
                # Put edge back
                edges.append((u, v))
                edges.sort()
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    return best_n, best_edges
