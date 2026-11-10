import time
import itertools
import random

# EVOLVE-BLOCK-START

def _has_k4(edges_set, n):
    """Check if graph has a K4. edges_set is a set of frozensets."""
    adj = [set() for _ in range(n)]
    for e in edges_set:
        u, v = e
        adj[u].add(v)
        adj[v].add(u)
    # For each edge (u,v), check if they share a common neighbor w,
    # then for each such w, check if v and w share a common neighbor x,
    # and if x is connected to u
    # Actually, simpler: enumerate triangles and check edges between them
    # Even simpler for small n: just check all 4-subsets
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        if (a,b) in edges_set and (a,c) in edges_set and (a,d) in edges_set and \
           (b,c) in edges_set and (b,d) in edges_set and (c,d) in edges_set:
            return True
    return False

def _has_independent_6(edges_set, n):
    """Check if graph has an independent set of size 6."""
    # Build non-adjacency list
    non_adj = [set(range(n)) for _ in range(n)]
    for i in range(n):
        non_adj[i].discard(i)
    for e in edges_set:
        u, v = e
        non_adj[u].discard(v)
        non_adj[v].discard(u)
    
    # Try to find independent set of size 6
    # Use backtracking
    best = []
    def backtrack(start, current):
        if len(current) == 6:
            return True
        # Pruning: if not enough vertices left
        if len(current) + (n - start) < 6:
            return False
        for v in range(start, n):
            # Check if v is non-adjacent to all in current
            if all(x in non_adj[v] for x in current):
                current.append(v)
                if backtrack(v + 1, current):
                    return True
                current.pop()
        return False
    
    return backtrack(0, [])

def _is_valid(edges_list, n):
    """Check both constraints."""
    edges_set = set()
    for e in edges_list:
        edges_set.add((min(e), max(e)))
    
    if _has_k4(edges_set, n):
        return False
    if _has_independent_6(edges_set, n):
        return False
    return True

def _paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
    return edges

def _try_extend(edges, n, deadline):
    """Try to add one vertex to the graph. Returns new edges and n+1 if successful, else None."""
    # Build adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Find all triangles in the graph
    triangles = []
    for i in range(n):
        for j in adj[i]:
            if j > i:
                for k in adj[i]:
                    if k > j and k in adj[j]:
                        triangles.append((i, j, k))
    
    # Find all independent 5-sets
    # This is expensive, but for n up to ~20 it's manageable
    # Use backtracking
    indep_5 = []
    non_adj = [set(range(n)) - {i} - adj[i] for i in range(n)]
    
    def find_indep(start, current):
        if len(current) == 5:
            indep_5.append(tuple(current))
            return
        if len(current) + (n - start) < 5:
            return
        for v in range(start, n):
            if all(x in non_adj[v] for x in current):
                current.append(v)
                find_indep(v + 1, current)
                current.pop()
    
    find_indep(0, [])
    
    # For the new vertex n, try all possible adjacency patterns
    # The new vertex must:
    # 1. Not form a K4: for every triangle (a,b,c), not all of a,b,c can be adjacent to n
    # 2. Not form an independent 6-set: for every independent 5-set S, not all of S can be non-adjacent to n
    
    # Forbidden: sets of 3 mutually adjacent vertices that would all connect to n
    forbidden_adj = set()
    for tri in triangles:
        forbidden_adj.add(frozenset(tri))
    
    # Forbidden: sets of 5 mutually non-adjacent vertices that would all NOT connect to n
    forbidden_nonadj = set()
    for s5 in indep_5:
        forbidden_nonadj.add(frozenset(s5))
    
    # Try subsets of [0..n-1] as adjacency of new vertex
    # Use random order for diversity
    candidates = list(range(1 << n))
    random.shuffle(candidates)
    
    for mask in candidates:
        if time.time() > deadline:
            return None
        
        adj_set = frozenset(i for i in range(n) if (mask >> i) & 1)
        nonadj_set = frozenset(range(n)) - adj_set
        
        # Check K4 constraint
        ok = True
        for tri in triangles:
            if frozenset(tri) <= adj_set:
                ok = False
                break
        if not ok:
            continue
        
        # Check independent 6 constraint
        for s5 in indep_5:
            if s5 <= nonadj_set:
                ok = False
                break
        if not ok:
            continue
        
        # Valid! Add the vertex
        new_edges = list(edges)
        for i in range(n):
            if (mask >> i) & 1:
                new_edges.append([i, n])
        return new_edges
    
    return None

def _random_search(n, deadline, best_n):
    """Random local search for a valid graph on n vertices."""
    for attempt in range(100000):
        if time.time() > deadline:
            break
        
        # Random graph
        edges = []
        adj = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    edges.append([i, j])
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Local search: flip edges to fix violations
        for step in range(5000):
            if time.time() > deadline:
                break
            
            if _is_valid(edges, n):
                return edges
            
            # Check for violations
            has_k4 = _has_k4(set((min(e), max(e)) for e in edges), n)
            has_ind6 = _has_independent_6(set((min(e), max(e)) for e in edges), n)
            
            if has_k4:
                # Find a K4 and remove an edge
                for quad in itertools.combinations(range(n), 4):
                    a, b, c, d = quad
                    if (a,b) in {(min(e),max(e)) for e in edges} and (a,c) in {(min(e),max(e)) for e in edges} and (a,d) in {(min(e),max(e)) for e in edges} and \
                       (b,c) in {(min(e),max(e)) for e in edges} and (b,d) in {(min(e),max(e)) for e in edges} and (c,d) in {(min(e),max(e)) for e in edges}:
                        # Remove a random edge from this K4
                        candidates = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                        to_remove = random.choice(candidates)
                        u, v = to_remove
                        edges.remove([u, v] if [u, v] in edges else [v, u])
                        adj[u].discard(v)
                        adj[v].discard(u)
                        break
                else:
                    break
            elif has_ind6:
                # Find an independent 6-set and add an edge
                non_adj = [set(range(n)) - {i} - adj[i] for i in range(n)]
                found = False
                for quad in itertools.combinations(range(n), 6):
                    if all(x in non_adj[y] for x, y in itertools.combinations(quad, 2)):
                        # Add a random edge within this set
                        u, v = random.sample(quad, 2)
                        edges.append([u, v])
                        adj[u].add(v)
                        adj[v].add(u)
                        found = True
                        break
                if not found:
                    break
            else:
                break
    
    return None

def construct():
    deadline = time.time() + 110  # Safety margin
    
    best_edges = _paley_17()
    best_n = 17
    
    # Try to extend from 17
    current_edges = list(best_edges)
    current_n = best_n
    
    while current_n < 35 and time.time() < deadline:
        result = _try_extend(current_edges, current_n, deadline)
        if result is not None:
            current_edges = result
            current_n += 1
            if current_n > best_n:
                best_n = current_n
                best_edges = list(current_edges)
        else:
            break
    
    # If we didn't reach high n, try random search for various n
    for target_n in range(best_n + 1, min(36, best_n + 10)):
        if time.time() > deadline:
            break
        result = _random_search(target_n, deadline, best_n)
        if result is not None:
            best_n = target_n
            best_edges = result
    
    return {'n': best_n, 'edges': best_edges}

# EVOLVE-BLOCK-END
