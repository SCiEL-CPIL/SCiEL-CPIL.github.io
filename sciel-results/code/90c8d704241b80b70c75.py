"""K4-free graph with no independent 6-set via simulated annealing on full edge set."""
import time
import random
import numpy as np

_START_TIME = time.time()
DEADLINE = 145.0  # safety margin before 150s kill

def _time_left():
    return DEADLINE - (time.time() - _START_TIME)

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_n = 0
    best_edges = []
    
    # Try to build graphs of increasing size using simulated annealing
    for target_n in [20, 25, 30, 33, 35, 36]:
        if _time_left() < 8:
            break
        
        # Try multiple random starts for this target size
        for trial in range(3):
            if _time_left() < 5:
                break
            
            rng = random.Random(trial * 1000 + target_n)
            n, edges = _anneal(target_n, rng)
            if n > best_n:
                best_n = n
                best_edges = edges
            if best_n >= 36:
                return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Paley graph on 17 vertices
    if best_n == 0:
        n = 17
        squares = {x * x % 17 for x in range(1, 17)}
        best_edges = [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % 17 in squares]
        best_n = n
    
    return {'n': best_n, 'edges': best_edges}

def _count_k4(adj):
    """Count K4s using vectorized operations. adj is n x n boolean numpy array."""
    n = adj.shape[0]
    count = 0
    # For each pair (a,b) that is an edge, count common neighbors that form triangles
    # K4 = (a,b,c,d) where all 6 edges exist
    # Efficient: for each edge (a,b), count triangles in the common neighborhood
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a, b]:
                continue
            # Common neighbors of a and b
            common = adj[a] & adj[b]
            # For each pair (c,d) in common, check if c-d is an edge
            # This gives K4 containing edge (a,b)
            idx = np.where(common)[0]
            idx = idx[idx > b]  # c > b to avoid double counting
            for i in range(len(idx)):
                c = idx[i]
                for j in range(i + 1, len(idx)):
                    d = idx[j]
                    if adj[c, d]:
                        count += 1
    return count

def _find_k4(adj):
    """Find one K4 if it exists. Returns (a,b,c,d) or None."""
    n = adj.shape[0]
    for a in range(n):
        for b in range(a + 1, n):
            if not adj[a, b]:
                continue
            common = np.where(adj[a] & adj[b])[0]
            common = common[common > b]
            for i in range(len(common)):
                c = common[i]
                for j in range(i + 1, len(common)):
                    d = common[j]
                    if adj[c, d]:
                        return (a, b, c, d)
    return None

def _count_ind6(adj):
    """Count independent 6-sets. Independent set in G = clique in complement."""
    n = adj.shape[0]
    comp = ~adj
    np.fill_diagonal(comp, False)
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            if not comp[a, b]:
                continue
            common = np.where(comp[a] & comp[b])[0]
            common = common[common > b]
            for i in range(len(common)):
                c = common[i]
                for j in range(i + 1, len(common)):
                    d = common[j]
                    if not comp[c, d]:
                        continue
                    common2 = np.where(comp[a] & comp[b] & comp[c] & comp[d])[0]
                    common2 = common2[common2 > d]
                    for k in range(len(common2)):
                        e = common2[k]
                        common3 = np.where(comp[a] & comp[b] & comp[c] & comp[d] & comp[e])[0]
                        common3 = common3[common3 > e]
                        if len(common3) > 0:
                            count += len(common3)
    return count

def _find_ind6(adj):
    """Find one independent 6-set if it exists. Returns tuple of 6 vertices or None."""
    n = adj.shape[0]
    comp = ~adj
    np.fill_diagonal(comp, False)
    for a in range(n):
        for b in range(a + 1, n):
            if not comp[a, b]:
                continue
            common = np.where(comp[a] & comp[b])[0]
            common = common[common > b]
            for i in range(len(common)):
                c = common[i]
                for j in range(i + 1, len(common)):
                    d = common[j]
                    if not comp[c, d]:
                        continue
                    common2 = np.where(comp[a] & comp[b] & comp[c] & comp[d])[0]
                    common2 = common2[common2 > d]
                    for k in range(len(common2)):
                        e = common2[k]
                        common3 = np.where(comp[a] & comp[b] & comp[c] & comp[d] & comp[e])[0]
                        common3 = common3[common3 > e]
                        if len(common3) > 0:
                            f = common3[0]
                            return (a, b, c, d, e, f)
    return None

def _anneal(target_n, rng):
    """Simulated annealing to find valid K4-free graph with no independent 6-set."""
    n = target_n
    # Initialize with random graph, density ~0.4
    p = 0.4
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i, j] = True
                adj[j, i] = True
    
    # Initial temperature
    T = 10.0
    T_min = 0.01
    cooling = 0.999
    
    best_energy = float('inf')
    best_adj = adj.copy()
    
    # Get initial energy
    k4_count = _count_k4(adj)
    ind6_count = _count_ind6(adj)
    energy = k4_count + ind6_count * 2  # weight independent sets more
    
    if energy == 0:
        return _extract_edges(adj)
    
    best_energy = energy
    best_adj = adj.copy()
    
    # Annealing loop
    while _time_left() > 2:
        T *= cooling
        if T < T_min:
            break
        
        # Randomly pick an edge to flip
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        
        old_val = adj[i, j]
        
        # Compute energy change approximately
        # Flip the edge
        adj[i, j] = not old_val
        adj[j, i] = not old_val
        
        # Quick check: count violations
        k4_count = _count_k4(adj)
        ind6_count = _count_ind6(adj)
        new_energy = k4_count + ind6_count * 2
        
        delta = new_energy - energy
        
        # Accept if better or with probability exp(-delta/T)
        if delta < 0 or rng.random() < np.exp(-delta / T):
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_adj = adj.copy()
                if energy == 0:
                    break
        else:
            # Reject: flip back
            adj[i, j] = old_val
            adj[j, i] = old_val
    
    # If we found a valid graph, return it
    if best_energy == 0:
        return _extract_edges(best_adj)
    
    # Try to repair the best found graph
    adj = best_adj.copy()
    for _ in range(500):
        if _time_left() < 1:
            break
        k4 = _find_k4(adj)
        if k4 is not None:
            a, b, c, d = k4
            # Remove a random edge from the K4
            pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            pi, pj = pairs[rng.randrange(6)]
            adj[pi, pj] = False
            adj[pj, pi] = False
            continue
        ind6 = _find_ind6(adj)
        if ind6 is not None:
            a, b, c, d, e, f = ind6
            pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),
                     (c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
            pi, pj = pairs[rng.randrange(15)]
            adj[pi, pj] = True
            adj[pj, pi] = True
            continue
        # Valid!
        return _extract_edges(adj)
    
    # Check if current is valid
    if _count_k4(adj) == 0 and _count_ind6(adj) == 0:
        return _extract_edges(adj)
    
    # Return best effort even if not fully valid (will be caught by evaluator)
    # Actually, we should only return valid graphs
    # Try one more repair pass
    for _ in range(200):
        if _time_left() < 1:
            break
        k4 = _find_k4(adj)
        if k4 is not None:
            a, b, c, d = k4
            pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
            pi, pj = pairs[rng.randrange(6)]
            adj[pi, pj] = False
            adj[pj, pi] = False
            continue
        ind6 = _find_ind6(adj)
        if ind6 is not None:
            a, b, c, d, e, f = ind6
            pairs = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),
                     (c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
            pi, pj = pairs[rng.randrange(15)]
            adj[pi, pj] = True
            adj[pj, pi] = True
            continue
        return _extract_edges(adj)
    
    # If we can't make it valid, return a smaller known-valid graph
    # Use Paley 17 as fallback
    n2 = 17
    squares = {x * x % 17 for x in range(1, 17)}
    edges = [[i, j] for i in range(n2) for j in range(i + 1, n2) if (j - i) % 17 in squares]
    return n2, edges

def _extract_edges(adj):
    """Extract edge list from adjacency matrix."""
    n = adj.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                edges.append([i, j])
    return n, edges
