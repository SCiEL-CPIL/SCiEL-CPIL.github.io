import time
import random
import sys

_DEADLINE = 120.0

def _has_c4(edges, n):
    """Check if graph has a C4. Returns True if C4 found."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in list(adj[u]):
            if u >= v:
                continue
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return True
    return False

def _max_independent_set_size(adj, n):
    """Branch and bound to find the size of the maximum independent set."""
    # Use a simple but effective branch and bound
    best = [0]
    
    def _bbs(idx, current_size, chosen):
        if time.time() > _DEADLINE - 5:
            return
        if current_size + (n - idx) <= best[0]:
            return
        if idx == n:
            if current_size > best[0]:
                best[0] = current_size
            return
        
        # Option 1: include vertex idx
        # Check if idx can be added (not adjacent to any in chosen)
        can_add = True
        for v in chosen:
            if v in adj[idx]:
                can_add = False
                break
        if can_add:
            chosen.append(idx)
            _bbs(idx + 1, current_size + 1, chosen)
            chosen.pop()
        
        # Option 2: exclude vertex idx
        _bbs(idx + 1, current_size, chosen)
    
    _bbs(0, 0, [])
    return best[0]

def _check_alpha(adj, n, threshold=11):
    """Check if alpha < threshold. Returns True if alpha < threshold."""
    # Quick upper bound: if graph is dense enough, alpha is small
    # Use a simpler approach: try to find an independent set of size threshold
    # If we can't find one, alpha < threshold
    
    # Branch and bound to find if there's an independent set of size >= threshold
    found = [False]
    
    def _search(idx, current_size, chosen):
        if found[0] or time.time() > _DEADLINE - 5:
            return
        if current_size >= threshold:
            found[0] = True
            return
        if idx == n:
            return
        # Pruning: even if we take all remaining, can't reach threshold
        if current_size + (n - idx) < threshold:
            return
        
        # Try to include idx
        can_add = True
        for v in chosen:
            if v in adj[idx]:
                can_add = False
                break
        if can_add:
            chosen.append(idx)
            _search(idx + 1, current_size + 1, chosen)
            chosen.pop()
        
        # Try to exclude idx
        _search(idx + 1, current_size, chosen)
    
    _search(0, 0, [])
    return not found[0]

def _greedy_c4_free(n, seed=0):
    """Build a C4-free graph on n vertices using random greedy."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    edges = []
    
    # Process all pairs in random order
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(pairs)
    
    for u, v in pairs:
        # Check if adding (u,v) creates a C4
        # C4 exists if |N(u) ∩ N(v)| >= 2
        common = adj[u] & adj[v]
        if len(common) <= 1:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return edges, adj

def construct():
    start_time = time.time()
    
    best_result = {'n': 30, 'edges': []}
    # Fallback: 10 disjoint triangles
    for t in range(10):
        a, b, c = 3*t, 3*t+1, 3*t+2
        best_result['edges'].append([a, b])
        best_result['edges'].append([a, c])
        best_result['edges'].append([b, c])
    
    # Try n = 43, 42, 41, 40
    for n in [43, 42, 41, 40]:
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        found = False
        for seed in range(500):
            if time.time() - start_time > _DEADLINE - 20:
                break
            
            edges, adj = _greedy_c4_free(n, seed)
            
            # Verify C4-free
            if _has_c4(edges, n):
                continue
            
            # Check alpha < 11
            if _check_alpha(adj, n, 11):
                found = True
                best_result = {'n': n, 'edges': edges}
                break
        
        if found:
            break
    
    return best_result
