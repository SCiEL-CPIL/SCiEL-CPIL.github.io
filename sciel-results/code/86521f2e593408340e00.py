import time
import random
import numpy as np

def _find_k222_fast(adj, n):
    """
    Fast detection of K2,2,2 in adjacency matrix adj (0/1 matrix, symmetric).
    Returns (u, v, w, x, y, z) or None.
    """
    # Precompute neighbor lists for speed
    neighbors = []
    for i in range(n):
        neighbors.append(np.where(adj[i] == 1)[0])
    
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] != 1:
                continue
            # Common neighbors of u and v
            nu = set(neighbors[u])
            nv = set(neighbors[v])
            S = list(nu & nv)
            S = [x for x in S if x != u and x != v]
            
            if len(S) < 4:
                continue
            
            S_set = set(S)
            # For each pair (w, x) in S, check if they have >= 2 common neighbors in S
            for i in range(len(S)):
                w = S[i]
                nw = set(neighbors[w])
                # Common neighbors of w and x that are in S
                for j in range(i + 1, len(S)):
                    x = S[j]
                    nx = set(neighbors[x])
                    common_wx = nw & nx & S_set
                    common_wx.discard(w)
                    common_wx.discard(x)
                    if len(common_wx) >= 2:
                        # Pick any two
                        candidates = list(common_wx)
                        y = candidates[0]
                        z = candidates[1]
                        return (u, v, w, x, y, z)
    return None

def _count_k222_fast(adj, n):
    """
    Count K2,2,2 subgraphs in adjacency matrix.
    """
    neighbors = []
    for i in range(n):
        neighbors.append(np.where(adj[i] == 1)[0])
    
    count = 0
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] != 1:
                continue
            nu = set(neighbors[u])
            nv = set(neighbors[v])
            S = list(nu & nv)
            S = [x for x in S if x != u and x != v]
            
            if len(S) < 4:
                continue
            
            S_set = set(S)
            for i in range(len(S)):
                w = S[i]
                nw = set(neighbors[w])
                for j in range(i + 1, len(S)):
                    x = S[j]
                    nx = set(neighbors[x])
                    common_wx = nw & nx & S_set
                    common_wx.discard(w)
                    common_wx.discard(x)
                    c = len(common_wx)
                    if c >= 2:
                        count += c * (c - 1) // 2
    return count

def _is_valid_adj(adj0, adj1, n):
    """Check validity using pre-built adjacency matrices."""
    if _find_k222_fast(adj0, n) is not None:
        return False
    if _find_k222_fast(adj1, n) is not None:
        return False
    return True

def _colors_to_adj(colors, n):
    adj0 = np.zeros((n, n), dtype=np.int8)
    adj1 = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i, j] = 1
                adj0[j, i] = 1
            else:
                adj1[i, j] = 1
                adj1[j, i] = 1
            idx += 1
    return adj0, adj1

def _adj_to_colors(adj0, adj1, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj0[i, j] == 1:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def _flip_edge(adj0, adj1, i, j):
    """Flip the color of edge (i,j) between adj0 and adj1."""
    if adj0[i, j] == 1:
        adj0[i, j] = 0
        adj0[j, i] = 0
        adj1[i, j] = 1
        adj1[j, i] = 1
    else:
        adj0[i, j] = 1
        adj0[j, i] = 1
        adj1[i, j] = 0
        adj1[j, i] = 0

def _paley_colors(n):
    """Generate Paley graph coloring for prime n = 4k+1."""
    if n <= 1 or n > 100:
        return None
    is_prime = True
    if n < 2:
        is_prime = False
    elif n % 2 == 0:
        is_prime = False
    else:
        for p in range(3, int(n**0.5) + 1, 2):
            if n % p == 0:
                is_prime = False
                break
    if not is_prime or n % 4 != 1:
        return None
    
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    for n in range(29, 16, -1):
        if time.time() > deadline:
            break
            
        best_colors = None
        found = False
        
        max_restarts = 150
        for r in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Different initialization strategies
            if r == 0 and n % 4 == 1:
                colors = _paley_colors(n)
                if colors is None:
                    colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            elif r < 20:
                # Random with slight bias
                bias = 0.5 + 0.1 * random.gauss(0, 1)
                bias = max(0.1, min(0.9, bias))
                colors = [1 if random.random() < bias else 0 for _ in range(n * (n - 1) // 2)]
            else:
                colors = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            
            adj0, adj1 = _colors_to_adj(colors, n)
            
            # Local search with simulated annealing
            stuck = 0
            T = 1.0  # temperature
            for step in range(3000):
                if time.time() > deadline:
                    break
                    
                # Find K2,2,2 in color 0
                k222_0 = _find_k222_fast(adj0, n)
                if k222_0 is not None:
                    u, v, w, x, y, z = k222_0
                    edges = [
                        (u, w), (u, x), (u, y), (u, z),
                        (v, w), (v, x), (v, y), (v, z),
                        (w, y), (w, z),
                        (x, y), (x, z)
                    ]
                    i, j = random.choice(edges)
                    if i > j:
                        i, j = j, i
                    
                    # Count violations before flip
                    viol_before = 0
                    if _find_k222_fast(adj0, n) is not None:
                        viol_before += 1
                    if _find_k222_fast(adj1, n) is not None:
                        viol_before += 1
                    
                    _flip_edge(adj0, adj1, i, j)
                    
                    # Count violations after flip
                    viol_after = 0
                    if _find_k222_fast(adj0, n) is not None:
                        viol_after += 1
                    if _find_k222_fast(adj1, n) is not None:
                        viol_after += 1
                    
                    # Accept if better or with probability exp(-delta/T)
                    if viol_after <= viol_before:
                        stuck = 0
                    else:
                        # Simulated annealing: accept with probability
                        prob = np.exp(-(viol_after - viol_before) / T)
                        if random.random() < prob:
                            stuck = 0
                        else:
                            _flip_edge(adj0, adj1, i, j)  # revert
                            stuck += 1
                    
                    # Cool down
                    T *= 0.999
                    continue
                
                # Find K2,2,2 in color 1
                k222_1 = _find_k222_fast(adj1, n)
                if k222_1 is not None:
                    u, v, w, x, y, z = k222_1
                    edges = [
                        (u, w), (u, x), (u, y), (u, z),
                        (v, w), (v, x), (v, y), (v, z),
                        (w, y), (w, z),
                        (x, y), (x, z)
                    ]
                    i, j = random.choice(edges)
                    if i > j:
                        i, j = j, i
                    
                    viol_before = 0
                    if _find_k222_fast(adj0, n) is not None:
                        viol_before += 1
                    if _find_k222_fast(adj1, n) is not None:
                        viol_before += 1
                    
                    _flip_edge(adj0, adj1, i, j)
                    
                    viol_after = 0
                    if _find_k222_fast(adj0, n) is not None:
                        viol_after += 1
                    if _find_k222_fast(adj1, n) is not None:
                        viol_after += 1
                    
                    if viol_after <= viol_before:
                        stuck = 0
                    else:
                        prob = np.exp(-(viol_after - viol_before) / T)
                        if random.random() < prob:
                            stuck = 0
                        else:
                            _flip_edge(adj0, adj1, i, j)
                            stuck += 1
                    
                    T *= 0.999
                    continue
                
                # No K2,2,2 found - verify validity
                if _is_valid_adj(adj0, adj1, n):
                    best_colors = _adj_to_colors(adj0, adj1, n)
                    found = True
                    break
                
                break

            if found:
                break
        
        if found:
            return {'n': n, 'colors': best_colors}
            
    # Fallback to n=17 Paley
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'colors': [0 if (j - i) % 17 in squares else 1 for i in range(17) for j in range(i + 1, 17)]}
