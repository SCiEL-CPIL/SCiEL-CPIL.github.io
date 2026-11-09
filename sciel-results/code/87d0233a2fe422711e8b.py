import time
import random
import numpy as np

_DEADLINE = 150.0

def _check_c6_exact(matrices, n):
    """Returns True if no color class contains a C6."""
    for c in range(4):
        M = matrices[c]
        # Get adjacency list
        adj = [[] for _ in range(n)]
        for i in range(n):
            # M[i] is a vector of 0s and 1s
            neighbors = np.nonzero(M[i])[0]
            adj[i] = neighbors.tolist()
            
        # Check for C6
        # For each edge (u,v), check if there is a path of length 4 from u to v not using edge (u,v) directly?
        # No, C6 is u-v-w-x-y-z-u.
        # Iterate over all pairs (u,v) with u<v.
        # If (u,v) is in G_c, check if there is a path of length 5 from u to v in G_c \ {(u,v)}?
        # No, path of length 5 from u to v means u-w-x-y-z-v.
        # This forms a cycle u-v-z-y-x-w-u? No.
        # Cycle: u-v-z-y-x-w-u. Edges: (u,v), (v,z), (z,y), (y,x), (x,w), (w,u).
        # So if we have edge (u,v), we need a path of length 5 from u to v in the graph with (u,v) removed.
        
        # This is expensive.
        # Alternative: Iterate over all 6-tuples? No.
        
        # Let's use the property: C6 exists iff there are 3 disjoint edges forming a "ladder"? No.
        
        # Simple DFS from each node to depth 6.
        for s in range(n):
            # DFS from s, trying to close a cycle of length 6
            # Stack: (current, start, depth, visited)
            stack = [(s, s, 0, {s})]
            while stack:
                u, start, d, vis = stack.pop()
                if d == 5:
                    if start in adj[u]:
                        return False # Found C6
                    continue
                for v in adj[u]:
                    if v == start:
                        if d == 5:
                            return False
                        continue
                    if v in vis:
                        continue
                    stack.append((v, start, d + 1, vis | {v}))
    return True

def _count_c6s(matrices, n):
    total = 0
    for c in range(4):
        M = matrices[c]
        M2 = M @ M
        M3 = M2 @ M
        M6 = M3 @ M3
        total += np.trace(M6)
    return total

def _solve_n(n, t_end):
    if n < 1:
        return None
    
    matrices = [np.zeros((n, n), dtype=np.int32) for _ in range(4)]
    edge_colors = {}
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, 3)
            edge_colors[(i,j)] = c
            matrices[c][i, j] = 1
            matrices[c][j, i] = 1
            
    current_energy = _count_c6s(matrices, n)
    if current_energy == 0:
        return edge_colors
        
    # Check exact C6 if energy is low
    if current_energy < 1000 and _check_c6_exact(matrices, n):
        return edge_colors

    T = 1.0
    T_min = 0.01
    cooling = 0.999
    best_energy = current_energy
    best_colors = {k: v for k, v in edge_colors.items()}
    
    edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    t_start = time.time()
    steps = 0
    last_check = 0
    
    while time.time() < t_end - 1.0:
        idx = random.randint(0, len(edges) - 1)
        u, v = edges[idx]
        old_c = edge_colors[(u,v)]
        new_c = random.randint(0, 3)
        if new_c == old_c:
            continue
            
        matrices[old_c][u, v] = 0
        matrices[old_c][v, u] = 0
        matrices[new_c][u, v] = 1
        matrices[new_c][v, u] = 1
        
        new_energy = _count_c6s(matrices, n)
        
        delta = new_energy - current_energy
        if delta < 0 or random.random() < np.exp(-delta / T):
            current_energy = new_energy
            edge_colors[(u,v)] = new_c
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = {k: v for k, v in edge_colors.items()}
                if best_energy == 0:
                    return best_colors
        else:
            matrices[old_c][u, v] = 1
            matrices[old_c][v, u] = 1
            matrices[new_c][u, v] = 0
            matrices[new_c][v, u] = 0
            
        T *= cooling
        if T < T_min:
            T = T_min
            
        steps += 1
        if steps - last_check > 500:
            last_check = steps
            if current_energy < 1000:
                if _check_c6_exact(matrices, n):
                    return edge_colors

    if best_energy == 0:
        return best_colors
    if best_energy < 1000 and _check_c6_exact(matrices, n):
        return best_colors
    return None

def construct():
    t0 = time.time()
    t_end = t0 + _DEADLINE - 5.0
    
    for n in [19, 18, 17, 16]:
        if time.time() > t_end:
            break
        res = _solve_n(n, t_end)
        if res is not None:
            colors = []
            for i in range(n):
                for j in range(i+1, n):
                    colors.append(res[(i,j)])
            return {'n': n, 'colors': colors}
            
    return {'n': 1, 'colors': []}
