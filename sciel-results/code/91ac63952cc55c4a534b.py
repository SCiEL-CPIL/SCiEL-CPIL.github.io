import random
import time
import sys
from collections import deque
import numpy as np

def construct(k, g):
    # If k or g are not 4 and 10, fallback to standard construction
    if k != 4 or g != 10:
        return _fallback_construct(k, g)
    
    deadline = time.time() + 115  # 115s limit to be safe (150s total, leave 35s margin)
    
    # Attempt to construct the Moore graph (n=242)
    try:
        n, edges = _construct_moore_candidate(deadline)
        if n is not None:
            # Verify girth
            if _verify_girth(n, edges, g):
                return n, edges
    except Exception:
        pass
    
    # Fallback to original Lazebnik-Ustimenko-Woldar construction
    return _fallback_construct(k, g)

def _construct_moore_candidate(deadline):
    # Build two trees of depth 4, rooted at 0 and 1
    edges = []
    edges.append((0, 1))
    
    # Left Tree
    l1 = [2, 3, 4]
    for node in l1: edges.append((0, node))
    
    l2 = []
    idx = 5
    for node in l1:
        for _ in range(3):
            edges.append((node, idx))
            l2.append(idx)
            idx += 1
            
    l3 = []
    for node in l2:
        for _ in range(3):
            edges.append((node, idx))
            l3.append(idx)
            idx += 1
            
    l4 = []
    for node in l3:
        for _ in range(3):
            edges.append((node, idx))
            l4.append(idx)
            idx += 1
            
    # Right Tree
    r1 = [121, 122, 123]
    for node in r1: edges.append((1, node))
    
    r2 = []
    idx = 124
    for node in r1:
        for _ in range(3):
            edges.append((node, idx))
            r2.append(idx)
            idx += 1
            
    r3 = []
    for node in r2:
        for _ in range(3):
            edges.append((node, idx))
            r3.append(idx)
            idx += 1
            
    r4 = []
    for node in r3:
        for _ in range(3):
            edges.append((node, idx))
            r4.append(idx)
            idx += 1
    
    # Build tree data structures
    def build_tree_data(root, level_lists):
        parents = {}
        depth = {root: 0}
        current_level = [root]
        for level_nodes in level_lists:
            next_level = []
            for i, node in enumerate(level_nodes):
                par = current_level[i // 3]
                parents[node] = par
                depth[node] = depth[par] + 1
                next_level.append(node)
            current_level = next_level
        return parents, depth
    
    parents_l, depth_l = build_tree_data(0, [l1, l2, l3, l4])
    parents_r, depth_r = build_tree_data(1, [r1, r2, r3, r4])
    
    # Build LCA with binary lifting
    def build_lca(parents, depth, nodes):
        n_nodes = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        max_depth = max(depth.values())
        log_max = max_depth.bit_length()
        
        up = [[-1] * n_nodes for _ in range(log_max)]
        
        for node in nodes:
            idx = node_to_idx[node]
            if node in parents:
                up[0][idx] = node_to_idx[parents[node]]
        
        for j in range(1, log_max):
            for i in range(n_nodes):
                if up[j-1][i] != -1:
                    up[j][i] = up[j-1][up[j-1][i]]
        
        def lca(u, v):
            u_idx = node_to_idx[u]
            v_idx = node_to_idx[v]
            
            if depth[u] < depth[v]:
                u_idx, v_idx = v_idx, u_idx
            
            diff = depth[u] - depth[v]
            for j in range(log_max):
                if diff & (1 << j):
                    u_idx = up[j][u_idx]
            
            if u_idx == v_idx:
                return nodes[u_idx]
            
            for j in range(log_max - 1, -1, -1):
                if up[j][u_idx] != up[j][v_idx]:
                    u_idx = up[j][u_idx]
                    v_idx = up[j][v_idx]
            
            return nodes[up[0][u_idx]]
        
        return lca
    
    left_nodes = [0] + l1 + l2 + l3 + l4
    right_nodes = [1] + r1 + r2 + r3 + r4
    lca_l = build_lca(parents_l, depth_l, left_nodes)
    lca_r = build_lca(parents_r, depth_r, right_nodes)
    
    N = 81
    D_L = np.zeros((N, N), dtype=np.int32)
    D_R = np.zeros((N, N), dtype=np.int32)
    
    for i in range(N):
        for j in range(i + 1, N):
            u = l4[i]
            v = l4[j]
            l = lca_l(u, v)
            d = depth_l[u] + depth_l[v] - 2 * depth_l[l]
            D_L[i, j] = d
            D_L[j, i] = d
            
            u = r4[i]
            v = r4[j]
            l = lca_r(u, v)
            d = depth_r[u] + depth_r[v] - 2 * depth_r[l]
            D_R[i, j] = d
            D_R[j, i] = d
            
    # Find Permutation sigma
    # We want min_{i!=j} D_L[i,j] + D_R[sigma[i], sigma[j]] >= 8
    
    best_sigma = None
    best_min_sum = -1
    
    def compute_min_sum_fast(sigma):
        """Compute min sum using numpy vectorization."""
        sigma_arr = np.array(sigma)
        min_s = np.inf
        for i in range(N):
            sums = D_L[i, i+1:] + D_R[sigma[i], sigma_arr[i+1:]]
            s = np.min(sums)
            if s < min_s:
                min_s = s
        return int(min_s)
    
    for restart in range(500):
        if time.time() > deadline - 10:
            break
            
        if restart == 0:
            sigma = list(range(N))
        elif restart == 1:
            sigma = list(range(N-1, -1, -1))
        else:
            sigma = list(range(N))
            random.shuffle(sigma)
        
        current_min = compute_min_sum_fast(sigma)
        
        if current_min >= 8:
            if current_min > best_min_sum:
                best_min_sum = current_min
                best_sigma = sigma[:]
            return 242, edges + [(l4[i], r4[sigma[i]]) for i in range(N)]
        
        # Local search
        for step in range(200):
            if time.time() > deadline - 10:
                break
                
            # Find worst pair
            min_sum = float('inf')
            min_i, min_j = 0, 1
            sigma_arr = np.array(sigma)
            for i in range(N):
                sums = D_L[i, i+1:] + D_R[sigma[i], sigma_arr[i+1:]]
                s = np.min(sums)
                if s < min_sum:
                    min_sum = s
                    min_idx = np.argmin(sums)
                    min_j = i + 1 + min_idx
            
            if min_sum >= 8:
                if min_sum > best_min_sum:
                    best_min_sum = min_sum
                    best_sigma = sigma[:]
                return 242, edges + [(l4[i], r4[sigma[i]]) for i in range(N)]
            
            # Try swaps to fix worst pair
            candidates = set()
            candidates.add(min_j)
            if min_i > 0: candidates.add(min_i - 1)
            if min_i < N-1: candidates.add(min_i + 1)
            if min_j > 0: candidates.add(min_j - 1)
            if min_j < N-1: candidates.add(min_j + 1)
            for _ in range(10):
                candidates.add(random.randint(0, N-1))
            
            best_k = -1
            best_new_min = min_sum - 1
            
            for k in candidates:
                if k == min_i:
                    continue
                
                sigma[min_i], sigma[k] = sigma[k], sigma[min_i]
                new_min = compute_min_sum_fast(sigma)
                
                if new_min > best_new_min:
                    best_new_min = new_min
                    best_k = k
                else:
                    sigma[min_i], sigma[k] = sigma[k], sigma[min_i]
            
            if best_k >= 0 and best_new_min > current_min:
                sigma[min_i], sigma[best_k] = sigma[best_k], sigma[min_i]
                current_min = best_new_min
            else:
                a, b = random.sample(range(N), 2)
                sigma[a], sigma[b] = sigma[b], sigma[a]
                current_min = compute_min_sum_fast(sigma)
                
    if best_sigma is not None and best_min_sum >= 8:
        return 242, edges + [(l4[i], r4[best_sigma[i]]) for i in range(N)]
        
    return None, None

def _verify_girth(n, edges, g):
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Check degree
    for i in range(n):
        if len(adj[i]) != 4:
            return False
            
    # Check simple graph
    edge_set = set()
    for u, v in edges:
        if u == v:
            return False
        if (min(u,v), max(u,v)) in edge_set:
            return False
        edge_set.add((min(u,v), max(u,v)))
    
    # Girth check: verify no cycle shorter than g
    for start in range(n):
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        queue = deque([start])
        
        while queue:
            u = queue.popleft()
            if dist[u] >= g:
                continue
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    queue.append(v)
                elif v != parent[u] and parent[v] != u:
                    cycle_len = dist[u] + dist[v] + 1
                    if cycle_len < g:
                        return False
    
    return True

def _fallback_construct(k, g):
    # Original program logic
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
