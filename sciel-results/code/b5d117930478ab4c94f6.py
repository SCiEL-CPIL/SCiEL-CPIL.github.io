import time
import random
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # --- Girth verification via non-backtracking walks from every vertex ---
    def verify_girth(n, edges, g, time_limit):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != k:
                return -1
            for a in adj[i]:
                if a == i:
                    return -1
        # check no repeated edges
        seen = set()
        for u, v in edges:
            if u > v:
                u, v = v, u
            if (u, v) in seen:
                return -1
            seen.add((u, v))

        # Non-backtracking BFS from each vertex
        for s in range(n):
            if time.time() > time_limit:
                return -1
            # state: (node, prev)  -> dist
            dist = {}
            queue = deque()
            for nb in adj[s]:
                dist[(s, nb)] = 1
                queue.append((s, nb))
            
            while queue:
                u, v = queue.popleft()
                d = dist[(u, v)]
                if d >= g:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == s:
                        if d + 1 < g:
                            return d + 1
                        continue
                    if (v, w) not in dist:
                        dist[(v, w)] = d + 1
                        queue.append((v, w))
        return g

    # --- Petersen graph data ---
    pet_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
    ]
    n_base = 10
    edge_index = {}
    for i, (u, v) in enumerate(pet_edges):
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    adj_base = [[] for _ in range(n_base)]
    for u, v in pet_edges:
        adj_base[u].append(v)
        adj_base[v].append(u)

    # --- Generate constraint matrix C ---
    walk_vectors = []
    max_L = g - 1
    
    for start_v in range(n_base):
        stack = [(start_v, -1, 0, [0] * 15)]
        while stack:
            node, prev, length, vec = stack.pop()
            if length > max_L:
                continue
            for nb in adj_base[node]:
                if nb == prev:
                    continue
                e_idx = edge_index[(node, nb)]
                u, v = pet_edges[e_idx]
                d = 1 if node == u else -1
                new_vec = vec.copy()
                new_vec[e_idx] += d
                new_l = length + 1
                if nb == start_v and new_l >= 3:
                    if any(x != 0 for x in new_vec):
                        walk_vectors.append(tuple(new_vec))
                if new_l < max_L:
                    stack.append((nb, node, new_l, new_vec))

    if not walk_vectors:
        return (0, [])

    unique_constraints = set()
    for c in walk_vectors:
        first_nz = -1
        for i in range(15):
            if c[i] != 0:
                first_nz = i
                break
        if first_nz != -1:
            if c[first_nz] < 0:
                c = tuple(-x for x in c)
            unique_constraints.add(c)

    C = np.array(list(unique_constraints), dtype=np.int64)
    if C.size == 0:
        return (0, [])

    def is_prime(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    # --- Voltage assignment solver with improved optimization ---
    def solve_voltage(m, max_iters, time_limit):
        def get_cost(v):
            residues = np.mod(C @ v, m)
            return np.sum(residues == 0)
        
        def get_soft_cost(v, beta=1.0):
            """Soft cost using log-sum-exp for smooth optimization"""
            residues = np.mod(C @ v, m).astype(np.float64)
            # Penalize zeros more heavily, use smooth approximation
            penalty = np.exp(-beta * (residues + (m - residues)) / m)
            return np.sum(penalty)
        
        # Multiple restarts with different strategies
        best_v = None
        best_cost = float('inf')
        
        for restart in range(5):
            if time.time() > time_limit:
                break
                
            if restart == 0:
                # Random initialization
                v = np.random.randint(1, m, size=15).astype(np.int64)
            elif restart == 1:
                # Greedy initialization
                v = np.zeros(15, dtype=np.int64)
                for j in range(15):
                    best_val = 1
                    best_c = float('inf')
                    for val in range(1, m):
                        v[j] = val
                        c = get_cost(v)
                        if c < best_c:
                            best_c = c
                            best_val = val
                    v[j] = best_val
            elif restart == 2:
                # Local search from random
                v = np.random.randint(1, m, size=15).astype(np.int64)
                # Quick local improvement
                improved = True
                while improved and time.time() < time_limit:
                    improved = False
                    for j in range(15):
                        old_val = v[j]
                        old_cost = get_cost(v)
                        for new_val in range(1, m):
                            if new_val == old_val:
                                continue
                            v[j] = new_val
                            new_cost = get_cost(v)
                            if new_cost < old_cost:
                                old_val = new_val
                                old_cost = new_cost
                                improved = True
                        v[j] = old_val
            else:
                v = np.random.randint(1, m, size=15).astype(np.int64)
            
            cur_cost = get_cost(v)
            if cur_cost < best_cost:
                best_cost = cur_cost
                best_v = v.copy()
            
            if best_cost == 0:
                return v, 0
            
            # Simulated annealing with adaptive temperature
            T = max(0.5, best_cost / 10.0)
            cooling = 0.999
            min_T = 0.01
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                    
                # Adaptive move selection
                if random.random() < 0.3:
                    # Multi-move perturbation
                    for _ in range(random.randint(2, 4)):
                        j = random.randint(0, 14)
                        v[j] = random.randint(1, m - 1)
                else:
                    j = random.randint(0, 14)
                    v[j] = random.randint(1, m - 1)
                
                new_cost = get_cost(v)
                delta = new_cost - best_cost
                
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        if new_cost == 0:
                            return v, 0
                else:
                    # Revert
                    if random.random() < 0.3:
                        v = best_v.copy()
                    else:
                        # Keep the bad move for exploration
                        pass
                
                T *= cooling
        
        return best_v, best_cost

    # --- Tree-closure fallback construction ---
    def tree_closure_construction(g, time_limit):
        """Build non-backtracking tree of depth (g-1)//2 and match boundary vertices"""
        depth = (g - 1) // 2
        if depth < 2:
            return None
        
        # Build tree
        nodes = [0]
        node_id = 1
        adj = {0: []}
        
        for d in range(depth):
            new_nodes = []
            for node in nodes:
                # Each internal node has 2 children (one edge used to connect to parent)
                for _ in range(2):
                    adj[node_id] = []
                    adj[node].append(node_id)
                    adj[node_id].append(node)
                    new_nodes.append(node_id)
                    node_id += 1
            nodes = new_nodes
        
        boundary = nodes
        n_boundary = len(boundary)
        
        if n_boundary < 2 or n_boundary % 2 != 0:
            return None
        
        # Greedy randomized matching
        random.shuffle(boundary)
        used = set()
        edges = set()
        
        # Add tree edges
        for u in adj:
            for v in adj[u]:
                if u < v:
                    edges.add((u, v))
        
        # Match boundary vertices
        for i in range(0, n_boundary, 2):
            a, b = boundary[i], boundary[i + 1]
            if a not in used and b not in used:
                # Check if this creates a short cycle
                # For now, just add the edge
                edges.add((a, b))
                used.add(a)
                used.add(b)
        
        # Verify girth
        n = node_id
        edge_list = list(edges)
        gc = verify_girth(n, edge_list, g, time_limit)
        
        if gc >= g:
            return (n, edge_list)
        
        return None

    # --- Main search ---
    min_m = 53
    current_m = min_m
    best_result = None
    
    # Try voltage lifts first
    while current_m < 2000 and remaining() > 15:
        if is_prime(current_m):
            attempts = 0
            max_attempts = 20
            while attempts < max_attempts and remaining() > 15:
                v, cost = solve_voltage(current_m, max_iters=2000, time_limit=deadline - 10)
                if cost == 0:
                    n = 10 * current_m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(current_m):
                            u_lift = u * current_m + layer
                            v_lift = v_base * current_m + (layer + vol) % current_m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))
                    gc = verify_girth(n, edges, g, deadline - 5)
                    if gc >= g:
                        return (n, edges)
                    else:
                        break
                attempts += 1
        current_m += 1
    
    # Fallback to tree closure if time remains
    if remaining() > 20:
        result = tree_closure_construction(g, deadline - 5)
        if result is not None:
            return result
    
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
