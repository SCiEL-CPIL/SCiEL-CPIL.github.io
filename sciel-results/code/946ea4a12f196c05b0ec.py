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
    # C @ v = 0 (mod m) for all non-backtracking closed walks of length < g
    # We want C @ v != 0 (mod m) for all such walks.
    
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

    # Normalize: make first non-zero element positive
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

    C_list = list(unique_constraints)
    C = np.array(C_list, dtype=np.int64)
    n_constraints = len(C_list)
    if n_constraints == 0:
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

    # --- Voltage assignment solver ---
    # Precompute column contributions for fast incremental cost updates
    
    def solve_voltage(m, max_iters, time_limit):
        # Precompute for each constraint row and each column the coefficient
        # This allows O(1) cost update when changing one variable
        
        # C is n_constraints x 15
        # For each row r, cost contribution is 1 if sum_j C[r,j]*v[j] == 0 mod m
        
        # Random restart with multiple initializations
        best_v = None
        best_cost = float('inf')
        
        for restart in range(5):
            if time.time() > time_limit:
                break
            
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            # Compute initial cost
            residues = np.mod(C @ v, m)
            cost = int(np.sum(residues == 0))
            
            if cost < best_cost:
                best_cost = cost
                best_v = v.copy()
            
            if cost == 0:
                return v, 0
            
            # Simulated annealing
            T = 1.0
            cooling = 0.9998
            min_T = 0.005
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    break
                
                j = random.randint(0, 14)
                old_val = int(v[j])
                new_val = random.randint(1, m - 1)
                if new_val == old_val:
                    continue
                
                # Delta cost: for each row r, check if C[r,j]*(new_val - old_val) changes residue from/to 0
                delta_j = (new_val - old_val) % m
                # New residues = old_residues + C[:,j] * delta_j (mod m)
                new_residues = (residues + C[:, j] * delta_j) % m
                new_cost = int(np.sum(new_residues == 0))
                
                delta = new_cost - cost
                if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                    v[j] = new_val
                    residues = new_residues
                    cost = new_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_v = v.copy()
                        if cost == 0:
                            return v, 0
                else:
                    pass  # reject, keep old state
                T *= cooling
        
        return best_v, best_cost

    # --- Main search ---
    min_m = 53  # Smallest prime >= 52 (Moore bound 512 / 10)
    current_m = min_m
    best_result = None
    
    while current_m < 2000 and remaining() > 15:
        if is_prime(current_m):
            attempts = 0
            max_attempts = 20
            while attempts < max_attempts and remaining() > 15:
                v, cost = solve_voltage(current_m, max_iters=5000, time_limit=deadline - 10)
                if cost == 0:
                    n = 10 * current_m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = int(v[e_idx])
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
    
    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
