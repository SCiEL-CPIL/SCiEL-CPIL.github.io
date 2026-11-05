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
            # We want shortest non-backtracking cycle through s
            # Actually, we want the shortest cycle in the whole graph.
            # Use BFS on the directed line graph (non-backtracking edges).
            # State is (node, prev_node). dist from (s, -1) which is "start".
            dist = {}
            # Initialize: from s, we can go to any neighbor.
            queue = deque()
            for nb in adj[s]:
                dist[(s, nb)] = 1
                queue.append((s, nb))
            # We are looking for a non-backtracking walk that returns to s.
            # A state (u, v) means we are at v, came from u.
            # We found a cycle if we reach a state (v, s) with dist >= 3.
            # Actually, we start at s with no "previous". We move to neighbor nb.
            # State (s, nb) at dist 1.
            # From (s, nb), we can go to any nnb in adj[nb] except s.
            # State (nb, nnb) at dist 2.
            # If nnb == s, we have a cycle of length 2? No, simple graph, no 2-cycles.
            # If we reach (v, s), the cycle length is dist[(v, s)].
            # But we must ensure it's a valid cycle (length >= 3).
            
            while queue:
                u, v = queue.popleft()
                d = dist[(u, v)]
                if d >= g:
                    continue
                for w in adj[v]:
                    if w == u:
                        continue
                    if w == s:
                        # Found a cycle of length d + 1
                        if d + 1 < g:
                            return d + 1
                        continue
                    if (v, w) not in dist:
                        dist[(v, w)] = d + 1
                        queue.append((v, w))
            # If we finished BFS from s without finding a cycle < g, continue
        return g  # or girth

    # Actually, the above BFS from each s finds the shortest cycle *through* s.
    # The global girth is the min over all s.
    # But the BFS above stops at the first cycle found from s? No, it continues.
    # But if we find ANY cycle < g from ANY s, we return it.
    # This is correct.

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
        # DFS
        # state: (node, prev, length, vec)
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
                    # Check non-zero
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

    # --- Voltage assignment solver ---
    # We want v in F_m^15 such that C @ v != 0 (mod m) for all rows.
    # Equivalent: for each row r, sum_j C[r,j] v[j] != 0 (mod m).
    
    def solve_voltage(m, max_iters, time_limit):
        # Random restart
        v = np.random.randint(1, m, size=15).astype(np.int64)
        
        def get_cost(v):
            # Number of violated constraints
            residues = np.mod(C @ v, m)
            return np.sum(residues == 0)
        
        best_v = v.copy()
        best_cost = get_cost(v)
        
        if best_cost == 0:
            return v, 0
        
        # Simulated annealing
        T = 1.0
        cooling = 0.9995
        min_T = 0.001
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            j = random.randint(0, 14)
            old_val = v[j]
            new_val = random.randint(1, m - 1)
            if new_val == old_val:
                continue
            v[j] = new_val
            new_cost = get_cost(v)
            delta = new_cost - best_cost
            if delta < 0 or (T > min_T and random.random() < np.exp(-delta / T)):
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_v = v.copy()
                    if new_cost == 0:
                        return v, 0
            else:
                v[j] = old_val
            T *= cooling
        
        return best_v, best_cost

    # --- Main search ---
    min_m = 53  # Smallest prime >= 52 (Moore bound 512 / 10)
    current_m = min_m
    best_result = None
    
    while current_m < 2000 and remaining() > 10:
        if is_prime(current_m):
            attempts = 0
            max_attempts = 30
            while attempts < max_attempts and remaining() > 10:
                v, cost = solve_voltage(current_m, max_iters=3000, time_limit=deadline - 5)
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
                    gc = verify_girth(n, edges, g, deadline - 3)
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
