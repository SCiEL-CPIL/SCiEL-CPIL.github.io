import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        for i in range(n):
            if len(adj[i]) != k:
                return -1
        
        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift with constraint-directed repair ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

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

        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        max_L = g - 1
        walk_vectors = set()

        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0] * 15)]
            while stack:
                current, prev, length, counts = stack.pop()

                if length >= max_L:
                    continue

                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1

                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1

                    if next_v == start_v and new_l >= 5:
                        walk_vectors.add(tuple(new_counts))

                    stack.append((next_v, current, new_l, new_counts))

        if not walk_vectors:
            return None

        unique_constraints = set()
        for c in walk_vectors:
            if all(x == 0 for x in c):
                continue
            first_non_zero = -1
            for i in range(15):
                if c[i] != 0:
                    first_non_zero = i
                    break
            if first_non_zero != -1:
                if c[first_non_zero] < 0:
                    c = tuple(-x for x in c)
                unique_constraints.add(c)

        C = np.array(list(unique_constraints), dtype=np.int64)
        if C.size == 0:
            return None

        constraint_nonzero_indices = []
        for i in range(C.shape[0]):
            nz = np.nonzero(C[i])[0]
            constraint_nonzero_indices.append(nz)

        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True

        def solve_voltage(m, max_iters=30000):
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = cost(v)
            
            if best_cost == 0:
                return v, 0

            T = 2.0
            cooling = 0.9998
            min_T = 0.005
            
            for it in range(max_iters):
                if remaining() < 5:
                    break
                
                if best_cost > 0:
                    residues = np.mod(C @ v, m)
                    violated_idx = np.nonzero(residues == 0)[0]
                    if len(violated_idx) == 0:
                        break
                    c_idx = violated_idx[random.randint(0, len(violated_idx)-1)]
                    
                    nz_indices = constraint_nonzero_indices[c_idx]
                    if len(nz_indices) == 0:
                        continue
                    j = nz_indices[random.randint(0, len(nz_indices)-1)]
                else:
                    j = random.randint(0, 14)

                old_val = v[j]
                v[j] = random.randint(1, m - 1)

                new_cost = cost(v)
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

        m = 53
        while m < 200 and remaining() > 20:
            if is_prime(m):
                found = False
                for restart in range(15):
                    if remaining() < 10:
                        break
                    
                    v, cost = solve_voltage(m, max_iters=8000)

                    if cost == 0:
                        n = 10 * m
                        edges = []
                        for u, v_base in pet_edges:
                            e_idx = edge_index[(u, v_base)]
                            vol = v[e_idx]
                            for layer in range(m):
                                u_lift = u * m + layer
                                v_lift = v_base * m + (layer + vol) % m
                                if u_lift < v_lift:
                                    edges.append((u_lift, v_lift))
                                else:
                                    edges.append((v_lift, u_lift))

                        gc = check_girth(n, edges, g, deadline - 3)
                        if gc >= g:
                            return (n, edges)

            m += 1

        return None

    def greedy_growth():
        if k != 3:
            return None

        for n in range(520, 600, 2):
            if remaining() < 10:
                break

            stubs = []
            for i in range(n):
                stubs.extend([i] * k)

            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if (u, v) in edges or (v, u) in edges:
                    valid = False
                    break
                edges.add((min(u, v), max(u, v)))

            if not valid:
                continue

            edges_list = list(edges)
            gc = check_girth(n, edges_list, g, deadline - 5)
            if gc >= g:
                return (n, edges_list)

        return None

    res = voltage_lift_petersen()
    if res is not None:
        return res

    res = greedy_growth()
    if res is not None:
        return res

    return (0, [])


if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
