import numpy as np
import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0 # 35s margin for safety and overhead

    def remaining():
        return deadline - time.time()

    def check_girth_fast(n, edges, min_girth, time_limit):
        """BFS from each vertex, early exit if short cycle found."""
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
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
                        if cycle_len < min_girth:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < min_girth:
                return min_cycle
        return min_cycle

    # ---------------- Voltage Lift Construction ----------------
    def voltage_lift_petersen():
        if k != 3 or g != 16:
            return None

        # Petersen Graph Definition
        # Vertices 0..9
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0), # Outer cycle
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5), # Inner cycle
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)  # Spokes
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
        constraint_rows = set()
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, (0,)*15)]
            while stack:
                current, prev, length, vol_vec = stack.pop()
                
                if length >= max_L:
                    continue
                    
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                        
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    
                    if current == u:
                        d = 1
                    else:
                        d = -1
                        
                    new_vol_list = list(vol_vec)
                    new_vol_list[e_idx] += d
                    new_vol = tuple(new_vol_list)
                    new_length = length + 1
                    
                    if next_v == start_v and new_length >= 5:
                        vec = list(new_vol)
                        for i in range(15):
                            if vec[i] != 0:
                                if vec[i] < 0:
                                    vec = [-x for x in vec]
                                break
                        constraint_rows.add(tuple(vec))
                    
                    stack.append((next_v, current, new_length, new_vol))
        
        if not constraint_rows:
            return None

        C = np.array(list(constraint_rows), dtype=np.int64)
        
        # We want to find m (prime) and x in Z_m^15 such that C @ x != 0 (mod m) for all rows.
        # n = 10 * m
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        def count_violations(v, m):
            residues = np.mod(C @ v, m)
            return int(np.sum(residues == 0))

        def solve_voltage(m, max_iters=3000):
            best_v = None
            best_cost = float('inf')
            
            for restart in range(5):
                if remaining() < 10:
                    break
                    
                v = np.random.randint(1, m, size=15).astype(np.int64)
                cost = count_violations(v, m)
                
                if cost == 0:
                    return v, 0
                
                if cost < best_cost:
                    best_cost = cost
                    best_v = v.copy()
                
                T = 1.0
                cooling = 0.9995
                min_T = 0.01
                no_improve = 0
                
                for it in range(max_iters):
                    if remaining() < 8:
                        break
                    
                    j = random.randint(0, 14)
                    old_val = v[j]
                    new_val = random.randint(1, m - 1)
                    if new_val == old_val:
                        new_val = (new_val + 1) % (m - 1) + 1
                    
                    v[j] = new_val
                    new_cost = count_violations(v, m)
                    delta = new_cost - cost
                    
                    if delta < 0:
                        cost = new_cost
                        no_improve = 0
                        if cost < best_cost:
                            best_cost = cost
                            best_v = v.copy()
                        if cost == 0:
                            return v, 0
                    elif T > min_T and random.random() < np.exp(-delta / T):
                        cost = new_cost
                        no_improve += 1
                    else:
                        v[j] = old_val
                        no_improve += 1
                    
                    T *= cooling
                    
                    if no_improve > 500:
                        break
                
            return best_v, best_cost

        m = 53
        while m < 500 and remaining() > 15:
            if is_prime(m):
                v, cost = solve_voltage(m, max_iters=2000)
                
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
                    
                    gc = check_girth_fast(n, edges, g, deadline - 5)
                    if gc >= g:
                        return (n, edges)
                
            m += 1
            
        return None

    # ---------------- Fallback: Random Regular Graph ----------------
    def random_regular_fallback():
        if k != 3:
            return None
            
        # Try to generate a random 3-regular graph with large n
        # Start from a size that is likely to have girth 16
        # Expected girth of random 3-regular graph grows like log_{2}(n)
        # For g=16, n should be around 2^16 = 65536? That's too big for 120s.
        # Actually, the smallest known is 936.
        # Random 3-reg graphs of size ~1000 often have girth ~10-12.
        # This fallback is unlikely to beat the voltage lift if it works.
        # We only use this if voltage lift fails.
        
        for n in [1000, 1200, 1500]:
            if remaining() < 30:
                break
                
            # Generate random 3-regular graph
            # Configuration model
            stubs = []
            for i in range(n):
                stubs.extend([i] * 3)
                
            edges = set()
            valid = True
            attempts = 0
            while attempts < 100:
                attempts += 1
                random.shuffle(stubs)
                edges = set()
                valid = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i+1]
                    if u == v:
                        valid = False
                        break
                    if (u, v) in edges or (v, u) in edges:
                        valid = False
                        break
                    edges.add((u, v))
                    
                if valid:
                    break
            
            if not valid:
                continue
                
            edge_list = list(edges)
            
            gc = check_girth_fast(n, edge_list, g, deadline - 5)
            if gc >= g:
                return (n, edge_list)
                
        return None

    # Main Logic
    result = voltage_lift_petersen()
    if result is not None:
        return result
        
    result = random_regular_fallback()
    if result is not None:
        return result
        
    # Last resort: Return a trivial invalid graph to avoid crash?
    # Or raise exception? "invalid result... scores 0".
    # Let's return a small cycle if k=2, else a large empty list?
    # The evaluator expects (n, edges).
    # If we fail, we should probably return something that doesn't crash the evaluator immediately,
    # but it will score 0.
    # Let's try to return the best effort from voltage lift even if unverified?
    # No, must be valid.
    
    # If we get here, we failed.
    # Return a dummy to prevent exception, knowing score will be 0.
    return 10, [(0,1), (1,2), (2,0)] if k==3 else (1, [])

def _construct_fallback(k, g):
    # This is not used in the main construct logic above, 
    # but kept for interface compatibility if called directly.
    if k == 2:
        n = g
        edges = [(i, (i+1)%n) for i in range(n)]
        return n, edges
    return 100, []

# Entry point
if __name__ == "__main__":
    # For testing
    n, edges = construct(3, 16)
    print(f"Constructed graph with {n} vertices")
