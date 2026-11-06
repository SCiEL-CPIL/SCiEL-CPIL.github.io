import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 320.0

    def time_left():
        return deadline - time.time()

    # 1. Construct Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    n_init = q
    G = [set() for _ in range(n_init)]
    edges = []
    for i in range(n_init):
        for j in range(i + 1, n_init):
            if (j - i) % q in residues:
                G[i].add(j)
                G[j].add(i)
                edges.append((i, j))
    
    current_n = n_init
    current_G = G
    current_edges = edges

    # Helper to count K4s in a subset S of current_G
    def count_k4(S):
        S_list = list(S)
        n_s = len(S_list)
        if n_s < 4:
            return 0
        count = 0
        # Iterate over all combinations of 4? No, O(N^4) is fine for small N.
        # But we can do it faster by iterating edges.
        # A K4 is (u,v,w,x) such that all pairs are edges.
        # Iterate u, v in S with edge.
        # Then w, x in (S \ {u,v}) that are common neighbors of u,v and edge between w,x.
        
        # Simpler: Just check all 4-subsets?
        # For N=40, 40C4 = 91390. Very fast.
        from itertools import combinations
        for u, v, w, x in combinations(S_list, 4):
            if (v in current_G[u] and w in current_G[u] and x in current_G[u] and
                w in current_G[v] and x in current_G[v] and
                x in current_G[w]):
                count += 1
        return count

    # Helper to count Independent K4s (K4 in complement) in a subset T
    def count_ik4(T):
        T_list = list(T)
        n_t = len(T_list)
        if n_t < 4:
            return 0
        count = 0
        from itertools import combinations
        for u, v, w, x in combinations(T_list, 4):
            if (v not in current_G[u] and w not in current_G[u] and x not in current_G[u] and
                w not in current_G[v] and x not in current_G[v] and
                x not in current_G[w]):
                count += 1
        return count

    def find_extension_sa():
        if time_left() < 2.0:
            return None
            
        n = current_n
        # A is the set of neighbors of the new vertex
        # We want E(A) = count_k4(A) + count_ik4(V \ A) == 0
        
        best_A = None
        best_cost = float('inf')
        
        # Multiple restarts
        max_restarts = 50
        restarts = 0
        
        while restarts < max_restarts and time_left() > 2.0:
            restarts += 1
            # Initialize A randomly
            # Probability of picking a vertex is 0.5
            A = set()
            for v in range(n):
                if random.random() < 0.5:
                    A.add(v)
            
            current_cost = count_k4(A) + count_ik4(set(range(n)) - A)
            
            if current_cost == 0:
                return A
            
            best_cost_restart = current_cost
            best_A_restart = A.copy()
            
            T = 1.0
            T_min = 0.01
            cooling = 0.95
            max_steps = 5000
            
            for step in range(max_steps):
                if time_left() < 1.0:
                    return best_A if best_cost_restart == 0 else None
                
                # Move: flip a random vertex
                v = random.choice(range(n))
                if v in A:
                    A.remove(v)
                else:
                    A.add(v)
                    
                new_cost = count_k4(A) + count_ik4(set(range(n)) - A)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    if current_cost == 0:
                        return A
                    if current_cost < best_cost_restart:
                        best_cost_restart = current_cost
                        best_A_restart = A.copy()
                else:
                    delta = new_cost - current_cost
                    if random.random() < math.exp(-delta / T):
                        current_cost = new_cost
                        if current_cost == 0:
                            return A
                        if current_cost < best_cost_restart:
                            best_cost_restart = current_cost
                            best_A_restart = A.copy()
                    else:
                        # Revert
                        if v in A:
                            A.remove(v)
                        else:
                            A.add(v)
                            
                T *= cooling
                if T < T_min:
                    break
            
            if best_cost_restart < best_cost:
                best_cost = best_cost_restart
                best_A = best_A_restart

        return best_A if best_cost == 0 else None

    # Try to extend to 42 vertices
    max_target = 42
    while current_n < max_target:
        if time_left() < 10.0:
            break
            
        extension = find_extension_sa()
        if extension is not None:
            new_v = current_n
            current_G.append(set(extension))
            for u in extension:
                current_G[u].add(new_v)
                current_edges.append((min(u, new_v), max(u, new_v)))
            current_n += 1
        else:
            # If we fail to extend, we might try to "fix" the graph?
            # Or just stop.
            # Given the time, stopping is safer to ensure validity.
            break

    return current_n, current_edges
