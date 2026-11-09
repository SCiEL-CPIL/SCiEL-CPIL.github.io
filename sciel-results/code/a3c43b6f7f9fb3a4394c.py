import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Try n from 30 down to 18
    # We can go higher, but 30 is a good target for a simple heuristic in 120s.
    # Exoo's record is 35. Let's try up to 32.
    for n in range(32, 17, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple restarts for this n
        max_restarts = 20
        found = False
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize random graph
            # p should be around 0.5
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local Search: Simulated Annealing
            T = 1.0
            best_cost = float('inf')
            best_adj = adj[:]
            
            # Helper to calculate cost
            def get_cost(adj_list):
                # Count K4 and I6
                k4_count = 0
                i6_count = 0
                
                # Check K4
                # Iterate over all 4-combinations
                for combo in itertools.combinations(range(n), 4):
                    u, v, w, x = combo
                    # Check if all edges exist
                    if (adj_list[u] & (1 << v)) and (adj_list[u] & (1 << w)) and (adj_list[u] & (1 << x)) and \
                       (adj_list[v] & (1 << w)) and (adj_list[v] & (1 << x)) and \
                       (adj_list[w] & (1 << x)):
                        k4_count += 1
                        if k4_count > 100: # Early exit for speed in cost calc
                            break
                if k4_count > 100:
                    return k4_count # + i6_count (approx)
                
                # Check I6
                for combo in itertools.combinations(range(n), 6):
                    # Check if no edges exist
                    is_indep = True
                    for i in range(6):
                        u = combo[i]
                        # Check if u connects to any other in combo
                        mask = 0
                        for j in range(6):
                            if i != j:
                                mask |= (1 << combo[j])
                        if adj_list[u] & mask:
                            is_indep = False
                            break
                    if is_indep:
                        i6_count += 1
                        if i6_count > 100:
                            break
                            
                return k4_count * 10 + i6_count # Weight K4 more? Or equal.
            
            # Optimize cost calculation using bitmasks more efficiently
            def fast_cost(adj_list):
                k4 = 0
                i6 = 0
                # K4 check
                for i in range(n):
                    # Only consider neighbors
                    neighbors = adj_list[i]
                    # Iterate pairs of neighbors j, k
                    # This is complex to iterate pairs from bitmask efficiently in pure python without precomputed lists.
                    # Let's stick to itertools for correctness, but optimize the inner check.
                    pass
                
                # Let's use a simpler heuristic for the loop
                # We will just flip edges and check violations locally.
                pass
                
            # Re-implementing a simpler local search loop
            # State: adj list
            # Move: flip one edge
            # Cost: number of K4 + number of I6
            
            # Precompute all 4-sets and 6-sets for fixed n
            all_4sets = list(itertools.combinations(range(n), 4))
            all_6sets = list(itertools.combinations(range(n), 6))
            
            def calc_violations(adj_list):
                k4 = 0
                for u, v, w, x in all_4sets:
                    if (adj_list[u] & (1 << v)) and (adj_list[u] & (1 << w)) and (adj_list[u] & (1 << x)) and \
                       (adj_list[v] & (1 << w)) and (adj_list[v] & (1 << x)) and \
                       (adj_list[w] & (1 << x)):
                        k4 += 1
                i6 = 0
                for s in all_6sets:
                    # Check independence
                    ok = True
                    for i in range(6):
                        u = s[i]
                        mask = 0
                        for j in range(6):
                            if i != j:
                                mask |= (1 << s[j])
                        if adj_list[u] & mask:
                            ok = False
                            break
                    if ok:
                        i6 += 1
                return k4, i6

            curr_adj = adj[:]
            curr_k4, curr_i6 = calc_violations(curr_adj)
            curr_cost = curr_k4 + curr_i6
            best_cost = curr_cost
            best_adj = curr_adj[:]
            
            T = 1.0
            T_min = 0.01
            steps = 5000
            
            for step in range(steps):
                if time.time() > deadline - 5:
                    break
                    
                # Pick random edge to flip
                i = random.randint(0, n - 1)
                j = random.randint(i + 1, n - 1)
                
                # Flip edge (i, j)
                curr_adj[i] ^= (1 << j)
                curr_adj[j] ^= (1 << i)
                
                new_k4, new_i6 = calc_violations(curr_adj)
                new_cost = new_k4 + new_i6
                
                delta = new_cost - curr_cost
                if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                    curr_cost = new_cost
                    if curr_cost < best_cost:
                        best_cost = curr_cost
                        best_adj = curr_adj[:]
                else:
                    # Revert
                    curr_adj[i] ^= (1 << j)
                    curr_adj[j] ^= (1 << i)
                    
                T *= 0.999
                if T < T_min:
                    T = T_min
                
                if best_cost == 0:
                    found = True
                    break
            
            if found:
                # Construct edge list
                edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if best_adj[i] & (1 << j):
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
            
        # If no graph found for this n, try smaller n
        # We continue the loop to try n-1
        
    # Fallback to seed if nothing better found
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
