import random
import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    n = 16
    num_edges = n * (n - 1) // 2  # 120
    num_k4 = 1820  # C(16, 4)
    
    # Precompute edge indices
    # Edge (i, j) with i < j
    # Index = i * (n - 1) - i * (i + 1) // 2 + j - i - 1 ? 
    # Simpler:
    edge_indices = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_indices[(i, j)] = idx
            idx += 1
            
    # Precompute K4s and which edges they use
    # k4_list[k] = list of 6 edge indices
    k4_list = []
    # edge_to_k4s[e] = list of k indices
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    k_idx = 0
    for quad in itertools.combinations(range(n), 4):
        i, j, k, l = quad
        edges = [
            edge_indices[(i, j)],
            edge_indices[(i, k)],
            edge_indices[(i, l)],
            edge_indices[(j, k)],
            edge_indices[(j, l)],
            edge_indices[(k, l)]
        ]
        k4_list.append(edges)
        for e in edges:
            edge_to_k4s[e].append(k_idx)
        k_idx += 1
        
    def is_valid(colors):
        # Check all K4s
        for edges in k4_list:
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            # Count distinct
            # Since max colours is small, we can just check distinctness
            # If len(set(...)) < 5, invalid
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                return False
        return True

    def count_violations(colors):
        v = 0
        for edges in k4_list:
            c0 = colors[edges[0]]
            c1 = colors[edges[1]]
            c2 = colors[edges[2]]
            c3 = colors[edges[3]]
            c4 = colors[edges[4]]
            c5 = colors[edges[5]]
            if len({c0, c1, c2, c3, c4, c5}) < 5:
                v += 1
        return v

    # We want to minimize number of colours used.
    # We will try to find a valid colouring with k colours.
    # We start with k=13.
    
    best_result = None
    best_k = 17 # Seed uses 17
    
    # Fallback: Seed solution
    # (i + j) % 17
    seed_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            seed_colors.append((i + j) % 17)
            
    if is_valid(seed_colors):
        best_result = seed_colors
        best_k = 17
        
    # Try to find better solutions with fewer colours
    for k in range(13, best_k):
        if time.time() > deadline - 20:
            break
            
        # Try multiple restarts for this k
        found_valid = False
        max_restarts = 50
        restarts = 0
        
        while restarts < max_restarts and not found_valid and time.time() < deadline - 10:
            # Initialize with random k colours
            colors = [random.randint(0, k - 1) for _ in range(num_edges)]
            
            # Simple local search
            # We use a penalty based on violations and number of colours used?
            # No, we fix k. We just want to satisfy constraints.
            # If we satisfy constraints, we check if all k colours are used.
            # If not, we can reduce k effectively.
            
            # Actually, if we find a valid colouring using colors from 0..k-1,
            # we can check the actual number of distinct colours used.
            
            # Let's use Simulated Annealing to minimize violations.
            # Temperature schedule
            T = 1.0
            cooling = 0.999
            step = 0
            
            current_violations = count_violations(colors)
            best_local_violations = current_violations
            best_local_colors = colors[:]
            
            while T > 0.01 and time.time() < deadline - 5:
                # Pick random edge
                e = random.randint(0, num_edges - 1)
                old_c = colors[e]
                new_c = random.randint(0, k - 1)
                
                if old_c == new_c:
                    continue
                
                colors[e] = new_c
                
                # Calculate delta violations
                # Only check K4s containing e
                delta_v = 0
                for k4_idx in edge_to_k4s[e]:
                    edges = k4_list[k4_idx]
                    c0 = colors[edges[0]]
                    c1 = colors[edges[1]]
                    c2 = colors[edges[2]]
                    c3 = colors[edges[3]]
                    c4 = colors[edges[4]]
                    c5 = colors[edges[5]]
                    if len({c0, c1, c2, c3, c4, c5}) < 5:
                        # This K4 is currently violated
                        # But we need to know if it WAS violated before
                        # This incremental check is tricky if we don't store state per K4.
                        # Given num_k4 is small (1820), full recount is ~1820 * 6 ops.
                        # 1820 * 6 = 10k ops.
                        # 1000 steps/s * 10k ops = 10M ops/s.
                        # Python is slower. Maybe 100-500 steps/s.
                        # 100 steps/s * 10k ops = 1M ops/s.
                        # 110s * 1M = 110M ops.
                        # 110M / 10k = 11k steps.
                        # This is feasible.
                        pass 
                
                # Let's just do full recount for simplicity and correctness, 
                # but optimize the inner loop.
                new_violations = count_violations(colors)
                
                if new_violations < current_violations:
                    current_violations = new_violations
                    if new_violations < best_local_violations:
                        best_local_violations = new_violations
                        best_local_colors = colors[:]
                else:
                    # Accept with probability
                    if random.random() < pow(2.718, -(new_violations - current_violations) / T):
                        current_violations = new_violations
                    else:
                        colors[e] = old_c # Revert
                
                T *= cooling
                step += 1
                
                if best_local_violations == 0:
                    # Found a valid colouring with at most k colours
                    # Check actual distinct colours
                    distinct = len(set(best_local_colors))
                    if distinct < best_k:
                        best_k = distinct
                        best_result = best_local_colors
                    found_valid = True
                    break
                    
            restarts += 1
            
        if found_valid:
            # We found a valid one with <= k colours.
            # We can continue to try smaller k? 
            # The loop iterates k downwards. 
            # If we found one with k=13, we break out of the for loop?
            # No, the for loop is `for k in range(13, best_k)`.
            # If we start with best_k=17, k goes 13, 14, 15, 16.
            # If we find valid at 13, we should stop because 13 is the minimum possible (lower bound).
            if best_k == 13:
                break

    # If best_result is still the seed (17 colours), or something better, return it.
    # The task asks to minimize.
    # If we found 13, return that.
    
    return best_result
