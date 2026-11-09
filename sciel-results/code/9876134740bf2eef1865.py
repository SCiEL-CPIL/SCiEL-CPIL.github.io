import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s limit to be safe
    
    n = 128
    
    # Precompute all 4-APs
    # A 4-AP is defined by (a, d) where a + 3d <= n
    # Positions are 1-indexed in problem, 0-indexed in array
    aps = []
    for a in range(n):
        for d in range(1, (n - 1 - a) // 3 + 1):
            if a + 3 * d < n:
                aps.append((a, a+d, a+2*d, a+3*d))
                
    def count_violations(colors):
        """Count 4-APs with < 3 distinct colors."""
        violations = 0
        for a, b, c, d in aps:
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                violations += 1
        return violations

    def is_valid(colors):
        return count_violations(colors) == 0

    best_solution = None
    best_k = 128
    
    # Try k=3 first
    if time.time() < deadline - 10:
        # Simulated Annealing for k=3
        colors = np.random.randint(0, 3, n)
        current_cost = count_violations(colors)
        
        if current_cost == 0:
            best_solution = colors
            best_k = 3
        else:
            T = 100.0
            T_min = 0.01
            cooling = 0.999
            max_iters = 500000
            iter_count = 0
            
            while T > T_min and time.time() < deadline - 10:
                # Pick a random position and change its color
                idx = np.random.randint(n)
                old_color = colors[idx]
                new_color = np.random.randint(0, 3)
                if new_color == old_color:
                    continue
                
                colors[idx] = new_color
                new_cost = count_violations(colors)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    if current_cost == 0:
                        best_solution = colors.copy()
                        best_k = 3
                        break
                else:
                    colors[idx] = old_color
                
                T *= cooling
                iter_count += 1

    # Try k=4 if k=3 failed or wasn't tried
    if best_k > 4 and time.time() < deadline - 5:
        # Greedy + Local Search for k=4
        # Initialize randomly
        colors = np.random.randint(0, 4, n)
        current_cost = count_violations(colors)
        
        if current_cost == 0:
            best_solution = colors
            best_k = 4
        else:
            # Local Search: Hill Climbing
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                # Shuffle order of indices to avoid bias
                indices = np.random.permutation(n)
                for idx in indices:
                    old_color = colors[idx]
                    best_neighbor_cost = current_cost
                    best_neighbor_color = old_color
                    
                    for new_color in range(4):
                        if new_color == old_color:
                            continue
                        colors[idx] = new_color
                        new_cost = count_violations(colors)
                        if new_cost < best_neighbor_cost:
                            best_neighbor_cost = new_cost
                            best_neighbor_color = new_color
                    colors[idx] = best_neighbor_color
                    
                    if best_neighbor_cost < current_cost:
                        current_cost = best_neighbor_cost
                        improved = True
                        if current_cost == 0:
                            best_solution = colors.copy()
                            best_k = 4
                            break
                
    # Fallback: If no solution found, use the seed logic or a simple greedy
    if best_solution is None:
        # Replicate seed logic to ensure validity, though it might use more colors
        colors = []
        for x in range(1, 129):
            for color in range(128):
                # Check 4-APs ending at x
                valid = True
                # d=1,2,... up to (x-1)//3
                for d in range(1, (x-1)//3 + 1):
                    if x - 3*d < 1: break
                    # Indices are 0-based in colors list, 1-based in x
                    # x is 1-based index in loop, so x-1 is 0-based index in colors
                    # AP: x-3d, x-2d, x-d, x
                    # In 0-based: x-1-3d, x-1-2d, x-1-d, x-1
                    c0 = colors[x-1-3*d]
                    c1 = colors[x-1-2*d]
                    c2 = colors[x-1-d]
                    if len({color, c0, c1, c2}) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    break
            else:
                # Should not happen if we have enough colors
                colors.append(0)
        best_solution = np.array(colors)
        best_k = len(set(best_solution))

    # Convert to list of ints as required
    return list(best_solution)

