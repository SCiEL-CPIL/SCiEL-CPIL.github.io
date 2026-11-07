import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 45  # Safety margin
    
    n = 14
    m = 35
    max_val = 10000
    
    # All edges are required since m = n*(n-1)/2
    all_edges = [list(e) for e in combinations(range(n), 2)]
    
    def score_values(vals):
        """
        Calculate max(distinct sums, distinct products) for the given values.
        """
        sums = set()
        prods = set()
        for i in range(n):
            for j in range(i+1, n):
                s = vals[i] + vals[j]
                p = vals[i] * vals[j]
                sums.add(s)
                prods.add(p)
        return max(len(sums), len(prods))
    
    # Baseline: 1 to 14
    base_vals = list(range(1, n + 1))
    best_score = score_values(base_vals)
    best_vals = base_vals[:]
    
    # Local Search / Simulated Annealing to improve
    # We try to swap or perturb values to increase collisions.
    # Since small integers generally have more additive and multiplicative collisions,
    # we search within a range of small integers.
    
    current_vals = base_vals[:]
    current_score = best_score
    
    # Simple Hill Climbing with random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Initialize with a random set of small distinct integers
        if restart == 0:
            cand = base_vals[:]
        else:
            # Generate a random set of 14 distinct integers in [1, 50]
            # Small numbers favor collisions
            cand = random.sample(range(1, 50), n)
            # Ensure sorted for consistency? Not strictly needed for set calculation.
        
        cand_score = score_values(cand)
        
        if cand_score < current_score:
            current_score = cand_score
            current_vals = cand[:]
            
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try swapping two elements
            for i in range(n):
                for j in range(i+1, n):
                    if time.time() > deadline:
                        break
                    # Swap i and j
                    cand[i], cand[j] = cand[j], cand[i]
                    new_score = score_values(cand)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        # Keep the swap
                    else:
                        # Revert
                        cand[i], cand[j] = cand[j], cand[i]
                if time.time() > deadline:
                    break
            
        if current_score < best_score:
            best_score = current_score
            best_vals = current_vals[:]
            
    # Final check: maybe a specific structured set is better?
    # Try 1..14 again just in case
    if score_values(base_vals) < best_score:
        best_vals = base_vals[:]
        best_score = score_values(base_vals)
        
    # Convert to int explicitly
    final_values = [int(v) for v in best_vals]
    final_edges = [list(e) for e in all_edges]
    
    return {'values': final_values, 'edges': final_edges}
