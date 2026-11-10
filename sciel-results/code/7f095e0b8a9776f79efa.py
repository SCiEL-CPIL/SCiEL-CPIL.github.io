"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction

def construct():
    target = Fraction(5, 31)
    start_time = time.time()
    deadline = start_time + 85.0 # Safety margin for 90s limit
    
    best_max_den = float('inf')
    best_solution = None

    # 1. Greedy Baseline
    r = target
    greedy_vals = []
    while r > 0:
        d = (r.denominator + r.numerator - 1) // r.numerator
        greedy_vals.append(d)
        r -= Fraction(1, d)
    if greedy_vals:
        best_max_den = max(greedy_vals)
        best_solution = list(greedy_vals)
    
    # 2. Meet-in-the-Middle Strategy
    # We split the search into two halves:
    # Left: Small denominators (1..M1)
    # Right: Large denominators (M2..M3)
    # We enumerate valid partial sums for Left and Right, then match them.
    
    # Heuristic range selection based on 5/31 ~ 0.161
    # If we pick d=8, rem ~ 0.035. If d=10, rem ~ 0.06.
    # We want to find solutions with max_den significantly lower than greedy (which is huge).
    # Let's try to bound the search space.
    
    # Parameters for MITM
    # Left side: denominators in [1, L_max]
    # Right side: denominators in [R_min, R_max]
    
    # Since we want to minimize max_den, we should try to find solutions where the largest denominator is small.
    # Let's try to find a solution with max_den <= B for various B.
    # Instead of binary search on B, we just run a bounded search and keep the best.
    
    # We'll use a dictionary to store partial sums from the "Left" side (small denoms).
    # Key: Fractional sum. Value: list of denominators.
    # To save memory, we only keep the list with the smallest max_den for a given sum? 
    # Actually, we want to minimize the GLOBAL max_den. 
    # If we fix the Left set, the Right set must sum to target - LeftSum.
    # The global max is max(max(Left), max(Right)).
    
    # Strategy:
    # 1. Enumerate all subsets of {1, ..., L_max} that sum to < target.
    #    Store in dict: sum_val -> list of denominators (keep only the one with min max_den? 
    #    No, we need distinct denominators. If we have two sets with same sum, 
    #    one with smaller max_den is better for the Left part, but might force larger Right part?
    #    Actually, if Left sum is fixed, Right sum is fixed. 
    #    To minimize max(Left, Right), we want Left max to be small AND Right max to be small.
    #    So for a given Left Sum, keeping the Left set with the smallest max_den is always optimal 
    #    because it leaves more room for Right to be small? Not necessarily, but it's a good heuristic.)
    
    # Let's try a specific split.
    # L_max = 50
    # R_min = L_max + 1
    # R_max = 500 (We hope to find a solution within this range. If not, we expand or fail.)
    
    # Actually, 5/31 is ~ 1/6. 
    # 1/7 + 1/13 + 1/15 ... 
    # Let's just try to find ANY solution with max_den < current best.
    
    # We will perform a DFS for the Left side (small denoms) and store partial sums.
    # Then a DFS for the Right side (large denoms) and check if (target - right_sum) is in LeftSums.
    
    # To handle the "distinct" constraint across the split:
    # We ensure Left uses [1, L_max] and Right uses [L_max+1, R_max]. This guarantees distinctness.
    
    L_max = 60
    R_max = 1000 # We will search up to this. If we find a solution with max < best, we update.
    
    left_sums = {} # Fraction -> list of denominators
    
    def dfs_left(start, end, remaining, current_list):
        nonlocal left_sums
        if time.time() > deadline:
            return
        
        if remaining == 0:
            # We found a complete left side? No, remaining is what's left for the RIGHT side.
            # Wait, the logic is:
            # We build Left Sum S_L.
            # We need S_R = Target - S_L.
            # So in dfs_left, we just accumulate S_L.
            # When we finish exploring a branch (or at any node), we record S_L.
            # But we need to ensure S_L < Target so S_R > 0.
            pass
            
        # Pruning: If remaining < 0, return
        if remaining < 0:
            return
            
        # Record current state
        # We store the sum of reciprocals of current_list.
        # To avoid recalculating, we can pass current_sum.
        # But Fraction addition is okay for small lists.
        
        # Let's restructure: 
        # dfs_left(d, max_d, current_sum, current_list)
        # We want to store current_sum in left_sums.
        
        if current_list:
            # Only store if this sum is achievable with a "better" (smaller max) left part?
            # Or just overwrite? 
            # If we have two left parts with same sum, the one with smaller max_den is preferable 
            # because it might allow a smaller right_max? 
            # Actually, if S_L is fixed, S_R is fixed. The constraint is just that Right must be formed by denoms > L_max.
            # The max_den of the solution is max(max(Left), max(Right)).
            # If we have Left1 (max=10) and Left2 (max=20) with same sum.
            # Right must be same set (or equivalent sum).
            # If Right has max=100.
            # Sol1 max = 100. Sol2 max = 100.
            # If Right has max=15.
            # Sol1 max = 15. Sol2 max = 20.
            # So smaller Left max is always better or equal.
            if current_sum not in left_sums or max(current_list) < max(left_sums[current_sum]):
                left_sums[current_sum] = list(current_list)
        
        # Next denominator
        next_d = d
        while next_d <= max_d:
            if time.time() > deadline:
                return
            new_sum = current_sum + Fraction(1, next_d)
            if new_sum >= target:
                break # Since denominators increase, sum will only increase
            dfs_left(next_d + 1, max_d, new_sum, current_list + [next_d])
            next_d += 1

    # Start Left DFS
    # We must allow empty left set? 
    # If Left is empty, Sum=0. Right must sum to 5/31.
    # Let's include empty set.
    left_sums[Fraction(0)] = []
    dfs_left(1, L_max, Fraction(0), [])
    
    # Now search Right side
    # Right denominators must be > L_max.
    # We want to find a set of denominators in [L_max+1, R_max] that sums to Target - S_L for some S_L in left_sums.
    
    # To be efficient, we can iterate through left_sums keys?
    # Or we can DFS the Right side and check if (Target - CurrentRightSum) is in left_sums.
    # The latter is better because we can prune based on R_max.
    
    def dfs_right(start, max_d, current_sum, current_list):
        nonlocal best_max_den, best_solution
        
        if time.time() > deadline:
            return
            
        # Check if we can complete with Left
        needed_left = target - current_sum
        if needed_left in left_sums:
            left_part = left_sums[needed_left]
            full_solution = left_part + current_list
            if len(full_solution) > 0 and len(full_solution) <= 256:
                # Check distinctness (guaranteed by split, but just in case)
                if len(set(full_solution)) == len(full_solution):
                    m = max(full_solution)
                    if m < best_max_den:
                        best_max_den = m
                        best_solution = full_solution
                        # We can't return immediately because we might find a smaller max_den?
                        # Actually, if we are searching R_max upwards, the first solution found 
                        # might not have the smallest max_den if the Left part was large.
                        # But we are iterating d upwards. 
                        # Wait, if we find a solution with max=m, any future solution in this branch 
                        # will have max >= m (since we add larger denoms).
                        # However, a different branch might have a smaller max.
                        # So we continue searching.
        
        # Pruning: 
        # The minimum possible sum we can add from here is 0 (if we stop).
        # The maximum possible sum we can add is bounded by sum of 1/(max_d+1) + ... 
        # But we just need to ensure we don't exceed target.
        
        # Next denominator
        next_d = start
        while next_d <= max_d:
            if time.time() > deadline:
                return
            new_sum = current_sum + Fraction(1, next_d)
            if new_sum >= target:
                break
            dfs_right(next_d + 1, max_d, new_sum, current_list + [next_d])
            next_d += 1

    # We run Right DFS. 
    # We can try different R_max values if we don't find a good solution.
    # But 1000 should be enough to beat greedy (which is likely > 1000).
    # Greedy for 5/31:
    # 1/7 = 0.142, rem = 0.019
    # 1/52 = 0.019, rem ~ 0
    # So greedy max is ~52? 
    # Let's check greedy max for 5/31.
    # 5/31 = 0.16129
    # d1 = ceil(31/5) = 7. 1/7 = 0.1428. Rem = 5/31 - 1/7 = (35-31)/217 = 4/217.
    # d2 = ceil(217/4) = 55. 1/55 = 0.01818. Rem = 4/217 - 1/55 = (220-217)/11935 = 3/11935.
    # d3 = ceil(11935/3) = 3979. 
    # So greedy max is 3979.
    # So if we find a solution with max < 3979, we win.
    # My R_max=1000 is a good target.
    
    dfs_right(L_max + 1, 1000, Fraction(0), [])
    
    # If we didn't find anything better than greedy, return greedy.
    # If we found something, return it.
    
    if best_solution is None:
        return greedy_vals
    return best_solution

# EVOLVE-BLOCK-END
