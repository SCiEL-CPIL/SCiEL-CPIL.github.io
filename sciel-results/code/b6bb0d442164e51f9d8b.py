"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    target = Fraction(5, 31)
    
    # Initial known solution
    best_solution = [8, 31, 248]
    best_max = 248
    
    def check_and_update(path):
        nonlocal best_solution, best_max
        # The last element of path is the new denominator
        if not path:
            return
        m = path[-1]
        if m < best_max:
            # Verify distinctness and sum
            if len(set(path)) == len(path):
                total = sum(Fraction(1, d) for d in path)
                if total == target:
                    best_max = m
                    best_solution = list(path)
    
    def max_possible_sum(min_d, terms_left):
        """Calculate the maximum sum of 'terms_left' distinct unit fractions 
        starting with denominator min_d."""
        s = Fraction(0, 1)
        for i in range(terms_left):
            s += Fraction(1, min_d + i)
        return s

    def dfs(current_sum, min_d, path, depth, max_depth):
        nonlocal best_max
        
        if time.time() > deadline:
            return
            
        # Pruning: if we can't reach the target even with max possible sum, stop
        if depth < max_depth:
            # We have max_depth - depth terms left to choose
            # But we can also stop early if the remainder is a unit fraction
            # Let's check if the remainder is a unit fraction first
            rem = target - current_sum
            if rem > 0:
                # Check if rem is 1/k for integer k > min_d
                if rem.numerator == 1:
                    k = rem.denominator
                    if k > min_d and k < best_max:
                        new_path = path + [k]
                        check_and_update(new_path)
                        # Note: we don't return here because a different set of 
                        # smaller denominators might exist? No, if rem is 1/k, 
                        # that's the only way to complete with 1 more term.
                        # But we might find a solution with MORE terms that has 
                        # a smaller max denominator? 
                        # If we use more terms, the denominators must be larger 
                        # than min_d, so the max denominator would be at least 
                        # the last one added. 
                        # Actually, adding more terms allows smaller individual 
                        # denominators to sum to the remainder, potentially 
                        # reducing the max denominator? 
                        # Example: 1/100 + 1/101 vs 1/50.5 (not int). 
                        # Generally, splitting a fraction 1/n into 1/a + 1/b 
                        # with a,b > n increases the max denominator. 
                        # So if we found a solution with max k, adding more terms 
                        # will likely result in max > k. 
                        # So we can prune this branch if we found a solution? 
                        # Not necessarily, because we might have skipped a 
                        # smaller min_d earlier? No, we iterate min_d upwards.
                        # So if we found a solution with k, any further solution 
                        # in this branch will have max >= k. 
                        # Since we are looking for min max, we can prune if 
                        # k == best_max? No, we want < best_max. 
                        # If k < best_max, we updated best_max. 
                        # Now any other solution in this subtree will have 
                        # max denominator >= k (since we add more terms, 
                        # denominators increase). 
                        # So we can return.
                        return

        if depth == max_depth:
            return

        # Pruning: check if max possible sum is sufficient
        terms_left = max_depth - depth
        max_sum = max_possible_sum(min_d, terms_left)
        if current_sum + max_sum < target:
            return

        # Iterate next denominator
        # Upper bound for d: 
        # 1/d < rem => d > 1/rem
        rem = target - current_sum
        if rem <= 0:
            return
        
        d_min = min_d
        # d_max is limited by best_max - 1
        d_max = best_max - 1
        
        # Also, we can't pick d such that 1/d >= rem (would overshoot)
        # So d > 1/rem
        d_lower_bound = int(1 / rem) + 1
        if d_lower_bound < d_min:
            d_lower_bound = d_min
            
        for d in range(d_lower_bound, d_max + 1):
            if time.time() > deadline:
                return
            
            new_sum = current_sum + Fraction(1, d)
            if new_sum >= target:
                # Since d is increasing, new_sum will only increase, 
                # but we want new_sum < target. 
                # Wait, if new_sum == target, we handled that above? 
                # No, above we checked if rem is unit fraction. 
                # If new_sum == target, then rem was 1/d. 
                # That case is handled by the unit fraction check at the start 
                # of the loop iteration? No, the check is for the *remaining* 
                # after adding d. 
                # Let's re-structure.
                
                # If new_sum == target, we have a solution.
                if new_sum == target:
                    check_and_update(path + [d])
                    # Since we found a solution with max d, and d is increasing, 
                    # any further d in this loop will be larger. 
                    # Any deeper recursion will have even larger denominators. 
                    # So we can break? 
                    # Yes, for this specific path, larger d means larger max. 
                    # So break.
                    break
                elif new_sum > target:
                    break
                else:
                    dfs(new_sum, d + 1, path + [d], depth + 1, max_depth)

    # Search with increasing max_depth
    # We start with small depths and go up. 
    # However, deeper depths allow smaller max denominators. 
    # So we should search deeper first? 
    # No, the search space grows exponentially. 
    # Let's try depths 3 to 8.
    
    # Optimization: 
    # If we find a solution with max M, we can prune all branches where 
    # the current max denominator >= M.
    
    max_terms_to_search = 8
    
    for max_d in range(3, max_terms_to_search + 1):
        if time.time() > deadline:
            break
        dfs(Fraction(0, 1), 2, [], 0, max_d)
        
    return best_solution
# EVOLVE-BLOCK-END
