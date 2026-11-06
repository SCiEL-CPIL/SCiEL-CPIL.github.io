"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit
    p, q = 5, 31  # target = 5/31
    
    # Known solution: 1/8 + 1/31 + 1/248 = 5/31
    best = [8, 31, 248]
    best_max = 248
    
    # We want to minimize the largest denominator.
    # Strategy: iterate possible maximum denominator M from small to large.
    # For each M, check if there exists a set of distinct denominators in [1, M]
    # with max = M (or <= M) summing to 5/31.
    # Actually, we can just search for ANY valid set with max denominator < current best_max.
    
    # Let's use a recursive search with tight bounds.
    # We'll search for k-term expansions for k=2,3,4,5,...
    # But we want to minimize the MAX denominator, not the count.
    # So we should iterate M from 7 upwards and check if a solution exists with all denominators <= M.
    
    # A solution exists with max denom <= M iff there's a subset of {1/1, 1/2, ..., 1/M}
    # that sums to 5/31. This is a subset sum problem which is hard, but M is small.
    
    # Better: direct search with pruning.
    # Sort denominators in increasing order: d1 < d2 < ... < dk <= M.
    # 1/d1 + ... + 1/dk = 5/31.
    
    # Let's search by increasing M.
    # For a given M, we need to find d1 < ... < dk <= M such that sum(1/di) = 5/31.
    
    # Since 5/31 ≈ 0.161, and 1/7 ≈ 0.143, 1/6 ≈ 0.167.
    # So d1 >= 7 (since 1/6 > 5/31 and we need sum = 5/31, if d1=6 then 1/6 > 5/31 already... wait)
    # 1/6 = 0.1667 > 5/31 = 0.1613. So d1 cannot be 6.
    # d1 must be >= 7.
    
    # Let's do a depth-first search with the constraint that all denominators <= M.
    # We'll try M = 7, 8, 9, ... until we find a solution.
    
    def search(max_denom, depth, current_sum_num, current_sum_den, chosen):
        """
        Search for remaining terms such that sum = 5/31 - current_sum.
        All denominators must be > last chosen and <= max_denom.
        current_sum = current_sum_num / current_sum_den
        """
        if depth > 10:  # safety
            return None
        
        # Remaining sum needed: 5/31 - current_sum
        # rem = 5/31 - current_sum_num/current_sum_den
        # rem = (5*current_sum_den - 31*current_sum_num) / (31*current_sum_den)
        rem_num = 5 * current_sum_den - 31 * current_sum_num
        rem_den = 31 * current_sum_den
        
        if rem_num <= 0:
            return None
        if rem_num == 0:
            return chosen[:]
        
        # rem = rem_num / rem_den
        # We need 1/d1 + 1/d2 + ... = rem_num/rem_den
        # Next denominator d must satisfy:
        # 1/d < rem_num/rem_den  =>  d > rem_den/rem_num
        # Also, if we have (depth_remaining) terms left, each <= 1/d, so:
        # depth_remaining / d >= rem_num/rem_den  =>  d <= depth_remaining * rem_den / rem_num
        
        last = chosen[-1] if chosen else 0
        d_min = max(last + 1, rem_den // rem_num + 1)
        if d_min > max_denom:
            return None
        
        # Upper bound on d: we need enough "room" for the remaining terms
        # If we pick d, the max sum of remaining (depth-1) terms is (depth-1)/(d+1) (approx)
        # Actually, the max sum of k terms all > d is 1/(d+1) + 1/(d+2) + ... 
        # But a simpler bound: if we pick d, the remaining sum is rem - 1/d.
        # This must be achievable by (depth-1) terms each >= 1/(max_denom).
        # So rem - 1/d <= (depth-1)/max_denom is necessary but not sufficient.
        # And rem - 1/d >= 0.
        
        # Also, the next term 1/d must be <= rem, so d >= rem_den/rem_num.
        # And the remaining (depth-1) terms must sum to rem - 1/d.
        # Each of those terms is at most 1/(d+1), so (depth-1)/(d+1) >= rem - 1/d.
        # This gives a bound on d.
        
        d_max = max_denom
        # Tighter bound: (depth-1) terms, each at most 1/(d+1)
        # So we need rem - 1/d <= (depth-1)/(d+1)
        # rem_num/rem_den - 1/d <= (depth-1)/(d+1)
        # (rem_num*d - rem_den)/(rem_den*d) <= (depth-1)/(d+1)
        # (rem_num*d - rem_den)*(d+1) <= (depth-1)*rem_den*d
        # This is a quadratic in d. Let's just iterate.
        
        for d in range(d_min, d_max + 1):
            if time.time() > deadline:
                return None
            
            # Check if 1/d <= rem
            if d * rem_num < rem_den:
                break  # 1/d > rem, no larger d will work either... wait, larger d means smaller 1/d
            # Actually 1/d decreases as d increases. So if 1/d > rem, we need larger d.
            # If 1/d <= rem, it's a candidate.
            
            # New remaining sum: rem - 1/d
            # = rem_num/rem_den - 1/d
            # = (rem_num*d - rem_den) / (rem_den*d)
            new_rem_num = rem_num * d - rem_den
            new_rem_den = rem_den * d
            
            if new_rem_num < 0:
                continue  # 1/d > rem, skip
            if new_rem_num == 0:
                return chosen + [d]
            
            # Check if remaining is achievable with (depth-1) terms
            # Min possible sum with (depth-1) terms > d: 1/(d+1) + ... 
            # Max possible sum with (depth-1) terms <= max_denom: (depth-1)/max_denom
            # Actually, the max sum is 1/(d+1) + 1/(d+2) + ... + 1/(d+depth-1)
            # The min sum is (depth-1)/max_denom (if all terms = max_denom, but they must be distinct)
            # Simpler: new_rem_num/new_rem_den must be <= (depth-1)/(d+1) (max sum of remaining)
            # And >= (depth-1)/max_denom (min sum, roughly)
            
            # Upper bound check: can (depth-1) terms sum to new_rem?
            # Max sum of (depth-1) distinct denominators > d is 1/(d+1) + 1/(d+2) + ... 
            # Let's compute this bound.
            max_possible = 0
            for i in range(1, depth):
                max_possible += 1.0 / (d + i)
            if new_rem_num / new_rem_den > max_possible + 1e-15:
                continue
            
            # Min sum: (depth-1) distinct denominators in [d+1, max_denom]
            # Min sum is 1/max_denom + 1/(max_denom-1) + ... 
            # But we need distinct, so min sum is sum of 1/max_denom, 1/(max_denom-1), ...
            min_possible = 0
            for i in range(depth - 1):
                min_possible += 1.0 / (max_denom - i)
            if new_rem_num / new_rem_den < min_possible - 1e-15:
                continue
            
            # Recurse
            result = search(max_denom, depth - 1, new_rem_num, new_rem_den, chosen + [d])
            if result is not None:
                return result
        
        return None
    
    # Try increasing max_denom
    # We know a solution exists with max_denom = 248.
    # Let's try from 7 up to 247.
    for M in range(7, best_max):
        if time.time() > deadline:
            break
        
        # Try different depths (number of terms)
        # Max depth: since 1/7 + 1/8 + ... is small, depth is limited.
        # 5/31 ≈ 0.161. 1/7 ≈ 0.143. So at least 2 terms? No, 1/7 < 5/31.
        # Actually, min term is 1/M. Max number of terms is roughly 5/31 * M.
        # For M=100, that's about 16 terms. But distinct denominators constrain this.
        
        # Let's try depths from 2 up to some reasonable limit.
        # For depth=2: 1/a + 1/b = 5/31. 
        # (a+b)/(ab) = 5/31 => 31(a+b) = 5ab => 5ab - 31a - 31b = 0
        # (5a-31)(5b-31) = 31^2 = 961
        # 961 = 1*961, 31*31. 
        # 5a-31=1 => a=32/5 not integer.
        # 5a-31=31 => a=12.4 not integer.
        # So no 2-term solution.
        
        # Try depths 3, 4, 5, ...
        for depth in range(3, min(M // 2 + 1, 20)):
            # Initial sum = 0
            result = search(M, depth, 0, 1, [])
            if result is not None:
                best = result
                best_max = M
                break
        else:
            continue
        break
    
    return best
# EVOLVE-BLOCK-END
