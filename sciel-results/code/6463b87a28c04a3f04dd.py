import time
from fractions import Fraction
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds to be safe against 90s limit
    
    n = 30
    # Precompute the terms as Fractions for exact arithmetic
    terms = [Fraction(1, i) for i in range(1, n + 1)]
    
    # dp[i] is a set of Fractions representing all possible sums using the first i terms
    # We start with sum 0 using 0 terms
    dp = [set() for _ in range(n + 1)]
    dp[0].add(Fraction(0, 1))
    
    for i in range(n):
        if time.time() > deadline:
            break
        term = terms[i]
        prev_set = dp[i]
        curr_set = dp[i + 1]
        
        # For each sum in the previous step, we can add -term, 0, or +term
        # We update the set for the next step
        for s in prev_set:
            curr_set.add(s - term)
            curr_set.add(s)
            curr_set.add(s + term)
            
    # If we didn't finish the DP loop, we fall back to a heuristic
    # But for n=30, the number of states might be large. 
    # Let's estimate the size. The number of distinct sums is bounded by 3^30 which is huge.
    # However, the sums are rationals with denominators dividing LCM(1..30).
    # LCM(1..30) is 2329089562800.
    # The number of distinct sums is much smaller than 3^30 but still potentially large.
    # Let's check if the DP set size explodes.
    
    # Actually, a better approach for exact minimization might be to use a different DP state or a meet-in-the-middle.
    # Meet-in-the-middle: split 30 terms into two sets of 15.
    # Generate all 3^15 sums for the first half and 3^15 for the second half.
    # 3^15 = 14348907. This is feasible.
    
    # Let's implement Meet-in-the-Middle instead if DP is too slow or memory heavy.
    # 3^15 is ~14 million. Storing 14 million Fractions might be memory intensive but doable.
    # Sorting 14 million items is also doable.
    
    # Let's restart with Meet-in-the-Middle which is more predictable for n=30.
    
    if time.time() > deadline:
         # Fallback
         return [0]*28 + [1, -1]

    n_half = n // 2
    # First half: indices 0 to n_half - 1 (terms 1/1 to 1/15)
    # Second half: indices n_half to n - 1 (terms 1/16 to 1/30)
    
    def generate_sums(start_idx, end_idx):
        """Generate all possible sums for terms from start_idx to end_idx-1"""
        sums = [Fraction(0, 1)]
        for i in range(start_idx, end_idx):
            term = terms[i]
            new_sums = []
            for s in sums:
                new_sums.append(s - term)
                new_sums.append(s)
                new_sums.append(s + term)
            sums = new_sums
            if time.time() > deadline:
                return None
        return sums

    sums1 = generate_sums(0, n_half)
    if sums1 is None:
        return [0]*28 + [1, -1]
        
    sums2 = generate_sums(n_half, n)
    if sums2 is None:
        return [0]*28 + [1, -1]
        
    # To find the pair (s1, s2) such that |s1 + s2| is minimized and non-zero.
    # We can sort sums2. Then for each s1 in sums1, we look for -s1 in sums2.
    # We want the value in sums2 closest to -s1.
    # Since we need exact non-zero, we check the closest values.
    
    # Sorting 14 million Fractions might be slow.
    # Alternative: Convert to a common denominator?
    # LCM(1..30) is 2329089562800.
    # We can represent each sum as an integer numerator divided by LCM.
    # Then we are just looking for integers.
    
    LCM = 1
    for i in range(1, n+1):
        LCM = LCM * i // __import__('math').gcd(LCM, i)
        
    # Convert sums to integers
    int_sums1 = [int(s * LCM) for s in sums1]
    int_sums2 = [int(s * LCM) for s in sums2]
    
    # Sort int_sums2
    int_sums2.sort()
    
    best_diff = float('inf')
    best_s1 = None
    best_s2 = None
    
    import bisect
    
    for s1 in int_sums1:
        if time.time() > deadline:
            break
        target = -s1
        # Find insertion point in int_sums2
        pos = bisect.bisect_left(int_sums2, target)
        
        # Check candidates around pos
        candidates = []
        if pos < len(int_sums2):
            candidates.append(int_sums2[pos])
        if pos > 0:
            candidates.append(int_sums2[pos-1])
            
        for s2 in candidates:
            diff = s1 + s2
            if diff != 0:
                abs_diff = abs(diff)
                if abs_diff < best_diff:
                    best_diff = abs_diff
                    best_s1 = s1
                    best_s2 = s2
                    
    if best_s1 is None:
        return [0]*28 + [1, -1]
        
    # Reconstruct coefficients
    # We need to find the specific coefficients that produce best_s1 and best_s2.
    # This is the hard part of MITM. We need to store the path or re-run DP to find the path.
    # Given the time constraints, let's just return a valid non-zero sum if we can't reconstruct quickly.
    # But the task asks for the coefficients.
    
    # To reconstruct, we can store the "choice" (which term was added) along with the sum.
    # Or, simpler: since we found the optimal value, we can use a DP that stores the parent pointer.
    # But MITM reconstruction is tricky.
    
    # Let's try a different approach for reconstruction:
    # We know the optimal sum S_opt = (best_s1 + best_s2) / LCM.
    # We can use a BFS/DFS from the start to find a path to S_opt? No, that's back to full search.
    
    # Alternative: Store the coefficients in the generation step?
    # That would blow up memory (14M * 15 bytes).
    
    # Let's try to reconstruct by doing a second pass DP that tracks the minimum "cost" or just finds one path to the target sum.
    # Actually, we can just run a local search starting from a random guess, but bias it towards the known optimal magnitude?
    # No, that's not guaranteed.
    
    # Let's refine the DP approach to store the path for the *best* sum only?
    # No, we don't know the best sum until we compare.
    
    # Okay, let's go back to the DP with sets, but store the coefficients?
    # For n=30, the number of unique sums is likely manageable if we prune?
    # No, we need exact.
    
    # Let's try a different strategy:
    # Use the MITM to find the optimal value.
    # Then, use a DP that only keeps sums that are "plausible" candidates for the final sum?
    # Or, simply, once we have the optimal numerator `best_num = best_s1 + best_s2`,
    # we can run a DP that finds *any* combination of coefficients that sums to `best_num`.
    # This is a subset sum problem with 3 elements per step.
    # We can do a DP where dp[i] is a set of integers (numerators) reachable using first i terms.
    # We stop when we find `best_num` in dp[n].
    # Then we backtrack.
    
    target_num = best_s1 + best_s2
    
    # DP for path reconstruction
    # dp[i] = set of reachable numerators
    # parent[i][num] = (prev_num, choice) where choice is -1, 0, 1 for term i
    # This might use too much memory if the set of numerators is large.
    # But the target is specific. We can do a bidirectional search or just a forward DP with pruning?
    # No, we just need to find ONE path to target_num.
    
    # Let's use a dictionary: dp[i] = {num: (prev_num, choice)}
    # We only store the first time we reach a num? No, we need to ensure we can reach the target.
    # Actually, since we know the target is reachable (from MITM), a simple BFS/DP forward will find it.
    
    dp_path = [dict() for _ in range(n + 1)]
    dp_path[0][0] = (None, None) # num 0 reached with no terms
    
    for i in range(n):
        if time.time() > deadline:
            break
        term_num = LCM // (i + 1) # Numerator of 1/(i+1) when denominator is LCM
        prev_dict = dp_path[i]
        curr_dict = dp_path[i + 1]
        
        for num, (prev_num, choice) in prev_dict.items():
            # Try -1, 0, 1
            for c in [-1, 0, 1]:
                new_num = num + c * term_num
                if new_num not in curr_dict:
                    curr_dict[new_num] = (num, c)
                    
                # Optimization: if we already found the target in the final step, we can stop early?
                # No, we need to fill the dict to allow backtracking.
                # But we can prune if the number of states becomes too large.
                # For n=30, the number of distinct numerators is bounded by range of sums.
                # Sum of 1/i is approx log(30) ~ 3.4.
                # Numerator range is approx 3.4 * LCM ~ 7 * 10^12.
                # This is too large for a dict if it fills up.
                
    # Check if target is reached
    if target_num in dp_path[n]:
        # Backtrack
        coeff = [0] * n
        curr_num = target_num
        for i in range(n - 1, -1, -1):
            prev_num, choice = dp_path[i + 1][curr_num]
            coeff[i] = choice
            curr_num = prev_num
        return coeff
    else:
        # Fallback
        return [0]*28 + [1, -1]

# EVOLVE-BLOCK-END
