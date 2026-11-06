import time
from fractions import Fraction

def construct():
    deadline = time.time() + 55
    target = Fraction(5, 31)
    
    # Known solution
    best = [8, 31, 248]
    best_max = 248
    
    def verify(dens):
        s = Fraction(0, 1)
        for d in dens:
            s += Fraction(1, d)
        return s == target

    # Search for smaller max denominator
    # C must be at least 31 because 31 | LCM(1..C)
    for C in range(31, 36):
        if time.time() > deadline - 5:
            break
            
        # We want to find a subset of {1, ..., C} summing to 5/31
        # Use DFS
        # To speed up, we can fix the number of terms or just let DFS find any
        
        # Precompute reciprocals
        recips = [Fraction(1, i) for i in range(1, C + 1)]
        
        # Pruning bounds:
        # For a given index i (next denominator to consider) and remaining terms k,
        # max_sum = sum of reciprocals of i, i+1, ..., i+k-1
        # min_sum = sum of reciprocals of C, C-1, ..., C-k+1
        
        # However, we don't know k in advance.
        # We can just do a standard subset sum DFS with pruning.
        
        def dfs(start_idx, current_sum, current_set):
            if time.time() > deadline - 2:
                return None
            
            if current_sum == target:
                return list(current_set)
            
            if current_sum > target:
                return None
            
            # If even taking all remaining smallest denominators (largest reciprocals)
            # doesn't reach target, prune.
            # Max possible add: sum of recips[start_idx : C]
            # This is an upper bound on what we can add.
            # If current_sum + sum(recips[start_idx:]) < target, prune.
            
            # Precompute suffix sums of reciprocals for quick upper bound check
            # suffix_sum[i] = sum(recips[i:])
            
            # Also, if current_sum + sum of largest possible k terms < target, prune?
            # We don't know k.
            
            for i in range(start_idx, C):
                if current_set[i]:
                    continue
                
                new_sum = current_sum + recips[i]
                if new_sum > target:
                    break # Since recips are decreasing, further i will be smaller, but wait.
                    # recips[i] = 1/(i+1).
                    # If we iterate i from start_idx to C-1, the denominators are start_idx+1, ..., C.
                    # The reciprocals are decreasing.
                    # So if new_sum > target, then for j > i, recips[j] < recips[i], so new_sum_j < new_sum.
                    # It might still be <= target. So we cannot break.
                    # We should continue.
                    
                current_set[i] = True
                res = dfs(i + 1, new_sum, current_set)
                current_set[i] = False
                
                if res is not None:
                    return res
                    
            return None
        
        # Initialize
        current_set = [False] * C
        # We can start from any denominator.
        # But to find a solution, we just need one.
        
        # Optimization: 
        # The target is 5/31 ~ 0.16.
        # If we start with small denominators, the sum grows fast.
        # If we start with large denominators, the sum grows slow.
        # We can iterate the first denominator in descending order?
        # Or just standard ascending.
        
        # Let's try to constrain the number of terms.
        # Min terms for C=31 is 5.
        # Max terms?
        # Min sum for k terms is sum of 1/C, 1/(C-1)...
        # For C=31, min sum for 5 terms is ~0.174 > 0.161.
        # So k must be at least 5?
        # Wait, 1/31+1/30+1/29+1/28+1/27 = 0.174.
        # Target is 0.161.
        # So 5 terms with largest denominators give 0.174 which is > 0.161.
        # So we can achieve 0.161 with 5 terms?
        # Yes, by picking slightly smaller denominators (larger reciprocals) or a mix.
        # Actually, if we pick smaller denominators, the sum increases.
        # So we can tune the sum from min_sum(5) to max_sum(5).
        # min_sum(5) = 0.174.
        # Target = 0.161.
        # Target < min_sum(5).
        # So 5 terms is NOT enough?
        # Wait. Min sum is achieved by largest denominators.
        # If Target < Min Sum, then we need MORE terms to get a smaller sum?
        # No, adding more terms INCREASES the sum.
        # So if Target < Min Sum for k terms, we need to check if we can get a smaller sum with k terms?
        # No, Min Sum is the smallest possible sum for k terms.
        # If Target < Min Sum(k), then it's impossible to reach Target with k terms.
        # We need a k such that Min Sum(k) <= Target <= Max Sum(k).
        
        # Let's check k=6 for C=31.
        # Min Sum(6) = 1/31+1/30+1/29+1/28+1/27+1/26 ≈ 0.174 + 0.038 = 0.212.
        # This is even larger.
        # So as k increases, Min Sum increases.
        # This implies that for C=31, the minimum possible sum for ANY number of terms is 1/31 (k=1).
        # And it increases with k.
        # So if Target < Min Sum(1), no solution.
        # Min Sum(1) = 1/31 ≈ 0.032.
        # Target = 0.161.
        # So we need k such that Min Sum(k) <= 0.161 <= Max Sum(k).
        
        # Let's re-evaluate Min Sum(k) for C=31.
        # Min Sum(k) uses the largest denominators: C, C-1, ..., C-k+1.
        # k=1: 1/31 = 0.032
        # k=2: 1/31 + 1/30 = 0.065
        # k=3: 1/31 + 1/30 + 1/29 = 0.101
        # k=4: 1/31 + 1/30 + 1/29 + 1/28 = 0.137
        # k=5: 1/31 + 1/30 + 1/29 + 1/28 + 1/27 = 0.174
        # Target 0.161 is between Min Sum(4) and Min Sum(5).
        # So k must be 4 or 5?
        # If k=4, Max Sum(4) = 1/1+1/2+1/3+1/4 = 2.08.
        # So [Min Sum(4), Max Sum(4)] = [0.137, 2.08]. Target 0.161 is in range.
        # So a solution with k=4 might exist for C=31.
        
        # If k=5, Max Sum(5) is even larger. Min Sum(5) = 0.174 > 0.161.
        # So k=5 is impossible for C=31?
        # Wait, Min Sum(5) is the SMALLEST sum with 5 terms.
        # If Target < Min Sum(5), then we cannot form Target with 5 terms.
        # So k=5 is impossible.
        # So we only need to check k=1, 2, 3, 4.
        
        # Let's check k=4 for C=31.
        # We need 4 distinct denominators from 1..31 summing to 5/31.
        
        # Let's modify the DFS to limit the number of terms.
        # Or just let it run, but the pruning "if current_sum > target: break" is wrong as noted before.
        # Correct pruning:
        # 1. If current_sum > target, prune.
        # 2. If current_sum + sum(recips[start_idx:]) < target, prune (even taking all remaining is not enough).
        # 3. If current_sum + sum of largest k possible < target? No.
        
        # Let's implement the DFS with these prunings.
        
        suffix_sums = [0] * (C + 1)
        for i in range(C - 1, -1, -1):
            suffix_sums[i] = suffix_sums[i + 1] + recips[i]
            
        def dfs(start_idx, current_sum, current_set, depth):
            if time.time() > deadline - 2:
                return None
            
            if current_sum == target:
                if depth > 0:
                    return list(current_set)
                return None # Need at least 1 term? Task says 1..256.
            
            if current_sum > target:
                return None
            
            # Prune if max possible remaining sum is too small
            if current_sum + suffix_sums[start_idx] < target:
                return None
            
            for i in range(start_idx, C):
                if current_set[i]:
                    continue
                
                new_sum = current_sum + recips[i]
                
                # If new_sum > target, we can skip this i?
                # No, because we might pick a larger denominator (smaller reciprocal) later?
                # No, we are picking i now. If 1/(i+1) makes sum > target, then any subset containing i has sum > target.
                # So we can skip i.
                if new_sum > target:
                    continue
                
                current_set[i] = True
                res = dfs(i + 1, new_sum, current_set, depth + 1)
                current_set[i] = False
                
                if res is not None:
                    return res
                    
            return None

        current_set = [False] * C
        res = dfs(0, Fraction(0, 1), current_set, 0)
        
        if res is not None:
            if verify(res):
                best = res
                best_max = C
                # Since we iterate C upwards, the first found is the minimum C.
                break

    return best
