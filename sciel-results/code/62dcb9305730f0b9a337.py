import time
import math
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin for 90s limit
    MAX_DEN = 10**7
    
    # We established k >= 8.
    # Try increasing k starting from 8.
    
    def find_solution(k):
        """
        Find k distinct denominators in [5, MAX_DEN] summing to 1.
        Uses DFS with pruning.
        """
        # To avoid deep recursion overhead and global state issues, 
        # we use a nested function.
        
        def dfs(rem, k_left, min_den):
            if time.time() > deadline:
                return None
            
            if k_left == 0:
                if rem == 0:
                    return []
                return None
            
            if k_left == 1:
                # 1/x = rem => x = 1/rem
                if rem.numerator == 1:
                    x = rem.denominator
                    if x > min_den and x <= MAX_DEN:
                        return [x]
                return None
            
            # Lower bound for next denominator d
            # We need 1/d <= rem => d >= 1/rem
            # 1/rem = rem.den / rem.num
            # d >= ceil(rem.den / rem.num)
            lower_d = max(min_den, math.ceil(rem.denominator / rem.numerator))
            
            if lower_d > MAX_DEN:
                return None
            
            # Upper bound for d
            # We need rem <= sum_{i=0}^{k_left-1} 1/(d+i)
            # Approx: rem <= k_left / d => d <= k_left / rem
            upper_d = min(MAX_DEN, int(k_left / rem) + 1)
            
            if lower_d > upper_d:
                return None
            
            # Iterate possible denominators
            for d in range(lower_d, upper_d + 1):
                if time.time() > deadline:
                    return None
                
                # Pruning 1: Max possible sum with k_left terms starting at d
                # If this sum is less than rem, we can't reach rem with any larger d either.
                # Calculate sum of 1/(d) + 1/(d+1) + ... + 1/(d+k_left-1)
                # For efficiency, we can approximate or just calculate since k_left is small.
                max_possible = 0.0
                for i in range(k_left):
                    max_possible += 1.0 / (d + i)
                
                if max_possible < float(rem) - 1e-12:
                    break  # Larger d will only decrease the sum further
                
                # Pruning 2: Check if remaining sum is achievable
                new_rem = rem - Fraction(1, d)
                if new_rem <= 0:
                    continue  # 1/d > rem, need larger d? 
                    # Wait, if 1/d > rem, then new_rem < 0.
                    # Since d is increasing, 1/d is decreasing.
                    # So if 1/d > rem, we should continue to larger d.
                    
                # Check if new_rem can be formed by k_left-1 terms starting from d+1
                # Max sum of k_left-1 terms starting from d+1 is approx (k_left-1)/(d+1)
                # If new_rem > (k_left-1)/(d+1), it's impossible.
                # As d increases, new_rem increases (1/d decreases) and (k_left-1)/(d+1) decreases.
                # So if this condition fails, it will fail for all larger d.
                if new_rem > Fraction(k_left - 1, d + 1):
                    break
                
                # Recursive step
                res = dfs(new_rem, k_left - 1, d + 1)
                if res is not None:
                    return [d] + res
            
            return None
        
        return dfs(Fraction(1), k, 5)

    # Try k = 8, 9, ...
    for k in range(8, 257):
        if time.time() > deadline - 5:
            break
        
        sol = find_solution(k)
        if sol is not None:
            # Verify solution
            if len(set(sol)) == len(sol) and all(5 <= d <= MAX_DEN for d in sol):
                s = sum(Fraction(1, d) for d in sol)
                if s == 1:
                    return sol
        else:
            # If find_solution returned None due to timeout or no solution, 
            # we might want to break if it's taking too long, 
            # but the inner dfs handles the deadline.
            # If it returns None quickly, we try next k.
            pass

    # Fallback: Greedy solution (not minimal, but valid)
    # This ensures we return *something* if DFS fails or times out.
    def greedy_solution():
        remaining = Fraction(1)
        denoms = []
        last = 4
        while remaining > 0 and len(denoms) < 256:
            # Find smallest d > last such that 1/d <= remaining
            # d >= 1/remaining
            d = max(last + 1, math.ceil(remaining.denominator / remaining.numerator))
            # Ensure d is within bounds
            if d > MAX_DEN:
                break
            denoms.append(d)
            remaining -= Fraction(1, d)
            last = d
        return denoms

    sol = greedy_solution()
    if sol:
        s = sum(Fraction(1, d) for d in sol)
        if s == 1 and len(set(sol)) == len(sol) and all(5 <= d <= MAX_DEN for d in sol):
            return sol

    raise RuntimeError("No solution found")
