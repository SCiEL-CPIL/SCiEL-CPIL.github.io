import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # 85 seconds, safe margin from 90s
    
    # We determined k=7 is impossible, so we start with k=8
    for k in range(8, 15):
        if time.time() > deadline:
            break
        solution = find_solution(k, start_time, deadline)
        if solution:
            return solution
            
    # Fallback if no small k found (should not happen)
    # Use a known valid solution or DP
    # 1/5 + 1/6 + 1/7 + 1/10 + 1/20 + 1/14 + 1/35 + ... 
    # Let's just return a valid set if found by a simpler method
    # But the DFS should find it.
    raise RuntimeError("No solution found within time limit")

def find_solution(k, start_time, deadline):
    target = Fraction(1, 1)
    # d[0] < d[1] < ... < d[k-1]
    # Sum 1/d[i] = 1
    # d[i] >= 5
    
    def dfs(idx, min_d, remaining, path):
        if time.time() > deadline:
            return None
        if idx == k:
            if remaining == 0:
                return path[:]
            return None
        
        # Max possible sum from this idx with smallest possible denominators
        # d[idx] >= min_d
        # d[idx+1] >= min_d+1 ...
        # This bound is tight and useful for pruning.
        # However, calculating sum of 1/(min_d+j) for j=0..k-idx-1 is cheap.
        
        # Lower bound for d[idx]:
        # remaining = 1/d[idx] + ... + 1/d[k-1]
        # remaining <= (k-idx) * (1/min_d)  => min_d <= (k-idx)/remaining
        # Also d[idx] >= min_d
        
        # Upper bound for d[idx]:
        # 1/d[idx] <= remaining => d[idx] >= 1/remaining
        # Also remaining = 1/d[idx] + sum_rest
        # sum_rest <= (k-idx-1) * 1/(d[idx]+1)
        # remaining <= 1/d[idx] + (k-idx-1)/(d[idx]+1)
        # This gives a lower bound on d[idx]? No, upper bound on 1/d[idx].
        
        # Let's use the bound: d[idx] <= (k-idx) / remaining
        # Because sum_{j=idx}^{k-1} 1/d[j] <= (k-idx)/d[idx]
        # So remaining <= (k-idx)/d[idx] => d[idx] <= (k-idx)/remaining
        
        max_d = int((k - idx) / remaining)
        if max_d < min_d:
            return None
            
        # Also d[idx] must be at least 1/remaining
        # But min_d is already >= 5 and increasing.
        
        for d in range(min_d, max_d + 1):
            # Check if 1/d is too small?
            # If 1/d + max_sum_rest < remaining, then d is too large?
            # No, if d is large, 1/d is small.
            # We need sum = remaining.
            # If 1/d > remaining, then d is too small? No, 1/d must be part of sum.
            # Actually, if 1/d > remaining, we can't subtract it.
            if Fraction(1, d) > remaining:
                continue
                
            new_remaining = remaining - Fraction(1, d)
            if new_remaining < 0:
                continue
                
            path.append(d)
            res = dfs(idx + 1, d + 1, new_remaining, path)
            if res:
                return res
            path.pop()
            
        return None

    return dfs(0, 5, target, [])

# EVOLVE-BLOCK-END
