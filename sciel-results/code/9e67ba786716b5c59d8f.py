"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute suffix sums for upper bounds
    # suffix_sum[i] = sum(1/d for d in denominators[i:])
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, denominators[i])
        
    best_sum = Fraction(0)
    best_indices = []
    
    # Initial good guesses to set a high bar for pruning
    # Guess 1: 1/2 + 1/3 + 1/7 + 1/43
    guess_indices = [0, 1, 5, 41] # Indices for 2, 3, 7, 43
    guess_sum = sum(Fraction(1, d) for d in [2, 3, 7, 43])
    if guess_sum < 1:
        best_sum = guess_sum
        best_indices = guess_indices
        
    # Guess 2: 1/2 + 1/3 + 1/8 + 1/24 + 1/30 + 1/40 + 1/48 ?
    # Let's try a few random-ish combinations or just rely on B&B
    # Another known good one: 1/2 + 1/3 + 1/8 + 1/24 + 1/30 + 1/40 + 1/42 ?
    # 1/2+1/3+1/8 = 23/24.
    # 23/24 + 1/24 = 1.
    # So we can't use 24 with 2,3,8.
    # 23/24 + 1/30 + 1/40 + 1/48 ...
    # 23/24 = 0.95833
    # 1/30 = 0.03333 -> 0.99166
    # 1/40 = 0.025 -> 1.016 (Too big)
    # 1/30 + 1/48 = 0.0333 + 0.0208 = 0.0541 -> 1.012 (Too big)
    # 1/30 + 1/60 (not in set)
    # 1/30 + 1/42 = 0.0333 + 0.0238 = 0.0571 -> 1.015
    # 1/30 + 1/48 + 1/50 (no)
    # 1/30 + 1/48 + 1/60 (no)
    # 1/30 + 1/48 + 1/70 (no)
    # 1/30 + 1/48 + 1/80 (no)
    # 1/30 + 1/48 + 1/90 (no)
    # 1/30 + 1/48 + 1/100 (no)
    # 1/30 + 1/48 + 1/120 (no)
    # 1/30 + 1/48 + 1/140 (no)
    # 1/30 + 1/48 + 1/160 (no)
    # 1/30 + 1/48 + 1/180 (no)
    # 1/30 + 1/48 + 1/200 (no)
    # 1/30 + 1/48 + 1/220 (no)
    # 1/30 + 1/48 + 1/240 (no)
    # 1/30 + 1/48 + 1/260 (no)
    # 1/30 + 1/48 + 1/280 (no)
    # 1/30 + 1/48 + 1/300 (no)
    # 1/30 + 1/48 + 1/320 (no)
    # 1/30 + 1/48 + 1/340 (no)
    # 1/30 + 1/48 + 1/360 (no)
    # 1/30 + 1/48 + 1/380 (no)
    # 1/30 + 1/48 + 1/400 (no)
    # 1/30 + 1/48 + 1/420 (no)
    # 1/30 + 1/48 + 1/440 (no)
    # 1/30 + 1/48 + 1/460 (no)
    # 1/30 + 1/48 + 1/480 (no)
    
    # It seems 1/2+1/3+1/7+1/43 is quite strong.
    
    sys.setrecursionlimit(10000)
    node_count = 0
    max_nodes = 5_000_000 # Safety limit
    
    current_indices = []
    
    def dfs(idx, current_sum):
        nonlocal best_sum, best_indices, node_count
        node_count += 1
        if node_count > max_nodes:
            return
            
        # Pruning: if current sum + all remaining <= best_sum, no point
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices.copy()
            return
            
        # Option 1: Include denominators[idx]
        new_sum = current_sum + Fraction(1, denominators[idx])
        if new_sum < 1:
            current_indices.append(idx)
            dfs(idx + 1, new_sum)
            current_indices.pop()
            
        # Option 2: Exclude denominators[idx]
        dfs(idx + 1, current_sum)
        
    dfs(0, Fraction(0))
    
    return best_indices
# EVOLVE-BLOCK-END
