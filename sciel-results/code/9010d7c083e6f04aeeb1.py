"""Rewritten program: branch-and-bound DFS over denominators [2..48] in descending order.

Goal: find a subset of indices (into denominators 2..48) whose reciprocal sum is
strictly below 1, minimizing the deficit 1 - sum.

Approach:
- Denominators are 2..48, so 47 items.
- We do a DFS in descending denominator order (largest denominators first? No — 
  actually the directive says "iterates denominators in descending order". Let's think:
  "descending order" of denominators means starting from 48 down to 2. But large 
  denominators contribute small fractions. To find sums close to 1, we likely want 
  to include many terms. Starting with large denominators (small contributions) 
  allows us to build up sums incrementally. However, for pruning, we want to 
  compute the maximum possible additional sum from remaining items.
  
  Actually, let's reconsider. The directive says: "iterates denominators in descending 
  order, pruning any branch where the current sum plus the sum of all remaining 
  reciprocals cannot exceed the best deficit found so far".
  
  Wait, the pruning condition is: if current_sum + sum_of_remaining < best_sum_found, 
  then we can't beat the best, so prune. But we want to MINIMIZE deficit, i.e., 
  MAXIMIZE sum (while keeping it < 1). So we want to find the maximum sum < 1.
  
  So the best deficit corresponds to the best (maximum) sum < 1.
  
  Pruning: if current_sum + remaining_max_sum <= best_sum, then even adding all 
  remaining won't beat best_sum, so prune.
  
  But we also need to ensure current_sum < 1 at the end. During search, if 
  current_sum >= 1, we must prune (since we need strictly below 1).
  
  Let's implement:
  - Precompute suffix sums: for each position i (in the ordered list), suffix_sum[i] 
    = sum of reciprocals from index i to end.
  - Order: descending denominators means indices for d=48, 47, ..., 2. But the 
    return value needs zero-based indices into the original array [2,3,...,48]. 
    So index 0 -> d=2, index 46 -> d=48.
    
  Let me define:
    denoms = list(range(2, 49))  # [2, 3, ..., 48]
    We'll search in order of descending denominators: 48, 47, ..., 2.
    For each denominator d, its original index is d - 2.
    
  - Use Fraction for exact arithmetic.
  - DFS: at each step, decide to include or exclude the current denominator.
  - Prune if current_sum >= 1 (invalid, since we need strictly below 1).
  - Prune if current_sum + suffix_sum[remaining] <= best_sum (can't improve).
  
  Since 47 items, worst case 2^47 is too big, but pruning should help significantly.
  
  Let's also try to get a good initial best_sum. The greedy approach (include if 
  it keeps sum < 1) gives a decent starting point.
  
  Let me implement this carefully.
"""

def construct():
    from fractions import Fraction
    
    denoms = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denoms)  # 47
    
    # Order: descending denominators: 48, 47, ..., 2
    # For each position in this order, store the original index and the fraction
    ordered = []
    for d in range(48, 1, -1):  # 48 down to 2
        idx = d - 2  # zero-based index into denoms
        frac = Fraction(1, d)
        ordered.append((idx, frac))
    
    # Precompute suffix sums: suffix_sum[i] = sum of ordered[i][1] + ... + ordered[n-1][1]
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + ordered[i][1]
    
    # Initial best: greedy
    best_sum = Fraction(0)
    best_indices = []
    current = Fraction(0)
    for idx, frac in ordered:
        if current + frac < 1:
            current += frac
            best_indices.append(idx)
    best_sum = current
    
    # DFS
    # We'll use iterative or recursive DFS. Recursive with pruning.
    # To avoid deep recursion issues, 47 levels should be fine.
    
    # We'll track the current selection as a list of indices
    current_selection = []
    
    def dfs(pos, current_sum):
        nonlocal best_sum, best_indices
        
        # Prune: if current_sum >= 1, invalid
        if current_sum >= 1:
            return
        
        # Prune: if we can't beat best_sum
        if current_sum + suffix_sum[pos] <= best_sum:
            return
        
        # Base case: all positions processed
        if pos == n:
            if current_sum > best_sum and current_sum < 1:
                best_sum = current_sum
                best_indices = list(current_selection)
            return
        
        # Option 1: include ordered[pos]
        idx, frac = ordered[pos]
        new_sum = current_sum + frac
        if new_sum < 1:
            current_selection.append(idx)
            dfs(pos + 1, new_sum)
            current_selection.pop()
        
        # Option 2: exclude ordered[pos]
        # But only if we can still potentially beat best_sum
        if current_sum + suffix_sum[pos + 1] > best_sum:
            dfs(pos + 1, current_sum)
    
    dfs(0, Fraction(0))
    
    # Ensure best_indices is not empty (it shouldn't be, but just in case)
    if not best_indices:
        # Fallback: just take index 0 (denom 2, frac 1/2)
        return [0]
    
    return best_indices
