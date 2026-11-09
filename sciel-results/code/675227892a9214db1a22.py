"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from itertools import combinations

def _check_valid(S):
    """
    Check if set S is 'reciprocal-subsum-free'.
    Condition: For any distinct x, y1, ..., yk in S,
    1/x != 1/y1 + ... + 1/yk.
    
    Using integer arithmetic to avoid floating point errors:
    1/x = sum(1/y_i)
    => 1/x = (sum(prod(y_j for j!=i) for i)) / prod(y_i)
    => prod(y_i) = x * sum(prod(y_j for j!=i) for i)
    """
    S = list(S)
    n = len(S)
    if n <= 1:
        return True
        
    # Precompute reciprocals as fractions? No, use the product formula.
    # For a candidate x in S, and a subset T of S \ {x}:
    # Check if 1/x == sum(1/t for t in T)
    
    # Optimization: 
    # 1/x = sum(1/t for t in T)
    # Multiply by x * prod(T):
    # prod(T) = x * sum( prod(T \ {t}) for t in T )
    
    # This is still expensive for large T.
    # However, note that if 1/x = sum(1/t), then sum(1/t) >= 1/max(T).
    # Also 1/x = sum(1/t) implies that the sum of reciprocals is exactly 1/x.
    
    # A faster check:
    # For each x in S:
    #   Let R = S \ {x}
    #   We want to know if there is a subset T of R such that sum(1/t for t in T) == 1/x.
    #   This is equivalent to: is 1/x a subset sum of the multiset {1/t for t in R}?
    
    # Since n is small (max 24), we can iterate over subsets?
    # 2^23 is too big.
    # But we can use the property: if 1/x = sum(1/t), then the sum is relatively small.
    
    # Let's use the integer identity:
    # sum(1/t for t in T) = 1/x
    # Let P = prod(T).
    # sum(P/t for t in T) = P/x
    # So P/x must be an integer, say K.
    # And sum(P/t for t in T) = K.
    
    # So for a fixed x and T, we can check:
    # 1. P = prod(T)
    # 2. K = P // x (if P % x != 0, invalid)
    # 3. Check if sum(P//t for t in T) == K. Note: P is divisible by t, so P//t is exact.
    
    # How to find T?
    # We can iterate over subsets of R.
    # To speed up, we can prune.
    # sum(1/t for t in T) = 1/x.
    # Since t >= x+1 (if we assume sorted? No, t can be smaller).
    # Actually, if t < x, then 1/t > 1/x.
    # If T contains any t < x, then sum(1/t) > 1/x unless there are other terms? No, reciprocals are positive.
    # So if any t in T is less than x, sum(1/t) > 1/x.
    # Therefore, all t in T must be > x.
    
    # This is a huge pruning!
    # For a given x, only elements in S that are strictly greater than x can be in T.
    
    # Let's re-verify:
    # 1/x = 1/t1 + 1/t2 + ...
    # Since 1/ti > 0, each 1/ti < 1/x.
    # Therefore 1/ti < 1/x => ti > x.
    # So yes, T must be a subset of {y in S | y > x}.
    
    # This makes the check much faster.
    
    for x in S:
        # Candidates for T are elements in S greater than x
        candidates = [y for y in S if y > x]
        
        # We need to check if any subset of 'candidates' sums to 1/x.
        # Using the integer check:
        # Let P = prod(T).
        # Check if P % x == 0.
        # Let K = P // x.
        # Check if sum(P // y for y in T) == K.
        
        # Since candidates are > x, and x >= 1.
        # If x is large, candidates are few.
        # If x is small, candidates are many, but 1/x is large, so T must be small?
        # Not necessarily.
        
        # We can use a recursive search or subset enumeration with pruning.
        # Since len(candidates) <= 24, and we prune based on sum.
        
        # Pruning:
        # sum(1/y for y in T) = 1/x.
        # If we build T incrementally, we can keep track of the current sum.
        # But floating point is risky.
        
        # Let's stick to the integer product check but enumerate subsets of candidates.
        # To be efficient, we can sort candidates descending?
        # Actually, let's just use a simple recursive backtracking with the integer check at the end?
        # Or better, use the property that sum(1/y) = 1/x.
        
        # Let's try a different approach for the check:
        # Multiply 1/x = sum(1/y) by x * prod(T):
        # prod(T) = x * sum( prod(T \ {y}) for y in T )
        
        # Let's implement a helper to check if 1/x is a subset sum of {1/y for y in candidates}.
        # We can use a set of possible sums? No, fractions.
        
        # Let's use the recursive check:
        def can_sum_to_one_over_x(cands, target_inv_x):
            # target_inv_x is 1/x. We want sum(1/y for y in T) == 1/x.
            # We represent fractions as (num, den) or just use the product trick.
            # Let's use the product trick in the recursion.
            # State: (index, current_product, current_sum_numer)
            # This is getting complicated.
            
            # Simpler: Iterate over subsets of cands.
            # len(cands) is at most 24.
            # But we can prune: if sum(1/y for y in cands[i:]) < 1/x, return False.
            # And if sum(1/y for y in cands[i:]) + current_sum < 1/x, etc.
            
            # Let's just use the integer check on the final subset.
            # To find the subset, we can use a simple DFS.
            # But since we are inside a loop for each x, and S is small, maybe we can just check all subsets?
            # 2^24 is 16 million. Too slow if done for each x.
            
            # But we established T must be from {y > x}.
            # If x is the max element, candidates is empty.
            # If x is small, candidates is large.
            
            # Let's optimize the subset check.
            # We want to know if there exists T subset of cands such that sum(1/y for y in T) == 1/x.
            
            # We can use a dictionary to store possible sums?
            # No, the sums are fractions.
            
            # Let's use the integer identity:
            # sum(1/y for y in T) = 1/x  <=>  prod(T) = x * sum(prod(T\{y}) for y in T)
            
            # Let's try a recursive search that builds T and checks the condition.
            # We can prune if the partial sum of reciprocals exceeds 1/x.
            
            cands = sorted(cands, reverse=True) # Larger y first (smaller 1/y)
            
            # Actually, smaller y (larger 1/y) first might prune faster if we exceed 1/x.
            cands = sorted(cands) # Smaller y first
            
            def dfs(idx, current_prod, current_sum_num, current_sum_den):
                # current_sum = current_sum_num / current_sum_den
                # We want current_sum == 1/x
                # i.e., current_sum_num * x == current_sum_den
                
                if idx == len(cands):
                    if current_sum_num * x == current_sum_den:
                        return True
                    return False
                
                # Option 1: Don't include cands[idx]
                if dfs(idx + 1, current_prod, current_sum_num, current_sum_den):
                    return True
                
                # Option 2: Include cands[idx]
                y = cands[idx]
                # New sum = current_sum + 1/y
                # New num = current_sum_num * y + current_sum_den
                # New den = current_sum_den * y
                new_num = current_sum_num * y + current_sum_den
                new_den = current_sum_den * y
                
                # Pruning: if new_sum > 1/x, return False
                # new_num / new_den > 1/x  =>  new_num * x > new_den
                if new_num * x > new_den:
                    return False
                
                if dfs(idx + 1, current_prod * y, new_num, new_den):
                    return True
                
                return False
            
            return dfs(0, 1, 0, 1)
        
        if can_sum_to_one_over_x(candidates, 1/x):
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    best_set = list(range(13, 25)) # Fallback
    best_size = len(best_set)
    
    # Greedy construction
    # Start with a good seed?
    # The seed {13..24} has size 12.
    # Let's try to build a set greedily.
    # Order: We want to maximize size.
    # Usually, larger numbers are "safer" because their reciprocals are smaller,
    # making it harder for them to be the sum of others?
    # Or is it harder for them to be formed by others?
    # If x is large, 1/x is small. It's hard to sum small reciprocals to get 1/x?
    # Wait, if x is large, candidates for T (y > x) are very few or none.
    # So large x are easy to include.
    # Small x are hard to include because they have many candidates y > x.
    # So we should include large x first?
    
    # Let's try greedy from 24 down to 1.
    S = []
    for x in range(24, 0, -1):
        if time.time() > deadline:
            break
        candidate_S = S + [x]
        if _check_valid(candidate_S):
            S.append(x)
    
    if len(S) > best_size:
        best_set = S
        best_size = len(S)
    
    # Try to improve by local search / swaps?
    # Or try different orders?
    # Maybe start with a subset of the seed and add smaller ones?
    
    # Let's try a second greedy: Start with {24, 23, ...} and try to add smaller ones.
    # This is what we did above.
    
    # Let's try another strategy:
    # Start with an empty set and add elements in increasing order?
    # Small x first. This will fail often.
    
    # Let's try a randomized restart?
    # Given the time limit, simple greedy might be enough to beat 12.
    # The knowledge says "monotone greedy + beam search + local add" yielded 13-15.
    
    # Let's implement a simple local search:
    # Try to add any missing element.
    # Try to swap an element with a missing one to increase size? (Size stays same, but might allow adding more later?)
    
    # Let's try to add elements not in S.
    missing = [i for i in range(1, 25) if i not in S]
    for x in missing:
        if time.time() > deadline:
            break
        candidate_S = S + [x]
        if _check_valid(candidate_S):
            S.append(x)
            if len(S) > best_size:
                best_set = S
                best_size = len(S)
    
    # Try to remove one element and add two?
    # This is expensive.
    
    # Let's try a different greedy order:
    # Maybe sort by some heuristic?
    # For now, let's return the best found.
    
    return best_set
# EVOLVE-BLOCK-END
