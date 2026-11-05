import time
import random
from itertools import combinations

_DEADLINE = 85.0  # seconds, safety margin before 90s limit

def _check_violation_full(A, x):
    """
    Full check if adding x to A creates any violation.
    A is a list of ints. x is an int not in A.
    Returns True if violation exists, False otherwise.
    """
    C = A + [x]
    n = len(C)
    
    # Check 1: Is x the mean of any subset of A (size >= 2)?
    # This is equivalent to: Is there a subset S of A, |S|>=2, such that sum(S) = |S|*x?
    # Let y_i = a_i - x. We need sum(y_i) = 0 for |S|>=2.
    
    # Check 2: Is any a in A the mean of any subset of (A \ {a}) U {x} (size >= 2)?
    # This is equivalent to: For each a in A, is there a subset S of (A \ {a}) U {x} such that a is the mean?
    # Let T = (A \ {a}) U {x}.
    # We can just run the check for each element in C.
    
    # To be efficient, we can just iterate through each element in C and check if it's a mean of others.
    # This is O(n) calls to a helper that does DP on n-1 elements.
    # n <= 48. n-1 <= 47.
    # DP size: sum of magnitudes.
    
    for i in range(n):
        target = C[i]
        others = C[:i] + C[i+1:]
        m = len(others)
        if m < 2:
            continue
            
        # Transform
        zs = [y - target for y in others]
        
        # Calculate max sum magnitude
        max_sum = sum(abs(z) for z in zs)
        if max_sum == 0:
            continue
            
        offset = max_sum
        mask = (1 << (2 * max_sum + 1)) - 1
        
        # dp[k] bitset
        dp = [0] * (m + 1)
        dp[0] = 1 << offset
        
        for z in zs:
            for k in range(m, 0, -1):
                if dp[k-1] == 0:
                    continue
                if z >= 0:
                    shifted = dp[k-1] << z
                else:
                    shifted = dp[k-1] >> (-z)
                dp[k] |= (shifted & mask)
                
                # Early exit
                if k >= 2 and (dp[k] & (1 << offset)):
                    return True
                    
        for k in range(2, m + 1):
            if dp[k] & (1 << offset):
                return True
                
    return False

def _check_violation_fast(A, x):
    """
    Fast heuristic check.
    A is a list of ints. x is an int.
    Returns True if violation is detected (definite violation), False if no violation found (might be valid).
    """
    # Check 1: x is mean of subset of A
    # k=2: 2x = a+b
    # k=3: 3x = a+b+c
    
    # We can maintain pair_sums and triple_sums in the outer loop to make this O(1) or O(N).
    # But here we just do a simple check.
    
    # Pair check
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            if A[i] + A[j] == 2 * x:
                return True
            if A[i] + A[j] == 3 * x: # Wait, 3x is sum of 3 elements.
                 pass
                 
    # This is getting complicated to do correctly in fast mode without precomputed sets.
    # Let's rely on the outer loop maintaining the sets.
    return False

def construct():
    start_time = time.time()
    best_set = []
    
    # Candidate pool
    candidates = list(range(1, 513))
    
    # We will try multiple random orders
    # Also try some structured orders
    
    orders = []
    
    # 1. Random orders
    rng = random.Random(42)
    for _ in range(200): # Adjust based on time
        if time.time() - start_time > _DEADLINE - 5:
            break
        rand_order = candidates[:]
        rng.shuffle(rand_order)
        orders.append(rand_order)
        
    # 2. Sorted orders
    orders.append(sorted(candidates))
    orders.append(sorted(candidates, reverse=True))
    
    # 3. "Multiplicative" order: sort by value, but prioritize primes?
    # Let's just stick to random + sorted.
    
    for order in orders:
        if time.time() - start_time > _DEADLINE - 5:
            break
            
        selected = []
        # Precomputed sums for fast check
        pair_sums = set()
        triple_sums = set()
        
        for x in order:
            if time.time() - start_time > _DEADLINE - 5:
                break
                
            # Fast Check
            valid = True
            
            # Check 1: x is mean of subset of selected
            # k=2: 2x in pair_sums
            if 2 * x in pair_sums:
                valid = False
            # k=3: 3x in triple_sums
            elif 3 * x in triple_sums:
                valid = False
            # k=4: 4x in quadruple_sums? (Too many to maintain)
            # If fast check passes, do full check.
            # Full check is expensive, so we only do it if fast check passes.
            
            if valid:
                # Check 2: Any a in selected is mean of subset involving x
                # k=1: a = (s + x)/2 => 2a = s + x => s = 2a - x
                # s must be in selected, s != a
                for a in selected:
                    s = 2 * a - x
                    if s in selected and s != a:
                        valid = False
                        break
                    # k=2: a = (s1 + s2 + x)/3 => 3a = s1 + s2 + x => s1+s2 = 3a - x
                    # Check if 3a - x is a sum of 2 distinct elements in selected
                    # We have pair_sums. But we need to ensure the pair doesn't involve a?
                    # Actually, the subset is from (selected \ {a}) U {x}.
                    # The pair s1, s2 must be from selected \ {a}.
                    # If 3a - x is in pair_sums, it might be formed by a.
                    # We can just do a quick check:
                    if valid:
                        target_sum = 3 * a - x
                        # Check if target_sum can be formed by 2 elements in selected \ {a}
                        # This is O(N^2) if done naively.
                        # For speed, we might skip k=2 in fast check and rely on full check.
                        # But full check is slow.
                        # Let's try to be smarter.
                        # If target_sum is in pair_sums, it's likely a violation.
                        # We can verify if the pair is valid.
                        if target_sum in pair_sums:
                            # Verify
                            found = False
                            for i in range(len(selected)):
                                for j in range(i+1, len(selected)):
                                    if selected[i] == a or selected[j] == a:
                                        continue
                                    if selected[i] + selected[j] == target_sum:
                                        found = True
                                        break
                                if found:
                                    break
                            if found:
                                valid = False
                                break
                            
            if not valid:
                continue
                
            # If fast check passed, run full check to be sure
            # This is the expensive part
            if _check_violation_full(selected, x):
                continue
                
            # Add x
            selected.append(x)
            
            # Update pair_sums
            for a in selected[:-1]: # selected[:-1] is the old selected
                pair_sums.add(a + x)
                
            # Update triple_sums
            # Add x + a + b for all pairs (a,b) in old selected
            # We can iterate over all pairs in old selected
            old_selected = selected[:-1]
            for i in range(len(old_selected)):
                for j in range(i+1, len(old_selected)):
                    triple_sums.add(old_selected[i] + old_selected[j] + x)
                    
            if len(selected) > len(best_set):
                best_set = selected[:]
                
            if len(selected) >= 48:
                break
                
    # Final verification
    if best_set:
        if _check_violation_full(best_set[:-1], best_set[-1]) if len(best_set) > 1 else False:
             # This check is incomplete. We need to verify the whole set.
             # But _check_violation_full was used during construction.
             # Let's do a final full check on the whole set just in case.
             pass
             
    # The _check_violation_full inside the loop ensures validity at each step.
    # So best_set should be valid.
    
    return best_set

# EVOLVE-BLOCK-END
