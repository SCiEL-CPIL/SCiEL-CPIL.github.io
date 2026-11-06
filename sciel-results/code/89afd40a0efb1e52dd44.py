"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _check_mean_bitset(target, others):
    """Check if target is the mean of any subset of >=2 other elements using bitset DP."""
    if len(others) < 2:
        return False
    # We need: sum(subset) = k * target for some k >= 2, where subset is from others
    # Equivalent: sum(subset - target) = 0 with |subset| >= 2
    # Let's use the direct approach: for each possible subset size k >= 2,
    # check if sum of k elements from others equals k * target
    
    n = len(others)
    # For each k from 2 to n, check if there's a subset of size k with sum = k*target
    # Use DP: dp[k][s] = True if we can pick k elements summing to s
    # But this is expensive for large n. Let's use a smarter approach.
    
    # Alternative: use the delta approach
    # Let d_i = other_i - target. We need sum of subset of d's = 0 with size >= 2.
    # But we also need to track the count.
    
    # Use bitset for sums, tracking min count
    # The deltas range from (1-target) to (512-target)
    # Sum range: [-511*48, 511*48] approx
    
    # Actually, let's just do a direct subset sum check with early termination
    # For small sets this is fine
    
    deltas = [y - target for y in others]
    
    # dp[s] = minimum number of elements needed to achieve sum s
    # Initialize with infinity
    INF = 1000
    # The sum can range from sum(negatives) to sum(positives)
    min_sum = sum(d for d in deltas if d < 0)
    max_sum = sum(d for d in deltas if d > 0)
    
    # Offset to make indices non-negative
    offset = -min_sum
    size = max_sum - min_sum + 1
    
    if size > 100000:  # Safety check
        # Fall back to simpler check
        return False
    
    dp = [INF] * size
    dp[offset] = 0  # Sum 0 with 0 elements
    
    for d in deltas:
        new_dp = dp[:]
        for s in range(size):
            if dp[s] < INF:
                ns = s + d
                if 0 <= ns < size:
                    if dp[s] + 1 < new_dp[ns]:
                        new_dp[ns] = dp[s] + 1
        dp = new_dp
    
    # Check if sum 0 is achievable with >= 2 elements
    return dp[offset] >= 2

def _can_add_bitset(x, S):
    """Check if adding x to S maintains the no-mean property."""
    # Check if x is mean of any subset of S
    if _check_mean_bitset(x, S):
        return False
    # Check if any element in S is mean of any subset of (S + [x])
    # This is more complex - need to check if adding x creates new means
    for a in S:
        others = [y for y in S if y != a] + [x]
        if _check_mean_bitset(a, others):
            return False
    return True

def _conflict_score(x, S):
    """Score how many conflicts x would create with S (lower is better)."""
    # Count potential arithmetic progressions involving x
    score = 0
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            a, b = S[i], S[j]
            # Check if x, a, b form an AP or related patterns
            if 2*a == b + x or 2*b == a + x or 2*x == a + b:
                score += 1
    return score

def construct():
    start = time.time()
    deadline = start + 85  # Safety margin before 90s limit
    best_set = []
    
    # Strategy 1: Greedy with conflict scoring
    for trial in range(30):
        if time.time() > deadline:
            break
        
        S = []
        candidates = list(range(1, 513))
        random.shuffle(candidates)
        
        # Sort candidates by initial conflict potential (prefer numbers with fewer AP conflicts)
        # Pre-compute: for each candidate, count how many pairs in 1..512 would form AP with it
        for x in candidates:
            if time.time() > deadline:
                break
            if x in S:
                continue
            
            # Quick check: can we add x?
            if _can_add_bitset(x, S):
                S.append(x)
        
        if len(S) > len(best_set):
            best_set = S[:]
    
    # Strategy 2: Local search / hill climbing on best_set
    if len(best_set) > 0 and time.time() < deadline - 10:
        current = best_set[:]
        current_set = set(current)
        
        for iteration in range(200):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding a random element
            candidates = [x for x in range(1, 513) if x not in current_set]
            random.shuffle(candidates)
            
            for x in candidates[:20]:
                if time.time() > deadline - 5:
                    break
                if _can_add_bitset(x, current):
                    current.append(x)
                    current_set.add(x)
                    improved = True
                    break
            
            if not improved and len(current) > 10:
                # Try removing a random element and replacing with two
                idx = random.randint(0, len(current) - 1)
                removed = current.pop(idx)
                current_set.discard(removed)
                
                candidates = [x for x in range(1, 513) if x not in current_set]
                random.shuffle(candidates)
                
                added_any = False
                for x in candidates[:10]:
                    if time.time() > deadline - 5:
                        break
                    if _can_add_bitset(x, current):
                        current.append(x)
                        current_set.add(x)
                        added_any = True
                        break
                
                if not added_any:
                    current.append(removed)
                    current_set.add(removed)
        
        if len(current) > len(best_set):
            best_set = current[:]
    
    return best_set
# EVOLVE-BLOCK-END
