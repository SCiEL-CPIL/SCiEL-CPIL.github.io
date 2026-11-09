import time
import random
import sys

def construct():
    deadline = time.time() + 55.0
    N = 512
    MAX_SIZE = 48
    
    # Precompute: for each element a, and each possible subset size k>=2,
    # we need to check if k*a can be formed as a sum of k distinct elements from [1..N]\{a}
    # But that's too expensive to precompute fully. Instead, we'll use a fast validity check.
    
    def is_valid_set(S):
        """Check if no element in S is the mean of any subset of size >= 2 of other elements."""
        n = len(S)
        if n < 3:
            return True
        
        S_set = set(S)
        
        for i, a in enumerate(S):
            others = [x for x in S if x != a]
            m = len(others)
            if m < 2:
                continue
            
            # For each k from 2 to m, check if k*a is a sum of k distinct elements from others
            # Use bitset DP: dp[k] is a bitset where bit s is set if sum s is achievable with k elements
            # Max sum of k elements from others <= k * N
            max_sum = k_max_sum = m * N  # upper bound
            # We only need up to m*a for each k, but let's be safe
            # Actually target is k*a, so we only need sums up to max(k*a for k in range(2,m+1)) = m*a
            target_max = m * a
            
            # Bitset DP: use Python integers as bitsets
            # dp[k] = bitset of achievable sums with exactly k elements
            dp = [0] * (m + 1)
            dp[0] = 1  # sum 0 with 0 elements
            
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
            
            for k in range(2, m + 1):
                target = k * a
                if target > target_max:
                    break
                if (dp[k] >> target) & 1:
                    return False
        
        return True
    
    def fast_validity_incremental(S, new_elem):
        """Check if adding new_elem to S keeps the set valid.
        Only need to check:
        1. new_elem is not the mean of any subset of S
        2. No element in S becomes the mean of a subset that includes new_elem
        """
        S_set = set(S)
        if new_elem in S_set:
            return False
        
        # Check 1: new_elem is not mean of any subset of S (size >= 2)
        m = len(S)
        if m >= 2:
            target_max = m * new_elem
            dp = [0] * (m + 1)
            dp[0] = 1
            for x in S:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
            for k in range(2, m + 1):
                target = k * new_elem
                if target > target_max:
                    break
                if (dp[k] >> target) & 1:
                    return False
        
        # Check 2: For each element a in S, check if a is mean of subset of (S \ {a}) U {new_elem}
        # The subset must have size >= 2 and be from OTHER elements (not a)
        # So subset is from (S \ {a}) U {new_elem}
        for a in S:
            others = [x for x in S if x != a]
            others.append(new_elem)
            m2 = len(others)
            if m2 < 2:
                continue
            target_max2 = m2 * a
            dp2 = [0] * (m2 + 1)
            dp2[0] = 1
            for x in others:
                for k in range(m2 - 1, -1, -1):
                    if dp2[k]:
                        dp2[k+1] |= (dp2[k] << x)
            for k in range(2, m2 + 1):
                target = k * a
                if target > target_max2:
                    break
                if (dp2[k] >> target) & 1:
                    return False
        
        return True
    
    def try_greedy(candidates, max_size=48):
        """Greedy: try to add each candidate in order, keep if valid."""
        current = []
        for x in candidates:
            if len(current) >= max_size:
                break
            if time.time() > deadline:
                break
            if fast_validity_incremental(current, x):
                current.append(x)
        return current
    
    def simulated_annealing(initial_set, iterations=200000):
        """Simulated annealing to maximize set size."""
        S = list(initial_set)
        S_set = set(S)
        available = [x for x in range(1, N + 1) if x not in S_set]
        
        best_set = list(S)
        best_size = len(S)
        
        T = 1.0
        cooling = 0.99995
        
        for it in range(iterations):
            if time.time() > deadline:
                break
            
            if len(S) >= MAX_SIZE:
                break
            
            # Choose a move: add, remove, or swap
            move_type = random.random()
            
            if len(S) < MAX_SIZE and move_type < 0.4:
                # Try to add
                if not available:
                    continue
                x = random.choice(available)
                if fast_validity_incremental(S, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = list(S)
                # For SA, we might accept invalid moves but that's tricky
                # Let's just do valid moves for simplicity
            elif move_type < 0.7 and len(S) > 1:
                # Try to remove
                idx = random.randrange(len(S))
                removed = S[idx]
                S.pop(idx)
                S_set.discard(removed)
                available.append(removed)
            else:
                # Try to swap
                if not available or len(S) == 0:
                    continue
                idx = random.randrange(len(S))
                removed = S[idx]
                x = random.choice(available)
                
                # Check if swap is valid
                S_temp = [y for i, y in enumerate(S) if i != idx]
                if fast_validity_incremental(S_temp, x):
                    S[idx] = x
                    S_set.discard(removed)
                    S_set.add(x)
                    available.remove(x)
                    available.append(removed)
                    if len(S) > best_size:
                        best_size = len(S)
                        best_set = list(S)
        
        T *= cooling
        return best_set
    
    # Strategy 1: Multiple random greedy passes with different orderings
    best_set = []
    best_size = 0
    
    for seed in range(300):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        result = try_greedy(candidates)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
        if best_size == MAX_SIZE:
            return best_set
    
    # Strategy 2: Simulated annealing from best greedy result
    if best_set and time.time() < deadline:
        improved = simulated_annealing(best_set, iterations=500000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == MAX_SIZE:
            return best_set
    
    # Strategy 3: Try structured initial sets
    if time.time() < deadline:
        # Large numbers
        upper = list(range(257, N + 1))
        rng = random.Random(999)
        rng.shuffle(upper)
        result = try_greedy(upper)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
        
        # Sparse
        sparse = list(range(1, N + 1, 3))
        rng.shuffle(sparse)
        result = try_greedy(sparse)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Strategy 4: More SA restarts
    if best_set and time.time() < deadline:
        for restart in range(5):
            if time.time() > deadline:
                break
            # Perturb
            perturbed = list(best_set)
            rng = random.Random(1234 + restart)
            if perturbed:
                for _ in range(min(3, len(perturbed))):
                    idx = rng.randrange(len(perturbed))
                    removed = perturbed.pop(idx)
                    avail = [x for x in range(1, N + 1) if x not in set(perturbed)]
                    if avail:
                        perturbed.append(rng.choice(avail))
            
            improved = simulated_annealing(perturbed, iterations=200000)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == MAX_SIZE:
                return best_set
    
    return best_set
