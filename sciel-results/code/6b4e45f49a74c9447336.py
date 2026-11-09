import time
import random

def construct():
    deadline = time.time() + 55.0
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def is_valid_incremental(S, x):
        """
        Checks if S U {x} is valid, given that S is valid.
        S is a list of integers.
        x is the new integer to add.
        """
        if x in S:
            return False
        
        # 1. Check if x is the mean of a subset of S
        # We need to check if k*x is a subset sum of S for any k >= 2
        # Subset sums of S
        # We can compute this on the fly or maintain it. 
        # For simplicity and correctness, we compute subset sums of S here.
        # Optimization: We only need to check sums up to max(S) * len(S) ? 
        # Actually, k*x <= sum(S). So k <= sum(S)/x.
        
        n = len(S)
        if n < 2:
            # If S has < 2 elements, x cannot be mean of subset of S (size >= 2)
            # But wait, if S has 2 elements, x can be mean.
            pass
            
        # Compute subset sums of S with cardinality
        # dp[k] is a bitset of sums achievable with k elements
        max_sum = sum(S)
        dp = [0] * (n + 1)
        dp[0] = 1
        
        for val in S:
            for k in range(n - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= (dp[k] << val)
                    
        for k in range(2, n + 1):
            target = k * x
            if target > max_sum:
                break
            if (dp[k] >> target) & 1:
                return False

        # 2. Check if any a in S becomes invalid due to x
        # For each a in S, check if k*a is a subset sum of (S \ {a}) U {x} for any k >= 2
        # This means sum(T) + x = k*a for some T subset of S \ {a}, |T| >= 1 (since x is one element, total size k = |T|+1 >= 2 => |T|>=1)
        # So sum(T) = k*a - x.
        # We need to check if k*a - x is a subset sum of S \ {a} for any k >= 2 and |T| >= 1.
        # Note: If |T|=0, then x = k*a. Since x != a (distinct) and x > 0, a > 0, this implies x is a multiple of a.
        # But the subset must have size >= 2. If T is empty, the subset is {x}, size 1. Not allowed.
        # So we require |T| >= 1.
        
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m < 1:
                continue
                
            # We need subset sums of 'others'
            # To avoid recomputing for every a, we could precompute, but m is small (N-1).
            # Let's compute dp for 'others'
            dp_o = [0] * (m + 1)
            dp_o[0] = 1
            for val in others:
                for k in range(m - 1, -1, -1):
                    if dp_o[k]:
                        dp_o[k+1] |= (dp_o[k] << val)
                        
            max_sum_o = sum(others)
            
            # k*a - x must be a sum of a subset of 'others' of size j >= 1.
            # And total subset size is j + 1 (for x).
            # Condition: subset size >= 2 => j + 1 >= 2 => j >= 1.
            # Also k >= 2.
            
            for k in range(2, m + 2): # Max size of subset is m+1
                target = k * a - x
                if target < 0:
                    continue
                if target > max_sum_o:
                    break # Since a > 0, k*a increases, but target = k*a - x. 
                          # If k*a - x > max_sum_o, larger k will also be > max_sum_o.
                          
                # We need to check if target is a sum of a subset of 'others' of size j such that j >= 1.
                # Also, the subset in the original condition has size k.
                # The subset is T U {x}. Size is |T| + 1.
                # So |T| = k - 1.
                # We need to check if target is a sum of a subset of 'others' of size exactly k - 1.
                # And k - 1 >= 1 => k >= 2.
                
                j = k - 1
                if j <= m:
                    if (dp_o[j] >> target) & 1:
                        return False
                        
        return True

    def try_greedy(candidates):
        current = []
        for x in candidates:
            if len(current) >= 48:
                break
            if time.time() > deadline:
                break
            if is_valid_incremental(current, x):
                current.append(x)
        return current

    def local_search(S):
        S = list(S)
        S_set = set(S)
        available = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= 48:
                    break
                if is_valid_incremental(S, x):
                    S.append(x)
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= 48:
                break
            
            # Try swapping
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    # S_temp is valid (subset of valid set)
                    for x in available:
                        if x in S_temp:
                            continue
                        # Check if S_temp U {x} is valid
                        # We can use is_valid_incremental(S_temp, x)
                        if is_valid_incremental(S_temp, x):
                            S = S_temp + [x]
                            S_set = set(S)
                            available.remove(x)
                            available.append(removed)
                            improved = True
                            break
                    if improved:
                        break
        return S

    # 1. Structured Greedy Passes
    # Descending
    desc = list(range(512, 0, -1))
    res = try_greedy(desc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Ascending
    asc = list(range(1, 513))
    res = try_greedy(asc)
    if len(res) > best_size:
        best_size = len(res)
        best_set = res
        
    # Random Passes
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        res = try_greedy(candidates)
        if len(res) > best_size:
            best_size = len(res)
            best_set = res
        if best_size == 48:
            return best_set

    # 2. Local Search
    if best_set and time.time() < deadline:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == 48:
            return best_set

    # 3. More Local Search with restarts
    if best_set and time.time() < deadline:
        for _ in range(5):
            if time.time() > deadline:
                break
            perturbed = list(best_set)
            rng = random.Random(time.time_ns())
            if perturbed:
                idx = rng.randrange(len(perturbed))
                perturbed.pop(idx)
            improved = local_search(perturbed)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == 48:
                return best_set

    return best_set
