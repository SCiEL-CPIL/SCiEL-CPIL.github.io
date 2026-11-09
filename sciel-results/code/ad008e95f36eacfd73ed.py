import time
import random

def construct():
    deadline = time.time() + 55.0  # Safety margin (90s wall, 55s for construct)
    best_set = []
    best_size = 0
    rng = random.Random(42)
    
    def is_valid_set(S):
        """Check if no element in S is the mean of any subset of size >= 2 of other elements."""
        n = len(S)
        if n < 3:
            return True
        
        for i, a in enumerate(S):
            others = S[:i] + S[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            # Fast check for k=2: a = (b+c)/2 => 2a = b+c
            # b, c must be distinct elements from others.
            pair_sums = set()
            for idx in range(m):
                b = others[idx]
                for jdx in range(idx + 1, m):
                    c = others[jdx]
                    pair_sums.add(b + c)
            
            if 2 * a in pair_sums:
                return False

            # Fast check for k=3: a = (b+c+d)/3 => 3a = b+c+d
            # b, c, d must be distinct elements from others.
            # We can reuse pair sums?
            # sum = b + (c+d).
            # Iterate b, then check if 3a - b is a sum of 2 distinct elements from others \ {b}.
            # This is O(N^3) if done naively, but N is small.
            # Or just generate all triple sums.
            triple_sums = set()
            for idx in range(m):
                b = others[idx]
                for jdx in range(idx + 1, m):
                    c = others[jdx]
                    for kdx in range(jdx + 1, m):
                        d = others[kdx]
                        triple_sums.add(b + c + d)
            
            if 3 * a in triple_sums:
                return False

            # DP for k >= 4
            # We only need to check if k*a is reachable with k elements from others, for k>=4.
            # We can initialize DP with 0 elements.
            dp = [0] * (m + 1)
            dp[0] = 1
            
            for x in others:
                for k in range(m - 1, -1, -1):
                    if dp[k]:
                        dp[k+1] |= (dp[k] << x)
                        
            for k in range(4, m + 1):
                target = k * a
                if target > sum(others):
                    break
                if (dp[k] >> target) & 1:
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
            new_set = current + [x]
            if is_valid_set(new_set):
                current = new_set
        return current

    def local_search(S, iterations=50):
        """Try to improve set S via add/swap operations."""
        S = list(S)
        n = len(S)
        if n == 0:
            return S
        
        # Build a set for O(1) membership
        S_set = set(S)
        # Candidates not in S
        available = [x for x in range(1, 513) if x not in S_set]
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            rng.shuffle(available)
            
            # Try adding
            for x in available:
                if len(S) >= 48:
                    break
                new_set = S + [x]
                if is_valid_set(new_set):
                    S = new_set
                    S_set.add(x)
                    available.remove(x)
                    improved = True
                    break
            
            if len(S) >= 48:
                break
            
            # Try swapping: remove one element, add another
            if not improved:
                rng.shuffle(S)
                for i in range(len(S)):
                    if time.time() > deadline:
                        break
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = S_set - {removed}
                    for x in available:
                        if x in S_temp_set:
                            continue
                        new_set = S_temp + [x]
                        if is_valid_set(new_set):
                            S = new_set
                            S_set = S_temp_set | {x}
                            available.remove(x)
                            available.append(removed)
                            improved = True
                            break
                    if improved:
                        break
        return S

    # Strategy 1: Multiple random greedy passes
    for seed in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates = list(range(1, 513))
        rng.shuffle(candidates)
        
        # Try a few different orderings
        for _ in range(3):
            if time.time() > deadline:
                break
            result = try_greedy(candidates)
            if len(result) > best_size:
                best_size = len(result)
                best_set = result
            if best_size == 48:
                return best_set
    
    # Strategy 2: Local search from best found
    if best_set and time.time() < deadline:
        improved = local_search(best_set)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        if best_size == 48:
            return best_set
    
    # Strategy 3: Try specific structured sets (e.g., large numbers, sparse)
    if time.time() < deadline:
        # Try picking from upper half
        upper = list(range(257, 513))
        rng.shuffle(upper)
        result = try_greedy(upper)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
        
        # Try sparse: step 2
        sparse = list(range(1, 513, 2))
        result = try_greedy(sparse)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
        
        # Try sparse: step 3
        sparse3 = list(range(1, 513, 3))
        result = try_greedy(sparse3)
        if len(result) > best_size:
            best_size = len(result)
            best_set = result
    
    # Strategy 4: More local search attempts with different random restarts
    if best_set and time.time() < deadline:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb best set
            perturbed = list(best_set)
            rng = random.Random(time.time_ns())
            # Remove a random element
            if perturbed:
                idx = rng.randrange(len(perturbed))
                removed = perturbed.pop(idx)
                # Try to add something
                available = [x for x in range(1, 513) if x not in set(perturbed)]
                rng.shuffle(available)
                for x in available[:50]:
                    new_set = perturbed + [x]
                    if is_valid_set(new_set):
                        perturbed = new_set
                        break
            if len(perturbed) > best_size:
                best_size = len(perturbed)
                best_set = perturbed
            # Local search on perturbed
            improved = local_search(perturbed, iterations=20)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved
            if best_size == 48:
                return best_set
    
    return best_set
