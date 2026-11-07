import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin (90s limit, leave 15s buffer)
    
    N = 256
    candidates = list(range(2, N + 1))
    n = len(candidates)  # 255 elements (2..256)
    
    # Index: element i corresponds to value 2+i
    # Precompute conflict bitsets
    # conflict[a] = set of (b,c) pairs such that a | b*c
    # For efficiency: for each a, precompute which pairs (b,c) in candidates conflict
    
    # Actually, let's think about it differently.
    # A triple (a,b,c) is bad if a|bc or b|ac or c|ab.
    # For a given set S, we need no bad triple within S.
    
    # Precompute: for each pair (i,j), the set of k such that (i,j,k) is a bad triple.
    # bad_triple[i][j] = bitmask of k values where (candidates[i], candidates[j], candidates[k]) is bad
    
    # Since n=255, we can use Python ints as bitmasks (arbitrary precision)
    
    # Precompute all bad pairs for each element
    # conflict_mask[a] = bitmask of all (b,c) pairs... no, that's too large.
    
    # Better: for each a, precompute the set of b such that there exists c with a|bc.
    # Actually, let's precompute: for each a, the bitmask of elements b such that 
    # adding a to a set containing b and c would create a conflict.
    
    # Let me use a different approach:
    # For each element a, precompute:
    #   partners[a] = set of b such that gcd(a,b) < a (i.e., a/b doesn't divide evenly in a useful way)
    # Actually, the condition a | b*c means: let g = gcd(a,b), then (a/g) | c.
    
    # Precompute for each pair (a,b): what values c make (a,b,c) a bad triple?
    # c must satisfy: a/gcd(a,b) | c, OR b/gcd(a,b) | c, OR c/gcd(a,c) | b, etc.
    
    # This is getting complex. Let me use a simpler bitset approach:
    # For each a, precompute conflict_with[a] = bitmask of all elements b such that
    # there exists some c where (a,b,c) is a bad triple.
    # This is an over-approximation but helps with pruning.
    
    # Actually, let me just precompute the full conflict structure more carefully.
    
    # For a set S to be valid, for every distinct a,b,c in S, a does not divide b*c.
    
    # Precompute: for each a, the set of "required c's" given b.
    # If a is in S and b is in S, then c cannot be in S if a | b*c.
    # a | b*c iff (a / gcd(a,b)) | c.
    
    # So: for each pair (a,b), define required_exclusion[a][b] = set of c such that (a/gcd(a,b)) | c.
    # If a and b are both in S, then no c in required_exclusion[a][b] can be in S.
    
    # Let's precompute this as bitmasks.
    
    vals = candidates  # [2, 3, ..., 256]
    n = len(vals)
    
    # Precompute required_exclusion as a list of lists of bitmasks
    # req_excl[a_idx][b_idx] = bitmask of c_idx such that vals[a_idx] | vals[b_idx] * vals[c_idx]
    
    # This is n^2 * n = 255^3 ~ 16.5M operations, which should be fast enough.
    
    print(f"Precomputing conflict structure for {n} elements...")
    precompute_start = time.time()
    
    req_excl = [[0] * n for _ in range(n)]
    
    for i in range(n):
        a = vals[i]
        for j in range(n):
            if i == j:
                continue
            b = vals[j]
            g = __import__('math').gcd(a, b)
            need = a // g  # c must be divisible by 'need'
            mask = 0
            if need > 1:
                for k in range(n):
                    if k == i or k == j:
                        continue
                    if vals[k] % need == 0:
                        mask |= (1 << k)
            req_excl[i][j] = mask
    
    precompute_time = time.time() - precompute_start
    print(f"Precompute took {precompute_time:.2f}s")
    
    if time.time() > deadline - 10:
        # Not enough time, return a simple answer
        # Upper half [129, 256] is an antichain, size 128
        return list(range(129, 257))
    
    # Now use this for a branch-and-bound / greedy approach
    
    def is_valid_set(S):
        """Check if set S (list of indices) is valid."""
        S_set = set(S)
        for i in S:
            for j in S:
                if i == j:
                    continue
                mask = req_excl[i][j]
                # Check if any k in S_set is in mask
                for k in S_set:
                    if (mask >> k) & 1:
                        return False
        return True
    
    def greedy_with_order(order):
        """Greedy: add elements in given order if valid."""
        S = []
        S_set = set()
        for idx in order:
            # Check if adding idx is valid
            valid = True
            for j in S:
                mask = req_excl[idx][j]
                for k in S_set:
                    if (mask >> k) & 1:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                S.append(idx)
                S_set.add(idx)
        return S
    
    def local_search(S, time_limit=5.0):
        """Try to improve set S by adding elements or swapping."""
        S = list(S)
        S_set = set(S)
        current_size = len(S)
        start_ls = time.time()
        
        improved = True
        while improved and time.time() < start_ls + time_limit:
            improved = False
            
            # Try to add any element
            for idx in range(n):
                if idx in S_set:
                    continue
                # Check if adding idx is valid
                valid = True
                for j in S:
                    mask = req_excl[idx][j]
                    for k in S_set:
                        if (mask >> k) & 1:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    S.append(idx)
                    S_set.add(idx)
                    current_size += 1
                    improved = True
                    break
            
            # If no add, try swaps
            if not improved:
                for i in range(len(S)):
                    removed = S[i]
                    S_temp = S[:i] + S[i+1:]
                    S_temp_set = S_set - {removed}
                    
                    for idx in range(n):
                        if idx in S_temp_set:
                            continue
                        # Check if adding idx to S_temp is valid
                        valid = True
                        for j in S_temp:
                            mask = req_excl[idx][j]
                            for k in S_temp_set:
                                if (mask >> k) & 1:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            S = S_temp + [idx]
                            S_set = S_temp_set | {idx}
                            improved = True
                            break
                    if improved:
                        break
        
        return S
    
    # Strategy 1: Start with upper half (antichain) and try to add lower elements
    upper_half = list(range(127, n))  # indices for 129..256
    best_set = greedy_with_order(upper_half)
    best_size = len(best_set)
    print(f"Upper half greedy: {best_size}")
    
    if time.time() < deadline - 10:
        improved = local_search(best_set, 10.0)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
            print(f"After local search: {best_size}")
    
    # Strategy 2: Greedy with various orderings
    if time.time() < deadline - 5:
        orders = [
            list(range(n, 0, -1)),  # Largest first
            list(range(n)),         # Smallest first
            sorted(range(n), key=lambda i: vals[i], reverse=True),
            sorted(range(n), key=lambda i: -vals[i]),
        ]
        
        for order in orders:
            if time.time() > deadline - 5:
                break
            s = greedy_with_order(order)
            if len(s) > best_size:
                best_set = s
                best_size = len(s)
                print(f"New best from greedy: {best_size}")
        
        # Randomized restarts
        random.seed(42)
        while time.time() < deadline - 5:
            order = list(range(n))
            random.shuffle(order)
            # Bias towards larger values
            order.sort(key=lambda i: random.random() * 0.5 + vals[i] / 256.0 * 0.5, reverse=True)
            s = greedy_with_order(order)
            if len(s) > best_size:
                best_set = s
                best_size = len(s)
                print(f"New best from random: {best_size}")
            
            if time.time() < deadline - 3:
                s = local_search(s, 2.0)
                if len(s) > best_size:
                    best_set = s
                    best_size = len(s)
                    print(f"New best after LS: {best_size}")
    
    # Convert indices back to values
    result = [vals[i] for i in best_set]
    return [int(x) for x in result]
