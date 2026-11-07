import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    MAX_SUM = 256 * 10 + 10 # Safe upper bound for sums
    
    # Precompute multiples bitsets
    # Mults[n] has bit k set if k is a multiple of n
    Mults = [0] * (MAX_SUM + 1)
    for n in range(1, MAX_SUM + 1):
        bits = 0
        for m in range(n, MAX_SUM + 1, n):
            bits |= (1 << m)
        Mults[n] = bits

    def can_add(S, x, sums_set, sums_bitset):
        # 1. Check divisibility with elements in S
        for a in S:
            if x % a == 0 or a % x == 0:
                return False, None, None
        
        # 2. Generate new sums
        # New sums are x and s+x for s in sums_set
        new_sums_set = set(sums_set)
        new_sums_set.add(x)
        for s in sums_set:
            new_sums_set.add(s + x)
        
        # Check if any new sum exceeds MAX_SUM? 
        # Max sum is sum(S) + x. If S has 10 elements of 256, sum is 2560.
        # MAX_SUM is 2570. So we are safe.
        
        new_sums_bitset = 0
        for s in new_sums_set:
            new_sums_bitset |= (1 << s)
            
        # 3. Check divisibility conflicts
        # A. Old sums (sums_set) vs New sums (new_sums_set)
        # Note: new_sums_set contains sums_set, so we just check new_sums_set against itself?
        # No, we need to check if any pair in new_sums_set divides another.
        # Since sums_set was valid, we only need to check:
        # - Any s in sums_set divides x? (Checked in step 1? No, step 1 checked a in S. s is sum. s can be larger than a.)
        # Wait, step 1 checked elements of S. 
        # s is a sum of elements in S.
        # If s divides x, then s and x are in new_sums_set and s|x.
        # If x divides s, then x and s are in new_sums_set and x|s.
        # So we MUST check sums_set against x.
        
        # Let's just check all pairs in new_sums_set using bitsets.
        # For each b in new_sums_set:
        #   Check if (new_sums_bitset & Mults[b]) has any bit other than b.
        
        # Optimization:
        # We know sums_set is valid.
        # So conflicts can only involve x or s+x.
        
        # Check x against sums_set
        if (sums_bitset & Mults[x]) != 0:
            return False, None, None
        if (new_sums_bitset & Mults[x]) != (1 << x): # x divides some other sum in new set?
             # Actually, if x divides s (s in sums_set), then s is in new_sums_bitset and Mults[x].
             # So this check covers x | s.
             # The check above (sums_bitset & Mults[x]) covers s | x? No, Mults[x] contains multiples of x.
             # If s is a multiple of x, s is in Mults[x]. So sums_bitset & Mults[x] != 0 means x|s.
             # What about s|x?
             # If s|x, then x is a multiple of s. So x is in Mults[s].
             # We need to check if any s in sums_set divides x.
             # Loop through sums_set? Or use bitsets?
             # We don't have a "Divisors" bitset.
             # But we can check: for each s in sums_set, if x % s == 0.
             # sums_set size is small (< 1000).
             pass

        # Let's stick to the robust O(|NewSums|) bitwise check.
        # It's fast enough.
        
        for b in new_sums_set:
            # Check if any other element in new_sums_set is a multiple of b
            # (new_sums_bitset & Mults[b]) will contain b itself.
            # If it contains anything else, it's a conflict.
            intersection = new_sums_bitset & Mults[b]
            if intersection != (1 << b):
                return False, None, None
                
        return True, new_sums_set, new_sums_bitset

    # Greedy to find initial solution
    best_set = []
    best_size = 0
    
    # Try to build a large set greedily
    # Order: Primes first? Large numbers first?
    # Let's try a few random orders and greedy
    import random
    
    candidates = list(range(1, 257))
    
    for _ in range(50):
        random.shuffle(candidates)
        current_set = []
        current_sums = set()
        current_bitset = 0
        
        for x in candidates:
            if time.time() > deadline:
                break
            ok, new_sums, new_bitset = can_add(current_set, x, current_sums, current_bitset)
            if ok:
                current_set.append(x)
                current_sums = new_sums
                current_bitset = new_bitset
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    # DFS to improve
    # We want to find a set larger than best_size
    # Search for size best_size + 1
    
    def dfs(S, sums_set, sums_bitset, remaining_candidates, target_size):
        nonlocal best_set, best_size, deadline
        
        if time.time() > deadline:
            return False
            
        if len(S) >= target_size:
            if len(S) > best_size:
                best_size = len(S)
                best_set = S[:]
            return True
            
        # Pruning
        if len(S) + len(remaining_candidates) < target_size:
            return False
            
        # Try adding each candidate
        for i, x in enumerate(remaining_candidates):
            if time.time() > deadline:
                return False
                
            ok, new_sums, new_bitset = can_add(S, x, sums_set, sums_bitset)
            if ok:
                # Recurse
                new_remaining = remaining_candidates[i+1:]
                # We can also pass a filtered remaining list?
                # For now, just pass the rest.
                
                if dfs(S + [x], new_sums, new_bitset, new_remaining, target_size):
                    return True
                    
        return False

    # Try to find best_size + 1
    # We need to order candidates well for DFS.
    # Sort candidates by "divisibility freedom"?
    
    # Let's try a few target sizes starting from best_size + 1
    # If we fail to find best_size+1, we are done.
    
    # To make DFS effective, we should order candidates.
    # Candidates that are less likely to conflict are better to try first?
    # Or candidates that allow more future extensions?
    
    # Let's try increasing target size
    target = best_size + 1
    if target <= 30: # Max possible
        # Prepare candidates
        # We can restrict candidates to those not in best_set? No, we are searching from scratch or extending?
        # The DFS above starts from empty S.
        # It's better to start from best_set and try to extend?
        # But best_set might be a local minimum.
        # Let's run DFS from scratch to find a set of size target.
        
        # To speed up, we can use the best_set as a guide?
        # No, just run DFS.
        
        # Optimization: If best_size is 7, target is 8.
        # Number of candidates 256.
        # Depth 8.
        # Branching factor is high.
        # We need strong pruning.
        
        # The `can_add` check is fast.
        # The main cost is the number of nodes.
        
        # Let's try to find a solution of size target.
        # We'll run DFS with a time limit.
        
        # To improve success rate, try multiple random orderings of candidates for DFS?
        # Or just one good ordering.
        
        # Ordering: Numbers with few divisors are better?
        # Let's sort candidates by number of divisors in 1..256.
        
        def num_divisors(n):
            count = 0
            for i in range(1, int(n**0.5) + 1):
                if n % i == 0:
                    count += 1
                    if i != n // i:
                        count += 1
            return count

        sorted_candidates = sorted(range(1, 257), key=num_divisors)
        
        # Try DFS with this ordering
        # But DFS from scratch with 256 candidates and depth 8 is huge.
        # We need to prune more.
        
        # Alternative: Local Search on best_set
        # Try to swap elements to allow extension?
        
        # Given the time limit, maybe just trying to extend best_set is safer?
        # But best_set is likely maximal (no single element can be added).
        
        # Let's try a hybrid:
        # 1. Get best_set (size K).
        # 2. Try to find a set of size K+1 using DFS.
        #    To make DFS feasible, we can restrict the search space.
        #    But we don't know which elements to keep.
        
        # Let's try a few random restarts of the Greedy with different heuristics.
        # The greedy already ran 50 times.
        
        # Maybe the max size is just 7 or 8.
        # If we can't find 8 quickly, we return 7.
        
        pass

    if best_set:
        return [int(x) for x in best_set]
    return [31, 32, 34]
