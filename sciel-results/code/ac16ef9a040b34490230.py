import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety

    def popcount(x):
        return bin(x).count('1')

    def to_result(n, adj_masks):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_masks[u] & (1 << v):
                    edges.append([u, v])
        return {'n': int(n), 'edges': edges}

    def is_regular_subset(S_mask, adj_masks, min_size=6):
        # S_mask is an integer representing the subset
        # Check if size >= min_size
        if popcount(S_mask) < min_size:
            return False
        
        # Get degrees of vertices in S
        # We need to check if all degrees are equal
        # Iterate through bits in S
        temp = S_mask
        first_deg = -1
        while temp:
            # Get lowest set bit
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = popcount(adj_masks[v] & S_mask)
            if first_deg == -1:
                first_deg = deg
            elif deg != first_deg:
                return False
            temp ^= lsb
        return True

    def find_bad_subset(adj_masks, n, min_k=6):
        # Iterate all subsets of size >= min_k
        # To be fast, we can iterate all 2^n masks, but that's 1M for n=20.
        # Checking each takes ~n popcounts.
        # Total ~20M ops. In Python, this might take 1-2 seconds.
        # If we do this in a repair loop, it's too slow.
        
        # Optimization: Only check subsets of size exactly min_k first?
        # Regular induced subgraphs of size > min_k often contain regular subgraphs of size min_k?
        # No, a regular graph of size 7 doesn't necessarily have a regular induced subgraph of size 6.
        # Example: C7 (cycle) is 2-regular. Induced subgraphs of size 6 are paths (not regular).
        # So we MUST check all k >= 6.
        
        # However, if we are repairing, we can just check k=6, then k=7, etc.
        # And stop at the first one found.
        
        # Let's try to find a bad subset by sampling? No, must be deterministic for validity.
        # But for repair, we just need ONE bad subset to fix.
        # If we fix all k=6, then check k=7, etc.
        
        # This function is used in the repair loop. It needs to be fast.
        # Let's restrict the search in the repair loop to k=6 initially.
        # If no k=6 bad subset, then check k=7, etc.
        
        # For the final validation, we need to be sure.
        
        # Let's implement a fast check for a specific k using combinations?
        # No, bitmask iteration over all masks is simpler and likely faster than itertools for n=20 if optimized.
        # But 1M iterations in Python is slow.
        
        # Alternative: Precompute popcount for all masks? No, 1M array is fine.
        # But we still have to iterate.
        
        # Let's try a different approach for repair:
        # Just check k=6 subsets using itertools.combinations but with bitmask degree calc.
        # C(20,6) = 38760. This is small.
        # C(20,7) = 77520.
        # C(20,8) = 125970.
        # C(20,9) = 167960.
        # C(20,10) = 184756.
        # Sum for k=6..20 is ~1M.
        
        # So iterating all subsets once is ~1M checks.
        # Each check: popcount(adj[v] & S) for v in S.
        # If we use itertools, we get S as a tuple. We need to convert to mask or just sum.
        # Converting tuple to mask is slow.
        
        # Let's stick to bitmask iteration but only for the *check* phase, not repair.
        # For repair, we will use a heuristic:
        # 1. Pick a random subset of size 6. Check if regular.
        # 2. If not, pick another.
        # 3. If we can't find one in X tries, assume no k=6 regular subsets?
        #    This is risky.
        
        # Better: Use the bitmask full scan but limit it.
        # Actually, for n=20, a full scan of 1M masks taking ~1s is acceptable if we do it a few times.
        # But repair loop might run 100-1000 times.
        
        # Let's try to construct a valid graph by starting with a known structure or random and repairing.
        # If repair is too slow, return n=10.
        
        # Let's try n=20 with a limited repair budget.
        
        pass

    # We will implement the repair using a specific strategy:
    # 1. Generate random graph.
    # 2. Loop:
    #    a. Find a bad subset (prioritize small k).
    #    b. If found, flip an edge in it.
    #    c. If not found, verify full validity (expensive) or trust if k=6..8 clean?
    #       Task says "No set of 6 or more".
    #       If we ensure no regular subsets of size 6, 7, 8, 9, 10, is it safe?
    #       Larger subsets are harder to be regular?
    #       Actually, if a subset S of size K is regular, it might contain a regular subset of size 6?
    #       Not necessarily.
    #       
    #       However, random graphs are unlikely to have large regular induced subgraphs if small ones are absent.
    #       But we must be correct.
    #       
    #       Let's do a full verification at the end.
    #       If it takes too long, we return a smaller n.

    def find_bad_subset_fast(adj_masks, n, max_k_check=10):
        # Check k=6, 7, ..., max_k_check
        # If none found, return None (optimistically)
        # We will do a full check in the final validation.
        
        # To speed up, we can just check k=6.
        # If k=6 is clean, check k=7.
        
        for k in range(6, max_k_check + 1):
            for S in itertools.combinations(range(n), k):
                # Convert S to mask
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                # Check regularity
                first_deg = -1
                is_reg = True
                for v in S:
                    deg = bin(adj_masks[v] & mask).count('1')
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return S
        return None

    def validate_full(adj_masks, n):
        # Check all k >= 6
        for k in range(6, n + 1):
            for S in itertools.combinations(range(n), k):
                mask = 0
                for v in S:
                    mask |= (1 << v)
                
                first_deg = -1
                is_reg = True
                for v in S:
                    deg = bin(adj_masks[v] & mask).count('1')
                    if first_deg == -1:
                        first_deg = deg
                    elif deg != first_deg:
                        is_reg = False
                        break
                if is_reg:
                    return False
        return True

    # Try n=20
    # We will try a few random restarts.
    # If we find a graph that passes a *partial* check (k=6..8) quickly, we do a full check.
    # If full check is too slow, we abort.
    
    best_result = None
    
    # Fallback valid graph: n=5 (empty)
    n_fallback = 5
    adj_fallback = [0] * n_fallback
    best_result = to_result(n_fallback, adj_fallback)
    
    # Try n=20
    n_target = 20
    if time.time() < deadline - 30:
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random graph
            adj_masks = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            # Repair loop
            repaired = False
            for step in range(500):
                if time.time() > deadline - 15:
                    break
                
                # Find bad subset (check k=6 first, then 7, 8)
                bad_S = find_bad_subset_fast(adj_masks, n_target, max_k_check=8)
                
                if bad_S is None:
                    # Looks good for k<=8.
                    # Let's try to validate fully, but with a time limit.
                    # If validation takes too long, we might skip.
                    # But for n=20, full validation is ~1M subsets.
                    # Each subset check is ~n ops. ~20M ops.
                    # In Python, ~2-5s.
                    # We have ~15s left. So we can afford 1-2 full validations.
                    
                    if validate_full(adj_masks, n_target):
                        result = to_result(n_target, adj_masks)
                        best_result = result
                        repaired = True
                        break
                    else:
                        # Found a bad subset in full validation?
                        # validate_full returns False but doesn't give the subset.
                        # We need to find it to repair.
                        # Let's modify validate_full to return the subset.
                        # Or just break and try another restart.
                        break
                else:
                    # Flip a random edge in bad_S
                    u, v = random.sample(list(bad_S), 2)
                    adj_masks[u] ^= (1 << v)
                    adj_masks[v] ^= (1 << u)
            
            if repaired:
                break

    # If n=20 failed, try n=15 or n=10?
    # The task asks to maximize n.
    # If we didn't get n=20, maybe we can get n=15?
    # But n=20 is the target. If we fail, n=5 is very low score.
    # Let's try to ensure we return a valid graph with n as large as possible.
    # If n=20 repair fails, try n=15.
    
    if best_result['n'] < 20 and time.time() < deadline - 10:
        n_target = 15
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            adj_masks = [0] * n_target
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    if random.random() < 0.5:
                        adj_masks[i] |= (1 << j)
                        adj_masks[j] |= (1 << i)
            
            for step in range(200):
                if time.time() > deadline - 5:
                    break
                bad_S = find_bad_subset_fast(adj_masks, n_target, max_k_check=8)
                if bad_S is None:
                    if validate_full(adj_masks, n_target):
                        result = to_result(n_target, adj_masks)
                        best_result = result
                        break
                    else:
                        break
                else:
                    u, v = random.sample(list(bad_S), 2)
                    adj_masks[u] ^= (1 << v)
                    adj_masks[v] ^= (1 << u)
            
            if best_result['n'] >= n_target:
                break

    return best_result
