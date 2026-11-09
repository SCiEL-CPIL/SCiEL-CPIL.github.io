import time
import random
import sys

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums_bitsets(S):
    """
    Compute bitsets of achievable sums for each cardinality.
    sums[k] is an integer where bit s is 1 if sum s is achievable with exactly k elements.
    """
    n = len(S)
    # Max possible sum is sum(S), which is at most 60 * 256 = 15360.
    # We need bitsets large enough.
    # sums[k] will be a Python int.
    sums = [0] * (n + 1)
    sums[0] = 1  # bit 0 is set (sum 0)
    
    for x in S:
        # Update in reverse to avoid using the element multiple times
        # Actually, we are building from scratch for each check, or updating incrementally?
        # For local search, we might want to update incrementally, but for simplicity and correctness in this scratch rewrite, 
        # let's recompute or use a careful update.
        # Recomputing is O(N^2) bitwise ops. N=60 -> 3600 ops. Very fast.
        pass
        
    # Let's do it properly incrementally in a helper, but for a single check:
    # sums[0] = 1
    # for x in S:
    #   for k in range(current_max_k, -1, -1):
    #      sums[k+1] |= sums[k] << x
    # But we need to be careful with the "exactly k" part.
    # Let's just recompute from scratch for each candidate set in the search.
    # It's safer and N is small.
    
    sums = [0] * (n + 1)
    sums[0] = 1
    for x in S:
        # We need to update sums[k] for k from current_max down to 0?
        # No, the standard subset sum DP is:
        # new_sums[k] = old_sums[k] | (old_sums[k-1] << x)
        # Since we are adding elements one by one, we can do:
        # for k in range(current_n, -1, -1):
        #    sums[k+1] |= sums[k] << x
        # This works if we start with sums[0]=1 and others 0, and add elements one by one.
        # Let's trace:
        # S = [a, b]
        # Init: sums[0]=1, sums[1]=0, sums[2]=0
        # Add a:
        #   k=0: sums[1] |= sums[0] << a -> sums[1] = 1<<a
        #   sums = [1, 1<<a, 0]
        # Add b:
        #   k=1: sums[2] |= sums[1] << b -> sums[2] = (1<<a) << b = 1<<(a+b)
        #   k=0: sums[1] |= sums[0] << b -> sums[1] = (1<<a) | (1<<b)
        #   sums = [1, 1<<a | 1<<b, 1<<(a+b)]
        # This is correct.
        
        # We need to know the current max cardinality to bound the loop.
        # Actually, we can just loop from current length of S processed so far down to 0.
        # But inside the loop `for x in S`, we don't have the "current length" easily if we just iterate.
        # Let's just loop k from n down to 0 for each x? No, that's inefficient.
        # Let's keep a counter.
        
        pass

    # Better implementation:
    sums = [0] * (n + 1)
    sums[0] = 1
    for i, x in enumerate(S):
        # i is the number of elements processed so far (excluding x)
        # Max cardinality possible with i elements is i.
        # We update sums[i+1] from sums[i], sums[i] from sums[i-1], etc.
        for k in range(i, -1, -1):
            sums[k+1] |= (sums[k] << x)
            
    return sums

def is_valid(S):
    """
    Check if S is a valid cardinality-separated set.
    """
    if not S:
        return True
    n = len(S)
    sums = compute_sums_bitsets(S)
    # Check pairwise disjointness
    for i in range(n + 1):
        if sums[i] == 0:
            continue
        for j in range(i + 1, n + 1):
            if sums[j] == 0:
                continue
            if sums[i] & sums[j]:
                return False
    return True

def greedy_construct(order):
    """
    Greedily add elements from the given order if they maintain validity.
    """
    S = []
    for x in order:
        if x in S:
            continue
        new_S = S + [x]
        if is_valid(new_S):
            S = new_S
    return S

def local_search(S, candidates, max_iters=1000):
    """
    Improve S by adding elements or swapping.
    """
    S = list(S)
    S_set = set(S)
    n = len(S)
    
    # Pre-compute sums for current S to speed up checks?
    # Maybe not, re-computation is fast enough for N~30.
    
    improved = True
    while improved and not is_done():
        improved = False
        
        # 1. Try to add any candidate
        # Shuffle candidates to avoid bias
        cand_list = list(candidates)
        random.shuffle(cand_list)
        
        for x in cand_list:
            if x in S_set:
                continue
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
                S_set.add(x)
                n += 1
                improved = True
                # Restart the loop to try adding more immediately
                break
        
        if improved:
            continue
            
        # 2. Try swaps if no addition was possible
        # This allows escaping local optima
        # Pick a random element to remove and a random element to add
        # To be more systematic, we can try all pairs, but that's O(N^2 * CheckCost).
        # Let's try random swaps.
        
        if n == 0:
            break
            
        # Try a few random swaps
        for _ in range(50):
            if is_done():
                break
            
            # Pick random element to remove
            idx = random.randint(0, n - 1)
            y = S[idx]
            
            # Pick random candidate to add
            # Filter candidates not in S
            available = [c for c in candidates if c not in S_set]
            if not available:
                break
                
            x = random.choice(available)
            
            new_S = S[:idx] + S[idx+1:] + [x]
            if is_valid(new_S):
                S = new_S
                S_set.remove(y)
                S_set.add(x)
                improved = True
                break
                
    return S

def construct():
    best_set = []
    best_len = 0
    
    all_candidates = list(range(1, 257))
    
    # Strategy 1: Greedy with different orders
    random.seed(42)
    
    # Ascending
    order = list(range(1, 257))
    S = greedy_construct(order)
    if len(S) > best_len:
        best_set = S
        best_len = len(S)
        
    # Descending
    order = list(range(256, 0, -1))
    S = greedy_construct(order)
    if len(S) > best_len:
        best_set = S
        best_len = len(S)
        
    # Random orders
    for _ in range(10):
        if is_done():
            break
        order = list(all_candidates)
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
    # Strategy 2: Local Search from best greedy solution
    if best_set:
        improved = local_search(best_set, all_candidates)
        if len(improved) > best_len:
            best_set = improved
            best_len = len(best_set)
            
    # Strategy 3: Random Restarts with Local Search
    # Start from random small sets and grow
    max_restarts = 50
    for i in range(max_restarts):
        if is_done():
            break
            
        # Start with a random subset of size 5-10
        size = random.randint(5, 10)
        start_set = random.sample(all_candidates, size)
        
        # Ensure start set is valid?
        # If not, shrink it or skip?
        # Let's just try to grow it. If it's invalid, local search might fix it via swaps?
        # Actually, local search assumes current state is valid to check "add".
        # If start is invalid, we should fix it or restart.
        
        # Let's make sure start is valid by greedy construction on a random order
        # but forcing the first few elements?
        # Simpler: Just use greedy construct to get a valid seed, then local search.
        
        order = list(all_candidates)
        random.shuffle(order)
        
        # Force a few random elements to be in the seed?
        # No, just run greedy, then local search.
        
        S = greedy_construct(order)
        
        # Add some random noise to the greedy result?
        # Or just run local search.
        
        improved = local_search(S, all_candidates)
        
        if len(improved) > best_len:
            best_set = improved
            best_len = len(best_set)
            
        # Also try local search starting from empty, but biased towards larger numbers?
        
    # Final check
    if best_set:
        if not is_valid(best_set):
            # Fallback
            best_set = list(range(128, 136)) # Size 8
        return [int(x) for x in best_set]
    else:
        return [1] # Fallback

# Optimization: The `is_valid` function recomputes sums from scratch every time.
# In `local_search`, we call `is_valid` many times.
# We can optimize `is_valid` to take the current sums and the new element?
# But the set changes (add/swap).
# For N=30, compute_sums_bitsets is ~900 ops. is_valid is ~900 checks.
# Total ~1800 ops per check.
# 1000 iterations * 1800 ops = 1.8M ops. Very fast in Python if optimized?
# Bitwise ops on big ints are fast.
# Let's stick to this for clarity and correctness first.

# One potential issue: `local_search` might get stuck.
# We can add a "kick" step: remove a random element to allow different additions.

def local_search_v2(S, candidates):
    """
    More aggressive local search with kicks.
    """
    S = list(S)
    S_set = set(S)
    n = len(S)
    best_S = S
    best_n = n
    
    for iter in range(500):
        if is_done():
            break
            
        # Try to add
        cand_list = [c for c in candidates if c not in S_set]
        random.shuffle(cand_list)
        
        added = False
        for x in cand_list:
            new_S = S + [x]
            if is_valid(new_S):
                S = new_S
                S_set.add(x)
                n += 1
                added = True
                if n > best_n:
                    best_S = S
                    best_n = n
                break
        
        if added:
            continue
            
        # Try swap
        if n > 0:
            # Random swap
            idx = random.randint(0, n - 1)
            y = S[idx]
            if cand_list:
                x = random.choice(cand_list)
                new_S = S[:idx] + S[idx+1:] + [x]
                if is_valid(new_S):
                    S = new_S
                    S_set.remove(y)
                    S_set.add(x)
                    if n > best_n: # n didn't change, but structure did
                        # We don't update best_n here as size is same
                        pass
                    # Continue search from this new state
                    continue
                    
        # If no add or swap worked, try a kick: remove a random element
        if n > 5:
            idx = random.randint(0, n - 1)
            y = S[idx]
            S = S[:idx] + S[idx+1:]
            S_set.remove(y)
            n -= 1
            
    return best_S

def construct():
    best_set = []
    best_len = 0
    
    all_candidates = list(range(1, 257))
    random.seed(42)
    
    # Initial greedy attempts
    for _ in range(5):
        if is_done():
            break
        order = list(all_candidates)
        random.shuffle(order)
        S = greedy_construct(order)
        if len(S) > best_len:
            best_set = S
            best_len = len(S)
            
    # Local search from best
    if best_set:
        improved = local_search_v2(best_set, all_candidates)
        if len(improved) > best_len:
            best_set = improved
            best_len = len(best_set)
            
    # Multiple random restarts with local search
    for i in range(100):
        if is_done():
            break
            
        # Random start
        order = list(all_candidates)
        random.shuffle(order)
        S = greedy_construct(order)
        
        # Local search
        improved = local_search_v2(S, all_candidates)
        
        if len(improved) > best_len:
            best_set = improved
            best_len = len(best_set)
            
    # Final verification
    if best_set:
        if not is_valid(best_set):
            best_set = list(range(128, 136))
        return [int(x) for x in best_set]
    else:
        return [1]
