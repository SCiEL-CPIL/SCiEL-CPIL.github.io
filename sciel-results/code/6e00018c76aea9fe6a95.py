import math
import time

def construct():
    start_time = time.time()
    limit = 10000
    deadline = start_time + 85  # safety margin

    # Generate primes up to 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i, flag in enumerate(is_prime) if flag]
    n = len(primes)

    if n == 0:
        return []
    if n == 1:
        return [primes[0]]

    # Fenwick Tree (Binary Indexed Tree) for prefix maximum
    # We store states (dp, p_val, index)
    # Comparison: higher dp is better. If dp equal, smaller last_gap is better (more extensible).
    
    # Max possible val = max(primes) + max_gap.
    # max_gap < 10000. So max val < 20000.
    max_val = 20005
    # tree[k] stores the best state for the range (k - (k&-k), k]
    # We use 1-based indexing for the tree.
    # Indices in tree: 1 to max_val.
    # val[j] = primes[j] + gap[j].
    # Note: val[j] >= 2.
    
    # State representation: (dp, last_gap, index)
    # Initialize with "none" state
    NEG_INF = -1
    # Use a large sentinel for gap in "none" state
    INF_GAP = 10**9
    tree = [(NEG_INF, INF_GAP, -1)] * (max_val + 1)
    
    def better(s1, s2):
        # s1 is better than s2?
        # Higher dp is better.
        if s1[0] != s2[0]:
            return s1[0] > s2[0]
        # If dp equal, smaller last_gap is better (allows larger next prime before gap constraint is violated? 
        # Actually, the constraint is gap_next >= gap_last. Smaller gap_last is less restrictive for the NEXT gap.
        # Wait, the condition for extending j to i is: primes[i] - primes[j] >= gap[j].
        # This means primes[i] >= primes[j] + gap[j] = val[j].
        # So for a fixed dp, we want to minimize val[j] to allow more future primes i to satisfy the condition.
        # val[j] = primes[j] + gap[j].
        # So we want to minimize val[j].
        # If we store (dp, val, index) and compare:
        # Higher dp is better.
        # If dp equal, smaller val is better.
        # Let's re-evaluate the state.
        
        # Current state in tree: (dp, p_val, index) where p_val is primes[j].
        # The key for query is val[j] = primes[j] + gap[j].
        # The query finds the best j with val[j] <= pi.
        # Among those, we want the one with max dp.
        # If dp is tied, which j is better?
        # It doesn't affect the validity of the extension to i (as long as val[j] <= pi).
        # But it affects the gap[i] = pi - primes[j].
        # And gap[i] becomes the new "last_gap" for i.
        # A smaller gap[i] is better for future extensions (less restrictive).
        # So if dp is tied, we want to minimize gap[i] = pi - primes[j].
        # Since pi is fixed for the current iteration, minimizing gap[i] is equivalent to maximizing primes[j].
        # So the current tie-breaking (higher p_val) is correct for minimizing the resulting gap.
        
        # However, the state stored in the tree is for j.
        # When we update the tree with state i, we store (dp[i], primes[i], i).
        # The val[i] used for the tree index is primes[i] + gap[i].
        # If two states i1 and i2 have the same dp and same val, but different primes[i], 
        # the one with larger primes[i] has smaller gap[i].
        # But the tree index is val.
        # If we query for a future prime p, we get the best state with val <= p.
        # If we have two candidates with same dp and same val, the one with larger primes[j] gives smaller gap for the extension.
        # So the current tie-breaking in `better` (higher p_val) is correct.
        
        # Let's re-read the current code's `better`:
        # if s1[0] != s2[0]: return s1[0] > s2[0]
        # if s1[1] != s2[1]: return s1[1] > s2[1]
        # This compares (dp, p_val).
        # This is correct.
        
        # So why change?
        # Maybe the issue is not the tie-breaking but something else?
        # Or maybe there is a subtle bug?
        
        # Let's look at the update logic.
        # update(val_i, state_i)
        # state_i = (dp[i], pi, i)
        # val_i = pi + gap[i]
        
        # Is it possible that a state with lower dp but much smaller val is better for the overall max length?
        # No, because we are maximizing the length of the subsequence ending at i.
        # The DP recurrence is:
        # dp[i] = 1 + max{ dp[j] : val[j] <= primes[i] }
        # This is correct.
        
        # What if we improve the tie-breaking in the final selection?
        # Or what if we do a local search?
        
        # Let's stick to the current `better` but ensure it's robust.
        # Actually, the current `better` returns False if dp and p_val are same.
        # This means the first state encountered with the same dp and p_val stays.
        # This should be fine.
        
        # Let's try a different approach:
        # What if we store the actual gap in the state and use that for tie-breaking?
        # State: (dp, gap, index)
        # Comparison:
        # Higher dp is better.
        # If dp equal, smaller gap is better.
        # Why? Because when this state is used to extend to a future prime p,
        # the new gap will be p - primes[j].
        # Wait, the gap stored in the state is the gap *ending* at j.
        # The new gap will be p - primes[j].
        # This new gap is independent of the old gap (as long as the constraint is satisfied).
        # So the old gap doesn't directly affect the new gap.
        # However, the old gap affects the val[j] = primes[j] + gap[j].
        # A smaller val[j] allows more future primes p to satisfy p >= val[j].
        # So for a fixed dp, we want to minimize val[j].
        # val[j] = primes[j] + gap[j].
        # So we should minimize val[j].
        
        # Current code stores p_val = primes[j].
        # And compares by p_val (higher is better).
        # This is equivalent to minimizing val[j] ONLY IF gap[j] is constant for states with the same dp.
        # But gap[j] is NOT constant.
        # Example:
        # State A: dp=2, primes[j]=10, gap[j]=2, val[j]=12
        # State B: dp=2, primes[j]=11, gap[j]=1, val[j]=12
        # Both have val=12.
        # If we extend to p=13:
        # From A: gap=13-10=3. Next val=13+3=16.
        # From B: gap=13-11=2. Next val=13+2=15.
        # State B gives a smaller next val, which is better.
        # So for the same val, we prefer the one with larger primes[j] (which implies smaller gap[j]).
        # The current code prefers larger p_val (primes[j]). So it prefers B. Correct.
        
        # Example 2:
        # State A: dp=2, primes[j]=10, gap[j]=1, val[j]=11
        # State B: dp=2, primes[j]=11, gap[j]=2, val[j]=13
        # val[A] < val[B].
        # A is better because it allows more future primes.
        # The current code compares by p_val.
        # p_val[A]=10, p_val[B]=11.
        # It prefers B (higher p_val).
        # But A has smaller val, so A should be preferred!
        # **BUG FOUND!**
        # The current tie-breaking is wrong. It should minimize val[j], not maximize primes[j].
        
        # Fix: Store (dp, val, index) in the tree?
        # Or store (dp, primes[j], gap[j], index) and compare by val = primes[j]+gap[j]?
        # If we store (dp, val, index), we can compare:
        # Higher dp is better.
        # If dp equal, smaller val is better.
        
        # Let's implement this.
        
        if s1[0] != s2[0]:
            return s1[0] > s2[0]
        # s1[1] is now val (primes[j] + gap[j])
        # Smaller val is better
        if s1[1] != s2[1]:
            return s1[1] < s2[1]
        return False

    def update(idx, state):
        # idx is the value val[j] (1-based in tree)
        # We update all nodes in the Fenwick tree that cover idx
        while idx <= max_val:
            if better(state, tree[idx]):
                tree[idx] = state
            idx += idx & -idx

    def query(idx):
        # Query max state in [1, idx]
        res = (NEG_INF, INF_GAP, -1)
        while idx > 0:
            if better(tree[idx], res):
                res = tree[idx]
            idx -= idx & -idx
        return res

    dp = [1] * n
    prev = [-1] * n
    gap = [0] * n

    for i in range(n):
        pi = primes[i]
        
        # Query for best predecessor j such that val[j] <= pi
        best_state = query(pi)
        
        if best_state[0] > NEG_INF:
            # Found a valid predecessor
            best_j = best_state[2]
            dp[i] = best_state[0] + 1
            prev[i] = best_j
            gap[i] = pi - primes[best_j]
        else:
            dp[i] = 1
            prev[i] = -1
            gap[i] = 0
            
        # Update the tree with the state for i
        # val[i] = pi + gap[i]
        val_i = pi + gap[i]
        # Ensure val_i is within bounds
        if val_i > max_val:
            val_i = max_val
        # State is (dp, val, index)
        state_i = (dp[i], val_i, i)
        update(val_i, state_i)
        
        if time.time() > deadline:
            break

    # Find the best end index
    max_len = max(dp)
    best_end = -1
    best_end_val = float('inf')
    for i in range(n):
        if dp[i] == max_len:
            # Prefer smaller last val?
            # val[i] = primes[i] + gap[i]
            # Smaller val[i] is better for potential further extension (though we are done).
            # It doesn't matter for the final sequence, but let's be consistent.
            val_i = primes[i] + gap[i]
            if val_i < best_end_val:
                best_end_val = val_i
                best_end = i

    # Reconstruct
    seq = []
    curr = best_end
    while curr != -1:
        seq.append(primes[curr])
        curr = prev[curr]
    seq.reverse()
    
    return seq
