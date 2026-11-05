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
    # Comparison: higher dp is better. If dp equal, higher p_val is better (smaller gap to future).
    
    # Max possible val = max(primes) + max_gap.
    # max_gap < 10000. So max val < 20000.
    max_val = 20005
    # tree[k] stores the best state for the range (k - (k&-k), k]
    # We use 1-based indexing for the tree.
    # Indices in tree: 1 to max_val.
    # val[j] = primes[j] + gap[j].
    # Note: val[j] >= 2.
    
    # State representation: (dp, p_val, index)
    # Initialize with "none" state
    NEG_INF = -1
    tree = [(NEG_INF, -1, -1)] * (max_val + 1)
    
    def better(s1, s2):
        # s1 is better than s2?
        if s1[0] != s2[0]:
            return s1[0] > s2[0]
        if s1[1] != s2[1]:
            return s1[1] > s2[1]
        # If dp and p_val are same, prefer larger index? 
        # Actually, if p_val is same, the gap to current prime is same.
        # But index matters for prev? No, prev is just the index.
        # Let's just return False to keep the first one found (or existing one).
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
        res = (NEG_INF, -1, -1)
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
        # val[j] = primes[j] + gap[j]
        # We need to map val[j] to the Fenwick tree index.
        # The values val[j] are integers.
        # We query up to pi.
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
        state_i = (dp[i], pi, i)
        update(val_i, state_i)
        
        if time.time() > deadline:
            break

    # Find the best end index
    max_len = max(dp)
    best_end = -1
    best_end_gap = float('inf')
    for i in range(n):
        if dp[i] == max_len:
            # Prefer smaller last gap?
            # It doesn't strictly matter for validity, but let's be consistent.
            if gap[i] < best_end_gap:
                best_end_gap = gap[i]
                best_end = i

    # Reconstruct
    seq = []
    curr = best_end
    while curr != -1:
        seq.append(primes[curr])
        curr = prev[curr]
    seq.reverse()
    
    return seq
