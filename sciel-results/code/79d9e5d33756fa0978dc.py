import time
import bisect

class FenwickMaxTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)  # 1-indexed, 0 is neutral for max

    def update(self, idx, val):
        # idx is 1-based in tree, but we pass 0-based shifted index?
        # Let's make update take 0-based index and convert to 1-based internally.
        # Or just use 1-based everywhere.
        # Let's use 1-based for Fenwick.
        # Input idx is 0-based.
        i = idx + 1
        while i <= self.size:
            if val > self.tree[i]:
                self.tree[i] = val
            i += i & (-i)

    def query(self, idx):
        # Query max in [0, idx] (0-based)
        # Convert to 1-based
        i = idx + 1
        res = 0
        while i > 0:
            if self.tree[i] > res:
                res = self.tree[i]
            i -= i & (-i)
        return res

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    limit = 10000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False

    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    n = len(primes)
    
    if n == 0:
        return []
        
    # Max possible length is small. 100 is safe.
    L_MAX = 100
    OFFSET = 10000
    # Key range: -10000 to 10000. Shifted: 0 to 20000.
    TREE_SIZE = 20005
    
    trees = [FenwickMaxTree(TREE_SIZE) for _ in range(L_MAX + 1)]
    
    # parent[i][L] stores the index j in primes that is the predecessor
    # for the best sequence of length L ending at i.
    # -1 if not reachable.
    parent = [[-1] * (L_MAX + 1) for _ in range(n)]
    
    best_len = 0
    best_end_idx = -1
    
    for i in range(n):
        p_i = primes[i]
        # dp_i[L] = min gap for sequence of length L ending at i
        # inf means unreachable
        dp_i = [float('inf')] * (L_MAX + 1)
        dp_i[1] = 0
        
        # Query trees to find best predecessors
        # For length L+1 ending at i, we look at trees[L]
        # Condition: key_j <= p_i  =>  primes[j] - gap_j <= p_i
        # We want max primes[j]
        
        for L in range(1, L_MAX):
            # Query tree for length L
            # We want to find max p_j such that p_j - gap_j <= p_i
            # This is equivalent to p_j <= p_i + gap_j
            # Wait, the condition in the tree update was:
            # key = p_j - gap_j.
            # We update tree[L] at index (key + OFFSET) with value p_j.
            # We query tree[L] for range [0, p_i + OFFSET].
            # This returns max p_j such that (p_j - gap_j) + OFFSET <= p_i + OFFSET
            # => p_j - gap_j <= p_i
            # => p_i - p_j >= gap_j
            # This is the correct condition.
            
            query_idx = p_i + OFFSET
            if query_idx >= TREE_SIZE:
                query_idx = TREE_SIZE - 1
                
            max_p_j = trees[L].query(query_idx)
            
            if max_p_j > 0:
                # Find index j
                j = bisect.bisect_left(primes, max_p_j)
                # Verify primes[j] == max_p_j
                if j < n and primes[j] == max_p_j:
                    gap = p_i - max_p_j
                    if gap < dp_i[L + 1]:
                        dp_i[L + 1] = gap
                        parent[i][L + 1] = j
        
        # Update trees with current i
        for L in range(1, L_MAX + 1):
            if dp_i[L] < float('inf'):
                key = p_i - dp_i[L]
                idx = key + OFFSET
                if 0 <= idx < TREE_SIZE:
                    trees[L].update(idx, p_i)
                
                # Track best
                if L > best_len:
                    best_len = L
                    best_end_idx = i
                elif L == best_len:
                    # Tie-breaking? Any is fine, or maybe prefer smaller last gap?
                    # Doesn't matter for length.
                    pass
        
        if time.time() > deadline:
            break

    if best_len == 0:
        return [primes[0]] if n > 0 else []

    # Backtrack
    seq = []
    curr_i = best_end_idx
    curr_L = best_len
    while curr_L > 0:
        seq.append(primes[curr_i])
        if curr_L > 1:
            curr_i = parent[curr_i][curr_L]
            curr_L -= 1
        else:
            break
    seq.reverse()
    
    return seq
