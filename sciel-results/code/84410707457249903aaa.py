"""Rewritten program: branch-and-bound over prime-exponent vectors for distinct subset products."""
import time

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _factorize(n, primes):
    vec = []
    for p in primes:
        if p * p > n:
            break
        e = 0
        while n % p == 0:
            e += 1
            n //= p
        vec.append(e)
    if n > 1:
        vec.append(1)
    return vec

def _is_independent(selected_vecs, new_vec):
    # Check if adding new_vec keeps all subset sums distinct.
    # Equivalent: no non-trivial integer combination of selected_vecs + new_vec equals zero
    # with coefficients in {-1,0,1} such that new_vec has coefficient 1 (or -1).
    # Simpler: maintain all subset-sum vectors of selected set; check new_vec not in any difference.
    # For small k this is fine.
    # We'll use: for each existing subset sum S, check if S == new_vec (conflict with empty)
    # or S - new_vec is a subset sum of previous (meaning two subsets collide).
    # Actually the cleanest: maintain set of all 2^k subset-sum vectors. When adding new_vec,
    # the new subset sums are old sums and old sums + new_vec. Conflict iff any old sum == new_vec + old sum'
    # i.e., new_vec = old_sum - old_sum' for some distinct old sums, OR new_vec = 0.
    # Since new_vec is nonzero, we just need: for all pairs (a,b) of old subset sums, a - b != new_vec.
    # That's O(4^k) which is too slow. Better: check if new_vec is in the set of all differences.
    # Alternative: just enumerate all 2^(k+1) subset sums of selected+new and check uniqueness.
    # For k up to ~12, 2^13 = 8192 sums, each a tuple of small ints. Feasible.
    pass

def _all_subset_sums_unique(vecs):
    """Check that all 2^len(vecs) subset sum vectors are distinct."""
    k = len(vecs)
    seen = set()
    # Use bitmasks
    for mask in range(1 << k):
        s = [0] * len(vecs[0])
        for i in range(k):
            if mask & (1 << i):
                v = vecs[i]
                for j in range(len(s)):
                    s[j] += v[j]
        seen.add(tuple(s))
        if len(seen) < mask + 1:
            return False
    return True

def _verify_distinct_products(nums):
    """Verify all 2^k subset products are distinct using exact integers."""
    k = len(nums)
    prods = set()
    for mask in range(1 << k):
        p = 1
        for i in range(k):
            if mask & (1 << i):
                p *= nums[i]
        if p in prods:
            return False
        prods.add(p)
    return True

def _branch_and_bound(nums, primes, deadline):
    """
    Find largest subset of nums (in [2,40]) with distinct subset products.
    Uses branch-and-bound on prime exponent vectors.
    """
    n = len(nums)
    # Precompute exponent vectors
    vecs = []
    for num in nums:
        v = _factorize(num, primes)
        vecs.append(tuple(v))
    
    # Determine max dimension
    max_dim = max(len(v) for v in vecs) if vecs else 0
    # Pad vectors to same length
    padded_vecs = []
    for v in vecs:
        padded = list(v) + [0] * (max_dim - len(v))
        padded_vecs.append(tuple(padded))
    
    best = []
    best_size = 0
    
    # Order: try smaller numbers first? Or primes first?
    # Primes are most "independent", so include them early.
    # Sort by: primes first, then composites
    def sort_key(i):
        num = nums[i]
        is_prime = all(num % d != 0 for d in range(2, int(num**0.5)+1)) if num >= 2 else False
        return (0 if is_prime else 1, num)
    
    order = sorted(range(n), key=sort_key)
    ordered_vecs = [padded_vecs[i] for i in order]
    ordered_nums = [nums[i] for i in order]
    
    # Current selection
    selected = []
    
    def dfs(idx, selected_vecs_list, selected_nums_list):
        nonlocal best, best_size
        if time.time() > deadline:
            return
        
        # Update best
        if len(selected_vecs_list) > best_size:
            best = list(selected_nums_list)
            best_size = len(selected_vecs_list)
        
        if idx == len(order):
            return
        
        # Pruning: if remaining + current <= best_size, can't improve
        remaining = len(order) - idx
        if len(selected_vecs_list) + remaining <= best_size:
            return
        
        # Try including ordered_vecs[idx]
        v = ordered_vecs[idx]
        num = ordered_nums[idx]
        
        # Check independence: all subset sums of selected + [v] must be distinct
        # Efficient check: maintain all subset sum vectors of selected
        # For now, just verify directly (k is small during search)
        candidate_vecs = selected_vecs_list + [v]
        if _all_subset_sums_unique(candidate_vecs):
            dfs(idx + 1, candidate_vecs, selected_nums_list + [num])
        
        # Try excluding
        dfs(idx + 1, selected_vecs_list, selected_nums_list)
    
    dfs(0, [], [])
    
    if best:
        return best
    return []

def construct():
    start = time.time()
    deadline = start + 55.0  # safety margin
    
    N = 40
    primes = _primes_up_to(N)
    
    nums = list(range(2, N + 1))
    
    result = _branch_and_bound(nums, primes, deadline)
    
    if not result:
        # Fallback: just primes
        result = [p for p in primes if p <= N]
    
    # Verify with exact integer products
    if not _verify_distinct_products(result):
        # Fallback to primes
        result = [p for p in primes if p <= N]
    
    # Ensure type is list of Python ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
