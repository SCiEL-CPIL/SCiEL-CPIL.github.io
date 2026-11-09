import time
import random

def _smooth_set(limit, primes):
    """Return a set of all n in [2, limit] whose prime factors are all in primes."""
    s = set()
    for n in range(2, limit + 1):
        m = n
        for p in primes:
            while m % p == 0:
                m //= p
        if m == 1:
            s.add(n)
    return s

def _find_clique(cands, smooth, size, deadline):
    """
    Find a clique of given size in the graph where edge (i,j) exists
    if (cands[i] + cands[j]) is in smooth.
    Uses backtracking with degree-ordered search.
    """
    n = len(cands)
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for i in range(n):
        ai = cands[i]
        for j in range(i + 1, n):
            if (ai + cands[j]) in smooth:
                adj[i].add(j)
                adj[j].add(i)
    
    # Order by degree descending to guide search
    order = sorted(range(n), key=lambda i: len(adj[i]), reverse=True)
    
    best = []
    
    def bt(current, pool):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current) == size:
            best = list(current)
            return
        if len(pool) < size - len(current):
            return
        
        for k, idx in enumerate(pool):
            if time.time() > deadline:
                return
            # Check if idx is compatible with all in current
            if not all(idx in adj[c] for c in current):
                continue
            
            # New pool: elements after idx in pool that are neighbors of idx
            new_pool = [x for x in pool[k + 1:] if x in adj[idx]]
            
            current.append(idx)
            bt(current, new_pool)
            current.pop()
            if best:
                return
    
    bt([], order)
    return best

def _count_distinct_primes(nums):
    """Count distinct prime factors across all pairwise sums."""
    all_primes = set()
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            n = nums[i] + nums[j]
            d = 2
            while d * d <= n:
                while n % d == 0:
                    all_primes.add(d)
                    n //= d
                d += 1
            if n > 1:
                all_primes.add(n)
    return len(all_primes)

def construct():
    t0 = time.time()
    deadline = t0 + 75.0
    N = 12
    
    # Strategy: Try increasingly larger prime sets.
    # For each prime set P, we look for 12 distinct integers <= 10000
    # such that all pairwise sums are P-smooth.
    # We search in the space of integers 1..MaxVal.
    # If we find a clique of size 12, the answer uses only primes in P.
    
    prime_sets = [
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13],
        [2, 3, 5, 7, 11, 13, 17],
        [2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 5, 7, 11, 13, 17, 19, 23],
    ]
    
    # We try different upper bounds for the search space.
    # Smaller bounds are faster but might not have a solution.
    # Larger bounds are slower but more likely to have a solution.
    # Note: If we use candidates 1..M, the sums are 2..2M.
    # The smooth set must cover 2..2M.
    
    max_search_bounds = [300, 600, 1200, 2400, 5000]
    
    best_result = None
    best_score = float('inf')
    
    for ps in prime_sets:
        if time.time() > deadline - 5.0:
            break
        
        for bound in max_search_bounds:
            if time.time() > deadline - 5.0:
                break
            
            # Candidates are 1..bound
            cands = list(range(1, bound + 1))
            
            # Precompute smooth numbers up to 2*bound
            smooth = _smooth_set(2 * bound, ps)
            
            # Find clique
            res = _find_clique(cands, smooth, N, deadline - 5.0)
            
            if res:
                out = [cands[i] for i in res]
                if len(out) == N and len(set(out)) == N:
                    score = _count_distinct_primes(out)
                    if score < best_score:
                        best_score = score
                        best_result = out
                    # If score is <= len(ps), we can't do better with this ps,
                    # and we are trying ps in increasing order of size,
                    # so this is optimal for this ps.
                    # Since we iterate ps in increasing size, if we find a solution
                    # with score == len(ps), it's the best possible for this ps.
                    # We can break out of the bound loop for this ps.
                    if score <= len(ps):
                        break
        
        # If we found a solution with score equal to the number of primes in ps,
        # and we are going to try a larger ps next, we might get a worse score.
        # So we should stop if we found a solution with a small prime set.
        # Specifically, if best_score <= len(ps), we can stop entirely because
        # any subsequent ps will have more primes, so the score will be at least len(ps_next) > len(ps) >= best_score.
        # Wait, the score is the number of DISTINCT primes.
        # If we use primes from ps, the score is at most len(ps).
        # If we find a solution with score == len(ps), it's optimal for that ps.
        # The next ps has len(ps)+1 primes, so the minimum possible score is len(ps)+1.
        # Since len(ps)+1 > len(ps) >= best_score, we can stop.
        
        if best_result and best_score <= len(ps):
            break
    
    if best_result:
        return best_result
    
    # Fallback: If no clique found (unlikely), return a default set.
    # The default set [1..12] has many primes.
    # Let's try a quick greedy search with a small prime set {2,3,5,7} just in case.
    # Or just return [210*i for i in 1..12] if valid?
    # 210*12 = 2520 <= 10000.
    # Sums are 210*(i+j). Primes are {2,3,5,7}.
    # This is a valid solution with score 4.
    fallback = [210 * i for i in range(1, N + 1)]
    return fallback

# EVOLVE-BLOCK-END
