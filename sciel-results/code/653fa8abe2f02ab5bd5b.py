"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from functools import lru_cache

def construct():
    deadline = time.time() + 82.0  # safety margin under 90s wall-clock
    N = 512
    MAX_K = 48

    # ---------- fast validity check using subset-sum DP ----------
    # For a set S and a candidate mean m (not in S), check if any subset of S
    # of size >= 2 has sum == len(subset)*m.
    # We do this by checking: for each k in 2..len(S), is there a subset of size k
    # with sum == k*m?
    # Use DP: dp[k][s] = can we form sum s with exactly k elements?
    # But s can be up to k*512, and k up to 48, so s up to ~24576.
    # That's 48 * 24576 ~ 1.2M booleans, feasible with bitsets.
    
    def is_valid_set(S):
        """Check if no element of S is the mean of a subset of >=2 OTHER elements."""
        n = len(S)
        S_sorted = sorted(S)
        for i, a in enumerate(S_sorted):
            others = S_sorted[:i] + S_sorted[i+1:]
            m = len(others)
            if m < 2:
                continue
            # Check if any subset of others of size k>=2 has sum == k*a
            # DP: dp[k] = set of achievable sums with k elements
            # Use a list of sets or bitsets. For speed, use a set of sums per k.
            # Actually, let's use a 2D approach: reachable[k] = set of sums
            # But that's slow. Let's use a different approach.
            
            # Alternative: for each k in 2..m, check if k*a is achievable as sum of k elements from others.
            # We can do a single DP pass: dp[j][s] using a list of bytearrays or sets.
            # Given m <= 47, and sums up to 47*512 = 24064, let's use a list of sets.
            
            # More efficient: use a 1D DP over elements, tracking (count, sum).
            # dp[count] = set of sums achievable with 'count' elements
            dp = [set() for _ in range(m + 1)]
            dp[0].add(0)
            for x in others:
                # iterate backwards to avoid using x multiple times
                for c in range(m, 0, -1):
                    if dp[c - 1]:
                        for s in dp[c - 1]:
                            dp[c].add(s + x)
            for k in range(2, m + 1):
                if k * a in dp[k]:
                    return False
        return True

    def is_valid_incremental(S, a):
        """Check if adding a to set S keeps validity. S does not contain a."""
        # We need to check:
        # 1. a is not the mean of any subset of >=2 elements of S
        # 2. No element of S is the mean of a subset of >=2 elements of S ∪ {a} that includes a
        #    (subsets not including a were already valid)
        
        n = len(S)
        if n < 2:
            # Case 1: check if a is mean of subset of S (need >=2 elements)
            if n < 2:
                return True
            # Check if a is mean of any subset of S of size k>=2
            dp = [set() for _ in range(n + 1)]
            dp[0].add(0)
            for x in S:
                for c in range(n, 0, -1):
                    if dp[c - 1]:
                        for s in dp[c - 1]:
                            dp[c].add(s + x)
            for k in range(2, n + 1):
                if k * a in dp[k]:
                    return False
            return True
        
        # Case 1: a is not mean of any subset of S of size >= 2
        dp = [set() for _ in range(n + 1)]
        dp[0].add(0)
        for x in S:
            for c in range(n, 0, -1):
                if dp[c - 1]:
                    for s in dp[c - 1]:
                        dp[c].add(s + x)
        for k in range(2, n + 1):
            if k * a in dp[k]:
                return False
        
        # Case 2: for each element b in S, check if b is mean of a subset of (S ∪ {a}) of size >= 2
        # that includes a. Subsets not including a were already valid.
        # A subset of S ∪ {a} that includes a has the form {a} ∪ T where T ⊆ S.
        # The subset has size 1 + |T| >= 2, so |T| >= 1.
        # Sum = a + sum(T), size = 1 + |T|.
        # We need: (a + sum(T)) / (1 + |T|) = b  =>  a + sum(T) = b * (1 + |T|)
        # => sum(T) = b*(1+|T|) - a
        # Let k = |T|, so k >= 1, and we need sum of k elements from S equals b*(1+k) - a.
        
        for b in S:
            for k in range(1, n + 1):
                target = b * (1 + k) - a
                if target < 0:
                    continue
                if target in dp[k]:
                    return False
        return True

    # ---------- Greedy construction with multiple orderings ----------
    def greedy_construct(order, current_time_fn):
        S = []
        for a in order:
            if len(S) >= MAX_K:
                break
            if current_time_fn() > deadline - 2.0:
                break
            if is_valid_incremental(S, a):
                S.append(a)
        return S

    # Generate candidate pool: 2^i * 3^j plus random fill
    pool = set()
    for i in range(10):  # 2^i up to 512
        for j in range(7):  # 3^j up to 729, but we cap at 512
            v = (2 ** i) * (3 ** j)
            if 1 <= v <= N:
                pool.add(v)
    
    # Add some random numbers to fill
    random.seed(42)
    for _ in range(200):
        pool.add(random.randint(1, N))
    
    pool = sorted(pool)
    
    # Try multiple orderings
    orderings = [
        pool,                          # ascending
        sorted(pool, reverse=True),    # descending
        list(pool[::2]) + list(pool[1::2]),  # even then odd
        sorted(pool, key=lambda x: x % 7),   # by mod 7
        sorted(pool, key=lambda x: x % 11),  # by mod 11
        sorted(pool, key=lambda x: (x % 5, x // 5)),  # grouped
    ]
    
    best = []
    for ordering in orderings:
        S = greedy_construct(ordering, time.time)
        if len(S) > len(best):
            best = S
        if len(best) >= MAX_K:
            break
    
    # ---------- Local search: remove-1/add-2, remove-2/add-3 ----------
    def local_search(S, current_time_fn):
        S = list(S)
        S_set = set(S)
        improved = True
        while improved:
            if current_time_fn() > deadline - 3.0:
                break
            improved = False
            
            # Try remove-1/add-2: remove one element, add two from pool
            candidates = [x for x in pool if x not in S_set]
            random.shuffle(candidates)
            
            for i in range(len(S)):
                if current_time_fn() > deadline - 3.0:
                    break
                S_no_i = [x for j, x in enumerate(S) if j != i]
                S_no_i_set = set(S_no_i)
                added = []
                for c in candidates:
                    if len(added) == 2:
                        break
                    if c in S_no_i_set:
                        continue
                    if is_valid_incremental(S_no_i, c):
                        added.append(c)
                        S_no_i_set.add(c)
                if len(added) == 2:
                    new_S = S_no_i + added
                    if len(new_S) > len(S):
                        S = new_S
                        S_set = set(S)
                        improved = True
                        break
            
            if current_time_fn() > deadline - 3.0:
                break
            
            # Try remove-2/add-3
            if len(S) >= 2:
                for i in range(len(S)):
                    if current_time_fn() > deadline - 3.0:
                        break
                    for j in range(i + 1, len(S)):
                        S_no_ij = [x for k, x in enumerate(S) if k != i and k != j]
                        S_no_ij_set = set(S_no_ij)
                        added = []
                        for c in candidates:
                            if len(added) == 3:
                                break
                            if c in S_no_ij_set:
                                continue
                            if is_valid_incremental(S_no_ij, c):
                                added.append(c)
                                S_no_ij_set.add(c)
                        if len(added) == 3:
                            new_S = S_no_ij + added
                            if len(new_S) > len(S):
                                S = new_S
                                S_set = set(S)
                                improved = True
                                break
                    if improved:
                        break
        
        return S

    if time.time() < deadline - 5.0:
        best = local_search(best, time.time)
    
    # Final verification
    if best and not is_valid_set(best):
        # Fallback: rebuild greedily
        best = greedy_construct(pool, time.time)
    
    return best
# EVOLVE-BLOCK-END
