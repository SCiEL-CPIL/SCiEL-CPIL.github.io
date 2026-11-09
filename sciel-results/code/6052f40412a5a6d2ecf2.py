"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from itertools import combinations
from collections import defaultdict

P = {'n': 256}
_DEADLINE = None

def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True

def _prime_factors(n):
    """Return set of prime factors of n."""
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors

def _valid_set(S):
    """Check if set S is valid: for all distinct a,b,c in S, a does not divide b*c."""
    S_list = list(S)
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            for k in range(j+1, m):
                c = S_list[k]
                if (b * c) % a == 0:
                    return False
    return True

def _can_add(S_list, x):
    """Check if x can be added to set S_list without violating the constraint.
    S_list is a list for speed."""
    m = len(S_list)
    for i in range(m):
        a = S_list[i]
        for j in range(i+1, m):
            b = S_list[j]
            if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                return False
    return True

def _precompute_divisibility_matrix(nums):
    """
    Precompute for each pair (i,j) whether nums[i] divides nums[j]*nums[k] for some k.
    Actually, we need: for a set S, x can be added iff for all distinct a,b in S:
      (a*b) % x != 0 AND (x*b) % a != 0 AND (x*a) % b != 0
    
    Key insight: a | b*c  iff  for every prime p, the exponent of p in a is <= 
    exponent of p in b + exponent of p in c.
    
    Precompute:
    - prime_exp[x][p] = exponent of prime p in x
    - For each x, precompute "conflict_with_pair" as a bitset over pairs
    """
    n = len(nums)
    idx_map = {v: i for i, v in enumerate(nums)}
    
    # Prime exponents for each number
    prime_exps = []
    for x in nums:
        factors = _prime_factors(x)
        exps = {}
        for p in factors:
            e = 0
            tmp = x
            while tmp % p == 0:
                e += 1
                tmp //= p
            exps[p] = e
        prime_exps.append(exps)
    
    # For each x, compute which pairs (a,b) would conflict with x
    # Conflict if: a|b*x OR b|a*x OR x|a*b
    # We'll store as a set of frozensets for quick lookup
    
    # Actually, let's use a different approach:
    # For each x, store the set of "blocking pairs" - pairs (a,b) such that {x,a,b} is invalid
    # Use bitsets for efficiency
    
    # Precompute: for each x, the set of y such that x|y*z for some z... 
    # This is getting complex. Let's use a simpler but effective approach:
    # Precompute conflict[x] = set of y such that there EXISTS z with {x,y,z} bad
    # But we need pairwise conflicts in the context of the current set.
    
    # Better: precompute for each pair (a,b) the set of x that conflict with this pair
    # pair_conflict[(a,b)] = set of x where {x,a,b} is invalid
    
    pair_conflict = defaultdict(set)
    for i, a in enumerate(nums):
        for j in range(i+1, n):
            b = nums[j]
            # For each x, check if {x,a,b} is bad
            for k, x in enumerate(nums):
                if x == a or x == b:
                    continue
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    pair_conflict[(a,b)].add(x)
    
    return pair_conflict, prime_exps

def _construct_fast():
    """
    Fast construction using:
    1. Greedy with multiple orderings (fast, O(1) conflict check via precomputed data)
    2. Local search with 1-for-1 and 1-for-2 swaps
    3. Exact branch-and-bound with bitset-based upper bounds
    """
    global _DEADLINE
    _DEADLINE = time.time() + 80  # 80s budget (10s safety margin)
    
    n = P['n']
    nums = list(range(2, n + 1))
    N = len(nums)
    
    # Precompute prime exponents
    prime_exps = []
    for x in nums:
        factors = _prime_factors(x)
        exps = {}
        for p in factors:
            e = 0
            tmp = x
            while tmp % p == 0:
                e += 1
                tmp //= p
            exps[p] = e
        prime_exps.append(exps)
    
    # Precompute: for each x, the set of (a,b) pairs that conflict with x
    # conflict_pairs[x_idx] = set of (a_idx, b_idx) such that {x, nums[a_idx], nums[b_idx]} is bad
    # This allows O(|pairs|) check for can_add
    
    # Actually, let's precompute differently:
    # For each x, precompute a "signature" that allows fast conflict checking
    # 
    # Key optimization: a | b*c  iff  for all primes p: exp_p(a) <= exp_p(b) + exp_p(c)
    # So x conflicts with pair (a,b) if:
    #   - x | a*b: for all p, exp_p(x) <= exp_p(a) + exp_p(b)
    #   - a | x*b: for all p, exp_p(a) <= exp_p(x) + exp_p(b)  
    #   - b | x*a: for all p, exp_p(b) <= exp_p(x) + exp_p(a)
    
    # Precompute conflict matrix: conflict[i][j] = True if nums[i] and nums[j] "conflict"
    # meaning there exists k such that {nums[i], nums[j], nums[k]} is bad
    # But this isn't quite right either - the conflict depends on the third element.
    
    # Let's use a direct approach with memoization:
    # For each x, precompute which pairs (a,b) in the full range [2,256] conflict with x
    # Store as a bitset (Python int) where bit i is set if pair involving nums[i] conflicts
    
    # Actually, the most effective approach for this size:
    # Precompute a 3D lookup or use the prime factor structure directly
    
    # Let me try a cleaner approach:
    # For each x in [2,256], compute "bad_pairs[x]" = set of (a,b) with a<b, a!=x, b!=x
    # such that {x,a,b} is invalid. Store as a frozenset of (a,b) tuples.
    
    bad_pairs = {}
    for x in nums:
        bp = set()
        for a in nums:
            if a == x:
                continue
            for b in nums:
                if b == x or b == a:
                    continue
                if (a * b) % x == 0 or (x * b) % a == 0 or (x * a) % b == 0:
                    # Normalize: store (min, max)
                    if a < b:
                        bp.add((a, b))
                    else:
                        bp.add((b, a))
        bad_pairs[x] = bp
    
    # Now _can_add can check: for each pair (a,b) in current set, is (a,b) in bad_pairs[x]?
    # This is O(|S|^2) set lookups, which is fast.
    
    def _can_add_fast(S_set, x):
        """Check if x can be added to S_set (a set)."""
        S_list = list(S_set)
        m = len(S_list)
        bp = bad_pairs[x]
        for i in range(m):
            a = S_list[i]
            for j in range(i+1, m):
                b = S_list[j]
                if a < b:
                    if (a, b) in bp:
                        return False
                else:
                    if (b, a) in bp:
                        return False
        return True
    
    # Greedy with multiple orderings
    def _greedy(order_func):
        S = set()
        for x in order_func(nums):
            if _can_add_fast(S, x):
                S.add(x)
        return S
    
    # Define orderings
    orderings = [
        lambda: sorted(nums, key=lambda x: -x),
        lambda: sorted(nums, key=lambda x: x),
        lambda: sorted(nums, key=lambda x: (len(_prime_factors(x)), x)),
        lambda: sorted(nums, key=lambda x: (x.bit_count(), x)),
        lambda: sorted(nums, key=lambda x: (-len(_prime_factors(x)), -x)),
        lambda: sorted(nums, key=lambda x: (sum(prime_exps[nums.index(x)].values()) if x in nums else 0, x)),
    ]
    
    # Add some random orderings
    import random
    random.seed(42)
    for _ in range(5):
        def _rand_order():
            shuffled = nums[:]
            random.shuffle(shuffled)
            return shuffled
        orderings.append(_rand_order)
    
    best_set = set()
    best_size = 0
    
    for order_func in orderings:
        if time.time() > _DEADLINE - 30:
            break
        S_greedy = _greedy(order_func)
        if len(S_greedy) > best_size:
            best_set = S_greedy
            best_size = len(S_greedy)
    
    # Local search: 1-for-1 and 1-for-2 swaps
    def _local_search(S, max_iters=200):
        best = set(S)
        best_size_local = len(best)
        candidates = [x for x in nums if x not in best]
        
        for it in range(max_iters):
            if time.time() > _DEADLINE - 25:
                break
            
            improved = False
            
            # Try 1-for-1 swap: remove one, add one
            for rem in list(best):
                if time.time() > _DEADLINE - 25:
                    break
                S2 = best - {rem}
                for x in candidates:
                    if x in S2:
                        continue
                    if _can_add_fast(S2, x):
                        S2.add(x)
                        if len(S2) > best_size_local:
                            best = S2
                            best_size_local = len(best)
                            candidates = [y for y in nums if y not in best]
                            improved = True
                            break
                if improved:
                    break
            
            # Try 1-for-2 swap: remove one, add two
            if not improved:
                for rem in list(best):
                    if time.time() > _DEADLINE - 25:
                        break
                    S2 = best - {rem}
                    added = []
                    for x in candidates:
                        if x in S2:
                            continue
                        if _can_add_fast(S2, x):
                            S2.add(x)
                            added.append(x)
                        if len(added) >= 2:
                            break
                    if len(S2) > best_size_local:
                        best = S2
                        best_size_local = len(best)
                        candidates = [y for y in nums if y not in best]
                        improved = True
                        break
            
            if not improved:
                break
        
        return best
    
    if time.time() < _DEADLINE - 20:
        best_set = _local_search(best_set)
        best_size = len(best_set)
    
    # Branch and bound with precomputed conflict data
    if time.time() < _DEADLINE - 15:
        # Use a smarter B&B:
        # State: current set (as frozenset for hashing), remaining candidates
        # Upper bound: len(current) + len(remaining)
        # But we can do better: for each remaining x, if it conflicts with current, it can't be added
        
        def _bb_upper_bound(current_set, remaining):
            """Compute upper bound on how many more we can add."""
            count = 0
            for x in remaining:
                if _can_add_fast(current_set, x):
                    count += 1
            return count
        
        # Sort candidates by "popularity" (fewer conflicts = more likely to be in optimal)
        def _conflict_degree(x, ref_set):
            """Count how many pairs in ref_set conflict with x."""
            ref_list = list(ref_set)
            m = len(ref_list)
            bp = bad_pairs[x]
            count = 0
            for i in range(m):
                a = ref_list[i]
                for j in range(i+1, m):
                    b = ref_list[j]
                    if a < b:
                        if (a, b) in bp:
                            count += 1
                    else:
                        if (b, a) in bp:
                            count += 1
            return count
        
        # Start B&B from the best set found so far, trying to improve
        # Or do a fresh B&B on a reduced candidate set
        
        # Strategy: take the current best, remove elements one by one,
        # and try to re-add different ones via B&B
        
        # Simpler: do B&B on candidates not in best_set, trying to replace some
        # elements of best_set
        
        # Let's do a direct B&B:
        # Universe: all numbers 2..256
        # We want max valid subset
        # Use the fact that primes are always valid as a baseline
        
        # For B&B, work with a sorted list of candidates
        # Order: try adding elements with fewer conflicts first
        
        all_candidates = nums[:]
        # Sort by conflict degree with an empty set (just number of bad pairs)
        all_candidates.sort(key=lambda x: len(bad_pairs[x]))
        
        def _bb_search(current_set, candidates, best_size_ref, best_set_ref, deadline):
            """Branch and bound search."""
            nonlocal best_size_ref, best_set_ref
            
            if time.time() > deadline:
                return
            
            if not candidates:
                if len(current_set) > best_size_ref:
                    best_size_ref = len(current_set)
                    best_set_ref = set(current_set)
                return
            
            # Upper bound
            ub = len(current_set) + len(candidates)
            if ub <= best_size_ref:
                return
            
            # Try to add the first candidate
            x = candidates[0]
            rest = candidates[1:]
            
            # Option 1: add x
            if _can_add_fast(current_set, x):
                _bb_search(current_set | {x}, rest, best_size_ref, best_set_ref, deadline)
            
            # Option 2: skip x
            _bb_search(current_set, rest, best_size_ref, best_set_ref, deadline)
        
        # B&B on a subset to keep it tractable
        # Use the top 30 least-conflicting candidates
        bb_candidates = all_candidates[:30]
        
        # Try B&B from empty set
        if time.time() < _DEADLINE - 20:
            _bb_search(set(), bb_candidates, best_size, best_set, _DEADLINE)
        
        # Also try B&B starting from a subset of the current best
        if time.time() < _DEADLINE - 15:
            # Remove the most "conflicting" element from best_set and try to rebuild
            if best_set:
                # Find element with highest conflict degree
                max_conf_el = max(best_set, key=lambda x: _conflict_degree(x, best_set - {x}))
                reduced = best_set - {max_conf_el}
                # Try to add elements from candidates
                extra_candidates = [x for x in nums if x not in reduced]
                extra_candidates.sort(key=lambda x: len(bad_pairs[x]))
                extra_candidates = extra_candidates[:20]
                
                _bb_search(reduced, extra_candidates, best_size, best_set, _DEADLINE)
    
    # Final verification
    S_list = sorted(best_set)
    if not _valid_set(S_list):
        # Fallback to primes
        S_list = sorted([x for x in nums if _is_prime(x)])
    
    return list(S_list)

def construct():
    return _construct_fast()
# EVOLVE-BLOCK-END
