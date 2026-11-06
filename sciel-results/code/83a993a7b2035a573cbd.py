"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = 88.0  # Safety margin from 90s limit

    best_set = []
    best_len = 0

    # Modular sieve: pre-filter candidates using small primes
    # For each prime p, compute which x in [1,512] are "safe"
    # x is unsafe if there exist distinct a,b in [1,512], a!=b, a!=x, b!=x
    # such that a + b ≡ 2x (mod p)
    # This is a necessary condition (not sufficient) for x to be a mean of {a,b}
    # We use multiple primes to reduce false positives

    def compute_modular_sieve(primes):
        """
        For each x in [1,512], check if 2x mod p can be written as a+b mod p
        for distinct a,b in [1,512] with a!=x, b!=x.
        If for ANY prime p this is impossible, x is definitely safe.
        We return the set of x that pass ALL primes (i.e., for every prime,
        there EXISTS a pair a,b that could form the mean mod p).
        Actually, we want to KEEP x only if for every prime, it's NOT the case
        that no pair works. Let me re-think.
        
        We want to eliminate x if for some prime p, 2x mod p CANNOT be expressed
        as a+b mod p for any distinct a,b in [1,512] with a,b != x.
        But that's hard to check efficiently.
        
        Alternative: For a given set S of selected elements, x is bad if
        x = (a1+...+ak)/k for some subset of S\{x}.
        Mod p: 2x ≡ a1+...+ak (mod p) is necessary for k=2.
        
        Let's use a simpler heuristic: precompute for each x, the set of primes
        p for which 2x mod p is NOT achievable as sum of 2 distinct elements
        from [1,512]\{x}. If x fails for at least one prime, it's likely safe.
        But we want to KEEP promising candidates, so we keep x if it passes
        the sieve (i.e., for all primes, 2x mod p IS achievable).
        
        Actually, the sieve should ELIMINATE x if it's clearly bad.
        x is clearly bad if for some prime p, 2x mod p is NOT achievable as
        sum of 2 distinct elements from the full range [1,512]\{x}.
        But that's almost always achievable for small p.
        
        Let me try a different approach: use the sieve to rank candidates.
        For each x, count how many primes p have 2x mod p NOT achievable as
        sum of 2 distinct elements from [1,512]\{x}. Higher count = safer.
        """
        # For each prime p, precompute achievable sums of 2 distinct elements
        achievable = {}
        for p in primes:
            # achievable[p] = set of (a+b) mod p for distinct a,b in [1,512]
            # But we need to exclude cases where a=x or b=x, which depends on x
            # For now, compute all achievable sums mod p from [1,512]
            sums = set()
            # This is O(512^2) per prime, which is too slow for many primes
            # Instead, note that for p <= 512, most residues are achievable
            # Let's just use a few small primes and a simpler criterion
            
            # For prime p, the set {a mod p : a in [1,512]} covers all residues
            # if 512 >= p. For p <= 512, all residues appear.
            # So a+b mod p can be any residue for distinct a,b if p is small enough.
            # Specifically, if p <= 512, for any target t mod p, we can find
            # a,b distinct with a+b ≡ t mod p unless p is very small.
            pass
        
        # Simpler: just use the fact that for large enough sets, most x are fine
        # Let's skip the complex sieve and use a simpler pre-filter
        return set(range(1, 513))

    def try_greedy_v2(order):
        selected = []
        n = 0
        dp_all = [1] # dp_all[0] = 1
        dp_without = {}
        
        for x in order:
            if time.time() - start_time > deadline:
                break
            if n >= 48:
                break
            
            valid = True
            
            # Check 1: Is x the mean of any subset of size k>=2 of selected?
            for k in range(2, n + 1):
                target = k * x
                if target > 24576:
                    break
                if (dp_all[k] >> target) & 1:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Check 2: For each a in selected, is a the mean of any subset of size k>=2
            # from (selected + {x}) \ {a}?
            for a in selected:
                dp_w = dp_without[a]
                # dp_w[k] = sums of k elements from selected \ {a}
                # We need to check if a is mean of k+1 elements from (selected \ {a}) ∪ {x}
                # i.e., (k+1)*a - x is a sum of k elements from selected \ {a}
                for k in range(1, n):
                    target = (k + 1) * a - x
                    if target < 0 or target > 24576:
                        continue
                    if (dp_w[k] >> target) & 1:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Valid, add x
            selected.append(x)
            
            # Save old state for x's exclusion set
            old_dp_all = list(dp_all)
            
            # Update dp_all
            new_dp_all = [0] * (n + 2)
            new_dp_all[0] = 1
            for k in range(1, n + 2):
                val = old_dp_all[k] if k <= n else 0
                if k - 1 >= 0 and k - 1 <= n:
                    val |= (old_dp_all[k-1] << x)
                new_dp_all[k] = val
            dp_all = new_dp_all
            
            # Update dp_without for existing elements
            new_dp_without = {}
            for a in selected[:-1]:
                old_dp_w = dp_without[a]
                new_list = [0] * (n + 2)
                new_list[0] = 1
                for k in range(1, n + 2):
                    val = old_dp_w[k] if k <= n - 1 else 0
                    if k - 1 >= 0 and k - 1 <= n - 1:
                        val |= (old_dp_w[k-1] << x)
                    new_list[k] = val
                new_dp_without[a] = new_list
            
            # For x, the set of others is the old selected set.
            # dp_without[x][k] = sums of k elements from selected[:-1] (which is old selected)
            new_dp_without[x] = old_dp_all + [0] * max(0, (n + 1) - len(old_dp_all))
            
            dp_without = new_dp_without
            n += 1
            
        return selected

    def validate_set(s):
        """Full bitset verification that no element is mean of subset of >=2 others."""
        n = len(s)
        if n < 3:
            return True
        
        # For each element a, check if a is mean of any subset of size >= 2 from others
        for i, a in enumerate(s):
            others = s[:i] + s[i+1:]
            m = len(others)
            if m < 2:
                continue
            
            # Build dp for others
            dp_others = [1]
            for x in others:
                new_dp = [0] * (len(dp_others) + 1)
                new_dp[0] = 1
                for k in range(1, len(dp_others) + 1):
                    val = dp_others[k-1] if k-1 < len(dp_others) else 0
                    if k-1 < len(dp_others):
                        val |= (dp_others[k-1] << x)
                    new_dp[k] = val
                dp_others = new_dp
            
            # Check all subset sizes k >= 2
            for k in range(2, m + 1):
                target = k * a
                if target > 24576:
                    break
                if (dp_others[k] >> target) & 1:
                    return False
        
        return True

    def local_search_full(initial_set, max_iters=200):
        """
        Local search on the full constraint.
        Tries to increase the size of the set by adding elements.
        """
        current_set = list(initial_set)
        current_set.sort()
        current_len = len(current_set)
        all_candidates = set(range(1, 513))
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            if time.time() - start_time > deadline:
                break
            improved = False
            iters += 1
            
            # Try to add an element
            candidates_to_try = list(all_candidates - set(current_set))
            random.shuffle(candidates_to_try)
            
            for new_x in candidates_to_try:
                if time.time() - start_time > deadline:
                    break
                
                if current_len < 48:
                    test_set = current_set + [new_x]
                    if validate_set(test_set):
                        current_set = test_set
                        current_len += 1
                        improved = True
                        break
                
        return current_set

    # Generate orders
    orders = []
    
    # Spread order
    spread = []
    lo, hi = 1, 512
    while lo <= hi:
        spread.append(lo)
        if lo != hi:
            spread.append(hi)
        lo += 1
        hi -= 1
    orders.append(spread)
    
    # Random shuffles
    for i in range(30):
        random.seed(i)
        cand = list(range(1, 513))
        random.shuffle(cand)
        orders.append(cand)
    
    # Ascending/Descending
    orders.append(list(range(1, 513)))
    orders.append(list(range(512, 0, -1)))
    
    for order in orders:
        if time.time() - start_time > deadline:
            break
        res = try_greedy_v2(order)
        
        # Apply local search to the result
        if len(res) > 0:
            res = local_search_full(res, max_iters=50)
            
        if len(res) > best_len:
            best_set = res
            best_len = len(res)
            if best_len >= 48:
                break
                
    return best_set

# EVOLVE-BLOCK-END
