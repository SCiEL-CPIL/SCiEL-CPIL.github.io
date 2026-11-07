# EVOLVE-BLOCK-START
import time
import random
import math

def _is_valid(S):
    """Full validity check (also enforces distinctness and range)."""
    if not S:
        return True
    if len(S) != len(set(S)):
        return False
    for x in S:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > 256:
            return False
    total = sum(S)
    # Build all nonempty subset sums
    sums = set()
    for x in S:
        new = set()
        for s in sums:
            new.add(s + x)
        sums |= new
        sums.add(x)
    # Check divisibility: no two distinct sums where one divides the other
    sl = sorted(sums)
    sset = sums
    for s in sl:
        if s == 0:
            continue
        m = 2 * s
        while m <= total:
            if m in sset:
                return False
            m += s
    return True

def _fast_divisibility_check(sums_mask, total):
    """Given a bitmask of sums (bit i set iff i in sums), check no divisibility pair.
    Only checks multiples up to total. Assumes 0 not in mask.
    Returns True if valid (no divisibility conflicts).
    """
    # Iterate over set bits
    m = sums_mask
    while m:
        # lowest set bit
        lb = m & (-m)
        s = lb.bit_length() - 1
        m ^= lb
        if s == 0:
            continue
        mult = 2 * s
        while mult <= total:
            if (sums_mask >> mult) & 1:
                return False
            mult += s
    return True

def _try_decompose(T, pool, deadline, rng, max_depth=30):
    """Given target sum set T (list of ints, sorted), try to find distinct elements
    from pool whose subset sums are exactly T.
    Returns list of elements or None.
    Uses backtracking: process T in descending order; for each t, try to assign a
    'new element' x such that x = t - (sum of some already-assigned smaller elements)?
    Actually simpler: think of it as: we need a set E such that subset_sums(E) == T.
    Equivalent: E must be a subset of T (each element is a subset sum of itself),
    and the subset-sum closure of E equals T.
    Strategy: pick elements from T greedily/backtracking such that closure matches T.
    """
    T_set = set(T)
    T_total = sum(T)  # not needed directly
    # Each element e in E must be in T (since e is a subset sum of {e}).
    # Also, the sum of all elements in E must be in T (the full set sum).
    # And for any subset, its sum must be in T.
    
    # Backtracking: build E by adding elements one by one from candidates in T.
    # When we add x, the new subset sums are {x} union {s+x for s in current_sums}.
    # All must be in T_set, and at the end current_sums must equal T_set.
    
    # Order candidates: try to match the largest elements of T first.
    # The largest element in T must be the sum of all elements in E (since it's the max subset sum).
    # So sum(E) = max(T).
    
    max_T = max(T)
    # We need E subset of T, sum(E) = max_T, and subset_sums(E) == T_set.
    
    # Candidates for E: elements of T (excluding max_T itself unless E has one element).
    # Actually if E = {max_T}, then subset_sums = {max_T}, so T must be {max_T}.
    
    # Use backtracking with pruning:
    # State: current_sums (set), remaining to add, current_total.
    # We must end with current_sums == T_set and current_total == max_T.
    
    # Sort T candidates for E: we'll try adding elements in some order.
    # Key insight: the elements of E must themselves be in T.
    # Also, the second largest in T is either the second largest element of E,
    # or the sum of all but the smallest element of E, etc.
    
    # Simpler backtracking:
    # Pick elements for E from T \ {max_T} (since max_T is the total, not an element unless |E|=1).
    # Actually max_T could be an element if E={max_T}.
    
    # Let's do: candidates = sorted(T, reverse=True)
    # We decide for each candidate whether to include it in E.
    # But that's 2^|T| which is too much.
    
    # Better: recursive with memo on (frozenset(current_sums), remaining_max_needed)
    # But current_sums can be large.
    
    # Alternative: since |T| is small (<=30 or so), and |E| <= 30,
    # we can do backtracking where at each step we pick the next element of E
    # from the remaining candidates in T, checking that all new sums are in T.
    
    # Let's use a different approach: 
    # The elements of E are a subset of T. Let's find E by:
    # 1. max_T must be in T (it is, by definition).
    # 2. The elements of E must sum to max_T.
    # 3. subset_sums(E) = T_set.
    
    # Backtracking over possible E:
    # Start with E = [], current_sums = set(), total = 0.
    # At each step, try adding a candidate x from T such that:
    #   - x not in E
    #   - x + current_sums subset of T_set (i.e., for all s in current_sums, s+x in T_set)
    #   - x in T_set
    #   - total + x <= max_T
    #   - if total + x == max_T, then current_sums | {x} | {s+x for s in current_sums} must equal T_set
    # Prune: the remaining sum needed is max_T - total. The smallest element we can still add
    # must be <= remaining, and the largest must be >= remaining/2 (if more than one element left).
    
    # This is still exponential but with strong pruning.
    
    candidates = sorted([x for x in T if x < max_T], reverse=True)
    # Also consider max_T itself as a candidate only if T == {max_T}
    
    best = None
    
    def backtrack(E, current_sums, total, idx):
        nonlocal best
        if best is not None:
            return True  # found
        if time.time() > deadline:
            return False
        if total == max_T:
            if current_sums == T_set:
                best = E[:]
                return True
            return False
        if len(E) >= max_depth:
            return False
        remaining = max_T - total
        # Try candidates from idx onwards
        for i in range(idx, len(candidates)):
            x = candidates[i]
            if x > remaining:
                continue
            # Check new sums
            new_sums = set(current_sums)
            new_sums.add(x)
            valid = True
            for s in current_sums:
                ns = s + x
                if ns not in T_set:
                    valid = False
                    break
                new_sums.add(ns)
            if not valid:
                continue
            # Also check: can we still reach max_T with remaining elements?
            # The sum of all remaining candidates from i+1 onwards must be >= remaining - x
            # (rough prune)
            E.append(x)
            if backtrack(E, new_sums, total + x, i + 1):
                return True
            E.pop()
        return False
    
    # Also try E = {max_T} if T == {max_T}
    if len(T) == 1:
        return [max_T]
    
    backtrack([], set(), 0, 0)
    if best is not None:
        return best
    return None

def _construct_T(rng, deadline, target_size):
    """Construct a candidate target sum set T of size ~target_size,
    with no divisibility pairs, in range 1..1280.
    Returns T as a sorted list, or None.
    """
    # We want T to be 'dense' enough to allow decomposition, but sparse enough
    # to avoid divisibility. 
    # Strategy: start with large numbers (less likely to divide others),
    # and ensure no divisibility.
    
    # Upper bound on sums: if we have k elements each <= 256, max sum = 256*k.
    # For k=10, max sum = 2560, but our range is 1..1280.
    # So for larger k, we need smaller elements.
    
    # Let's pick T elements from 1..1280 with no divisibility, size = target_size.
    # Use greedy: start with large numbers, add if no divisibility conflict.
    
    # To make decomposition easier, T should be 'closed' under some structure.
    # A good heuristic: T should contain a chain of 'building blocks'.
    
    # Alternative: generate T as the subset sums of a random small set, then verify.
    # This guarantees decomposability by construction!
    
    # Let's try: pick a random set E of size target_size from 1..256,
    # compute its subset sums T, check T has no divisibility pairs and all sums <= 1280.
    # If valid, return T (and we know E is a valid answer).
    # But this is just the original problem. The point of T-based search is to
    # explore T that might be decomposable by a *different* E.
    
    # For now, let's use the T-based approach as a way to *verify* and *search*:
    # Generate random T (no divisibility, size ~target_size), try to decompose.
    
    # Generate T: random subset of 1..1280 with no divisibility.
    # To bias toward decomposability, include some small numbers.
    
    T = set()
    # Add some small numbers
    for x in range(1, 20):
        if rng.random() < 0.5:
            # Check divisibility with existing
            ok = True
            for y in T:
                if y % x == 0 or x % y == 0:
                    ok = False
                    break
            if ok:
                T.add(x)
    
    # Add larger numbers
    candidates = list(range(20, 1281))
    rng.shuffle(candidates)
    for x in candidates:
        if len(T) >= target_size:
            break
        if time.time() > deadline:
            break
        ok = True
        for y in T:
            if y % x == 0 or x % y == 0:
                ok = False
                break
        if ok:
            T.add(x)
    
    if len(T) < 3:
        return None
    
    return sorted(T)

def _decompose_from_T(T, pool, deadline, rng):
    """Given T, try to find E such that subset_sums(E) == T.
    Uses the backtracking in _try_decompose.
    """
    E = _try_decompose(T, pool, deadline, rng)
    return E

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit
    
    pool = list(range(1, 257))
    best_set = []
    best_size = 0
    
    # Phase 1: Quick greedy to get a baseline
    # (Same as before but with bitmask)
    current = []
    cur_sums_mask = 0  # bit i set iff i is a subset sum
    cur_total = 0
    cur_sums_list = set()
    
    # Greedy: try to add elements
    for x in sorted(pool, reverse=True):
        if time.time() > deadline:
            break
        # Check if adding x is valid
        new_mask = cur_sums_mask | (1 << x)
        new_total = cur_total + x
        # Check new sums: for each s in cur_sums, s+x must not create divisibility
        # and must be <= 1280
        valid = True
        for s in cur_sums_list:
            ns = s + x
            if ns > 1280:
                valid = False
                break
            if (new_mask >> ns) & 1:
                # ns already a sum - that's fine, it's just the same value
                pass
            # Check divisibility: does any existing sum divide ns, or ns divide any existing sum?
            # ns divides existing: existing = k*ns for k>=2, but existing < ns+x <= 1280
            # Actually ns > s, so ns can't divide s. But ns could divide a larger existing sum?
            # No, ns is new. We need: no existing sum t such that t|ns or ns|t.
            # ns|t means t = k*ns, t <= 1280. 
            # t|ns means ns = k*t, k>=2, ns <= 1280.
            # Let's just check all existing sums.
            if valid:
                for t in cur_sums_list:
                    if t != ns and t != 0 and ns != 0:
                        if ns % t == 0 or t % ns == 0:
                            valid = False
                            break
            if not valid:
                break
        if valid:
            # Also check x against all existing sums
            for t in cur_sums_list:
                if t != x and (x % t == 0 or t % x == 0):
                    valid = False
                    break
        if valid:
            # Update
            current.append(x)
            new_sums_list = set(cur_sums_list)
            new_sums_list.add(x)
            for s in cur_sums_list:
                new_sums_list.add(s + x)
            cur_sums_list = new_sums_list
            cur_total = new_total
            cur_sums_mask = 0
            for s in cur_sums_list:
                cur_sums_mask |= (1 << s)
            if len(current) >= 30:
                break
    
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Phase 2: T-based backtracking search
    # Try to find larger sets by constructing T and decomposing.
    for restart in range(200):
        if time.time() > deadline:
            break
        rng = random.Random(restart * 104729)
        
        # Target size: start from best_size+1, go up
        target = min(best_size + 1 + (restart % 5), 30)
        
        T = _construct_T(rng, deadline, target)
        if T is None:
            continue
        
        # Try to decompose T
        E = _decompose_from_T(T, pool, deadline, rng)
        if E is not None and len(E) > best_size:
            # Verify
            if _is_valid(E):
                best_size = len(E)
                best_set = E[:]
    
    # Phase 3: Local search on best_set (swap-based)
    if best_set and time.time() < deadline - 3:
        current = list(best_set)
        used = set(current)
        available = [x for x in pool if x not in used]
        
        improved = True
        while improved and time.time() < deadline - 1:
            improved = False
            rng = random.Random(int(time.time() * 1000) % 100000)
            rng.shuffle(available)
            
            # Try swaps
            for i in range(len(current)):
                if time.time() > deadline - 1:
                    break
                for x in available[:50]:
                    if time.time() > deadline - 1:
                        break
                    removed = current[i]
                    test = current[:i] + current[i+1:] + [x]
                    if _is_valid(test):
                        current = test
                        used.discard(removed)
                        used.add(x)
                        available.remove(x)
                        available.append(removed)
                        improved = True
                        break
                if improved:
                    break
            
            # Try adds
            if not improved:
                for x in available:
                    if time.time() > deadline - 1:
                        break
                    test = current + [x]
                    if _is_valid(test):
                        current = test
                        used.add(x)
                        available.remove(x)
                        improved = True
                        break
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Final validation
    if best_set and not _is_valid(best_set):
        # Fallback
        best_set = []
        for x in sorted(pool, reverse=True):
            test = best_set + [x]
            if _is_valid(test):
                best_set = test
    
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
