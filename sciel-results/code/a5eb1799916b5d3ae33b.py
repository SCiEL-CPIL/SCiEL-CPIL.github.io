"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

_DEADLINE_SECONDS = 85  # safety margin before 90s kill

def _precompute_gcd_table(n):
    """Precompute gcd for all pairs in [1..n]."""
    table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            g = gcd(i, j)
            table[i][j] = g
            table[j][i] = g
    return table

def _build_conflict_graph(n, table):
    """
    Build conflict graph: for each x, conflicts[x] is a set of (a,b) pairs
    such that (a,b,x) forms an equal-gcd triple.
    Also build pair_conflicts: for each pair (a,b), the set of x that conflict with it.
    """
    conflicts = [set() for _ in range(n + 1)]
    # For each triple (a,b,c), check if all three pairwise gcds are equal
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            for c in range(b + 1, n + 1):
                g_ab = table[a][b]
                g_ac = table[a][c]
                g_bc = table[b][c]
                if g_ab == g_ac == g_bc:
                    # This triple is bad; each element conflicts with the pair of the other two
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))
    return conflicts

def _can_add_fast(x, selected, conflicts):
    """
    Check if adding x to selected keeps the set valid using precomputed conflicts.
    For each conflict pair (a,b) of x, check if both a and b are in selected.
    """
    selected_set = set(selected)
    for (a, b) in conflicts[x]:
        if a in selected_set and b in selected_set:
            return False
    return True

def _greedy_pass(order, conflicts):
    """Greedy pass: add elements in given order if valid."""
    selected = []
    selected_set = set()
    for x in order:
        if _can_add_fast(x, selected, conflicts):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_improve_add(selected, candidates, conflicts, deadline):
    """Try to add excluded candidates."""
    improved = list(selected)
    in_set = set(improved)
    to_try = [c for c in candidates if c not in in_set]
    random.shuffle(to_try)
    for x in to_try:
        if time.time() > deadline:
            break
        if _can_add_fast(x, improved, conflicts):
            improved.append(x)
            in_set.add(x)
    return improved

def _local_improve_swap(selected, candidates, conflicts, deadline, max_swaps=200):
    """Try swap-based local search: remove one, add one or more."""
    improved = list(selected)
    in_set = set(improved)
    excluded = [c for c in candidates if c not in in_set]
    
    for _ in range(max_swaps):
        if time.time() > deadline:
            break
        if not excluded:
            break
        
        # Pick a random element to remove
        idx = random.randint(0, len(improved) - 1)
        removed = improved[idx]
        improved.pop(idx)
        in_set.discard(removed)
        
        # Try to add multiple excluded elements
        added_any = False
        remaining = [e for e in excluded if e not in in_set]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            if _can_add_fast(x, improved, conflicts):
                improved.append(x)
                in_set.add(x)
                added_any = True
        
        if not added_any or len(improved) < len(selected):
            # Revert
            improved.insert(idx, removed)
            in_set.add(removed)
        else:
            # Update excluded list
            excluded = [c for c in candidates if c not in in_set]
    
    return improved

def _construct_prime_powers(n):
    """
    Construct a set based on prime powers and 1.
    Select numbers that are powers of distinct primes, plus 1.
    This creates structural diversity in gcd patterns.
    """
    result = set()
    result.add(1)
    
    # Generate prime powers up to n
    # For each prime p, add p^k for k >= 1 while p^k <= n
    # But to ensure distinct primes, we only take one power per prime
    # Actually, let's be more inclusive: add all prime powers, but ensure
    # that for any triple, the gcds are distinct
    
    # Strategy: Add 1, then for each prime, add the smallest prime power
    # This ensures that gcd of two numbers from different primes is 1
    # gcd of two powers of the same prime is the smaller power
    # We need to avoid triples where all three pairwise gcds are equal
    
    # Let's add: 1, and for each prime p <= n, add p (the prime itself)
    # Then optionally add some prime squares if they don't create conflicts
    
    # First, get all primes up to n
    primes = []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    # Add all primes
    for p in primes:
        result.add(p)
    
    # Try to add prime squares
    for p in primes:
        p2 = p * p
        if p2 <= n:
            result.add(p2)
    
    # Try to add prime cubes
    for p in primes:
        p3 = p * p * p
        if p3 <= n:
            result.add(p3)
    
    return list(result)

def _factor_count(n):
    """Count total prime factors with multiplicity."""
    cnt = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            cnt += 1
            n //= d
        d += 1
    if n > 1:
        cnt += 1
    return cnt

def _verify_set(selected, table):
    """Verify that the set has no equal-gcd triples."""
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            for k in range(j + 1, len(selected)):
                a, b, c = selected[i], selected[j], selected[k]
                g_ab = table[a][b]
                g_ac = table[a][c]
                g_bc = table[b][c]
                if g_ab == g_ac == g_bc:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    all_nums = list(range(1, 129))
    
    # Precompute gcd table
    table = _precompute_gcd_table(128)
    
    # Build conflict graph for fast checking
    conflicts = _build_conflict_graph(128, table)
    
    # Precompute factor counts for heuristic ordering
    fc = {x: _factor_count(x) for x in all_nums}
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Prime power construction (directive-based)
    if time.time() < deadline:
        s = _construct_prime_powers(128)
        # Verify and clean if needed
        if _verify_set(s, table):
            if len(s) > best_len:
                best_set, best_len = s, len(s)
        else:
            # Greedily remove elements that cause conflicts
            s_valid = []
            for x in s:
                if _can_add_fast(x, s_valid, conflicts):
                    s_valid.append(x)
            if len(s_valid) > best_len:
                best_set, best_len = s_valid, len(s_valid)
    
    # Strategy 2: Sequential greedy
    if time.time() < deadline:
        s = _greedy_pass(all_nums, conflicts)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 3: Reverse order greedy
    if time.time() < deadline:
        s = _greedy_pass(list(reversed(all_nums)), conflicts)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 4: Prime-factor-rich first greedy
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (-fc[x], x))
        s = _greedy_pass(order, conflicts)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 5: Prime-factor-poor first greedy
    if time.time() < deadline:
        order = sorted(all_nums, key=lambda x: (fc[x], x))
        s = _greedy_pass(order, conflicts)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 6: Many random greedy passes with local improvement
    rng = random.Random(42)
    num_random_passes = 2000
    for i in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order, conflicts)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, conflicts, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved2 = _local_improve_swap(best_set, all_nums, conflicts, deadline, max_swaps=50)
                    if len(improved2) > best_len:
                        best_set, best_len = improved2, len(improved2)
    
    # Final local improvement
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, conflicts, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
        if time.time() < deadline:
            improved2 = _local_improve_swap(best_set, all_nums, conflicts, deadline, max_swaps=100)
            if len(improved2) > best_len:
                best_set, best_len = improved2, len(improved2)
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
