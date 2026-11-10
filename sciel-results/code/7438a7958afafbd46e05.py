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

def _precompute_bad_pairs(n, table):
    """
    Precompute for each x, the list of pairs (u, v) with u < v
    such that (u, v, x) is a bad triple.
    i.e., gcd(u, v) == gcd(u, x) == gcd(v, x).
    """
    bad_pairs = [[] for _ in range(n + 1)]
    for u in range(1, n + 1):
        for v in range(u + 1, n + 1):
            g_uv = table[u][v]
            # Find all x such that gcd(u, x) == g_uv and gcd(v, x) == g_uv
            # This is equivalent to:
            # x is a multiple of g_uv
            # gcd(u/g_uv, x/g_uv) == 1
            # gcd(v/g_uv, x/g_uv) == 1
            u_prime = u // g_uv
            v_prime = v // g_uv
            # x = k * g_uv
            # gcd(u_prime, k) == 1 and gcd(v_prime, k) == 1
            # So k must be coprime to u_prime and v_prime.
            # i.e., k is coprime to lcm(u_prime, v_prime) ? No, just coprime to both.
            # Let L = lcm(u_prime, v_prime). Then gcd(k, L) == 1 is sufficient?
            # If gcd(k, u_prime)==1 and gcd(k, v_prime)==1, then gcd(k, lcm(u_prime, v_prime))==1.
            # Yes.
            # So we need to find k such that 1 <= k * g_uv <= n and gcd(k, L) == 1.
            # Then x = k * g_uv.
            # And we add (u, v) to bad_pairs[x].
            
            # Optimization: Iterate through multiples of g_uv
            # x = m * g_uv
            # Condition: gcd(u, m*g_uv) == g_uv => gcd(u/g_uv, m) == 1
            #            gcd(v, m*g_uv) == g_uv => gcd(v/g_uv, m) == 1
            
            # Let's just check all x in [1, n] for this pair (u, v)?
            # That is O(n) per pair. Total pairs ~ 8000. 8000 * 128 = 1M ops. Fast.
            for x in range(1, n + 1):
                if x == u or x == v:
                    continue
                if table[u][x] == g_uv and table[v][x] == g_uv:
                    bad_pairs[x].append((u, v))
    return bad_pairs

def _can_add_fast(x, selected_set, bad_pairs):
    """Check if adding x to selected_set keeps the set valid using precomputed bad pairs."""
    for u, v in bad_pairs[x]:
        if u in selected_set and v in selected_set:
            return False
    return True

def _greedy_pass(order, bad_pairs):
    """Greedy pass: add elements in given order if valid."""
    selected = []
    selected_set = set()
    for x in order:
        if _can_add_fast(x, selected_set, bad_pairs):
            selected.append(x)
            selected_set.add(x)
    return selected

def _local_improve_add(selected, candidates, bad_pairs, deadline):
    """Try to add excluded candidates."""
    improved = list(selected)
    in_set = set(improved)
    to_try = [c for c in candidates if c not in in_set]
    random.shuffle(to_try)
    for x in to_try:
        if time.time() > deadline:
            break
        if _can_add_fast(x, in_set, bad_pairs):
            improved.append(x)
            in_set.add(x)
    return improved

def _local_improve_swap(selected, candidates, bad_pairs, deadline, max_swaps=1000):
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
        in_set.remove(removed)
        
        # Try to add multiple excluded elements
        added_any = False
        remaining = [e for e in excluded if e not in in_set]
        random.shuffle(remaining)
        for x in remaining:
            if time.time() > deadline:
                break
            if _can_add_fast(x, in_set, bad_pairs):
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

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE_SECONDS
    
    all_nums = list(range(1, 129))
    
    # Precompute gcd table
    table = _precompute_gcd_table(128)
    
    # Precompute bad pairs
    bad_pairs = _precompute_bad_pairs(128, table)
    
    best_set = []
    best_len = 0
    
    # Strategy 1: Sequential greedy
    if time.time() < deadline:
        s = _greedy_pass(all_nums, bad_pairs)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 2: Reverse order greedy
    if time.time() < deadline:
        s = _greedy_pass(list(reversed(all_nums)), bad_pairs)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
    
    # Strategy 3: Many random greedy passes with local improvement
    rng = random.Random(42)
    num_random_passes = 500
    for i in range(num_random_passes):
        if time.time() > deadline:
            break
        order = all_nums[:]
        rng.shuffle(order)
        s = _greedy_pass(order, bad_pairs)
        if len(s) > best_len:
            best_set, best_len = s, len(s)
            if time.time() < deadline:
                improved = _local_improve_add(best_set, all_nums, bad_pairs, deadline)
                if len(improved) > best_len:
                    best_set, best_len = improved, len(improved)
                if time.time() < deadline:
                    improved2 = _local_improve_swap(best_set, all_nums, bad_pairs, deadline, max_swaps=200)
                    if len(improved2) > best_len:
                        best_set, best_len = improved2, len(improved2)
    
    # Final local improvement
    if time.time() < deadline:
        improved = _local_improve_add(best_set, all_nums, bad_pairs, deadline)
        if len(improved) > best_len:
            best_set, best_len = improved, len(improved)
        if time.time() < deadline:
            improved2 = _local_improve_swap(best_set, all_nums, bad_pairs, deadline, max_swaps=500)
            if len(improved2) > best_len:
                best_set, best_len = improved2, len(improved2)
    
    # Ensure nonempty
    if not best_set:
        best_set = [1]
    
    return best_set
# EVOLVE-BLOCK-END
