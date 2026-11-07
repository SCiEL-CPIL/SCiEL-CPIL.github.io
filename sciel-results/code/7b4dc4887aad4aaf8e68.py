"""Suite task: large integer sets with no two-other-product divisibility."""
import time
import random
from math import isqrt, gcd

NMAX = 256
UNIVERSE = list(range(2, NMAX + 1))
UNIV_IDX = {v: i for i, v in enumerate(UNIVERSE)}  # value -> 0-based index
UNIV_SIZE = len(UNIVERSE)  # 255


def _precompute_conflicts():
    """Precompute conflict structures for fast validity checking.
    
    Returns a dict with:
      - 'req': req[x][b] = x // gcd(x, b), so x | b*c iff req[x][b] | c
      - 'conflict_mask': conflict_mask[x] = bitmask of y such that x|y*z for SOME z != y in universe
        (used for quick conflict detection in removal)
      - 'divides_mask': divides_mask[x] = bitmask of y such that x | y
    """
    req = {}
    for x in range(2, NMAX + 1):
        req[x] = {}
        for b in range(2, NMAX + 1):
            if x == b:
                continue
            d = x // gcd(x, b)
            req[x][b] = d
    
    # Precompute: for each x, the bitmask of y values (in universe) such that x | y
    divides_mask = {}
    for x in range(2, NMAX + 1):
        mask = 0
        for y in range(2, NMAX + 1):
            if y % x == 0:
                mask |= (1 << UNIV_IDX[y])
        divides_mask[x] = mask
    
    # For each x and each b, precompute the bitmask of c such that req[x][b] | c
    # i.e., x | b*c
    # req_divides_mask[x][b] = bitmask of c where req[x][b] | c
    req_divides_mask = {}
    for x in range(2, NMAX + 1):
        req_divides_mask[x] = {}
        for b in range(2, NMAX + 1):
            if x == b:
                req_divides_mask[x][b] = 0
                continue
            d = req[x][b]
            mask = 0
            for c in range(2, NMAX + 1):
                if c % d == 0:
                    mask |= (1 << UNIV_IDX[c])
            req_divides_mask[x][b] = mask
    
    return {
        'req': req,
        'divides_mask': divides_mask,
        'req_divides_mask': req_divides_mask,
    }


def _valid_addition_fast(x, S_list, precomp):
    """Check if x can be added to set S_list without violating the constraint.
    Uses bitmask operations for speed.
    """
    n = len(S_list)
    if n == 0:
        return True
    if n == 1:
        # Only two elements, no triple possible
        return True
    if n == 2:
        # Check all three conditions for the pair
        b, c = S_list[0], S_list[1]
        # x | b*c?
        if (b * c) % x == 0:
            return False
        # b | x*c?
        if (x * c) % b == 0:
            return False
        # c | x*b?
        if (x * b) % c == 0:
            return False
        return True
    
    req = precomp['req']
    req_divides_mask = precomp['req_divides_mask']
    
    # Build bitmask of current set
    S_mask = 0
    for v in S_list:
        S_mask |= (1 << UNIV_IDX[v])
    
    for i in range(n):
        b = S_list[i]
        for j in range(n):
            if i == j:
                continue
            c = S_list[j]
            # x | b*c ?  -> req[x][b] | c
            d = req[x][b]
            if c % d == 0:
                return False
            # b | x*c ?  -> req[b][x] | c
            d2 = req[b][x]
            if c % d2 == 0:
                return False
            # c | x*b ?  -> req[c][x] | b
            d3 = req[c][x]
            if b % d3 == 0:
                return False
    return True


def _valid_addition_bitmask(x, S_mask, S_list, precomp):
    """Faster check using precomputed bitmasks when S is large."""
    n = len(S_list)
    if n < 3:
        return _valid_addition_fast(x, S_list, precomp)
    
    req = precomp['req']
    req_divides_mask = precomp['req_divides_mask']
    
    # x | b*c for some b,c in S, b!=c:
    # For each b in S, need to check if there exists c in S, c != b, with req[x][b] | c
    # req_divides_mask[x][b] gives all c where req[x][b] | c
    # We need (req_divides_mask[x][b] & S_mask) to contain some c != b
    # i.e., (req_divides_mask[x][b] & S_mask) != 0 AND not just {b}
    for b in S_list:
        mask_c = req_divides_mask[x][b] & S_mask
        # Remove b itself from consideration
        if mask_c & ~(1 << UNIV_IDX[b]):
            return False
    
    # b | x*c for some b,c in S, b!=c:
    # For each b in S, need c in S, c != b, with req[b][x] | c
    for b in S_list:
        mask_c = req_divides_mask[b][x] & S_mask
        if mask_c & ~(1 << UNIV_IDX[b]):
            return False
    
    # c | x*b for some b,c in S, b!=c:
    # For each c in S, need b in S, b != c, with req[c][x] | b
    for c in S_list:
        mask_b = req_divides_mask[c][x] & S_mask
        if mask_b & ~(1 << UNIV_IDX[c]):
            return False
    
    return True


def _greedy_remove(precomp, deadline, order='largest'):
    """Start with all numbers, iteratively remove elements involved in conflicts.
    Uses bitmask for efficient conflict detection."""
    S = set(range(2, NMAX + 1))
    S_mask = 0
    for v in S:
        S_mask |= (1 << UNIV_IDX[v])
    
    req = precomp['req']
    req_divides_mask = precomp['req_divides_mask']
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        to_remove = set()
        S_list = list(S)
        n = len(S_list)
        
        if n < 3:
            break
        
        # For each x, check if it's in a conflict using bitmask
        for x in S_list:
            if x in to_remove:
                continue
            x_idx = UNIV_IDX[x]
            x_bit = 1 << x_idx
            S_without_x = S_mask & ~x_bit
            
            # Check: exists b,c in S\{x}, b!=c, such that x|b*c
            found = False
            for b in S_list:
                if b == x or b in to_remove:
                    continue
                b_idx = UNIV_IDX[b]
                b_bit = 1 << b_idx
                # Need c in S\{x}, c != b, with req[x][b] | c
                mask_c = req_divides_mask[x][b] & S_without_x
                if mask_c & ~b_bit:
                    found = True
                    break
            
            if not found:
                # Check: exists b,c in S\{x}, b!=c, such that b|x*c
                for b in S_list:
                    if b == x or b in to_remove:
                        continue
                    b_idx = UNIV_IDX[b]
                    b_bit = 1 << b_idx
                    mask_c = req_divides_mask[b][x] & S_without_x
                    if mask_c & ~b_bit:
                        found = True
                        break
            
            if not found:
                # Check: exists b,c in S\{x}, b!=c, such that c|x*b
                for c in S_list:
                    if c == x or c in to_remove:
                        continue
                    c_idx = UNIV_IDX[c]
                    c_bit = 1 << c_idx
                    mask_b = req_divides_mask[c][x] & S_without_x
                    if mask_b & ~c_bit:
                        found = True
                        break
            
            if found:
                to_remove.add(x)
        
        if to_remove:
            for x in to_remove:
                S.discard(x)
            S_mask = 0
            for v in S:
                S_mask |= (1 << UNIV_IDX[v])
            changed = True
    
    return list(S)


def _greedy_add_largest(precomp, deadline):
    """Greedy: add numbers from largest to smallest if valid."""
    chosen = []
    chosen_set = set()
    chosen_mask = 0
    for x in range(NMAX, 1, -1):
        if time.time() > deadline:
            break
        if _valid_addition_fast(x, chosen, precomp):
            chosen.append(x)
            chosen_set.add(x)
            chosen_mask |= (1 << UNIV_IDX[x])
    return chosen


def _greedy_add_smallest(precomp, deadline):
    """Greedy: add numbers from smallest to largest if valid."""
    chosen = []
    chosen_set = set()
    for x in range(2, NMAX + 1):
        if time.time() > deadline:
            break
        if _valid_addition_fast(x, chosen, precomp):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _greedy_add_random(precomp, deadline, seed=42):
    """Greedy: add numbers in random order if valid."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    rng.shuffle(candidates)
    chosen = []
    chosen_set = set()
    for x in candidates:
        if time.time() > deadline:
            break
        if _valid_addition_fast(x, chosen, precomp):
            chosen.append(x)
            chosen_set.add(x)
    return chosen


def _greedy_add_weighted(precomp, deadline, seed=42):
    """Greedy: add numbers weighted by 'freedom' (how many others they conflict with).
    Numbers with fewer conflicts are added first."""
    rng = random.Random(seed)
    candidates = list(range(2, NMAX + 1))
    
    # Weight: prefer numbers that are "less conflicting" - e.g., primes and large numbers
    # Simple heuristic: weight = number of divisors in [2,256] (fewer = better)
    weights = {}
    for x in range(2, NMAX + 1):
        div_count = sum(1 for d in range(2, NMAX + 1) if x % d == 0 or d % x == 0)
        weights[x] = 1.0 / (1 + div_count)
    
    # Build weighted list
    weighted_candidates = [(weights[x], x) for x in candidates]
    
    # Partial sort: do multiple passes, each time picking the highest-weight available
    chosen = []
    chosen_set = set()
    remaining = set(range(2, NMAX + 1))
    
    while remaining and time.time() < deadline:
        # Pick a few top-weighted candidates and try them
        top = sorted(remaining, key=lambda x: -weights[x])[:10]
        rng.shuffle(top)
        added = False
        for x in top:
            if _valid_addition_fast(x, chosen, precomp):
                chosen.append(x)
                chosen_set.add(x)
                remaining.discard(x)
                added = True
                break
        if not added:
            # Try any remaining
            for x in list(remaining)[:20]:
                if _valid_addition_fast(x, chosen, precomp):
                    chosen.append(x)
                    chosen_set.add(x)
                    remaining.discard(x)
                    added = True
                    break
            if not added:
                break
    
    return chosen


def _local_search(S, precomp, deadline, max_iters=500, seed=99999):
    """Local search: try adding, removing, swapping elements."""
    S = list(S)
    S_set = set(S)
    S_mask = 0
    for v in S:
        S_mask |= (1 << UNIV_IDX[v])
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        
        op = rng.random()
        
        if op < 0.4 and len(S) < NMAX:
            # Try adding
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                rng.shuffle(candidates)
                added = False
                for x in candidates[:40]:
                    if _valid_addition_fast(x, S, precomp):
                        S.append(x)
                        S_set.add(x)
                        S_mask |= (1 << UNIV_IDX[x])
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        added = True
                        break
                if added:
                    continue
        
        elif op < 0.7 and len(S) > 2:
            # Try removing one and adding one (swap)
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            S_temp_mask = S_mask & ~(1 << UNIV_IDX[removed])
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                rng.shuffle(candidates)
                improved = False
                for x in candidates[:50]:
                    if _valid_addition_fast(x, S_temp, precomp):
                        S = S_temp + [x]
                        S_set = set(S)
                        S_mask = S_temp_mask | (1 << UNIV_IDX[x])
                        if len(S) > best_size:
                            best = S[:]
                            best_size = len(S)
                        improved = True
                        break
                if not improved:
                    S = S_temp + [removed]
                    S_set = set(S)
                    S_mask = S_temp_mask | (1 << UNIV_IDX[removed])
        
        else:
            # Try removing a random element to allow re-growth
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                S_mask &= ~(1 << UNIV_IDX[removed])
                # Try to add back something better
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:40]:
                        if _valid_addition_fast(x, S, precomp):
                            S.append(x)
                            S_set.add(x)
                            S_mask |= (1 << UNIV_IDX[x])
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            break
    
    return best


def _simulated_annealing(S, precomp, deadline, max_iters=4000, T_start=100.0, T_end=0.5, seed=55555):
    """Simulated annealing for the set optimization."""
    S = list(S)
    S_set = set(S)
    best = S[:]
    best_size = len(S)
    rng = random.Random(seed)
    
    current_temp = T_start
    cooling = (T_end / T_start) ** (1.0 / max(1, max_iters))
    
    for _it in range(max_iters):
        if time.time() > deadline:
            break
        current_temp *= cooling
        
        op = rng.random()
        
        if op < 0.35 and len(S) < NMAX:
            # Add
            candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition_fast(x, S, precomp):
                    S.append(x)
                    S_set.add(x)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
                    continue
        
        elif op < 0.65 and len(S) > 2:
            # Swap: remove one, add one
            i = rng.randrange(len(S))
            removed = S[i]
            S_temp = S[:i] + S[i+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if candidates:
                x = rng.choice(candidates)
                if _valid_addition_fast(x, S_temp, precomp):
                    S = S_temp + [x]
                    S_set = set(S)
                    if len(S) > best_size:
                        best = S[:]
                        best_size = len(S)
        
        elif op < 0.85 and len(S) > 3:
            # Remove two, try to add two different ones
            i, j = rng.sample(range(len(S)), 2)
            i, j = min(i, j), max(i, j)
            S_temp = S[:i] + S[i+1:j] + S[j+1:]
            S_temp_set = set(S_temp)
            candidates = [x for x in range(2, NMAX + 1) if x not in S_temp_set]
            if len(candidates) >= 2:
                x = rng.choice(candidates)
                if _valid_addition_fast(x, S_temp, precomp):
                    S_temp2 = S_temp + [x]
                    S_temp2_set = S_temp_set | {x}
                    remaining = [y for y in candidates if y != x]
                    if remaining:
                        y = rng.choice(remaining)
                        if _valid_addition_fast(y, S_temp2, precomp):
                            S = S_temp2 + [y]
                            S_set = set(S)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
        
        else:
            # Remove one (to escape local optima)
            if S:
                i = rng.randrange(len(S))
                removed = S[i]
                S.pop(i)
                S_set.discard(removed)
                # Try to add something
                candidates = [x for x in range(2, NMAX + 1) if x not in S_set]
                if candidates:
                    rng.shuffle(candidates)
                    for x in candidates[:30]:
                        if _valid_addition_fast(x, S, precomp):
                            S.append(x)
                            S_set.add(x)
                            if len(S) > best_size:
                                best = S[:]
                                best_size = len(S)
                            break
    
    return best


def _multi_start_local_search(precomp, deadline, seeds=None):
    """Run local search from multiple random starts."""
    if seeds is None:
        seeds = [1, 7, 13, 42, 99, 256, 777, 1234]
    
    best = None
    best_size = 0
    
    for seed in seeds:
        if time.time() > deadline:
            break
        # Random greedy start
        rng = random.Random(seed)
        candidates = list(range(2, NMAX + 1))
        rng.shuffle(candidates)
        S = []
        S_set = set()
        for x in candidates:
            if time.time() > deadline:
                break
            if _valid_addition_fast(x, S, precomp):
                S.append(x)
                S_set.add(x)
        
        # Local search refinement
        S_refined = _local_search(S, precomp, deadline, max_iters=300, seed=seed * 1000)
        if S_refined and len(S_refined) > best_size:
            best = S_refined
            best_size = len(S_refined)
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # 90s kill, 8s safety margin
    
    precomp = _precompute_conflicts()
    
    best = None
    best_size = 0
    
    def consider(c):
        nonlocal best, best_size
        if c and len(c) > best_size:
            best = c
            best_size = len(c)
    
    # 1. Greedy add (largest first) - tends to keep large numbers
    consider(_greedy_add_largest(precomp, deadline))
    
    # 2. Greedy add (smallest first)
    if time.time() < deadline - 10:
        consider(_greedy_add_smallest(precomp, deadline))
    
    # 3. Greedy add (weighted)
    if time.time() < deadline - 10:
        consider(_greedy_add_weighted(precomp, deadline))
    
    # 4. Greedy remove from full set (bitmask-accelerated)
    if time.time() < deadline - 15:
        consider(_greedy_remove(precomp, deadline - 15))
    
    # 5. Multiple random greedy starts with local search
    if time.time() < deadline - 20:
        consider(_multi_start_local_search(precomp, deadline - 20, seeds=[1, 7, 13, 42, 99, 256, 777]))
    
    # 6. Simulated annealing from best so far
    if best and time.time() < deadline - 10:
        for sa_seed in [55555, 12345, 77777, 31337]:
            if time.time() > deadline - 5:
                break
            consider(_simulated_annealing(best, precomp, deadline - 5, max_iters=3000, seed=sa_seed))
    
    # 7. More local search iterations on best
    if best and time.time() < deadline - 5:
        consider(_local_search(best, precomp, deadline - 5, max_iters=1500, seed=424242))
    
    # 8. Final: try to add any remaining valid elements
    if best and time.time() < deadline - 2:
        S = list(best)
        S_set = set(S)
        for x in range(2, NMAX + 1):
            if x not in S_set:
                if _valid_addition_fast(x, S, precomp):
                    S.append(x)
                    S_set.add(x)
        consider(S)
    
    # 9. One more round of local search + final add
    if best and time.time() < deadline - 1:
        S = _local_search(best, precomp, deadline - 0.5, max_iters=500, seed=999999)
        S_set = set(S)
        for x in range(2, NMAX + 1):
            if x not in S_set:
                if _valid_addition_fast(x, S, precomp):
                    S.append(x)
                    S_set.add(x)
        consider(S)
    
    if best is None:
        # Fallback: all primes
        def is_prime(n):
            if n < 2:
                return False
            if n < 4:
                return True
            if n % 2 == 0 or n % 3 == 0:
                return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0:
                    return False
                i += 6
            return True
        best = [x for x in range(2, NMAX + 1) if is_prime(x)]
    
    return [int(x) for x in best]
