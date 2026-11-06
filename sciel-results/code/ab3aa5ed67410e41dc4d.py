import math
import random
import time
from fractions import Fraction

# ---- primes up to 1000 (fixed) ----
def _primes_upto(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = [False] * len(sieve[start:n + 1:step])
    return [i for i, v in enumerate(sieve) if v]

_PRIMES = _primes_upto(1000)
NUM_PRIMES = len(_PRIMES)
_COSTS = [Fraction(1, p) for p in _PRIMES]
MAX_CHECK = 100000

def _seed_solution():
    """The i-th prime from 13 upward covers the integer i."""
    primes_from_13 = [p for p in _PRIMES if p >= 13]
    assign = [0] * NUM_PRIMES
    for i, p in enumerate(primes_from_13):
        idx = _PRIMES.index(p)
        r = (i + 1) % p
        if r == 0:
            r = 1
        assign[idx] = r
    return assign

def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)

def _coverage_length(assign):
    """Full recompute of coverage length (for final validation)."""
    covered = bytearray(MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _build_cover_state(assign):
    """Build cover_count array from assign. Returns (cover_count, L)."""
    cover_count = [0] * (MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            cover_count[n] += 1
    # Find L: largest n such that all 1..n have cover_count > 0
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if cover_count[n] > 0:
            L = n
        else:
            break
    return cover_count, L

def _add_class(cover_count, idx, r):
    """Add prime idx with residue r. Returns (num_newly_covered, max_newly_covered)."""
    p = _PRIMES[idx]
    newly_covered = 0
    max_new = 0
    for n in range(r, MAX_CHECK + 1, p):
        if cover_count[n] == 0:
            newly_covered += 1
            if n > max_new:
                max_new = n
        cover_count[n] += 1
    return newly_covered, max_new

def _remove_class(cover_count, idx, r):
    """Remove prime idx with residue r. Returns (num_newly_uncovered, min_newly_uncovered)."""
    p = _PRIMES[idx]
    newly_uncovered = 0
    min_unc = MAX_CHECK + 1
    for n in range(r, MAX_CHECK + 1, p):
        cover_count[n] -= 1
        if cover_count[n] == 0:
            newly_uncovered += 1
            if n < min_unc:
                min_unc = n
    return newly_uncovered, min_unc

def _recompute_L(cover_count):
    """Recompute coverage length from cover_count."""
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if cover_count[n] > 0:
            L = n
        else:
            break
    return L

def _simulated_annealing_incremental(assign, time_limit):
    """
    Simulated annealing with incremental coverage updates.
    Each move is O(p) instead of O(MAX_CHECK), enabling far more iterations.
    """
    start = time.time()
    current = assign[:]
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    cover_count, current_L = _build_cover_state(current)
    
    best = current[:]
    best_L = current_L
    best_cost = current_cost
    
    T = 50.0
    cooling = 0.9999  # Very slow cooling for more exploration
    
    # Track which primes are used for quick access
    used_set = set(i for i in range(NUM_PRIMES) if current[i] != 0)
    
    iteration = 0
    while time.time() - start < time_limit:
        iteration += 1
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap residue of a used prime
            if not used_set:
                continue
            i = random.choice(list(used_set))
            p = _PRIMES[i]
            old_r = current[i]
            r = random.randint(1, p)
            if r == old_r:
                continue
            
            # Remove old, add new
            _, min_unc_old = _remove_class(cover_count, i, old_r)
            newly_cov, max_new = _add_class(cover_count, i, r)
            current[i] = r
            
            new_L = _recompute_L(cover_count)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best = current[:]
                    best_L = current_L
                    best_cost = current_cost
            else:
                # Revert
                _remove_class(cover_count, i, r)
                _add_class(cover_count, i, old_r)
                current[i] = old_r
                
        elif move_type < 0.8:
            # Add a prime
            unused = [i for i in range(NUM_PRIMES) if i not in used_set]
            if not unused:
                continue
            i = random.choice(unused)
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            p = _PRIMES[i]
            r = random.randint(1, p)
            
            newly_cov, max_new = _add_class(cover_count, i, r)
            current[i] = r
            used_set.add(i)
            old_cost = current_cost
            current_cost += _COSTS[i]
            
            new_L = _recompute_L(cover_count)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best = current[:]
                    best_L = current_L
                    best_cost = current_cost
            else:
                # Revert
                _remove_class(cover_count, i, r)
                current[i] = 0
                used_set.discard(i)
                current_cost = old_cost
                
        else:
            # Remove a prime
            if not used_set:
                continue
            i = random.choice(list(used_set))
            r = current[i]
            
            _, min_unc = _remove_class(cover_count, i, r)
            current[i] = 0
            used_set.discard(i)
            old_cost = current_cost
            current_cost -= _COSTS[i]
            
            new_L = _recompute_L(cover_count)
            delta_L = new_L - current_L
            
            if delta_L >= 0 or random.random() < math.exp(delta_L / T):
                current_L = new_L
                if current_L > best_L:
                    best = current[:]
                    best_L = current_L
                    best_cost = current_cost
            else:
                # Revert
                _add_class(cover_count, i, r)
                current[i] = r
                used_set.add(i)
                current_cost = old_cost
        
        T *= cooling
        if T < 0.1:
            break
    
    return best, best_L

def _local_search_incremental(assign, time_limit):
    """
    Hill-climbing local search with incremental updates.
    Tries all single-residue changes and accepts if L improves.
    """
    start = time.time()
    current = assign[:]
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    cover_count, current_L = _build_cover_state(current)
    best = current[:]
    best_L = current_L
    
    used_set = set(i for i in range(NUM_PRIMES) if current[i] != 0)
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try swapping residues for used primes
        for i in list(used_set):
            if time.time() - start > time_limit:
                break
            p = _PRIMES[i]
            old_r = current[i]
            
            for r in range(1, p + 1):
                if r == old_r:
                    continue
                
                _remove_class(cover_count, i, old_r)
                _add_class(cover_count, i, r)
                new_L = _recompute_L(cover_count)
                
                if new_L > current_L:
                    current[i] = r
                    current_L = new_L
                    improved = True
                    if current_L > best_L:
                        best = current[:]
                        best_L = current_L
                    break
                else:
                    # Revert
                    _remove_class(cover_count, i, r)
                    _add_class(cover_count, i, old_r)
            
            if time.time() - start > time_limit:
                break
        
        # Try adding unused primes
        if not improved:
            unused = [i for i in range(NUM_PRIMES) if i not in used_set]
            for i in unused:
                if time.time() - start > time_limit:
                    break
                if current_cost + _COSTS[i] > Fraction(1, 1):
                    continue
                
                p = _PRIMES[i]
                best_r = -1
                best_new_L = current_L
                
                for r in range(1, p + 1):
                    _add_class(cover_count, i, r)
                    new_L = _recompute_L(cover_count)
                    _remove_class(cover_count, i, r)
                    
                    if new_L > best_new_L:
                        best_new_L = new_L
                        best_r = r
                
                if best_r != -1:
                    _add_class(cover_count, i, best_r)
                    current[i] = best_r
                    used_set.add(i)
                    current_cost += _COSTS[i]
                    current_L = best_new_L
                    improved = True
                    if current_L > best_L:
                        best = current[:]
                        best_L = current_L
        
        # Try removing primes that don't hurt
        if not improved:
            for i in list(used_set):
                if time.time() - start > time_limit:
                    break
                r = current[i]
                
                _remove_class(cover_count, i, r)
                new_L = _recompute_L(cover_count)
                
                if new_L >= current_L:
                    current[i] = 0
                    used_set.discard(i)
                    current_cost -= _COSTS[i]
                    current_L = new_L
                    improved = True
                else:
                    # Revert
                    _add_class(cover_count, i, r)
    
    return best, best_L

def construct():
    start = time.time()
    deadline = start + 135  # Safety margin before 150s kill
    
    # Start with seed solution
    best_assign = _seed_solution()
    best_L = _coverage_length(best_assign)
    
    if not _is_valid(best_assign):
        best_assign = [0] * NUM_PRIMES
        best_L = 0
    
    # Phase 1: Local search refinement
    if time.time() < deadline - 60:
        ls_assign, ls_L = _local_search_incremental(best_assign, min(50, deadline - time.time() - 10))
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
    
    # Phase 2: Simulated annealing with incremental updates
    if time.time() < deadline - 10:
        sa_assign, sa_L = _simulated_annealing_incremental(best_assign, min(70, deadline - time.time() - 5))
        if sa_L > best_L and _is_valid(sa_assign):
            best_assign = sa_assign
            best_L = sa_L
    
    # Phase 3: Another local search pass on the SA result
    if time.time() < deadline - 5:
        ls2_assign, ls2_L = _local_search_incremental(best_assign, min(15, deadline - time.time() - 2))
        if ls2_L > best_L and _is_valid(ls2_assign):
            best_assign = ls2_assign
            best_L = ls2_L
    
    # Final validation
    if _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # Fallback to seed
        return _seed_solution()
