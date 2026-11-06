import math
import random
import time
from fractions import Fraction
from collections import deque

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

# Precompute reciprocal costs as Fractions
_COSTS = [Fraction(1, p) for p in _PRIMES]

MAX_CHECK = 100000

def _coverage_length(assign):
    """
    assign: list of residues. 0 means unused. 1..p means residue r (covering r, r+p, ...).
    Return largest L such that every 1..L is covered.
    """
    covered = bytearray(MAX_CHECK + 1)
    for idx, r in enumerate(assign):
        if r == 0:
            continue
        p = _PRIMES[idx]
        # Mark coverage
        # Optimization: only mark up to a reasonable limit? No, we need exact L.
        # But we can stop early if we know we've covered up to MAX_CHECK.
        for n in range(r, MAX_CHECK + 1, p):
            covered[n] = 1
    
    L = 0
    for n in range(1, MAX_CHECK + 1):
        if covered[n]:
            L = n
        else:
            break
    return L

def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)

def _seed_solution():
    """
    A strong starting point.
    Strategy: Use small primes to cover dense regions, larger primes to fill gaps.
    The previous seed used primes >= 13 to cover 1..~900.
    Let's try a slightly better heuristic seed:
    Greedy set cover for the first 2000 numbers, prioritizing small primes.
    """
    assign = [0] * NUM_PRIMES
    current_cost = Fraction(0)
    covered = bytearray(2001) # Check up to 2000 for seed
    
    # Sort primes by size (smaller primes cover more numbers)
    prime_indices = list(range(NUM_PRIMES))
    prime_indices.sort(key=lambda i: _PRIMES[i])
    
    # Greedy: pick the prime/residue that covers the most uncovered numbers in 1..2000
    # We will iterate a few times or just do a single pass?
    # A single pass of "best residue for each prime" is weak.
    # Let's do a simple greedy construction.
    
    # Actually, the seed in the current program is quite specific.
    # Let's stick to a robust heuristic:
    # 1. Use p=2, r=1 (covers odds) or r=2 (covers evens). Odds are better for density? No, 1/2 cost.
    # Let's try to cover 1..1000 efficiently.
    
    # We will use a simple greedy loop:
    # While there is uncovered number in 1..1000:
    #   Find first uncovered n.
    #   Try all unused primes p.
    #   For each p, find residue r such that n is in the class (r = n % p, if 0 then p).
    #   Check if adding (p, r) is valid (cost <= 1).
    #   Pick the one that maximizes coverage of uncovered numbers in 1..1000.
    
    target_L = 1000
    covered = bytearray(target_L + 1)
    
    while True:
        # Find first uncovered
        first_uncov = -1
        for n in range(1, target_L + 1):
            if not covered[n]:
                first_uncov = n
                break
        if first_uncov == -1:
            break
            
        best_prime_idx = -1
        best_residue = -1
        best_gain = 0
        
        for i in range(NUM_PRIMES):
            if assign[i] != 0:
                continue
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
                
            p = _PRIMES[i]
            # Residue must cover first_uncov
            r = first_uncov % p
            if r == 0:
                r = p
            # Note: Our assign uses 1..p. r is in 1..p.
            
            # Count gain
            gain = 0
            for n in range(r, target_L + 1, p):
                if not covered[n]:
                    gain += 1
                    
            if gain > best_gain:
                best_gain = gain
                best_prime_idx = i
                best_residue = r
                
        if best_prime_idx == -1:
            break
            
        assign[best_prime_idx] = best_residue
        current_cost += _COSTS[best_prime_idx]
        p = _PRIMES[best_prime_idx]
        r = best_residue
        for n in range(r, target_L + 1, p):
            covered[n] = 1
            
    return assign

def _local_search(assign, time_limit):
    """
    Hill-climbing local search with efficient coverage updates.
    We maintain a coverage count array to speed up L calculation.
    """
    start = time.time()
    current = assign[:]
    current_L = _coverage_length(current)
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    # Precompute coverage counts for the current state
    # counts[n] = number of classes covering n
    counts = [0] * (MAX_CHECK + 1)
    for idx, r in enumerate(current):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            counts[n] += 1
            
    def update_counts(i, old_r, new_r):
        p = _PRIMES[i]
        if old_r != 0:
            for n in range(old_r, MAX_CHECK + 1, p):
                counts[n] -= 1
        if new_r != 0:
            for n in range(new_r, MAX_CHECK + 1, p):
                counts[n] += 1
                
    def calc_L():
        L = 0
        for n in range(1, MAX_CHECK + 1):
            if counts[n] > 0:
                L = n
            else:
                break
        return L
        
    # Recompute initial L using counts to be safe
    current_L = calc_L()
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        used_primes = [i for i in range(NUM_PRIMES) if current[i] != 0]
        
        # 1. Swap residues
        for i in used_primes:
            if time.time() - start > time_limit:
                return current, current_L
            p = _PRIMES[i]
            old_r = current[i]
            
            # Try a subset of residues? Or all?
            # For small p, all. For large p, maybe random?
            # Let's try all for p < 100, and random for larger?
            # Actually, just try all. It's fast enough with update_counts.
            for r in range(1, p + 1):
                if r == old_r:
                    continue
                current[i] = r
                update_counts(i, old_r, r)
                new_L = calc_L()
                if new_L > current_L:
                    current_L = new_L
                    improved = True
                    break
                else:
                    # Revert
                    current[i] = old_r
                    update_counts(i, r, old_r)
                    
        # 2. Add unused primes
        if not improved:
            unused_primes = [i for i in range(NUM_PRIMES) if current[i] == 0]
            for i in unused_primes:
                if time.time() - start > time_limit:
                    return current, current_L
                if current_cost + _COSTS[i] > Fraction(1, 1):
                    continue
                p = _PRIMES[i]
                best_r = -1
                best_L = current_L
                for r in range(1, p + 1):
                    current[i] = r
                    update_counts(i, 0, r)
                    new_L = calc_L()
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                    else:
                        # Revert to keep counts clean for next r?
                        # Yes, revert
                        current[i] = 0
                        update_counts(i, r, 0)
                        
                if best_r != -1:
                    # Apply the best
                    current[i] = best_r
                    update_counts(i, 0, best_r)
                    current_L = best_L
                    current_cost += _COSTS[i]
                    improved = True
                else:
                    # Ensure it's 0
                    current[i] = 0
                    
        # 3. Remove primes
        if not improved:
            for i in used_primes[:]:
                if time.time() - start > time_limit:
                    return current, current_L
                old_r = current[i]
                current[i] = 0
                update_counts(i, old_r, 0)
                new_L = calc_L()
                if new_L >= current_L:
                    current_L = new_L
                    current_cost -= _COSTS[i]
                    improved = True
                else:
                    # Revert
                    current[i] = old_r
                    update_counts(i, 0, old_r)
    
    return current, current_L

def _simulated_annealing(assign, time_limit):
    """
    Simulated annealing with efficient coverage updates.
    """
    start = time.time()
    current = assign[:]
    current_L = _coverage_length(current)
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    # Precompute coverage counts
    counts = [0] * (MAX_CHECK + 1)
    for idx, r in enumerate(current):
        if r == 0:
            continue
        p = _PRIMES[idx]
        for n in range(r, MAX_CHECK + 1, p):
            counts[n] += 1
            
    def update_counts(i, old_r, new_r):
        p = _PRIMES[i]
        if old_r != 0:
            for n in range(old_r, MAX_CHECK + 1, p):
                counts[n] -= 1
        if new_r != 0:
            for n in range(new_r, MAX_CHECK + 1, p):
                counts[n] += 1
                
    def calc_L():
        L = 0
        for n in range(1, MAX_CHECK + 1):
            if counts[n] > 0:
                L = n
            else:
                break
        return L
        
    current_L = calc_L()
    best = current[:]
    best_L = current_L
    
    T = 1.0
    cooling = 0.995
    
    while time.time() - start < time_limit:
        if T < 0.01:
            break
            
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap residue
            used = [i for i in range(NUM_PRIMES) if current[i] != 0]
            if not used:
                continue
            i = random.choice(used)
            p = _PRIMES[i]
            old_r = current[i]
            r = random.randint(1, p)
            if r == old_r:
                continue
            current[i] = r
            update_counts(i, old_r, r)
            new_L = calc_L()
            delta_L = new_L - current_L
            current_L = new_L
        elif move_type < 0.8:
            # Add prime
            unused = [i for i in range(NUM_PRIMES) if current[i] == 0]
            if not unused:
                continue
            i = random.choice(unused)
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            p = _PRIMES[i]
            r = random.randint(1, p)
            current[i] = r
            update_counts(i, 0, r)
            new_L = calc_L()
            delta_L = new_L - current_L
            current_L = new_L
            current_cost += _COSTS[i]
        else:
            # Remove prime
            used = [i for i in range(NUM_PRIMES) if current[i] != 0]
            if not used:
                continue
            i = random.choice(used)
            old_r = current[i]
            current[i] = 0
            update_counts(i, old_r, 0)
            new_L = calc_L()
            delta_L = new_L - current_L
            current_L = new_L
            current_cost -= _COSTS[i]
            
        if delta_L >= 0:
            pass
        else:
            if random.random() < math.exp(delta_L / T):
                pass
            else:
                # Revert
                if move_type < 0.5:
                    current[i] = old_r
                    update_counts(i, r, old_r)
                elif move_type < 0.8:
                    current[i] = 0
                    update_counts(i, r, 0)
                    current_cost -= _COSTS[i]
                else:
                    current[i] = old_r
                    update_counts(i, 0, old_r)
                    current_cost += _COSTS[i]
                current_L = calc_L()
                
        if current_L > best_L:
            best = current[:]
            best_L = current_L
            
        T *= cooling
        
    return best, best_L

def construct():
    start = time.time()
    deadline = start + 145  # Safety margin
    
    # 1. Start with seed
    best_assign = _seed_solution()
    best_L = _coverage_length(best_assign)
    
    if not _is_valid(best_assign):
        best_assign = [0] * NUM_PRIMES
        best_L = 0
        
    # 2. Local Search
    if time.time() < deadline - 40:
        ls_assign, ls_L = _local_search(best_assign, min(30, deadline - time.time() - 10))
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
            
    # 3. Simulated Annealing
    if time.time() < deadline - 10:
        sa_assign, sa_L = _simulated_annealing(best_assign, min(30, deadline - time.time() - 5))
        if sa_L > best_L and _is_valid(sa_assign):
            best_assign = sa_assign
            best_L = sa_L
            
    # Final validation
    if _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # Fallback to seed
        return [[_PRIMES[i], r] for i, r in enumerate(_seed_solution()) if r != 0]

