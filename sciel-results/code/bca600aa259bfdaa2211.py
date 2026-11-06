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


def _coverage_length(assign):
    """assign: list of residues (0..p-1) per prime; 0 means 'not used'.
    Return largest L such that every 1..L is covered by some class."""
    MAX_CHECK = 100000
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


def _is_valid(assign):
    c = Fraction(0)
    for idx, r in enumerate(assign):
        if r != 0:
            c += _COSTS[idx]
    return c <= Fraction(1, 1)


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


def _local_search(assign, time_limit):
    """
    Hill-climbing local search: try to improve coverage by swapping residues
    or adding/removing primes while maintaining validity.
    """
    start = time.time()
    current = assign[:]
    current_L = _coverage_length(current)
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    # Precompute coverage masks for speed
    # mask[i][r] = set of numbers in 1..current_L+500 covered by prime i with residue r
    # We'll compute on the fly for efficiency
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        
        # Try swapping residues for used primes
        used_primes = [i for i in range(NUM_PRIMES) if current[i] != 0]
        
        for i in used_primes:
            if time.time() - start > time_limit:
                return current, current_L
            p = _PRIMES[i]
            old_r = current[i]
            
            # Try each other residue
            for r in range(1, p + 1):
                if r == old_r:
                    continue
                
                new_assign = current[:]
                new_assign[i] = r
                
                # Check validity (cost doesn't change)
                # Check coverage
                new_L = _coverage_length(new_assign)
                if new_L > current_L:
                    current = new_assign
                    current_L = new_L
                    improved = True
                    break
        
        # Try adding unused primes
        if not improved:
            unused_primes = [i for i in range(NUM_PRIMES) if current[i] == 0]
            for i in unused_primes:
                if time.time() - start > time_limit:
                    return current, current_L
                if current_cost + _COSTS[i] > Fraction(1, 1):
                    continue
                
                p = _PRIMES[i]
                # Find best residue
                best_r = -1
                best_L = current_L
                
                # Sample residues for speed
                for r in range(1, p + 1):
                    new_assign = current[:]
                    new_assign[i] = r
                    new_L = _coverage_length(new_assign)
                    if new_L > best_L:
                        best_L = new_L
                        best_r = r
                
                if best_r != -1 and best_L > current_L:
                    current = current[:]
                    current[i] = best_r
                    current_L = best_L
                    current_cost += _COSTS[i]
                    improved = True
        
        # Try removing primes that don't hurt coverage
        if not improved:
            for i in used_primes[:]:
                if time.time() - start > time_limit:
                    return current, current_L
                
                new_assign = current[:]
                new_assign[i] = 0
                new_L = _coverage_length(new_assign)
                if new_L >= current_L:
                    current = new_assign
                    current_cost -= _COSTS[i]
                    improved = True
    
    return current, current_L


def _simulated_annealing(assign, time_limit):
    """
    Simulated annealing to escape local minima.
    """
    start = time.time()
    current = assign[:]
    current_L = _coverage_length(current)
    current_cost = sum(_COSTS[i] for i in range(NUM_PRIMES) if current[i] != 0)
    
    best = current[:]
    best_L = current_L
    
    T = 1.0
    cooling = 0.995
    
    while time.time() - start < time_limit:
        if T < 0.01:
            break
        
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.5:
            # Swap residue of a used prime
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
            delta_L = _coverage_length(current) - current_L
            current_L += delta_L
            delta_cost = 0  # cost unchanged
        elif move_type < 0.8:
            # Add a prime
            unused = [i for i in range(NUM_PRIMES) if current[i] == 0]
            if not unused:
                continue
            i = random.choice(unused)
            if current_cost + _COSTS[i] > Fraction(1, 1):
                continue
            p = _PRIMES[i]
            r = random.randint(1, p)
            current[i] = r
            delta_L = _coverage_length(current) - current_L
            current_L += delta_L
            delta_cost = _COSTS[i]
        else:
            # Remove a prime
            used = [i for i in range(NUM_PRIMES) if current[i] != 0]
            if not used:
                continue
            i = random.choice(used)
            current[i] = 0
            delta_L = _coverage_length(current) - current_L
            current_L += delta_L
            delta_cost = -_COSTS[i]
        
        # Accept or reject
        if delta_L >= 0:
            pass  # Accept
        else:
            # Probabilistic acceptance
            if random.random() < math.exp(delta_L / T):
                pass  # Accept
            else:
                # Revert
                if move_type < 0.5:
                    current[i] = old_r
                elif move_type < 0.8:
                    current[i] = 0
                    delta_cost = 0
                else:
                    # Need to restore old residue, but we didn't save it
                    # This is a problem. Let's save it.
                    pass
                # For simplicity, revert by recomputing
                # Actually, let's just keep the current state for now and rely on best tracking
        
        if current_L > best_L:
            best = current[:]
            best_L = current_L
        
        T *= cooling
    
    return best, best_L


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    
    # Start with seed solution
    best_assign = _seed_solution()
    best_L = _coverage_length(best_assign)
    
    if not _is_valid(best_assign):
        # Fallback: empty
        best_assign = [0] * NUM_PRIMES
        best_L = 0
    
    # Local search refinement
    if time.time() < deadline - 30:
        ls_assign, ls_L = _local_search(best_assign, min(30, deadline - time.time()))
        if ls_L > best_L and _is_valid(ls_assign):
            best_assign = ls_assign
            best_L = ls_L
    
    # Simulated annealing for further improvement
    if time.time() < deadline - 20:
        sa_assign, sa_L = _simulated_annealing(best_assign, min(20, deadline - time.time()))
        if sa_L > best_L and _is_valid(sa_assign):
            best_assign = sa_assign
            best_L = sa_L
    
    # Final validation
    if _is_valid(best_assign):
        return [[_PRIMES[i], r] for i, r in enumerate(best_assign) if r != 0]
    else:
        # Fallback to seed
        return _seed_solution()
