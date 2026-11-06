"""Find the maximum-size dissociated subset of [1..22].

A subset S is dissociated (reciprocal) iff every subset of S has a distinct
sum of reciprocals: the map A -> sum_{a in A} 1/a is injective over all 2^|S|
sub-sums (including the empty sum = 0).

Strategy:
  - Exact integer arithmetic via LCM(1..22).
  - Fast randomized greedy construction with local search (add/swap/2-opt).
  - Global deadline guard.
  - Optimized collision checking using set intersection.
"""
import time
import random
from math import gcd
from functools import reduce

_T0 = time.time()
_BUDGET = 55.0  # Wall-clock guard (evaluator kills at 60s; keep 5s margin)
_N = 22

# LCM(1..22) for exact integer representation of reciprocal sums
def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM = reduce(_lcm, range(1, _N + 1))
# Reciprocal of i as integer: LCM // i
_REC = [0] + [_LCM // i for i in range(1, _N + 1)]



# Large prime for modular pre-check (must be > max possible subset sum to avoid false negatives)
# Max subset sum of reciprocals * LCM < 2^32, so a prime > 2^32 works
_MOD = (1 << 34) + 11  # 17179869193, prime? Let's use a known prime
# Actually let's use a verified large prime
_MOD = 18446744073709551557  # 2^64 - 2^32 + 1, a known prime (GMP prime)
# Modular reciprocals: w_i = MOD * inv(i, MOD) % MOD
def _modinv(a, m):
    """Compute modular inverse of a mod m using extended Euclidean."""
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        return None
    return x % m

def _extended_gcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = _extended_gcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

# Precompute modular weights
_MODW = [0] * (_N + 1)
for _i in range(1, _N + 1):
    _MODW[_i] = _modinv(_i, _MOD)


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums distinct."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums = sums | shifted
    return True


def _greedy_construct(order):
    """Greedy construction: pick elements in 'order' if they fit."""
    current_set = []
    sums = {0}
    for idx in order:
        v = _REC[idx]
        # Check if adding v causes collision: (sums + v) must be disjoint from sums
        if not (sums & {s + v for s in sums}):
            current_set.append(idx)
            sums |= {s + v for s in sums}
    return current_set, sums


def _try_add(current_set, sums, candidate):
    """Try to add candidate to current_set. Returns (new_set, new_sums) or (None, None)."""
    v = _REC[candidate]
    if sums & {s + v for s in sums}:
        return None, None
    new_sums = sums | {s + v for s in sums}
    new_set = current_set + [candidate]
    return new_set, new_sums


def _compute_sums(element_set):
    """Compute all subset sums for a given element set. Returns None if not dissociated."""
    sums = {0}
    for e in element_set:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return None  # Not dissociated
        sums |= shifted
    return sums


def _simulated_annealing(current_set, sums, deadline):
    """Simulated annealing to find larger dissociated sets."""
    best_set = list(current_set)
    best_sums = sums
    best_size = len(best_set)
    
    unused = set(range(1, _N + 1)) - set(current_set)
    
    T = 5.0  # Initial temperature
    cooling = 0.97
    min_T = 0.01
    max_iter = 5000
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if T < min_T:
            break
        
        # Choose a random move
        r = random.random()
        if r < 0.25 and unused:
            # Add a random unused element
            cand = random.choice(list(unused))
            new_set = current_set + [cand]
            new_sums = _compute_sums(new_set)
            if new_sums is not None:
                if len(new_set) > best_size:
                    best_set = new_set
                    best_sums = new_sums
                    best_size = len(new_set)
                current_set = new_set
                sums = new_sums
                unused.discard(cand)
            # If invalid, reject
        elif r < 0.55 and unused and current_set:
            # Swap: remove random from current, add random from unused
            rem = random.choice(current_set)
            add = random.choice(list(unused))
            new_set = [x for x in current_set if x != rem] + [add]
            new_sums = _compute_sums(new_set)
            if new_sums is not None:
                # Always accept valid swaps (same size, explores neighborhood)
                current_set = new_set
                sums = new_sums
                unused.discard(add)
                unused.add(rem)
                if len(new_set) > best_size:
                    best_set = new_set
                    best_sums = new_sums
                    best_size = len(new_set)
            # If invalid, reject
        elif r < 0.80 and len(current_set) >= 2 and len(unused) >= 2:
            # 2-opt: remove two, add two
            rem1, rem2 = random.sample(current_set, 2)
            add1, add2 = random.sample(list(unused), 2)
            new_set = [x for x in current_set if x not in (rem1, rem2)] + [add1, add2]
            new_sums = _compute_sums(new_set)
            if new_sums is not None:
                current_set = new_set
                sums = new_sums
                unused.discard(add1)
                unused.discard(add2)
                unused.add(rem1)
                unused.add(rem2)
                if len(new_set) > best_size:
                    best_set = new_set
                    best_sums = new_sums
                    best_size = len(new_set)
            # If invalid, reject
        elif r < 0.95 and len(current_set) >= 3 and len(unused) >= 3:
            # 3-opt: remove three, add three
            rem1, rem2, rem3 = random.sample(current_set, 3)
            add1, add2, add3 = random.sample(list(unused), 3)
            new_set = [x for x in current_set if x not in (rem1, rem2, rem3)] + [add1, add2, add3]
            new_sums = _compute_sums(new_set)
            if new_sums is not None:
                current_set = new_set
                sums = new_sums
                for a in (add1, add2, add3):
                    unused.discard(a)
                for r2 in (rem1, rem2, rem3):
                    unused.add(r2)
                if len(new_set) > best_size:
                    best_set = new_set
                    best_sums = new_sums
                    best_size = len(new_set)
            # If invalid, reject
        else:
            # Try to add any available element (greedy completion)
            for cand in list(unused):
                new_set, new_sums = _try_add(current_set, sums, cand)
                if new_set is not None:
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_sums = new_sums
                        best_size = len(new_set)
                    current_set = new_set
                    sums = new_sums
                    unused.discard(cand)
                    break
        
        T *= cooling
    
    # Final phase: try to add any remaining elements to best
    unused_final = set(range(1, _N + 1)) - set(best_set)
    for cand in list(unused_final):
        if time.time() > deadline:
            break
        new_set, new_sums = _try_add(best_set, best_sums, cand)
        if new_set is not None:
            best_set = new_set
            best_sums = new_sums
            best_size = len(best_set)
    
    return best_set, best_sums, best_size > len(current_set)


def construct():
    """Return a nonempty subset of [1..22] maximising dissociated cardinality."""
    deadline = _T0 + _BUDGET
    
    # Initial baseline: Greedy descending
    best_set, best_sums = _greedy_construct(list(range(_N, 0, -1)))
    best_size = len(best_set)
    
    # Ensure we have a valid nonempty set
    if not best_set:
        best_set = [1]
        best_sums = {0, _REC[1]}
        best_size = 1

    # Randomized restarts with simulated annealing
    while time.time() < deadline - 2.0:
        # Generate a random order
        order = list(range(1, _N + 1))
        random.shuffle(order)
        
        # Greedy construct
        curr_set, curr_sums = _greedy_construct(order)
        
        # If this greedy result is better, update best
        if len(curr_set) > best_size:
            best_set = curr_set
            best_sums = curr_sums
            best_size = len(best_set)

        # Try to improve the current best with simulated annealing
        if time.time() < deadline - 5.0:
            new_set, new_sums, improved = _simulated_annealing(best_set, best_sums, deadline)
            if improved:
                best_set = new_set
                best_sums = new_sums
                best_size = len(best_set)
        
        # Also try SA on the current greedy result if it's close to best
        if len(curr_set) >= best_size - 1 and time.time() < deadline - 5.0:
            new_set2, new_sums2, improved2 = _simulated_annealing(curr_set, curr_sums, deadline)
            if improved2 and len(new_set2) > best_size:
                best_set = new_set2
                best_sums = new_sums2
                best_size = len(best_set)

    # Final phase: try to add any remaining elements to best
    unused = set(range(1, _N + 1)) - set(best_set)
    for cand in list(unused):
        if time.time() > deadline - 1.0:
            break
        new_set, new_sums = _try_add(best_set, best_sums, cand)
        if new_set is not None:
            best_set = new_set
            best_sums = new_sums
            best_size = len(best_set)

    # Final verification
    if not _is_dissociated(best_set):
        return [1]
        
    return best_set
