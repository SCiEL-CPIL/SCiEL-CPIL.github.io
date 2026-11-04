"""Construct a finite set U of nonnegative integers containing 0 to maximize
the sumset/difference-set exponent lower bound.

Score = 1 + log(|U-U| / |U+U|) / log(2*max(U) + 1)
"""
import time
import math
import random
import numpy as np

# Deadline guard: evaluator kills at 150s, use 100s to be safe
_START_TIME = time.time()
DEADLINE = 100.0  # seconds


def _score(U):
    """Compute the score for a set U (list of nonnegative ints containing 0)."""
    U = sorted(set(U))
    if 0 not in U:
        return -1.0
    n = len(U)
    if n < 2:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 0.0
    # Compute sumset and difference set using set operations (faster for small n)
    U_set = set(U)
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    n_sums = len(sums)
    n_diffs = len(diffs)
    ratio = n_diffs / n_sums
    if ratio <= 0:
        return 0.0
    denom = math.log(2 * max_u + 1)
    if denom <= 0:
        return 0.0
    return 1.0 + math.log(ratio) / denom


def _geometric_set(base, m):
    """Return U = {floor(base^k) for k=0..m} as a sorted list of unique nonneg ints."""
    U = set()
    val = 1  # base^0
    for k in range(m + 1):
        U.add(int(round(val)))
        val *= base
    # Ensure 0 is in the set
    U.add(0)
    # Remove any duplicates and sort
    return sorted(U)


def _linear_recurrence_set(coeffs, init_vals, length):
    """
    Generate a set from a linear recurrence.
    
    coeffs: list of coefficients [c1, c2, ..., ck] for the recurrence
            u[n] = c1*u[n-1] + c2*u[n-2] + ... + ck*u[n-k]
    init_vals: initial values [u[0], u[1], ..., u[k-1]]
    length: number of terms to generate
    
    Returns sorted list of unique nonnegative integers containing 0.
    """
    k = len(coeffs)
    if len(init_vals) < k:
        return [0]
    
    # Generate the recurrence
    seq = list(init_vals)
    for i in range(k, length):
        val = 0
        for j, c in enumerate(coeffs):
            val += c * seq[i - 1 - j]
        seq.append(val)
    
    # Filter to nonnegative integers and ensure 0 is present
    U = set()
    for x in seq:
        if x >= 0:
            U.add(int(x))
    U.add(0)
    
    return sorted(U)


def _fibonacci_set(n_terms):
    """Generate first n_terms Fibonacci numbers starting from F(1)=1, F(2)=1."""
    if n_terms <= 0:
        return [0]
    if n_terms == 1:
        return [0, 1]
    
    fibs = [1, 1]
    for i in range(2, n_terms):
        fibs.append(fibs[-1] + fibs[-2])
    
    U = set(fibs)
    U.add(0)
    return sorted(U)


def _evaluate_linear_recurrence_sweep():
    """
    Sweep over linear recurrence parameters to find good sets.
    
    We try recurrences of form: u[n] = a*u[n-1] + b*u[n-2]
    with various initial conditions and lengths.
    """
    best_score = -1.0
    best_set = None
    
    # Try various recurrence coefficients
    # Characteristic equation: x^2 - a*x - b = 0
    # Roots must have modulus > 1 for growth
    coeff_pairs = []
    
    # Fibonacci-like: x^2 - x - 1 = 0
    for a in range(1, 8):
        for b in range(-3, 4):
            # Check characteristic roots have modulus > 1
            # x = (a ± sqrt(a^2 + 4b)) / 2
            disc = a * a + 4 * b
            if disc < 0:
                # Complex roots, modulus = sqrt(|b|) if b < 0... actually modulus = sqrt(a^2/4 + |b|)
                # For x^2 - ax - b = 0 with complex roots, modulus = sqrt(b) if b > 0
                # Actually for x^2 + px + q = 0, complex roots have modulus sqrt(q)
                # Here x^2 - ax - b = 0, so q = -b
                if b > 0:
                    mod = math.sqrt(b)
                    if mod > 1.0:
                        coeff_pairs.append((a, b))
                continue
            
            sqrt_disc = math.sqrt(disc)
            r1 = (a + sqrt_disc) / 2.0
            r2 = (a - sqrt_disc) / 2.0
            if abs(r1) > 1.0 and abs(r2) > 0.5:  # At least one root grows, other not too small
                coeff_pairs.append((a, b))
    
    # Also try some specific known good recurrences
    coeff_pairs.append((1, 1))   # Fibonacci
    coeff_pairs.append((2, 1))   # x^2 - 2x - 1 = 0
    coeff_pairs.append((3, 1))
    coeff_pairs.append((2, 2))
    coeff_pairs.append((1, 2))
    
    for a, b in coeff_pairs:
        for length in range(4, 15):
            if time.time() - _START_TIME > DEADLINE - 5:
                return best_set, best_score
            
            # Try different initial conditions
            for init in [[1, 1], [1, 2], [1, 3], [2, 1], [1, 0], [2, 2]]:
                U = _linear_recurrence_set([a, b], init, length)
                if len(U) < 2:
                    continue
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    # Try Fibonacci specifically with more terms
    for n in range(4, 16):
        if time.time() - _START_TIME > DEADLINE - 5:
            return best_set, best_score
        U = _fibonacci_set(n)
        if len(U) < 2:
            continue
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U[:]
    
    return best_set, best_score


def _evaluate_geometric_sweep():
    """Sweep over base b in [1.5, 4.0] and m in [2, 12] to find best geometric set."""
    best_score = -1.0
    best_set = None
    
    # Coarse sweep over base
    bases = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0]
    
    for b in bases:
        for m in range(2, 13):
            if time.time() - _START_TIME > DEADLINE - 5:
                return best_set, best_score
            U = _geometric_set(b, m)
            if len(U) < 2:
                continue
            s = _score(U)
            if s > best_score:
                best_score = s
                best_set = U[:]
    
    # Fine sweep around the best base found
    if best_set is not None and best_score > 0:
        # Refine around the best base
        for b in [1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95, 3.1, 3.2, 3.3, 3.4, 3.6, 3.7, 3.8, 3.9]:
            for m in range(2, 13):
                if time.time() - _START_TIME > DEADLINE - 5:
                    return best_set, best_score
                U = _geometric_set(b, m)
                if len(U) < 2:
                    continue
                s = _score(U)
                if s > best_score:
                    best_score = s
                    best_set = U[:]
    
    return best_set, best_score


def _evaluate_powers_of_two():
    """Evaluate powers of 2 sets for various n, return the best."""
    best_score = -1.0
    best_set = None
    for n in range(2, 20):
        U = [0] + [1 << i for i in range(n)]
        s = _score(U)
        if s > best_score:
            best_score = s
            best_set = U
    return best_set, best_score


def _generate_dynamic_candidates(current_set, max_val):
    """Generate candidate elements based on arithmetic neighborhoods and gaps."""
    candidates = set()
    U = sorted(current_set)
    n = len(U)
    
    # Neighborhoods of existing elements
    for elem in U:
        for delta in range(-4, 5):
            c = elem + delta
            if c > 0:
                candidates.add(c)
    
    # Gaps between consecutive elements
    for i in range(n - 1):
        gap = U[i+1] - U[i]
        if gap > 2:
            # Try midpoints and fractions
            candidates.add(U[i] + gap // 2)
            candidates.add(U[i] + gap // 3)
            candidates.add(U[i] + 2 * gap // 3)
    
    # Values beyond max
    for c in range(max_val + 1, max_val + 20):
        candidates.add(c)
    
    # Small values not yet in set
    for c in range(1, min(max_val + 1, 20)):
        if c not in current_set:
            candidates.add(c)
    
    return candidates


def _try_local_search(initial_set, time_limit=30.0):
    """Local search with dynamic candidate generation."""
    best_set = list(initial_set)
    best_score = _score(best_set)
    deadline = time.time() + time_limit
    max_val = max(best_set) if best_set else 1
    max_val = max(max_val, 10)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        candidates = _generate_dynamic_candidates(best_set, max_val)
        
        for c in candidates:
            if c in best_set:
                continue
            new_set = sorted(best_set + [c])
            s = _score(new_set)
            if s > best_score + 1e-9:
                best_set = new_set
                best_score = s
                improved = True
                max_val = max(best_set)
        
        if time.time() >= deadline:
            break
        
        # Try removing an element (only if size > 2)
        if len(best_set) > 2:
            for i, elem in enumerate(best_set):
                if elem == 0:
                    continue
                new_set = best_set[:i] + best_set[i+1:]
                s = _score(new_set)
                if s > best_score + 1e-9:
                    best_set = new_set
                    best_score = s
                    improved = True
                    max_val = max(best_set) if best_set else 1
        
        if time.time() >= deadline:
            break
    
    return best_set, best_score


def _generate_parametric_candidates():
    """Generate structured parametric candidate sets."""
    candidates = []
    
    # Known good sets from inspirations
    candidates.extend([
        [0, 1, 3],
        [0, 1, 4, 9, 11],
        [0, 1, 3, 7, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 12, 17, 25],
        [0, 1, 2, 4, 8],
        [0, 1, 2, 5, 11],
        [0, 1, 3, 7, 14],
        [0, 1, 3, 8, 13],
        [0, 2, 3, 7, 12],
        [0, 1, 4, 11, 13],
        [0, 1, 4, 9, 13],
    ])
    
    # Mixed-radix constructions
    for base in [2, 3, 4, 5]:
        for m in range(3, 8):
            U = [0]
            val = 1
            for k in range(m):
                U.append(val)
                val *= base
            candidates.append(sorted(set(U)))
    
    # Arithmetic progressions with varying step
    for start in [1, 2, 3]:
        for step in [2, 3, 4, 5]:
            for length in range(4, 10):
                U = [0] + [start + i * step for i in range(length)]
                candidates.append(sorted(set(U)))
    
    # Geometric with perturbation
    for base in [1.8, 2.0, 2.2, 2.5, 3.0]:
        for m in range(4, 10):
            U = {0}
            val = 1.0
            for k in range(m):
                U.add(int(round(val)))
                val *= base
            candidates.append(sorted(U))
    
    return candidates


def _try_random_search(deadline):
    """Random local search with parametric candidates and dynamic refinement."""
    best_score = -1
    best_set = [0, 1, 4, 9, 11]
    
    # Evaluate all parametric candidates
    for c in _generate_parametric_candidates():
        s = _score(c)
        if s > best_score:
            best_score = s
            best_set = c[:]
    
    random.seed(42)
    
    while time.time() < deadline - 5:
        # Start from best found or a random structured set
        if random.random() < 0.3:
            current = best_set[:]
        else:
            n = random.randint(4, 9)
            max_val = random.randint(15, 80)
            other_elements = random.sample(range(1, max_val), n - 1)
            current = sorted([0] + other_elements)
        
        current_score = _score(current)
        if current_score < 0:
            continue
        
        # Local search on current set
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            candidates = _generate_dynamic_candidates(current, max(current) if current else 1)
            
            for c in candidates:
                if c in current:
                    continue
                new_set = sorted(current + [c])
                new_score = _score(new_set)
                if new_score > current_score + 1e-9:
                    current = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = current[:]
                    break
            
            if time.time() >= deadline - 5:
                break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(1, len(current)):
                    new_set = [current[j] for j in range(len(current)) if j != i]
                    new_score = _score(new_set)
                    if new_score > current_score + 1e-9:
                        current = new_set
                        current_score = new_score
                        improved = True
                        if current_score > best_score:
                            best_score = current_score
                            best_set = current[:]
                        break
    
    return best_set, best_score


def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    best_set = None
    best_score = -1.0
    
    # Phase 1: Evaluate linear recurrence sets (Fibonacci-like structures)
    if time.time() - _START_TIME < DEADLINE - 10:
        rec_set, rec_score = _evaluate_linear_recurrence_sweep()
        if rec_set is not None and rec_score > best_score:
            best_set = rec_set
            best_score = rec_score
    
    # Phase 2: Evaluate geometric progression sets
    if time.time() - _START_TIME < DEADLINE - 10:
        geo_set, geo_score = _evaluate_geometric_sweep()
        if geo_set is not None and geo_score > best_score:
            best_set = geo_set
            best_score = geo_score
    
    # Phase 3: Evaluate powers of 2
    if time.time() - _START_TIME < DEADLINE - 10:
        pow2_set, pow2_score = _evaluate_powers_of_two()
        if pow2_set is not None and pow2_score > best_score:
            best_set = pow2_set
            best_score = pow2_score
    
    # Phase 4: Local search from the best set found so far
    if best_set is not None and time.time() - _START_TIME < DEADLINE - 20:
        remaining = DEADLINE - (time.time() - _START_TIME) - 20
        improved_set, improved_score = _try_local_search(best_set, time_limit=min(remaining, 25))
        if improved_score > best_score:
            best_set = improved_set
            best_score = improved_score
    
    # Phase 5: Random search with parametric candidates if time permits
    if time.time() - _START_TIME < DEADLINE - 10:
        remaining = DEADLINE - (time.time() - _START_TIME) - 10
        rand_set, rand_score = _try_random_search(time.time() + min(remaining, 45))
        if rand_score > best_score:
            best_set = rand_set
            best_score = rand_score
    
    # Phase 6: Final local polish on the best set
    if best_set is not None and time.time() - _START_TIME < DEADLINE - 5:
        remaining = DEADLINE - (time.time() - _START_TIME) - 5
        if remaining > 5:
            final_set, final_score = _try_local_search(best_set, time_limit=remaining)
            if final_score > best_score:
                best_set = final_set
                best_score = final_score
    
    # Safety: ensure 0 is in the set
    if best_set is None:
        best_set = [0, 1, 4, 9, 11]
    if 0 not in best_set:
        best_set = [0] + sorted(best_set)
    
    return best_set
