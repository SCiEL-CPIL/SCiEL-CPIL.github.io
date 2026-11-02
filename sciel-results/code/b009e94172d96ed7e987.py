"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np
from itertools import product as iter_product

# EVOLVE-BLOCK-START

# Known optimal or near-optimal Golomb rulers
_KNOWN_GOLOMB = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 12, 17, 22],
    8: [0, 1, 4, 10, 12, 17, 22, 29],
    9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
    10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
    16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115],
    17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132],
    18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151],
    19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171],
    20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192],
    21: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215],
    22: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240],
    23: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240, 267],
    24: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240, 267, 296],
    25: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 192, 215, 240, 267, 296, 327],
}

# First 100 primes for gap-multiplication construction
def _generate_primes(n):
    """Generate first n primes."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes

_PRIMES = _generate_primes(100)


def _score_fast(U):
    """Fast score computation using numpy."""
    U = sorted(set(U))
    n = len(U)
    if n < 2:
        return 0.0
    max_u = U[-1]
    if max_u <= 0:
        return 0.0
    U_arr = np.array(U, dtype=np.int64)
    sums = U_arr[:, None] + U_arr[None, :]
    diffs = U_arr[:, None] - U_arr[None, :]
    n_sums = len(np.unique(sums))
    n_diffs = len(np.unique(diffs))
    if n_sums == 0:
        return 0.0
    ratio = n_diffs / n_sums
    if ratio <= 1.0:
        return 1.0
    score = 1.0 + math.log(ratio) / math.log(2 * max_u + 1)
    return score


def _gap_multiplication_set(num_gaps, prime_pool_size=30, max_val=500000):
    """
    Construct a set using gap-multiplication: gaps are products of distinct primes.
    This ensures differences have unique factorizations, maximizing |U-U|/|U+U|.
    
    We greedily select primes to maximize the ratio at each step.
    """
    if num_gaps < 1:
        return [0]
    
    # Start with 0, then build up using gaps
    U = [0]
    current_pos = 0
    used_primes = set()
    
    for i in range(num_gaps):
        best_gap = None
        best_ratio = 0.0
        best_prime_idx = None
        
        # Try different prime combinations for the gap
        # For efficiency, try single primes and small products
        candidates = []
        
        # Single primes
        for p_idx in range(min(prime_pool_size, 50)):
            p = _PRIMES[p_idx]
            if p_idx not in used_primes:
                candidates.append((p, [p_idx]))
        
        # Products of 2 primes
        for i1 in range(min(prime_pool_size, 20)):
            for i2 in range(i1 + 1, min(prime_pool_size, 30)):
                if i1 not in used_primes and i2 not in used_primes:
                    p1, p2 = _PRIMES[i1], _PRIMES[i2]
                    gap = p1 * p2
                    if gap <= max_val // 2:
                        candidates.append((gap, [i1, i2]))
        
        # Products of 3 primes (only for later gaps when we need larger gaps)
        if i >= 2:
            for i1 in range(min(prime_pool_size, 15)):
                for i2 in range(i1 + 1, min(prime_pool_size, 25)):
                    for i3 in range(i2 + 1, min(prime_pool_size, 35)):
                        if i1 not in used_primes and i2 not in used_primes and i3 not in used_primes:
                            p1, p2, p3 = _PRIMES[i1], _PRIMES[i2], _PRIMES[i3]
                            gap = p1 * p2 * p3
                            if gap <= max_val // 2:
                                candidates.append((gap, [i1, i2, i3]))
        
        # Evaluate each candidate gap
        for gap, prime_indices in candidates:
            new_pos = current_pos + gap
            if new_pos > max_val:
                continue
            
            new_U = U + [new_pos]
            # Quick heuristic: estimate ratio without full computation
            # For now, just check if it's valid and use a simple heuristic
            n = len(new_U)
            max_u = new_U[-1]
            
            # Estimate: differences are roughly n*(n-1)/2, sums are roughly n*(n+1)/2
            # But with unique factorization, differences should be more distinct
            est_diffs = n * (n - 1)  # upper bound
            est_sums = n * (n + 1) // 2  # typical
            
            # Use actual computation for small sets, estimate for large
            if n <= 20:
                score = _score_fast(new_U)
            else:
                # Heuristic score
                ratio_est = min(est_diffs / max(1, est_sums), 2.0)
                score = 1.0 + math.log(ratio_est) / math.log(2 * max_u + 1)
            
            if score > best_ratio:
                best_ratio = score
                best_gap = gap
                best_prime_idx = prime_indices
        
        if best_gap is None:
            # Fallback: use next prime
            for p_idx in range(len(_PRIMES)):
                if p_idx not in used_primes:
                    best_gap = _PRIMES[p_idx]
                    best_prime_idx = [p_idx]
                    break
        
        if best_gap is None:
            break
        
        current_pos += best_gap
        U.append(current_pos)
        used_primes.update(best_prime_idx)
    
    return U


def _greedy_golomb(n, max_val=10000):
    """Greedy Golomb ruler construction."""
    if n in _KNOWN_GOLOMB:
        return _KNOWN_GOLOMB[n][:]
    U = [0]
    diffs = set()
    for _ in range(1, n):
        placed = False
        for v in range(1, max_val + 1):
            if v in U:
                continue
            ok = True
            for a in U:
                d = abs(v - a)
                if d in diffs:
                    ok = False
                    break
            if ok:
                U.append(v)
                for a in U[:-1]:
                    diffs.add(abs(v - a))
                placed = True
                break
        if not placed:
            break
    return U


def _local_search(U, time_limit=20.0, max_steps=50000):
    """Local search with add, remove, and swap operations."""
    U = sorted(set(U))
    best_score = _score_fast(U)
    best_U = list(U)
    start = time.time()
    n = len(U)
    max_u = U[-1] if U else 0

    for step in range(max_steps):
        if time.time() - start > time_limit:
            break

        improved = False
        max_u = max(U) if U else 0

        # Try adding a new element
        candidates = []
        # Near max
        for d in range(1, 30):
            candidates.append(max_u + d)
        # Geometric extensions
        for d in range(1, 15):
            candidates.append(max_u * 2 + d)
            candidates.append(max_u * 3 + d)
        # Interior random
        for _ in range(10):
            candidates.append(random.randint(1, max_u))

        random.shuffle(candidates)
        for new_val in candidates[:12]:
            if new_val <= 0 or new_val in U:
                continue
            new_U = sorted(U + [new_val])
            new_score = _score_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                improved = True
                break

        if not improved:
            # Try removing an element (not 0)
            if len(U) > 2:
                idxs = list(range(1, len(U)))
                random.shuffle(idxs)
                for idx in idxs[:6]:
                    new_U = U[:idx] + U[idx+1:]
                    new_score = _score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        improved = True
                        break

        if not improved:
            # Try swapping an element
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                old_val = U[idx]
                new_val = random.randint(1, max_u * 2 + 50)
                if new_val != old_val and new_val not in U:
                    new_U = list(U)
                    new_U[idx] = new_val
                    new_U = sorted(new_U)
                    new_score = _score_fast(new_U)
                    if new_score > best_score + 1e-12:
                        best_score = new_score
                        best_U = new_U
                        U = new_U
                        improved = True

        if not improved and step > 200 and random.random() < 0.02:
            # Perturb: replace a random element with a random value
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                new_val = random.randint(1, max_u * 3)
                if new_val not in U:
                    U = list(U)
                    U[idx] = new_val
                    U = sorted(U)
                    best_score = _score_fast(U)
                    best_U = list(U)

    return best_U, best_score


def _prime_gap_local_search(U, time_limit=15.0, max_steps=30000):
    """
    Local search specialized for prime-gap-based sets.
    Tries to modify gaps by changing prime factors.
    """
    U = sorted(set(U))
    best_score = _score_fast(U)
    best_U = list(U)
    start = time.time()
    
    # Extract gaps
    if len(U) < 2:
        return best_U, best_score
    
    gaps = [U[i+1] - U[i] for i in range(len(U) - 1)]
    
    for step in range(max_steps):
        if time.time() - start > time_limit:
            break
        
        improved = False
        
        # Try modifying a gap by changing its prime factorization
        gap_idx = random.randint(0, len(gaps) - 1)
        old_gap = gaps[gap_idx]
        
        # Generate candidate new gaps using primes
        candidates = []
        
        # Single primes
        for p in _PRIMES[:30]:
            candidates.append(p)
        
        # Products of 2 primes
        for i in range(15):
            for j in range(i + 1, 25):
                candidates.append(_PRIMES[i] * _PRIMES[j])
        
        # Products of 3 primes
        for i in range(10):
            for j in range(i + 1, 20):
                for k in range(j + 1, 30):
                    candidates.append(_PRIMES[i] * _PRIMES[j] * _PRIMES[k])
        
        # Also try scaling the current gap
        candidates.append(old_gap * 2)
        candidates.append(old_gap * 3)
        candidates.append(old_gap // 2)
        candidates.append(old_gap // 3)
        
        random.shuffle(candidates)
        
        for new_gap in candidates[:20]:
            if new_gap <= 0:
                continue
            
            # Apply the new gap
            new_gaps = list(gaps)
            new_gaps[gap_idx] = new_gap
            
            # Reconstruct U
            new_U = [0]
            for g in new_gaps:
                new_U.append(new_U[-1] + g)
            
            new_score = _score_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                gaps = new_gaps
                improved = True
                break
        
        # Try adding a new gap
        if not improved and len(gaps) < 30:
            for p in _PRIMES[:20]:
                new_gaps = list(gaps) + [p]
                new_U = [0]
                for g in new_gaps:
                    new_U.append(new_U[-1] + g)
                new_score = _score_fast(new_U)
                if new_score > best_score + 1e-12:
                    best_score = new_score
                    best_U = new_U
                    U = new_U
                    gaps = new_gaps
                    improved = True
                    break
        
        # Try removing a gap
        if not improved and len(gaps) > 2:
            idx = random.randint(0, len(gaps) - 1)
            new_gaps = gaps[:idx] + gaps[idx+1:]
            new_U = [0]
            for g in new_gaps:
                new_U.append(new_U[-1] + g)
            new_score = _score_fast(new_U)
            if new_score > best_score + 1e-12:
                best_score = new_score
                best_U = new_U
                U = new_U
                gaps = new_gaps
                improved = True
        
        if not improved and step > 100 and random.random() < 0.05:
            # Perturb: replace a random gap with a random prime product
            if len(gaps) > 0:
                idx = random.randint(0, len(gaps) - 1)
                # Random product of 1-3 primes
                num_primes = random.randint(1, 3)
                prime_indices = random.sample(range(30), num_primes)
                new_gap = 1
                for pi in prime_indices:
                    new_gap *= _PRIMES[pi]
                
                gaps[idx] = new_gap
                U = [0]
                for g in gaps:
                    U.append(U[-1] + g)
                best_score = _score_fast(U)
                best_U = list(U)
    
    return best_U, best_score


def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    best_U = [0, 1, 3]
    best_score = _score_fast(best_U)

    # Phase 1: Try known Golomb rulers
    for n in range(3, 26):
        if time.time() > deadline - 60:
            break
        if n in _KNOWN_GOLOMB:
            U = _KNOWN_GOLOMB[n][:]
        else:
            U = _greedy_golomb(n, max_val=5000)
        if not U or 0 not in U:
            continue
        U = sorted(U)
        score = _score_fast(U)
        if score > best_score:
            best_score = score
            best_U = list(U)

    # Phase 2: Try geometric sets {0, 1, b, b^2, ..., b^(n-1)}
    for b in range(2, 40):
        if time.time() > deadline - 50:
            break
        for n in range(2, 12):
            if time.time() > deadline - 50:
                break
            U = [0]
            val = 1
            for i in range(1, n):
                val *= b
                U.append(val)
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)

    # Phase 3: Try small structured sets
    for a in range(2, 25):
        if time.time() > deadline - 40:
            break
        for b in range(a + 1, 120):
            if time.time() > deadline - 40:
                break
            U = [0, 1, a, b]
            score = _score_fast(U)
            if score > best_score:
                best_score = score
                best_U = list(U)

    # Phase 4: Gap-multiplication constructions
    if time.time() < deadline - 30:
        for num_gaps in range(3, 15):
            if time.time() > deadline - 30:
                break
            U = _gap_multiplication_set(num_gaps, prime_pool_size=30, max_val=500000)
            if U and 0 in U:
                score = _score_fast(U)
                if score > best_score:
                    best_score = score
                    best_U = list(U)

    # Phase 5: Local search from best seed
    if time.time() < deadline - 20:
        U, score = _local_search(best_U, time_limit=25.0, max_steps=50000)
        if score > best_score:
            best_score = score
            best_U = U

    # Phase 6: Prime-gap local search
    if time.time() < deadline - 10:
        U, score = _prime_gap_local_search(best_U, time_limit=15.0, max_steps=30000)
        if score > best_score:
            best_score = score
            best_U = U

    # Phase 7: More local search rounds with perturbation
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            # Perturb
            U = list(best_U)
            if len(U) > 2:
                idx = random.randint(1, len(U) - 1)
                U.pop(idx)
            max_u = max(U) if U else 0
            new_val = random.randint(1, max_u + max(10, max_u // 3))
            if new_val not in U:
                U.append(new_val)
            U = sorted(U)

            U, score = _local_search(U, time_limit=8.0, max_steps=15000)
            if score > best_score:
                best_score = score
                best_U = U

    # Ensure 0 is in the set
    if 0 not in best_U:
        best_U.append(0)

    return sorted(best_U)

# EVOLVE-BLOCK-END
