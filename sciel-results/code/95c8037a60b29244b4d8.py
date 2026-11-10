"""NumPy-accelerated doubly-covered prefix: SA escape + bottleneck local search."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return [i for i in range(n + 1) if sieve[i]]

_PRIMES = _primes_up_to(1000)
_NPRIMES = len(_PRIMES)
_PRIMES_ARR = np.array(_PRIMES, dtype=np.int64)

def _build_count(residues, L_MAX):
    """Build count array for x in [1, L_MAX] using NumPy slicing."""
    count = np.zeros(L_MAX + 1, dtype=np.int32)
    for i, p in enumerate(_PRIMES):
        r = residues[i]
        start = p if r == 0 else r
        if start <= L_MAX:
            count[start::p] += 1
    return count

def _find_prefix_L(count, L_MAX):
    """Find largest L such that count[x] >= 2 for all x in [1..L]."""
    L = 0
    for x in range(1, L_MAX + 1):
        if count[x] >= 2:
            L = x
        else:
            break
    return L

def _score(residues):
    """Compute the actual score (largest L with full double coverage)."""
    L_MAX = 100000
    count = _build_count(residues, L_MAX)
    return _find_prefix_L(count, L_MAX)

def _greedy_construct(deadline):
    """
    Greedy construction: for each prime, pick the residue that maximizes
    coverage of under-covered positions in a working window.
    """
    residues = [0] * _NPRIMES
    L_MAX = 4000
    count = np.zeros(L_MAX + 1, dtype=np.int32)

    # Process primes in increasing order (small primes cover more positions)
    for idx in range(_NPRIMES):
        if time.time() > deadline:
            break
        p = _PRIMES[idx]
        best_r = 0
        best_gain = -1

        # For large primes, only a few residues matter in the window
        if p <= L_MAX:
            # Try all residues for small primes
            for r in range(p):
                start = p if r == 0 else r
                if start > L_MAX:
                    continue
                # Count positions with count < 2 that would be covered
                positions = range(start, L_MAX + 1, p)
                gain = 0
                for x in positions:
                    if count[x] < 2:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
        else:
            # p > L_MAX: each residue covers at most one position in [1, L_MAX]
            # Pick the residue r such that r (or r+p if r==0) is an under-covered position
            for r in range(1, min(p, L_MAX + 1)):
                if count[r] < 2:
                    best_r = r
                    best_gain = 1
                    break

        residues[idx] = best_r
        start = p if best_r == 0 else best_r
        if start <= L_MAX:
            count[start::p] += 1

    current_L = _find_prefix_L(count, L_MAX)
    return residues, current_L

def _simulated_annealing(residues, deadline, L_MAX=5000):
    """
    Simulated annealing to escape local optima.
    Bias toward small primes (higher impact). Accept moves that don't break
    the prefix; accept breaking moves with low probability at high T.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)

    best_residues = list(residues)
    best_L = current_L

    T = 10.0
    T_min = 0.01
    cooling = 0.9997
    no_improve = 0
    no_improve_limit = 20000

    # Weighted index selection favoring small primes
    # Precompute weights: smaller primes get higher weight
    weights = np.array([1.0 / _PRIMES[i] for i in range(_NPRIMES)], dtype=np.float64)
    weights /= weights.sum()

    while time.time() < deadline:
        idx = np.random.choice(_NPRIMES, p=weights)
        p = _PRIMES[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        if new_r == old_r:
            continue

        old_start = p if old_r == 0 else old_r
        new_start = p if new_r == 0 else new_r

        # Quick check: does the move break the current prefix?
        breaks = False
        if old_start <= current_L:
            for x in range(old_start, current_L + 1, p):
                if count[x] == 2 and new_start <= x and (x - new_start) % p == 0:
                    # x is covered by new residue too, so net change is 0
                    pass
                elif count[x] == 2:
                    breaks = True
                    break

        if breaks:
            # Only accept with low probability
            if random.random() > 0.05 * T:
                T *= cooling
                if T < T_min:
                    T = T_min
                no_improve += 1
                if no_improve > no_improve_limit:
                    T = min(T * 3.0, 10.0)
                    no_improve = 0
                continue
            # Accept breaking move (rare)
            count[old_start::p] -= 1
            count[new_start::p] += 1
            residues[idx] = new_r
            # Recompute prefix
            current_L = _find_prefix_L(count, L_MAX)
        else:
            # Safe move: apply and check for extension
            count[old_start::p] -= 1
            count[new_start::p] += 1
            residues[idx] = new_r
            # Check for extension
            x = current_L + 1
            while x <= L_MAX and count[x] >= 2:
                current_L = x
                x += 1

        if current_L > best_L:
            best_L = current_L
            best_residues = list(residues)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > no_improve_limit:
                T = min(T * 3.0, 10.0)
                no_improve = 0

        T *= cooling
        if T < T_min:
            T = T_min

    return best_residues, best_L

def _local_search_targeted(residues, deadline, L_MAX=5000):
    """
    Targeted local search: only consider flips that affect the bottleneck.
    Two-tier: first small primes, then larger primes if no improvement.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)

    improved = True
    stall_count = 0
    max_stall = 30
    tier = 0  # 0 = small primes, 1 = all primes

    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break

        gap = current_L + 1
        if gap > L_MAX:
            break

        if count[gap] >= 2:
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            tier = 0
            continue

        # Determine which primes to try based on tier
        if tier == 0:
            prime_range = [i for i in range(_NPRIMES) if _PRIMES[i] <= 200]
        else:
            prime_range = list(range(_NPRIMES))

        best_move = None
        best_gain = 0

        for i in prime_range:
            p = _PRIMES[i]
            current_r = residues[i]
            desired_r = gap % p

            if current_r == desired_r:
                continue

            # Check safety: all positions in [1, current_L] losing coverage must have count >= 3
            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break

            if not safe:
                continue

            # Simulate the flip
            new_start = p if desired_r == 0 else desired_r

            count[old_start::p] -= 1
            count[new_start::p] += 1

            # Find new L
            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1

            # Undo
            count[new_start::p] -= 1
            count[old_start::p] += 1

            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)

        # If no move found in current tier, switch to next tier
        if best_move is None and tier == 0:
            tier = 1
            stall_count = 0
            continue

        if best_move is None:
            # Try 2-flips among small primes
            small_idx = [i for i in range(_NPRIMES) if _PRIMES[i] <= 150]
            for i1 in range(len(small_idx)):
                if time.time() > deadline:
                    break
                i = small_idx[i1]
                p = _PRIMES[i]
                current_r1 = residues[i]
                desired_r1 = gap % p

                if current_r1 == desired_r1:
                    continue

                old_start1 = p if current_r1 == 0 else current_r1
                new_start1 = p if desired_r1 == 0 else desired_r1

                # Check safety for first flip
                safe1 = True
                for x in range(old_start1, current_L + 1, p):
                    if count[x] < 3:
                        safe1 = False
                        break
                if not safe1:
                    continue

                # Apply first flip
                count[old_start1::p] -= 1
                count[new_start1::p] += 1

                for i2 in range(i1 + 1, len(small_idx)):
                    j = small_idx[i2]
                    q = _PRIMES[j]
                    current_r2 = residues[j]
                    desired_r2 = gap % q

                    if current_r2 == desired_r2:
                        continue

                    old_start2 = q if current_r2 == 0 else current_r2
                    new_start2 = q if desired_r2 == 0 else desired_r2

                    # Check safety for second flip (after first applied)
                    safe2 = True
                    for x in range(old_start2, current_L + 1, q):
                        if count[x] < 3:
                            safe2 = False
                            break
                    if not safe2:
                        continue

                    # Apply second flip
                    count[old_start2::q] -= 1
                    count[new_start2::q] += 1

                    new_L = current_L
                    while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                        new_L += 1

                    # Undo second flip
                    count[new_start2::q] -= 1
                    count[old_start2::q] += 1

                    gain = new_L - current_L
                    if gain > best_gain:
                        best_gain = gain
                        best_move = (i, desired_r1, j, desired_r2, new_L)

                # Undo first flip
                count[new_start1::p] -= 1
                count[old_start1::p] += 1

        if best_move is not None:
            if len(best_move) == 3:
                i, new_r, new_L = best_move
                p = _PRIMES[i]
                old_r = residues[i]
                old_start = p if old_r == 0 else old_r
                new_start = p if new_r == 0 else new_r

                count[old_start::p] -= 1
                count[new_start::p] += 1

                residues[i] = new_r
                current_L = new_L
            else:
                i, r1, j, r2, new_L = best_move
                p = _PRIMES[i]
                q = _PRIMES[j]
                old_r1 = residues[i]
                old_r2 = residues[j]
                old_start1 = p if old_r1 == 0 else old_r1
                new_start1 = p if r1 == 0 else r1
                old_start2 = q if old_r2 == 0 else old_r2
                new_start2 = q if r2 == 0 else r2

                count[old_start1::p] -= 1
                count[new_start1::p] += 1
                count[old_start2::q] -= 1
                count[new_start2::q] += 1

                residues[i] = r1
                residues[j] = r2
                current_L = new_L

            improved = True
            stall_count = 0
            tier = 0  # Reset to small primes after improvement

    return residues, current_L

def _polish_pass(residues, deadline, L_MAX=10000):
    """
    Final polish pass with larger L_MAX to extend the prefix further.
    Tries all primes, not just small ones.
    """
    residues = list(residues)
    count = _build_count(residues, L_MAX)
    current_L = _find_prefix_L(count, L_MAX)

    improved = True
    stall_count = 0
    max_stall = 15

    while improved and time.time() < deadline:
        improved = False
        stall_count += 1
        if stall_count > max_stall:
            break

        gap = current_L + 1
        if gap > L_MAX:
            break

        if count[gap] >= 2:
            while gap <= L_MAX and count[gap] >= 2:
                current_L = gap
                gap += 1
            improved = True
            stall_count = 0
            continue

        # Try all primes, prioritizing small ones
        best_move = None
        best_gain = 0

        for i in range(_NPRIMES):
            p = _PRIMES[i]

            # Skip very large primes if time is tight
            if time.time() > deadline - 5 and p > 300:
                continue

            current_r = residues[i]
            desired_r = gap % p

            if current_r == desired_r:
                continue

            old_start = p if current_r == 0 else current_r
            safe = True
            for x in range(old_start, current_L + 1, p):
                if count[x] < 3:
                    safe = False
                    break

            if not safe:
                continue

            new_start = p if desired_r == 0 else desired_r

            count[old_start::p] -= 1
            count[new_start::p] += 1

            new_L = current_L
            while new_L + 1 <= L_MAX and count[new_L + 1] >= 2:
                new_L += 1

            count[new_start::p] -= 1
            count[old_start::p] += 1

            gain = new_L - current_L
            if gain > best_gain:
                best_gain = gain
                best_move = (i, desired_r, new_L)

        if best_move is not None:
            i, new_r, new_L = best_move
            p = _PRIMES[i]
            old_r = residues[i]
            old_start = p if old_r == 0 else old_r
            new_start = p if new_r == 0 else new_r

            count[old_start::p] -= 1
            count[new_start::p] += 1

            residues[i] = new_r
            current_L = new_L
            improved = True
            stall_count = 0

    return residues, current_L

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit

    # Phase 1: Greedy construction
    residues, L = _greedy_construct(deadline - 80)
    best_residues = list(residues)
    best_L = L

    # Phase 2: Simulated annealing to escape local optima
    if time.time() < deadline - 40:
        residues, L = _simulated_annealing(residues, deadline - 40, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)

    # Phase 3: Targeted local search
    if time.time() < deadline - 20:
        residues, L = _local_search_targeted(best_residues, deadline - 20, L_MAX=5000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)

    # Phase 4: Polish pass with larger range
    if time.time() < deadline - 5:
        residues, L = _polish_pass(best_residues, deadline - 5, L_MAX=10000)
        if L > best_L:
            best_L = L
            best_residues = list(residues)

    # Final verification
    final_L = _score(best_residues)
    if final_L > best_L:
        best_L = final_L

    return best_residues

# EVOLVE-BLOCK-END
