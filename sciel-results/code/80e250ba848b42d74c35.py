# EVOLVE-BLOCK-START
import time
import math
import random
import numpy as np
from numpy.fft import fft, ifft

# ---------------------------------------------------------------------------
# Scoring (FFT-based, exact for the sizes we use)
# ---------------------------------------------------------------------------
def _score_set_fft(U):
    """Compute the score for a finite set U using FFT for speed."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    total_len = 4 * max_u + 1
    fft_len = 1
    while fft_len < total_len:
        fft_len *= 2

    indicator = np.zeros(fft_len, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0

    freq = fft(indicator)

    sum_conv = ifft(freq * freq).real
    sumset_size = int(np.sum(sum_conv[:2 * max_u + 1] > 1e-9))

    autocorr = ifft(freq * np.conj(freq)).real
    if max_u >= 1:
        pos_lags = autocorr[1:max_u + 1] > 1e-9
        neg_lags = autocorr[fft_len - max_u:fft_len - 0][::-1] > 1e-9
        diffset_size = 1 + int(np.sum(pos_lags)) + int(np.sum(neg_lags))
    else:
        diffset_size = 1

    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
    log_ratio = math.log(ratio)
    log_denom = math.log(2 * max_u + 1)
    if log_denom == 0:
        return 1.0
    return 1.0 + log_ratio / log_denom


def _score_set(U):
    """Compute the score for a finite set U containing 0."""
    U = sorted(set(U))
    n = len(U)
    if n == 0:
        return 0.0
    max_u = max(U)
    if max_u == 0:
        return 1.0

    if n <= 25:
        sumset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
        diffset = set()
        for a in U:
            for b in U:
                diffset.add(a - b)
        ratio = len(diffset) / len(sumset)
        if ratio <= 0:
            return 0.0
        log_ratio = math.log(ratio)
        log_denom = math.log(2 * max_u + 1)
        if log_denom == 0:
            return 1.0
        return 1.0 + log_ratio / log_denom
    return _score_set_fft(U)


# ---------------------------------------------------------------------------
# Seed families
# ---------------------------------------------------------------------------
def _golomb_ruler(n):
    """Known (near-)optimal Golomb rulers: unique pairwise differences."""
    known = {
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 10, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    }
    if n in known:
        return known[n][:]
    if n <= 1:
        return [0]
    ruler = [0]
    for _ in range(1, n):
        pos = 1
        while True:
            cand = ruler + [pos]
            diffs = set()
            valid = True
            for j in range(len(cand)):
                for k in range(j + 1, len(cand)):
                    d = cand[k] - cand[j]
                    if d in diffs:
                        valid = False
                        break
                    diffs.add(d)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


def _sidon_ruler(n):
    """Greedy Sidon (B2) set: few distinct pairwise sums, differences near-maximal."""
    ruler = [0]
    for _ in range(1, n):
        pos = 1
        while True:
            cand = ruler + [pos]
            sums = set()
            valid = True
            for i in range(len(cand)):
                for j in range(i, len(cand)):
                    s = cand[i] + cand[j]
                    if s in sums:
                        valid = False
                        break
                    sums.add(s)
                if not valid:
                    break
            if valid:
                ruler.append(pos)
                break
            pos += 1
    return ruler


# ---------------------------------------------------------------------------
# Dual-lattice / modular-congruence construction
# ---------------------------------------------------------------------------
def _dual_lattice(m1, m2, max_val):
    """
    Construct U = {x in [0, max_val] : x ≡ 0 (mod m1) AND x ≡ 0 (mod m2)}.
    This is equivalent to x ≡ 0 (mod lcm(m1, m2)), giving a single arithmetic
    progression.  For a richer structure we use a *union* of two residue classes:
        U = {x : x ≡ r1 (mod m1)} ∪ {x : x ≡ r2 (mod m2)}  ∩ [0, max_val]
    The intersection of the two lattices (via CRT) produces a sparse set with
    predictable |U+U| and |U-U|.
    """
    U = set()
    # First lattice: multiples of m1
    for x in range(0, max_val + 1, m1):
        U.add(x)
    # Second lattice: offset by r2, step m2
    r2 = m2 // 2 if m2 >= 2 else 1
    for x in range(r2, max_val + 1, m2):
        U.add(x)
    U = sorted(U)
    if 0 not in U:
        U = [0] + U
    return U


def _dual_lattice_rich(m1, m2, r1, r2, max_val):
    """
    Richer dual-lattice: U = {x in [0,max_val] : x ≡ r1 (mod m1)} ∪ {x : x ≡ r2 (mod m2)}.
    The overlap region (where both congruences hold) is governed by CRT,
    creating a structured pattern with controlled sumset/difference-set growth.
    """
    U = set()
    for x in range(r1 % m1, max_val + 1, m1):
        if x >= 0:
            U.add(x)
    for x in range(r2 % m2, max_val + 1, m2):
        if x >= 0:
            U.add(x)
    U = sorted(U)
    if 0 not in U:
        U = [0] + U
    return U


def _modular_grid_search(max_val, deadline):
    """
    Grid search over (m1, m2, r1, r2) to find a good dual-lattice set.
    """
    best_U = None
    best_score = -float('inf')
    # Search over moduli
    for m1 in range(2, 30):
        if time.time() > deadline:
            break
        for m2 in range(2, 30):
            if time.time() > deadline:
                break
            if m1 == m2:
                continue
            # Try a few residue combinations
            for r1 in range(0, min(m1, 4)):
                for r2 in range(0, min(m2, 4)):
                    U = _dual_lattice_rich(m1, m2, r1, r2, max_val)
                    if len(U) < 3:
                        continue
                    s = _score_set(U)
                    if s > best_score:
                        best_score = s
                        best_U = U[:]
    return best_U, best_score


# ---------------------------------------------------------------------------
# Simulated Annealing optimizer
# ---------------------------------------------------------------------------
def _simulated_annealing(U, max_iters=10000, deadline=None, T_init=5.0, T_min=0.01, alpha=0.9995):
    """
    Simulated annealing over set elements: move/add/remove one element per step.
    Accepts worse moves with probability exp(-ΔE/T) to escape local optima.
    """
    U = sorted(set(U))
    current_score = _score_set(U)
    best_score = current_score
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 300
    T = T_init
    n = len(U)
    max_size = 65
    min_size = 3

    for it in range(max_iters):
        if deadline and time.time() > deadline:
            break
        T *= alpha
        if T < T_min:
            T = T_min

        # Choose a random move type
        move_type = random.random()
        new_U = None

        if move_type < 0.55 and n > 1:
            # Move an element
            idx = random.randint(1, n - 1)  # skip index 0 (value 0)
            old_val = U[idx]
            # Gaussian-ish perturbation
            delta = int(random.gauss(0, 30))
            new_val = max(1, old_val + delta)
            if new_val != old_val and new_val not in U:
                new_U = U[:]
                new_U[idx] = new_val
                new_U = sorted(set(new_U))
                if 0 not in new_U:
                    new_U = [0] + new_U
                    new_U = sorted(new_U)
        elif move_type < 0.75 and n < max_size:
            # Add an element
            new_val = random.randint(1, upper_bound)
            if new_val not in U:
                new_U = U + [new_val]
                new_U = sorted(new_U)
        elif move_type < 0.90 and n > min_size:
            # Remove an element
            idx = random.randint(1, n - 1)
            new_U = U[:idx] + U[idx + 1:]
        elif move_type < 0.95:
            # Swap two elements
            if n >= 3:
                i, j = random.sample(range(1, n), 2)
                new_U = U[:]
                new_U[i], new_U[j] = new_U[j], new_U[i]
                new_U = sorted(set(new_U))
                if len(new_U) == n and 0 in new_U:
                    pass
                else:
                    new_U = None
        else:
            # Large random jump
            new_U = U[:]
            for _ in range(random.randint(1, 3)):
                idx = random.randint(1, len(new_U) - 1)
                new_U[idx] = random.randint(1, upper_bound)
            new_U = sorted(set(new_U))
            if 0 not in new_U:
                new_U = [0] + sorted(new_U)

        if new_U is None or len(new_U) < 2:
            continue

        new_score = _score_set(new_U)
        delta = new_score - current_score

        if delta > 0 or random.random() < math.exp(delta / T):
            current_score = new_score
            U = new_U
            n = len(U)
            max_u = max(U)
            upper_bound = max_u + 300
            if current_score > best_score:
                best_score = current_score
                best_U = U[:]

    return sorted(set(best_U)), best_score


# ---------------------------------------------------------------------------
# Local search (kept as fallback / polish)
# ---------------------------------------------------------------------------
def _local_search(U, max_iters=5000, deadline=None, polish=False):
    U = sorted(set(U))
    best_score = _score_set(U)
    best_U = U[:]
    max_u = max(U)
    upper_bound = max_u + 200
    tol = 1e-12 if polish else 1e-9
    move_range = 80 if polish else 40
    n_rand = 20 if polish else 12
    max_size = 70 if polish else 60
    stall_count = 0
    stall_limit = 50

    for _ in range(max_iters):
        if deadline and time.time() > deadline:
            break
        improved = False

        for i in range(len(U)):
            if U[i] == 0:
                continue
            old_val = U[i]
            candidates = set()
            for offset in range(-move_range, move_range + 1):
                nv = old_val + offset
                if nv > 0:
                    candidates.add(nv)
            for _ in range(n_rand):
                candidates.add(random.randint(1, upper_bound))
            for new_val in sorted(candidates):
                if new_val == old_val:
                    continue
                new_U = U[:]
                new_U[i] = new_val
                if len(set(new_U)) != len(new_U):
                    continue
                s = _score_set(new_U)
                if s > best_score + tol:
                    best_score = s
                    best_U = new_U[:]
                    U = new_U[:]
                    max_u = max(U)
                    upper_bound = max_u + 200
                    improved = True
                    break
            if improved:
                break

        if not improved:
            if len(U) < max_size:
                for _ in range(30):
                    new_val = random.randint(1, upper_bound)
                    if new_val in U:
                        continue
                    new_U = U + [new_val]
                    s = _score_set(new_U)
                    if s > best_score + tol:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

            if not improved and len(U) > 3:
                for i in range(len(U)):
                    if U[i] == 0:
                        continue
                    new_U = U[:i] + U[i + 1:]
                    s = _score_set(new_U)
                    if s > best_score + tol:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        break

        if not improved:
            stall_count += 1
            if stall_count >= stall_limit:
                break
            if random.random() < 0.2:
                new_U = U[:]
                for _ in range(min(4, len(new_U))):
                    idx = random.randint(1, len(new_U) - 1)
                    new_U[idx] = random.randint(1, upper_bound)
                new_U_set = set(new_U)
                if len(new_U_set) == len(new_U):
                    s = _score_set(new_U)
                    if s > best_score + tol:
                        best_score = s
                        best_U = new_U[:]
                        U = new_U[:]
                        max_u = max(U)
                        upper_bound = max_u + 200
                        improved = True
                        stall_count = 0
            if not improved:
                break
        else:
            stall_count = 0

    return sorted(set(best_U)), best_score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before the 150s limit

    random.seed(42)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    candidates = []

    # --- Sidon / B2 sets ---
    for n in range(3, 20):
        if time.time() > deadline - 80:
            break
        candidates.append(_sidon_ruler(n))

    # --- Golomb rulers ---
    for n in range(3, 20):
        if time.time() > deadline - 80:
            break
        candidates.append(_golomb_ruler(n))

    # --- Dual-lattice / modular-congruence sets (NEW structural family) ---
    for max_val in [100, 200, 400, 800]:
        if time.time() > deadline - 80:
            break
        U_latt, s_latt = _modular_grid_search(max_val, deadline - 80)
        if U_latt and s_latt > best_score:
            best_score = s_latt
            best_U = U_latt
        if U_latt:
            candidates.append(U_latt)

    # --- Arithmetic progressions ---
    for base in [2, 3, 5, 7, 11]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 30):
            U.append(i * base)
        candidates.append(U[:min(len(U), 30)])

    # --- Powers-based sets ---
    for p in [2, 3]:
        if time.time() > deadline - 80:
            break
        U = [0]
        for i in range(1, 15):
            U.append(p ** i)
            if U[-1] > 20000:
                break
        candidates.append(U)

    # --- Random sets ---
    for trial in range(20):
        if time.time() > deadline - 80:
            break
        n = random.randint(3, 40)
        max_val = random.randint(20, 600)
        U = [0] + random.sample(range(1, max_val + 1), n - 1)
        candidates.append(sorted(U))

    # --- Fibonacci-based ---
    for n in range(5, 20):
        if time.time() > deadline - 80:
            break
        U = [0]
        fib = [1, 1]
        for i in range(2, 30):
            fib.append(fib[-1] + fib[-2])
        for i in range(1, min(n, len(fib))):
            if fib[i] not in U:
                U.append(fib[i])
        candidates.append(sorted(U))

    # Evaluate all candidates
    scored = []
    for U in candidates:
        if time.time() > deadline - 50:
            break
        s = _score_set(U)
        scored.append((s, U))
        if s > best_score:
            best_score = s
            best_U = U[:]

    scored.sort(key=lambda x: -x[0])

    # --- Simulated Annealing on top candidates (NEW optimizer) ---
    seen = []
    for s, U in scored[:8]:
        if time.time() > deadline - 15:
            break
        key = tuple(U)
        if key in seen:
            continue
        seen.append(key)
        if s < best_score - 0.05:
            continue
        # Run SA with moderate iterations
        sa_U, sa_score = _simulated_annealing(U, max_iters=5000, deadline=deadline - 15)
        if sa_score > best_score:
            best_score = sa_score
            best_U = sa_U

    # --- Final hill-climb polish on the best result ---
    if time.time() < deadline - 5 and len(best_U) > 2:
        polished_U, polished_score = _local_search(best_U, max_iters=6000, deadline=deadline - 3, polish=True)
        if polished_score > best_score:
            best_score = polished_score
            best_U = polished_U

    return sorted(set(best_U))
# EVOLVE-BLOCK-END
