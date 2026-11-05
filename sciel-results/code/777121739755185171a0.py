"""Optimized program for sum-product extremal sets."""
import time
import random
import math
import numpy as np


def _fast_objective(A):
    """Compute max(|A+A|, |A*A|) using NumPy for speed."""
    A = np.array(A, dtype=np.int64)
    n = len(A)
    # Sumset: all pairs (i,j) with i <= j
    idx_i, idx_j = np.triu_indices(n)
    sums = A[idx_i] + A[idx_j]
    # Product set
    prods = A[idx_i] * A[idx_j]
    return max(len(np.unique(sums)), len(np.unique(prods)))


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin under 90s

    random.seed(123)
    np.random.seed(123)

    # --- Build lattice candidates: 2^i * 3^j ---
    lattice_candidates = set()
    for i in range(0, 14):
        for j in range(0, 14):
            v = (2 ** i) * (3 ** j)
            if 1 <= v <= 10000:
                lattice_candidates.add(v)
    lattice_candidates = sorted(lattice_candidates)
    small_lattice = [v for v in lattice_candidates if v <= 100]

    best_set = None
    best_obj = float('inf')

    def try_candidate(A):
        nonlocal best_set, best_obj
        if len(A) != 20:
            return
        if len(set(A)) != 20:
            return
        if any(x < 1 or x > 10000 for x in A):
            return
        obj = _fast_objective(A)
        if obj < best_obj:
            best_obj = obj
            best_set = list(A)

    # --- Generate diverse initial candidates ---
    # 1. Smallest lattice points
    if len(lattice_candidates) >= 20:
        try_candidate(sorted(lattice_candidates[:20]))

    # 2. Spread lattice selections
    for start in range(0, max(1, len(lattice_candidates) - 19), 2):
        cand = sorted(lattice_candidates[start:start + 20])
        if len(set(cand)) == 20:
            try_candidate(cand)

    # 3. Random lattice selections
    for _ in range(50):
        cand = sorted(random.sample(lattice_candidates, 20))
        try_candidate(cand)

    # 4. Small lattice random
    if len(small_lattice) >= 20:
        for _ in range(30):
            cand = sorted(random.sample(small_lattice, 20))
            try_candidate(cand)

    # 5. Mixed: some lattice + some small integers
    for _ in range(30):
        k = random.randint(10, 18)
        lat_part = random.sample(lattice_candidates[:60], min(k, 20))
        remaining = 20 - len(lat_part)
        if remaining > 0:
            small_ints = [x for x in range(1, 200) if x not in lat_part]
            if len(small_ints) >= remaining:
                int_part = random.sample(small_ints, remaining)
                cand = sorted(lat_part + int_part)
                if len(set(cand)) == 20:
                    try_candidate(cand)

    # 6. Pure small integers
    for _ in range(30):
        cand = sorted(random.sample(range(1, 100), 20))
        try_candidate(cand)

    # 7. Arithmetic progressions
    for start in range(1, 10):
        for step in range(1, 15):
            cand = [start + i * step for i in range(20)]
            if max(cand) <= 10000:
                try_candidate(sorted(cand))

    # 8. Geometric-like sequences
    for base in [2, 3, 4, 5]:
        geo = [base ** i for i in range(1, 21)]
        if all(x <= 10000 for x in geo):
            try_candidate(sorted(geo))

    # 9. Multiplicative structures: a * 2^i
    for a in range(1, 20):
        seq = [a * (2 ** i) for i in range(20)]
        if all(x <= 10000 for x in seq):
            try_candidate(sorted(seq))

    # 10. Multiplicative: a * 3^i
    for a in range(1, 10):
        seq = [a * (3 ** i) for i in range(20)]
        if all(x <= 10000 for x in seq):
            try_candidate(sorted(seq))

    # 11. a * 6^i
    for a in range(1, 5):
        seq = [a * (6 ** i) for i in range(20)]
        if all(x <= 10000 for x in seq):
            try_candidate(sorted(seq))

    if best_set is None:
        best_set = list(range(1, 21))
        best_obj = _fast_objective(best_set)

    # --- Simulated Annealing with multiple strategies ---
    current_set = best_set[:]
    current_obj = best_obj
    T = 10.0
    cooling = 0.997
    min_T = 0.2

    # Precompute lattice indices for fast lookup
    lattice_set = set(lattice_candidates)
    lattice_arr = np.array(lattice_candidates)

    while time.time() < deadline:
        if T < min_T:
            # Restart from best
            current_set = best_set[:]
            current_obj = best_obj
            T = 10.0

        new_set = current_set[:]
        move_type = random.random()

        if move_type < 0.45:
            # Perturb one element
            i = random.randint(0, 19)
            orig = current_set[i]
            if random.random() < 0.5:
                scale = max(2, int(0.1 * orig))
                delta = random.randint(-scale, scale)
            else:
                scale = max(3, int(0.25 * orig))
                delta = random.randint(-scale, scale)
            new_val = orig + delta
            if new_val < 1 or new_val > 10000 or new_val in current_set:
                continue
            new_set[i] = new_val

        elif move_type < 0.65:
            # Replace with lattice point
            i = random.randint(0, 19)
            orig = current_set[i]
            r = random.random()
            if r < 0.3:
                # Small lattice
                if small_lattice:
                    new_val = random.choice(small_lattice[:40])
                else:
                    new_val = random.randint(1, 50)
            elif r < 0.6:
                # Lattice near current
                mask = (lattice_arr >= 0.5 * orig) & (lattice_arr <= 2.0 * orig)
                candidates = lattice_arr[mask]
                if len(candidates) > 0:
                    new_val = int(random.choice(candidates))
                else:
                    new_val = random.randint(1, 200)
            else:
                # Any lattice
                new_val = random.choice(lattice_candidates)
            if new_val in current_set:
                continue
            new_set[i] = new_val

        elif move_type < 0.80:
            # Swap two elements
            i, j = random.sample(range(20), 2)
            orig_i, orig_j = current_set[i], current_set[j]
            si = random.randint(-max(2, int(0.15 * orig_i)), max(2, int(0.15 * orig_i)))
            sj = random.randint(-max(2, int(0.15 * orig_j)), max(2, int(0.15 * orig_j)))
            new_i, new_j = orig_i + si, orig_j + sj
            if (1 <= new_i <= 10000 and 1 <= new_j <= 10000 and
                    new_i != new_j and new_i not in current_set and new_j not in current_set):
                new_set[i] = new_i
                new_set[j] = new_j
            else:
                continue

        elif move_type < 0.90:
            # Replace 2 elements with lattice points
            i, j = random.sample(range(20), 2)
            existing = set(current_set)
            new_vals = set()
            for _ in range(30):
                v = random.choice(lattice_candidates[:80])
                if v not in existing and v not in new_vals:
                    new_vals.add(v)
                    if len(new_vals) == 2:
                        break
            if len(new_vals) != 2:
                continue
            nv = list(new_vals)
            new_set[i] = nv[0]
            new_set[j] = nv[1]

        else:
            # Replace 3 elements with lattice points (bigger jump)
            indices = random.sample(range(20), 3)
            existing = set(current_set)
            new_vals = set()
            for _ in range(50):
                v = random.choice(lattice_candidates[:100])
                if v not in existing and v not in new_vals:
                    new_vals.add(v)
                    if len(new_vals) == 3:
                        break
            if len(new_vals) != 3:
                continue
            nv = list(new_vals)
            for idx, val in zip(indices, nv):
                new_set[idx] = val

        new_set.sort()
        if len(set(new_set)) != 20:
            continue
        if any(x < 1 or x > 10000 for x in new_set):
            continue

        new_obj = _fast_objective(new_set)

        if new_obj <= current_obj:
            current_set = new_set
            current_obj = new_obj
            if new_obj < best_obj:
                best_obj = new_obj
                best_set = new_set[:]
        else:
            delta_obj = new_obj - current_obj
            if delta_obj < 30:
                accept_prob = math.exp(-delta_obj / max(T, 0.01))
                if random.random() < accept_prob:
                    current_set = new_set
                    current_obj = new_obj

        T *= cooling

    # --- Final local search polish ---
    if time.time() < deadline - 5.0:
        polish_start = time.time()
        polish_deadline = min(deadline - 1.0, polish_start + 10.0)
        current = best_set[:]
        current_obj = best_obj
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            for i in range(20):
                orig = current[i]
                # Try small perturbations
                for delta in range(-10, 11):
                    if delta == 0:
                        continue
                    new_val = orig + delta
                    if new_val < 1 or new_val > 10000 or new_val in current:
                        continue
                    trial = current[:]
                    trial[i] = new_val
                    trial.sort()
                    obj = _fast_objective(trial)
                    if obj < current_obj:
                        current = trial
                        current_obj = obj
                        improved = True
                        break
                if improved:
                    break
        if current_obj < best_obj:
            best_obj = current_obj
            best_set = current[:]

    return best_set
