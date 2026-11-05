"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math


def _set_sizes(A):
    """Compute |A+A| and |A*A| for set A (with equal elements allowed)."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        for j in range(i, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return len(sums), len(prods)


def _objective(A):
    """Return max(|A+A|, |A*A|)."""
    s, p = _set_sizes(A)
    return max(s, p)


# EVOLVE-BLOCK-START
def _fast_objective(A):
    """Compute max(|A+A|, |A*A|) using arrays for speed."""
    n = len(A)
    sums = set()
    prods = set()
    for i in range(n):
        ai = A[i]
        sums.add(ai + ai)
        prods.add(ai * ai)
        for j in range(i + 1, n):
            aj = A[j]
            sums.add(ai + aj)
            prods.add(ai * aj)
    return max(len(sums), len(prods))


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin under 60s

    random.seed(42)

    best_set = list(range(1, 21))
    best_obj = _fast_objective(best_set)

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

    # --- Lattice-based seeds: 2^i * 3^j keeps products in a structured subgroup ---
    # This is the key structural idea: multiplicative lattice dramatically reduces
    # product set size, while SA then optimizes the sum set.
    lattice_candidates = []
    for i in range(0, 14):
        for j in range(0, 14):
            v = (2 ** i) * (3 ** j)
            if 1 <= v <= 10000:
                lattice_candidates.append(v)
    # Deduplicate and sort
    lattice_candidates = sorted(set(lattice_candidates))

    # Generate lattice-based initial sets: pick 20 from the lattice
    # Prefer smaller values to keep sums small
    if len(lattice_candidates) >= 20:
        # Take the 20 smallest lattice points
        try_candidate(sorted(lattice_candidates[:20]))
        # Take a spread across the lattice
        for start in range(0, len(lattice_candidates) - 19, 3):
            cand = sorted(lattice_candidates[start:start + 20])
            if len(set(cand)) == 20:
                try_candidate(cand)
        # Random selections from lattice
        for _ in range(30):
            cand = sorted(random.sample(lattice_candidates, 20))
            try_candidate(cand)

    # --- Mixed lattice + small integers ---
    small_lattice = [v for v in lattice_candidates if v <= 50]
    for _ in range(20):
        if len(small_lattice) >= 20:
            cand = sorted(random.sample(small_lattice, 20))
            try_candidate(cand)

    # --- Diverse structured initializations ---
    # Consecutive
    try_candidate(list(range(1, 21)))
    # Arithmetic progressions
    for start in range(1, 5):
        for step in range(1, 10):
            cand = [start + i * step for i in range(20)]
            if max(cand) <= 200:
                try_candidate(sorted(cand))
    # Powers of 2
    geo2 = [2 ** i for i in range(1, 21)]
    if all(x <= 10000 for x in geo2):
        try_candidate(sorted(geo2))
    # Powers of 3
    geo3 = [3 ** i for i in range(1, 21)]
    if all(x <= 10000 for x in geo3):
        try_candidate(sorted(geo3))
    # Random small sets
    for _ in range(50):
        cand = sorted(random.sample(range(1, 100), 20))
        try_candidate(cand)

    # --- Simulated Annealing refinement ---
    current_set = best_set[:]
    current_obj = best_obj
    T = 15.0
    cooling = 0.995
    min_T = 0.3

    while time.time() < deadline:
        if T < min_T:
            # Restart from best
            current_set = best_set[:]
            current_obj = best_obj
            T = 15.0

        new_set = current_set[:]
        move_type = random.random()

        if move_type < 0.50:
            # Perturb one element (local or medium)
            i = random.randint(0, 19)
            orig = current_set[i]
            if random.random() < 0.6:
                scale = max(2, int(0.15 * orig))
                delta = random.randint(-scale, scale)
            else:
                scale = max(5, int(0.4 * orig))
                delta = random.randint(-scale, scale)
            new_val = orig + delta
            if new_val < 1 or new_val > 10000 or new_val in current_set:
                continue
            new_set[i] = new_val

        elif move_type < 0.70:
            # Replace one element with a lattice point (exploit structure)
            i = random.randint(0, 19)
            # Pick from lattice points near current value, or small lattice
            if random.random() < 0.5:
                # Small lattice value
                if small_lattice:
                    new_val = random.choice(small_lattice[:30])
                else:
                    new_val = random.randint(1, 50)
            else:
                # Lattice point near current
                orig = current_set[i]
                candidates = [v for v in lattice_candidates
                              if 0.5 * orig <= v <= 2.0 * orig]
                if candidates:
                    new_val = random.choice(candidates)
                else:
                    new_val = random.randint(1, 100)
            if new_val in current_set:
                continue
            new_set[i] = new_val

        elif move_type < 0.85:
            # Swap two elements
            i, j = random.sample(range(20), 2)
            orig_i, orig_j = current_set[i], current_set[j]
            si = random.randint(-max(2, int(0.1 * orig_i)), max(2, int(0.1 * orig_i)))
            sj = random.randint(-max(2, int(0.1 * orig_j)), max(2, int(0.1 * orig_j)))
            new_i, new_j = orig_i + si, orig_j + sj
            if (1 <= new_i <= 10000 and 1 <= new_j <= 10000 and
                    new_i != new_j and new_i not in current_set and new_j not in current_set):
                new_set[i] = new_i
                new_set[j] = new_j
            else:
                continue

        else:
            # Replace 2 elements with lattice points
            i, j = random.sample(range(20), 2)
            existing = set(current_set)
            new_vals = set()
            for _ in range(20):
                v = random.choice(small_lattice[:40]) if small_lattice else random.randint(1, 60)
                if v not in existing and v not in new_vals:
                    new_vals.add(v)
                    if len(new_vals) == 2:
                        break
            if len(new_vals) != 2:
                continue
            nv = list(new_vals)
            new_set[i] = nv[0]
            new_set[j] = nv[1]

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
            if delta_obj < 50:
                accept_prob = math.exp(-delta_obj / max(T, 0.01))
                if random.random() < accept_prob:
                    current_set = new_set
                    current_obj = new_obj

        T *= cooling

    return best_set
# EVOLVE-BLOCK-END
