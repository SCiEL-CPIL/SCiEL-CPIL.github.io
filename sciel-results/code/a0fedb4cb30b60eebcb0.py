import time
import random
import itertools

def _is_sidon(s):
    """Check if sorted list s has all unordered pair sums (i<=j) unique."""
    n = len(s)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            sm = s[i] + s[j]
            if sm in seen:
                return False
            seen.add(sm)
    return True

def _max_sidon_size(S):
    """Find largest subset of S that is Sidon (all pair sums i<=j unique)."""
    S = sorted(S)
    n = len(S)
    for k in range(n, 0, -1):
        for comb in itertools.combinations(S, k):
            if _is_sidon(list(comb)):
                return k
    return 0

def construct():
    t0 = time.time()
    deadline = t0 + 82  # 8s safety margin before 90s kill

    # Precompute all 12-element subsets? No, too many.
    # Strategy: dense small-range sets with controlled perturbations.
    # Key: maximize additive collisions to reduce max Sidon subset.

    best_val = 10
    best_set = list(range(12))

    # Phase 1: Systematic search over dense structures
    # Try sets that are nearly arithmetic progressions or have many 3-term APs

    # 1a: Contiguous blocks with one or two perturbations
    for start in range(0, 10):
        if time.time() > deadline - 8:
            break
        base = list(range(start, start + 12))
        if base[-1] > 128:
            continue
        v = _max_sidon_size(base)
        if v < best_val:
            best_val = v
            best_set = base[:]
            if best_val == 4:
                return best_set

    # 1b: Pairs of small dense blocks
    # e.g., {0,1,2,3,4,5,6, 20,21,22,23,24} type structures
    for a_start in range(0, 5):
        for b_start in range(a_start + 10, a_start + 25):
            if time.time() > deadline - 8:
                break
            cand = list(range(a_start, a_start + 6)) + list(range(b_start, b_start + 6))
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue
            v = _max_sidon_size(cand)
            if v < best_val:
                best_val = v
                best_set = cand[:]
                if best_val == 4:
                    return best_set

    # 1c: Three small blocks
    for a in range(0, 3):
        for b in range(a + 5, a + 12):
            for c in range(b + 5, b + 12):
                if time.time() > deadline - 8:
                    break
                cand = list(range(a, a + 4)) + list(range(b, b + 4)) + list(range(c, c + 4))
                cand = sorted(set(cand))
                if len(cand) != 12:
                    continue
                v = _max_sidon_size(cand)
                if v < best_val:
                    best_val = v
                    best_set = cand[:]
                    if best_val == 4:
                        return best_set

    # Phase 2: Local search / swap-based refinement
    # Start from best_set and try single-element swaps, double swaps
    rng = random.Random(12345)

    def _try_improve(cand):
        nonlocal best_val, best_set
        cand = sorted(set(cand))
        if len(cand) != 12:
            return False
        v = _max_sidon_size(cand)
        if v < best_val:
            best_val = v
            best_set = cand[:]
            if best_val == 4:
                return True
        return False

    # 2a: Single element replacement from best_set
    for i in range(12):
        if time.time() > deadline - 15:
            break
        orig = best_set[i]
        for new_val in range(0, 129):
            if new_val == orig:
                continue
            cand = best_set[:]
            cand[i] = new_val
            if _try_improve(cand):
                return best_set

    # 2b: Random local search with restarts
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        if restart == 0:
            cand = best_set[:]
        else:
            # New random start: dense set with random perturbations
            start = rng.randint(0, 8)
            cand = list(range(start, start + 12))
            if cand[-1] > 128:
                start = 0
                cand = list(range(12))
            # Perturb 2-4 elements
            n_pert = rng.randint(2, 4)
            indices = rng.sample(range(12), n_pert)
            for idx in indices:
                cand[idx] = rng.randint(0, min(128, start + 30))
            cand = sorted(set(cand))
            if len(cand) != 12:
                continue

        # Iterate: try single swaps, keep if improvement
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                orig = cand[i]
                for new_val in range(0, 129):
                    if new_val == orig:
                        continue
                    trial = cand[:]
                    trial[i] = new_val
                    trial = sorted(set(trial))
                    if len(trial) != 12:
                        continue
                    v = _max_sidon_size(trial)
                    if v < best_val:
                        best_val = v
                        best_set = trial[:]
                        cand = trial[:]
                        improved = True
                        if best_val == 4:
                            return best_set
                    elif v == best_val and v < 6:
                        # Accept equal value to explore plateau
                        if rng.random() < 0.3:
                            cand = trial[:]
                            improved = True
                        break

    # Phase 3: Targeted structured search
    # Try sets with many 3-term arithmetic progressions
    if time.time() < deadline - 3:
        # Pattern: pairs at various spacings
        for d1 in range(1, 8):
            for d2 in range(d1 + 1, d1 + 10):
                if time.time() > deadline - 3:
                    break
                # Build set with APs of length 3
                base_pts = [0, d1, 2*d1, 3*d1, 4*d1, 5*d1]
                # Add more points at different spacing
                extra = [d2, 2*d2, 3*d2, 4*d2, 5*d2, 6*d2]
                cand = sorted(set(base_pts + extra))
                cand = [x for x in cand if 0 <= x <= 128]
                if len(cand) >= 12:
                    cand = cand[:12]
                if len(cand) == 12:
                    v = _max_sidon_size(cand)
                    if v < best_val:
                        best_val = v
                        best_set = cand[:]
                        if best_val == 4:
                            return best_set

    # Phase 4: Exhaustive small-range search if time permits
    # Search all 12-element subsets of [0, N] for small N
    # N=15: C(16,12)=1820, N=16: C(17,12)=6188, N=17: C(18,12)=18564
    if time.time() < deadline - 3:
        for N in range(15, 20):
            if time.time() > deadline - 3:
                break
            total = 0
            for comb in itertools.combinations(range(N + 1), 12):
                total += 1
                if total % 500 == 0 and time.time() > deadline - 3:
                    break
                v = _max_sidon_size(list(comb))
                if v < best_val:
                    best_val = v
                    best_set = list(comb)
                    if best_val == 4:
                        return best_set

    return best_set
