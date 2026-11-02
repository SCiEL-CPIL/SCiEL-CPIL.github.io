"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 10.0


def _generate_candidates():
    """Generate all nonzero integer vectors in R^11 with squared norm <= 16.
    We include weight 2 (squared norm 4) and weight 3 (squared norm 9) vectors,
    each with a symmetry-breaking constraint (first nonzero coordinate positive).
    """
    cands = []
    # Weight 2: two coordinates are ±1, rest 0
    for i, j in itertools.combinations(range(D), 2):
        # symmetry: sign[i] = +1 (first nonzero is positive)
        for sj in (1, -1):
            v = [0] * D
            v[i] = 1
            v[j] = sj
            cands.append(v)
    # Weight 3: three coordinates are ±1, rest 0
    for i, j, k in itertools.combinations(range(D), 3):
        # symmetry: sign[i] = +1
        for sj in (1, -1):
            for sk in (1, -1):
                v = [0] * D
                v[i] = 1
                v[j] = sj
                v[k] = sk
                cands.append(v)
    return np.array(cands, dtype=np.int64)


def _inner_products(A, B):
    """Compute inner products between rows of A and rows of B."""
    return A @ B.T


def _greedy_select(cands, order, time_limit):
    """Greedy max-min selection: pick candidate maximizing min inner product
    with already-selected set (we want orthogonal, so inner product = 0)."""
    selected = []
    selected_idx = set()
    n = len(cands)

    # Precompute all-pairs inner products? Too big. Compute on the fly.
    # For efficiency, maintain a running array of min inner product per candidate.
    min_ip = np.full(n, np.inf)  # min |inner product| with selected set
    # Actually we want inner product == 0. Let's track: a candidate is valid
    # if inner product with ALL selected is 0. We'll just check on the fly.

    selected_arr = []  # list of vectors

    for idx in order:
        if time.time() > time_limit:
            break
        v = cands[idx]
        # Check orthogonality with all selected
        if not selected_arr:
            ok = True
        else:
            # vectorized check
            ok = True
            # chunked for speed
            for start in range(0, len(selected_arr), 512):
                chunk = selected_arr[start:start+512]
                ips = chunk @ v
                if np.any(ips != 0):
                    ok = False
                    break
        if ok:
            selected_arr.append(v)
            selected_idx.add(idx)

    return np.array(selected_arr, dtype=np.int64) if selected_arr else None


def _greedy_min_degree(cands, order, time_limit):
    """Greedy selection: at each step, pick the candidate that has the fewest
    conflicts (non-orthogonal) with remaining candidates, among valid ones."""
    n = len(cands)
    remaining = set(order)
    selected_arr = []

    while remaining and time.time() < time_limit:
        # For each remaining candidate, check if orthogonal to all selected
        valid = []
        for idx in remaining:
            v = cands[idx]
            if not selected_arr:
                valid.append(idx)
                continue
            ok = True
            for start in range(0, len(selected_arr), 512):
                chunk = selected_arr[start:start+512]
                ips = chunk @ v
                if np.any(ips != 0):
                    ok = False
                    break
            if ok:
                valid.append(idx)

        if not valid:
            break

        # Pick the one with min degree (fewest conflicts with other remaining)
        # Approximate: pick the one that conflicts with the fewest other valid candidates
        # For speed, just pick the first valid one (or use a simple heuristic)
        # Use min-degree: count how many other valid candidates are non-orthogonal
        best_idx = valid[0]
        best_degree = float('inf')

        # Sample if too many
        if len(valid) > 200:
            sample = random.sample(valid, 200)
        else:
            sample = valid

        for idx in sample:
            v = cands[idx]
            degree = 0
            for other in valid:
                if other == idx:
                    continue
                if v @ cands[other] != 0:
                    degree += 1
                    if degree >= best_degree:
                        break
            if degree < best_degree:
                best_degree = degree
                best_idx = idx

        selected_arr.append(cands[best_idx])
        remaining.discard(best_idx)

    return np.array(selected_arr, dtype=np.int64) if selected_arr else None


def _local_improve(selected, cands, time_limit):
    """Try to improve by adding more vectors or swapping.
    Simple: try to add any candidate that's orthogonal to all selected."""
    if len(selected) == 0:
        return selected
    n = len(cands)
    improved = True
    while improved and time.time() < time_limit:
        improved = False
        for idx in range(n):
            if time.time() > time_limit:
                break
            v = cands[idx]
            ok = True
            for start in range(0, len(selected), 512):
                chunk = selected[start:start+512]
                ips = chunk @ v
                if np.any(ips != 0):
                    ok = False
                    break
            if ok:
                selected = np.vstack([selected, v])
                improved = True
                break
    return selected


def construct():
    start_time = time.time()
    time_limit = start_time + DEADLINE_SECONDS - SAFETY_MARGIN

    cands = _generate_candidates()
    n = len(cands)

    best_result = None
    best_size = 0

    # Multiple random restarts with different orderings
    rng = random.Random(42)

    # Try several strategies
    # Strategy 1: random order greedy
    for trial in range(50):
        if time.time() > time_limit:
            break
        order = list(range(n))
        rng.shuffle(order)
        result = _greedy_select(cands, order, time_limit)
        if result is not None and len(result) > best_size:
            best_size = len(result)
            best_result = result

    # Strategy 2: sorted by norm (prefer weight-2 first, then weight-3)
    norms = np.sum(cands * cands, axis=1)
    order_by_norm = np.argsort(norms)
    result = _greedy_select(cands, order_by_norm, time_limit)
    if result is not None and len(result) > best_size:
        best_size = len(result)
        best_result = result

    # Strategy 3: local improvement on best found
    if best_result is not None and time.time() < time_limit:
        improved = _local_improve(best_result, cands, time_limit)
        if len(improved) > best_size:
            best_result = improved
            best_size = len(improved)

    if best_result is None or len(best_result) == 0:
        # Fallback: return seed result
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return best_result
# EVOLVE-BLOCK-END
