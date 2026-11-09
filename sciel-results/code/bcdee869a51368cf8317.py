import numpy as np
import time
from itertools import combinations, product

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s limit

    # Generate candidate vectors with entries in {-1, 0, 1} and weight 1..4
    candidates = []
    for weight in range(1, 5):
        for combo in combinations(range(d), weight):
            for signs in product([-1, 1], repeat=weight):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n_cands = len(candidates)

    # Precompute squared norms
    norms_sq = np.sum(candidates * candidates, axis=1)

    # Greedy selection for different target max norms
    best_points = None
    best_k = 0

    for target_norm in [2, 3, 4, 5, 6]:
        if time.time() > deadline - 30:
            break

        # Filter candidates with norm^2 <= target_norm
        mask = norms_sq <= target_norm
        cands = candidates[mask]
        cands_norms = norms_sq[mask]
        n = len(cands)

        if n == 0:
            continue

        # Sort by norm (prefer lower norms)
        order = np.argsort(cands_norms)
        cands = cands[order]
        cands_norms = cands_norms[order]

        # Greedy: try to add each candidate if compatible with all selected
        selected = []
        selected_norms = []

        for i in range(n):
            if time.time() > deadline - 30:
                break

            v = cands[i]
            v_norm = int(cands_norms[i])

            if not selected:
                selected.append(v)
                selected_norms.append(v_norm)
                continue

            # Vectorized check: dist^2 = ||v||^2 + ||p||^2 - 2*v.p
            sel_arr = np.array(selected)
            dots = sel_arr @ v
            dist_sqs = v_norm + np.array(selected_norms) - 2 * dots

            if np.all(dist_sqs >= target_norm):
                selected.append(v)
                selected_norms.append(v_norm)

        k = len(selected)
        if k > best_k:
            best_k = k
            best_points = np.array(selected, dtype=np.int64)

    if best_points is None:
        # Fallback: weight-2 vectors only
        w2 = []
        for combo in combinations(range(d), 2):
            for signs in product([-1, 1], repeat=2):
                v = np.zeros(d, dtype=np.int64)
                for idx, s in zip(combo, signs):
                    v[idx] = s
                w2.append(v)
        best_points = np.array(w2, dtype=np.int64)

    return best_points
