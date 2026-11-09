import numpy as np
import time
import random
from itertools import combinations

def _squared_norm(v):
    return int(np.dot(v, v))

def _squared_dist(a, b):
    d = a - b
    return int(np.dot(d, d))

def kissing_number11() -> np.ndarray:
    d = 11
    deadline = time.time() + 580.0  # safety margin before 630s kill

    best_points = None
    best_k = 0

    # Start with the known 220-point solution (weight-2 vectors)
    base_points = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [-1, 1]:
                for s2 in [-1, 1]:
                    p = np.zeros(d, dtype=np.int64)
                    p[i] = s1
                    p[j] = s2
                    base_points.append(p)
    base_points = np.array(base_points, dtype=np.int64)
    best_points = base_points.copy()
    best_k = len(base_points)  # 220

    # Try greedy selection from larger candidate pools
    # Candidate pool: vectors with weight 1, 2, 3, 4 (all sign combinations)
    # Weight w in dimension 11: C(11,w) * 2^w candidates
    # w=1: 11*2 = 22
    # w=2: 55*4 = 220
    # w=3: 165*8 = 1320
    # w=4: 330*16 = 5280
    # Total: ~6842 candidates

    def generate_candidates(weights):
        cands = []
        for w in weights:
            for combo in combinations(range(d), w):
                for mask in range(2 ** w):
                    v = np.zeros(d, dtype=np.int64)
                    for idx, pos in enumerate(combo):
                        if mask & (1 << idx):
                            v[pos] = 1
                        else:
                            v[pos] = -1
                    cands.append(v)
        return np.array(cands, dtype=np.int64)

    # Try different weight combinations
    weight_sets = [
        [1, 2],      # 242 candidates
        [2, 3],      # 1540 candidates
        [1, 2, 3],   # 1562 candidates
        [2, 3, 4],   # 6820 candidates
        [1, 2, 3, 4] # 6842 candidates
    ]

    rng = random.Random(42)

    for ws in weight_sets:
        if time.time() > deadline - 10:
            break

        cands = generate_candidates(ws)
        n = len(cands)

        # Precompute norms
        norms = np.array([_squared_norm(c) for c in cands], dtype=np.int64)

        # Greedy selection: sort by norm, then add if compatible
        # A point p can be added to set S if:
        #   min_{q in S} dist(p,q)^2 >= max(max_norm(S), norm(p))
        # Since we process in order of increasing norm, max_norm is non-decreasing

        for restart in range(20):
            if time.time() > deadline - 5:
                break

            # Random permutation of candidates
            order = list(range(n))
            rng.shuffle(order)

            selected = []
            selected_norms = []
            min_dist_sq = float('inf')  # track the current minimum pairwise distance among selected

            for idx in order:
                if time.time() > deadline - 5:
                    break

                p = cands[idx]
                pn = norms[idx]

                # Check against all selected points
                compatible = True
                current_max_norm = max(selected_norms) if selected else 0
                required_min_dist = max(min_dist_sq if min_dist_sq != float('inf') else 0, current_max_norm, pn)

                for sp in selected:
                    d_sq = _squared_dist(p, sp)
                    if d_sq < required_min_dist:
                        compatible = False
                        break
                    if d_sq < min_dist_sq:
                        min_dist_sq = d_sq

                if compatible:
                    selected.append(p)
                    selected_norms.append(pn)

            if len(selected) > best_k:
                best_k = len(selected)
                best_points = np.array(selected, dtype=np.int64)

    # Also try the pure weight-2 solution as a fallback
    if best_points is None:
        best_points = base_points
        best_k = 220

    return np.array(best_points, dtype=np.int64)

# EVOLVE-BLOCK-END
