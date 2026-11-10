# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize the number of integer grid points in {0..7}^3 such that no five are
    co-spherical (which also forbids five co-planar).

    A point p is "bad" for an accepted set S iff p lies on the sphere through
    some 4-tuple of S.  By symmetry this is exactly the constraint the evaluator
    enforces: for any 5 selected points, the point added last lies on a sphere
    through 4 earlier ones.  We therefore maintain the set incrementally and
    reject a candidate iff it sits on a sphere determined by 4 accepted points.

    Deterministic (np.random.seed(0)) and time-bounded (deadline guard).
    """
    n = 8
    deadline = time.time() + 595.0   # safety margin well under the 630s kill
    grid = np.array(list(itertools.product(range(n), range(n), range(n))),
                    dtype=np.float64)
    sq = (grid ** 2).sum(axis=1)      # |g|^2 for every grid point

    def can_add(arr, p, psq):
        """True iff adding p (row vector) keeps S (arr) sphere-free."""
        m = len(arr)
        if m < 4:
            return True
        # rows [q - p, |q|^2 - |p|^2]; p is co-spherical with any 4 of S iff
        # some 4x4 sub-determinant vanishes (integer coords => |det| >= 1).
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = sq[:m] - psq
        for c in itertools.combinations(range(m), 4):
            if abs(np.linalg.det(w[list(c)])) < 0.5:
                return False
        return True

    def greedy(order):
        """Greedy over a fixed point order; returns selected indices."""
        S = []
        for i in order:
            p = grid[i]
            if can_add(grid[S], p, sq[i]):
                S.append(i)
        return S

    def anneal(S, budget):
        """Simulated-annealing: randomly drop a point, then greedily re-add."""
        S = list(S)
        T = 1.0
        cooling = 0.999
        rng = np.random
        while time.time() < budget:
            if len(S) >= 2:
                a = int(rng.randint(len(S)))
                b = int(rng.randint(len(S)))
                if a != b:
                    S[a], S[b] = S[b], S[a]
            k = len(S)
            if k >= 2:
                drop = int(rng.randint(k))
                S.pop(drop)
            # greedy re-add from the shuffled remaining points
            rem = [i for i in order if i not in set(S)]
            rng.shuffle(rem)
            S2 = list(S)
            for i in rem:
                p = grid[i]
                if can_add(grid[S2], p, sq[i]):
                    S2.append(i)
            if len(S2) >= len(S):
                S = S2
            T *= cooling
        return S

    np.random.seed(0)
    rng = np.random

    # --- Phase 1: randomized greedy restarts (fast, cheap) ---
    best = None
    best_order = None
    while time.time() < deadline - 120.0:
        order = rng.permutation(n ** 3).astype(np.int64)
        S = greedy(order)
        if best is None or len(S) > len(best):
            best = S
            best_order = order
        if len(best) >= 45:
            break

    # --- Phase 2: simulated-annealing polish from the best greedy seed ---
    if best is not None and time.time() < deadline - 30.0:
        S = anneal(best, deadline - 30.0)
        if len(S) > len(best):
            best = S

    return np.array(best, dtype=np.int64)


# EVOLVE-BLOCK-END
