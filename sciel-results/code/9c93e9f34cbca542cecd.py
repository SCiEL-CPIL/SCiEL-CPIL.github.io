# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    N = n ** 3
    deadline = time.time() + 620

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    sq_norms = (grid ** 2).sum(axis=1)

    def lift_matrix(p, arr):
        """Return m x 4 matrix: rows (q - p, |q|^2 - |p|^2) for q in arr."""
        m = len(arr)
        if m == 0:
            return np.empty((0, 4), dtype=np.float64)
        W = np.empty((m, 4), dtype=np.float64)
        W[:, 0:3] = arr.astype(np.float64) - p.astype(np.float64)
        W[:, 3] = sq_norms[list(arr[:, 0]) * n * n + list(arr[:, 1]) * n + list(arr[:, 2])] - (p ** 2).sum()
        return W

    def count_violations_fast(p, S):
        """Count 4-tuples in S whose det < 0.5 in absolute value."""
        m = len(S)
        if m < 4:
            return 0
        W = lift_matrix(p, S)
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return 0
        sub = W[combos]
        dets = np.linalg.det(sub)
        return int(np.sum(np.abs(dets) < 0.5))

    def total_violations(S):
        """Total number of violating 5-tuples in S (each counted once, from the last-added perspective in the test, but we count all 5-sets with det < 0.5).
        Actually the test checks: for each p at index i (with at least 4 earlier points), for every 4 earlier points q1..q4, check det. 
        This is equivalent to: for each 5-subset {q1..q5}, check if any point p in the subset, when used as the "pivot" with the other 4, gives |det| < 0.5.
        But actually the determinant condition is symmetric: 5 points are co-spherical iff for ANY choice of one as p and the other 4 as q's, the 4x4 det is 0.
        So we just need to count 5-sets that are co-spherical.
        
        However, the test checks a specific ordering. Let me re-read.
        "for each point p with at least 4 earlier points, and every 4 earlier points q1..q4, form the 4x4 matrix with rows (q - p, |q|^2 - |p|^2); if |np.linalg.det| < 0.5 for any such matrix, the result is invalid."
        
        So for a valid set, for EVERY point p (in the returned order) and EVERY 4-tuple of earlier points, the det must be >= 0.5.
        This is stricter than just "no 5 co-spherical points" because the ordering matters? No, the determinant condition is symmetric. If 5 points are co-spherical, then for ANY choice of one as pivot, the det is 0. If they are not co-spherical, then for ANY choice of one as pivot, the det is non-zero (and since integer, |det| >= 1).
        
        So the condition is equivalent to: no 5 points are co-spherical. The ordering doesn't matter for the validity, only for the check procedure.
        
        So total_violations should count the number of 5-subsets that are co-spherical.
        """
        k = len(S)
        if k < 5:
            return 0
        S_idx = np.arange(k)
        count = 0
        for i in range(k):
            # For each point i, check 4-tuples from indices < i (to match the test's "earlier points")
            earlier = S_idx[:i]
            m = len(earlier)
            if m < 4:
                continue
            W = lift_matrix(S[i], S[earlier])
            combos = np.array(list(itertools.combinations(range(m), 4)))
            if len(combos) == 0:
                continue
            sub = W[combos]
            dets = np.linalg.det(sub)
            count += int(np.sum(np.abs(dets) < 0.5))
        return count

    def greedy_init():
        S = []
        for p in grid:
            if count_violations_fast(p, np.array(S)) == 0:
                S.append(p)
        return np.array(S)

    # Greedy with random restarts
    np.random.seed(0)
    best = None
    best_k = 0

    for restart in range(10):
        if time.time() > deadline - 30:
            break
        order = grid[np.random.permutation(N)]
        S = []
        for p in order:
            if time.time() > deadline - 30:
                break
            if count_violations_fast(p, np.array(S)) == 0:
                S.append(p)
        S = np.array(S)
        if len(S) > best_k:
            best_k = len(S)
            best = S

    if best is None:
        return grid[:0].astype(np.int64)

    # Simulated annealing to try to increase k
    T0 = 5.0
    T1 = 0.01
    S = best.copy()
    k = len(S)
    best_SA = S.copy()
    best_k_SA = k

    while time.time() < deadline - 10:
        # Try to increase k by 1
        target_k = k + 1
        if target_k > N:
            break

        # Start from a random set of size target_k
        S_curr = grid[np.random.choice(N, size=target_k, replace=False)]
        E_curr = total_violations(S_curr)

        for step in range(5000):
            if time.time() > deadline - 10:
                break

            # Pick a random point in S_curr to swap out
            i = np.random.randint(target_k)
            p_out = S_curr[i]

            # Pick a random point to swap in (not already in S_curr)
            p_in = grid[np.random.randint(N)]
            if np.any(np.all(S_curr == p_in, axis=1)):
                continue

            # Compute delta_E
            S_minus = np.delete(S_curr, i, axis=0)
            v_in = count_violations_fast(p_in, S_minus)
            v_out = count_violations_fast(p_out, S_minus)
            delta_E = v_in - v_out

            if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / max(T0 * (T1 / T0) ** (step / 5000), 1e-8)):
                S_curr = S_curr.copy()
                S_curr[i] = p_in
                E_curr += delta_E

                if E_curr == 0 and target_k > best_k_SA:
                    best_k_SA = target_k
                    best_SA = S_curr.copy()
                    k = target_k
                    # Reset S for next attempt
                    S = S_curr.copy()

        # If we found a valid set of size target_k, continue with that
        if E_curr == 0:
            k = target_k
            S = S_curr.copy()
            if k > best_k:
                best = S
                best_k = k

    # Final check: return the best valid set
    if best_k_SA > best_k:
        best = best_SA
        best_k = best_k_SA

    # Ensure best is valid
    if best is not None and len(best) > 0:
        if total_violations(best) == 0:
            return best.astype(np.int64)
        else:
            # Fallback: return a smaller valid subset
            S = []
            for p in best:
                if count_violations_fast(p, np.array(S)) == 0:
                    S.append(p)
            return np.array(S).astype(np.int64)

    return best.astype(np.int64) if best is not None else grid[:0].astype(np.int64)


# EVOLVE-BLOCK-END
