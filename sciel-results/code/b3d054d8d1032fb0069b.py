"""Rewritten program: histogram-based max tracking + multi-restart simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import floor

N = 256
DEADLINE_MARGIN = 2.0  # seconds of safety margin before the 90s kill

def _build_incidence():
    """Build the binary incidence matrix for all APs in [1..256]."""
    rows = []
    for start in range(1, N + 1):
        max_step = N - start + 1
        for step in range(1, max_step + 1):
            length = (N - start) // step + 1
            if length < 1:
                continue
            row = np.zeros(N, dtype=np.float64)
            for k in range(length):
                pos = start + k * step - 1
                if pos < N:
                    row[pos] = 1.0
            rows.append(row)
    return np.array(rows, dtype=np.float64)

_INC = _build_incidence()
_NUM_APS = _INC.shape[0]
_APS_PER_POS = [np.nonzero(_INC[:, j] == 1.0)[0] for j in range(N)]
_MAX_SUM = N

def _max_abs_sum(signs):
    sums = _INC @ signs
    return float(np.max(np.abs(sums)))

def construct():
    t_start = time.time()
    deadline = 90.0 - DEADLINE_MARGIN

    default_signs = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)])
    best_signs = default_signs.copy()
    best_score = _max_abs_sum(best_signs)

    rng = np.random.default_rng(42)
    max_restarts = 200

    try:
        for restart in range(max_restarts):
            if time.time() > deadline:
                break

            if restart == 0:
                signs = rng.choice([-1.0, 1.0], size=N)
            elif restart % 4 == 1:
                signs = rng.choice([-1.0, 1.0], size=N)
            elif restart % 4 == 2:
                signs = best_signs.copy()
                nf = int(rng.integers(3, 12))
                fi = rng.choice(N, size=nf, replace=False)
                signs[fi] *= -1.0
            else:
                signs = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
                nf = int(rng.integers(10, 30))
                fi = rng.choice(N, size=nf, replace=False)
                signs[fi] *= -1.0

            sums = _INC @ signs
            # Build histogram: count of APs at each |sum| level
            abs_sums = np.abs(sums).astype(np.int64)
            hist = np.bincount(abs_sums, minlength=_MAX_SUM + 1).astype(np.int64)
            
            # Track current max
            cur_max = int(np.max(abs_sums)) if _NUM_APS > 0 else 0
            current_score = float(cur_max)

            T = 4.0
            cooling_rate = 0.9997
            T_end = 0.05
            max_steps = 500000

            step_count = 0
            best_local_signs = signs.copy()
            best_local_score = current_score

            while True:
                if time.time() > deadline:
                    break
                if step_count > max_steps:
                    break

                step_count += 1

                j = int(rng.integers(0, N))
                old_val = signs[j]
                delta_change = -2.0 * old_val

                idxs = _APS_PER_POS[j]
                n_idxs = len(idxs)
                if n_idxs == 0:
                    continue

                old_sums = sums[idxs]
                new_sums = old_sums + delta_change

                old_abs = np.abs(old_sums).astype(np.int64)
                new_abs = np.abs(new_sums).astype(np.int64)

                # Update histogram
                # For each unique value in old_abs, decrement; for new_abs, increment
                u_old, c_old = np.unique(old_abs, return_counts=True)
                u_new, c_new = np.unique(new_abs, return_counts=True)
                hist[u_old] -= c_old
                hist[u_new] += c_new

                # Update sums and signs
                sums[idxs] = new_sums
                signs[j] = -old_val

                # Update cur_max
                new_max = cur_max
                # Check if any new_abs value exceeds cur_max
                if new_abs.size > 0:
                    local_max = int(np.max(new_abs))
                    if local_max > new_max:
                        new_max = local_max
                # If cur_max count dropped to 0, scan down
                if hist[cur_max] == 0:
                    # Find new max by scanning down from cur_max
                    while new_max > 0 and hist[new_max] == 0:
                        new_max -= 1
                cur_max = new_max
                new_score = float(cur_max)

                delta = new_score - current_score
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    current_score = new_score
                    if new_score < best_local_score:
                        best_local_score = new_score
                        best_local_signs = signs.copy()
                else:
                    # Reject
                    signs[j] = old_val
                    sums[idxs] = old_sums
                    # Undo histogram
                    hist[u_new] -= c_new
                    hist[u_old] += c_old
                    # cur_max may need to go back up
                    if hist[cur_max] == 0:
                        # scan up from cur_max
                        while cur_max < _MAX_SUM and hist[cur_max] == 0:
                            cur_max += 1
                    elif int(np.max(new_abs)) > cur_max:
                        # This shouldn't happen since we're undoing
                        pass
                    # Actually after undo, cur_max should be at least as large as before
                    # Let's just recompute if needed
                    if hist[cur_max] == 0:
                        while cur_max < _MAX_SUM and hist[cur_max] == 0:
                            cur_max += 1

                T *= cooling_rate
                if T < T_end:
                    break

            if best_local_score < best_score:
                best_score = best_local_score
                best_signs = best_local_signs.copy()

    except Exception:
        pass

    if best_signs is None:
        best_signs = default_signs

    result = [int(s) for s in best_signs]
    return result

# EVOLVE-BLOCK-END
