"""Rewritten program: dense incidence matrix + multi-restart simulated annealing."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import floor

N = 256
DEADLINE_MARGIN = 2.0  # seconds of safety margin before the 90s kill

def _build_incidence():
    """Build the binary incidence matrix for all APs in [1..256].
    Rows = APs, columns = positions 0..255.
    Entry (r, c) = 1 if position c+1 is in AP r.
    """
    rows = []
    for start in range(1, N + 1):
        max_step = N - start + 1
        for step in range(1, max_step + 1):
            # length of this AP
            length = (N - start) // step + 1
            # skip singletons? No, task says include singletons.
            # But singletons have sum = sign[start-1], which is always +/-1.
            # Including them adds N rows that are just unit vectors.
            # We'll include them for completeness but they won't be the bottleneck.
            if length < 1:
                continue
            row = np.zeros(N, dtype=np.float64)
            for k in range(length):
                pos = start + k * step - 1  # 0-indexed
                if pos < N:
                    row[pos] = 1.0
            rows.append(row)
    return np.array(rows, dtype=np.float64)

# Precompute the incidence matrix at module load time
_INC = _build_incidence()
_NUM_APS = _INC.shape[0]

def _max_abs_sum(signs):
    """Compute max absolute signed sum over all APs.
    signs: array of shape (256,) with values -1 or +1.
    """
    sums = _INC @ signs  # shape (_NUM_APS,)
    return float(np.max(np.abs(sums)))

def construct():
    t_start = time.time()
    deadline = 90.0 - DEADLINE_MARGIN

    # Multi-restart simulated annealing
    best_signs = None
    best_score = float('inf')

    # Initial random seed
    rng = np.random.default_rng(42)

    # Number of restarts we can afford; we'll decide dynamically
    max_restarts = 50
    restarts_done = 0

    for restart in range(max_restarts):
        if time.time() > deadline:
            break

        # Initialize: random signs
        if restart == 0:
            # Start with the seed program's random signs for consistency?
            # Actually, let's just use a fresh random start each restart.
            # For restart 0, use a deterministic random for reproducibility if needed,
            # but the task says minimize, so any good start is fine.
            signs = rng.choice([-1.0, 1.0], size=N)
        else:
            # Perturb the best found so far, or fresh random?
            # Fresh random for diversity, but also try perturbations of best.
            if restart % 2 == 0:
                signs = rng.choice([-1.0, 1.0], size=N)
            else:
                # Start from best known and flip a few random positions
                signs = best_signs.copy()
                num_flips = rng.integers(5, 20)
                flip_idx = rng.choice(N, size=num_flips, replace=False)
                signs[flip_idx] *= -1.0

        current_score = _max_abs_sum(signs)

        # Simulated annealing parameters
        # Temperature schedule: exponential cooling
        T0 = 5.0
        T_end = 0.01
        # We'll do a fixed number of steps per restart, but check time
        # Estimate: each step involves a matrix-vector product of 256xNUM_APS.
        # NUM_APS is large (~256*128/2 ~ 16384 + more). Let's measure.
        
        # Actually, let's do a simple SA loop with time-based termination.
        # Each iteration: flip one random position, compute new score.
        # To speed up, we can compute the change in score analytically for the flipped position.
        # But for simplicity and to use the vectorized approach, let's just recompute.
        # However, recomputing _INC @ signs for each step is O(NUM_APS * N) which is ~16384*256 ~ 4M ops per step.
        # That's too slow for many steps. Let's use an incremental update.
        
        # Incremental update:
        # When we flip position j, the change in sum for AP r is 2 * signs[j] * _INC[r, j] (before flip).
        # Actually, if we flip signs[j] from s to -s, the change in sum_r is -2*s for APs where _INC[r,j]=1.
        # So we can maintain the sums array and update it incrementally.
        
        sums = _INC @ signs  # current sums, shape (NUM_APS,)
        current_score = float(np.max(np.abs(sums)))

        # SA parameters
        T = T0
        cooling_rate = 0.999
        min_steps = 1000
        max_steps = 50000

        step_count = 0
        best_local_signs = signs.copy()
        best_local_score = current_score

        while True:
            if time.time() > deadline:
                break
            if step_count > max_steps:
                break

            step_count += 1

            # Choose a random position to flip
            j = rng.integers(0, N)
            old_val = signs[j]
            new_val = -old_val

            # Update sums incrementally
            # For all APs that include position j, the sum changes by -2 * old_val
            # _INC[:, j] is a column vector with 1s where position j is in the AP
            mask = _INC[:, j]
            if np.any(mask):
                sums[mask] -= 2.0 * old_val

            signs[j] = new_val

            new_score = float(np.max(np.abs(sums)))

            # Accept or reject
            delta = new_score - current_score
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_score = new_score
                if new_score < best_local_score:
                    best_local_score = new_score
                    best_local_signs = signs.copy()
            else:
                # Reject: undo the flip
                signs[j] = old_val
                if np.any(mask):
                    sums[mask] += 2.0 * old_val

            # Cool down
            T *= cooling_rate
            if T < T_end:
                break

        # End of SA loop for this restart
        if best_local_score < best_score:
            best_score = best_local_score
            best_signs = best_local_signs.copy()

        restarts_done += 1

    # Convert to list of Python ints -1 or +1
    result = [int(s) for s in best_signs]
    return result

# EVOLVE-BLOCK-END
