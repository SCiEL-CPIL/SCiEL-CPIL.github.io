"""Low-discrepancy signs on all arithmetic progressions, n=256.

Strategy:
- Start from a Thue-Morse seed (known low-discrepancy structure) plus a few
  structured/random restarts.
- Use a NumPy vectorized evaluator that computes the maximum absolute prefix
  sum over every (start, step) AP in one batched operation. This is far faster
  than the pure-Python while-loop evaluator, so we can run many more
  constraint-directed simulated-annealing steps in the same wall-clock budget.
- Keep the constraint-directed move: with high probability flip an index that
  lies on the currently worst AP; otherwise flip randomly to escape.
"""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before the 60s kill

    # --- Vectorized evaluator ------------------------------------------------
    # For every step d in [1, n] and start a in [0, n), compute the prefix
    # sums of signs[a], signs[a+d], signs[a+2d], ... and take the max abs.
    # We do this in a batched NumPy op: build a matrix M of shape (n, n) where
    # M[a, k] = signs[a + k*d] if a + k*d < n else 0, but that depends on d.
    # Simpler: for each d, build an (n, n) matrix of the "d-stride" sequences
    # starting at each a, cumsum along the length axis, take max abs.
    # Total cost: sum_d n * (n/d) ~ n^2 * H_n ~ 400k adds, done in NumPy.

    def max_discrepancy(signs_np, return_argmax=False):
        """signs_np: int8 array of length n with values in {-1, +1}."""
        max_disc = 0
        best_start = 0
        best_step = 1
        for d in range(1, n + 1):
            # For each start a in [0, n), the sequence is signs[a], signs[a+d], ...
            # Length is floor((n - 1 - a) / d) + 1.
            # Build matrix of shape (n, max_len) where max_len = ceil(n/d).
            max_len = (n - 1) // d + 1
            # indices: row a, col k -> a + k*d, valid if < n
            cols = np.arange(max_len)
            rows = np.arange(n)[:, None]
            idx = rows + d * cols[None, :]
            valid = idx < n
            idx_clipped = np.clip(idx, 0, n - 1)
            M = signs_np[idx_clipped]  # shape (n, max_len)
            M[~valid] = 0
            prefix = np.cumsum(M, axis=1)
            row_max = np.max(np.abs(prefix), axis=1)
            # Only consider starts a where the AP has at least one element (always true)
            # but the "true" max for start a should ignore the padded zeros.
            # Since we zeroed invalid entries, cumsum after the last valid element
            # stays at the final sum, which is fine (it's a valid prefix sum).
            # Actually the padded zeros don't change the prefix sums after the last
            # valid element, so row_max is correct.
            d_max = int(np.max(row_max))
            if d_max > max_disc:
                max_disc = d_max
                # Find the start that achieves it
                best_start = int(np.argmax(row_max))
                best_step = d
        if return_argmax:
            return max_disc, best_start, best_step
        return max_disc

    # --- Initializations -----------------------------------------------------
    def thue_morse(n):
        s = np.zeros(n, dtype=np.int8)
        s[0] = 1
        for i in range(1, n):
            if i % 2 == 0:
                s[i] = s[i // 2]
            else:
                s[i] = -s[i // 2]
        return s

    rng = random.Random(42)

    inits = []
    inits.append(thue_morse(n))
    # Alternating
    alt = np.ones(n, dtype=np.int8)
    alt[1::2] = -1
    inits.append(alt)
    # Period-3 pattern
    p3 = np.ones(n, dtype=np.int8)
    p3[2::3] = -1
    inits.append(p3)
    # A few random
    for _ in range(3):
        inits.append(np.array([rng.choice([-1, 1]) for _ in range(n)], dtype=np.int8))

    best_signs = None
    best_disc = float('inf')

    for init in inits:
        d = max_discrepancy(init)
        if d < best_disc:
            best_disc = d
            best_signs = init.copy()

    # --- Constraint-directed Simulated Annealing -----------------------------
    current = best_signs.copy()
    current_disc = best_disc
    _, cur_start, cur_step = max_discrepancy(current, return_argmax=True)

    temperature = 4.0
    cooling_rate = 0.99985
    min_temp = 0.05
    stall_counter = 0
    max_stall = 8000

    while time.time() < deadline - 1.0:
        # Choose index to flip
        if rng.random() < 0.7 and cur_step > 0:
            # Flip an index on the current worst AP
            ap_indices = list(range(cur_start, n, cur_step))
            if ap_indices:
                idx = ap_indices[rng.randint(0, len(ap_indices) - 1)]
            else:
                idx = rng.randint(0, n - 1)
        else:
            idx = rng.randint(0, n - 1)

        current[idx] = -current[idx]
        new_disc, new_start, new_step = max_discrepancy(current, return_argmax=True)
        delta = new_disc - current_disc

        if delta < 0 or (delta > 0 and rng.random() < math.exp(-delta / temperature)):
            current_disc = new_disc
            cur_start = new_start
            cur_step = new_step
            if new_disc < best_disc:
                best_disc = new_disc
                best_signs = current.copy()
                stall_counter = 0
            else:
                stall_counter += 1
        else:
            current[idx] = -current[idx]  # revert
            stall_counter += 1

        temperature = max(min_temp, temperature * cooling_rate)
        if stall_counter >= max_stall:
            # Restart from best with a small perturbation
            current = best_signs.copy()
            for _ in range(rng.randint(3, 8)):
                current[rng.randint(0, n - 1)] *= -1
            current_disc, cur_start, cur_step = max_discrepancy(current, return_argmax=True)
            temperature = 3.0
            stall_counter = 0

    # Return as a list of Python ints
    return [int(x) for x in best_signs]

# EVOLVE-BLOCK-END
