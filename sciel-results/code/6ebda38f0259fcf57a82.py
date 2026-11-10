# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy (structural change):
    - Use an exponential-decay kernel in the "step + return" cost, which is
      a different modelling family than the power-law used before.
    - Combine multiple kernel widths (a multi-scale view) via a weighted blend.
    - Include a prize-power term to let high-value nodes dominate.
    - Add a gentle origin-proximity bonus to stabilise the first move.
    - All entries are forced to be positive and finite; non-positive become 1e-9.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # safety margin before 300s kill

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)

    n = len(prizes)

    # Basic sanity: finite values and correct shape
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9
    if prizes.shape != (n,) or distance_matrix.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9

    # Cost of moving i -> j -> 0 (step plus return)
    # This is the incremental length if we visit j from i and then must return to 0.
    step_return = distance_matrix + distance_matrix[:, 0][None, :]  # shape (n, n)

    # Ensure no zero/negative costs
    eps = 1e-9
    step_return_safe = np.maximum(step_return, eps)

    # Prize column for broadcasting over rows
    p = prizes[None, :]  # shape (1, n)

    # Multi-scale exponential kernels with different decay rates (widths)
    # A smaller sigma means sharper local focus; larger sigma is more global.
    sigmas = [0.08, 0.15, 0.30, 0.50]
    weights = [0.45, 0.30, 0.15, 0.10]  # favour sharper kernels

    H_blend = np.zeros((n, n), dtype=float)

    # Prize exponent: raise prize to a power to amplify high-value nodes
    prize_exp = 1.6

    for sigma, w in zip(sigmas, weights):
        if time.time() > deadline - 5.0:
            break
        # Exponential decay kernel: exp(-cost / sigma)
        kernel = np.exp(-step_return_safe / sigma)
        # Combine with prize
        term = (p ** prize_exp) * kernel
        H_blend += w * term

    # Origin-proximity bonus: favour nodes close to the origin for early moves
    dist_to_origin = distance_matrix[:, 0]  # shape (n,)
    origin_bonus = 1.0 / (1.0 + dist_to_origin[None, :] / 0.2)  # shape (1, n), decays over ~0.2 units

    # Apply a gentle multiplicative bonus (max ~1.5x for nodes at origin)
    H_blend *= (1.0 + 0.3 * origin_bonus)

    # Normalise so that the maximum entry is 1 (keeps scale stable)
    max_val = np.max(H_blend)
    if max_val > 0 and np.isfinite(max_val):
        H_blend /= max_val

    # Time guard: if we are close to deadline, just return what we have
    if time.time() > deadline - 1.0:
        H_blend = np.where(np.isfinite(H_blend) & (H_blend > 0), H_blend, 1e-9)
        return H_blend

    # Ensure all entries are positive and finite
    H_blend = np.where(np.isfinite(H_blend) & (H_blend > 0), H_blend, 1e-9)

    # Final shape check
    if H_blend.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9

    return H_blend
# EVOLVE-BLOCK-END
