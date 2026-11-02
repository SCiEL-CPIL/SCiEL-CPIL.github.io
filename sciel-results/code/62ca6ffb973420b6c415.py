"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog


def _autoconv(x):
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f), n=2 * n)
    return ac[:n]


def _score(x):
    ac = _autoconv(x)
    return np.max(np.abs(ac[1:]))


def construct():
    t0 = time.time()
    deadline = 110.0

    # Start from a good continuous initial guess
    x = np.ones(N)
    # Add a sinusoidal perturbation to break symmetry
    x += 0.3 * np.cos(2 * np.pi * 2 * np.arange(N) / N)

    best_x = x.copy()
    best_score = _score(x)

    # Subgradient method to minimize max |ac_k| for k != 0
    # f(x) = max_{k in {1..N-1}} |ac_k(x)|
    # subgradient: at k*, g = sign(ac_{k*}) * d ac_{k*} / dx
    # d ac_k / dx = x[k:] + x[:N-k] (shifted) + x[:N-k] (reversed shift) ... actually
    # ac_k = sum_{i} x_i x_{i+k}
    # d ac_k / dx_j = x_{j+k} (if j+k < N) + x_{j-k} (if j-k >= 0)

    t_max = 20000
    for t in range(1, t_max + 1):
        if time.time() > deadline:
            break

        ac = _autoconv(x)
        # Find argmax |ac_k| for k in 1..N-1
        k_star = np.argmax(np.abs(ac[1:])) + 1
        ac_k = ac[k_star]
        sign_ac = np.sign(ac_k)

        # Compute subgradient: g[j] = sign_ac * (x[j+k] if j+k<N else 0) + sign_ac * (x[j-k] if j-k>=0 else 0)
        g = np.zeros(N)
        if k_star < N:
            g[:N - k_star] += sign_ac * x[k_star:]
            g[k_star:] += sign_ac * x[:N - k_star]

        # Step size: use diminishing step
        eta = 1.0 / np.sqrt(t + 1)

        # Update x
        x = x - eta * g
        # Project onto sum = 1 (normalize)
        x = x + (1.0 - np.sum(x)) / N

        # Round to signed steps: use a soft rounding or just keep continuous
        # Actually, the task says "signed step heights" - so x should be discrete?
        # But the seed returns np.ones(N) which are all +1. So heights are just values.
        # Let's keep continuous for optimization, then round at the end if needed.
        # Actually, let's check: the score is max |autoconv| / (sum x)^2? No, the score is
        # "maximum absolute autoconvolution divided by squared integral".
        # So score = max_{k!=0} |ac_k| / (sum(x))^2
        # We need to minimize this ratio.

        # Recompute score with normalization
        cur_score = _score(x) / (np.sum(x))**2 if np.sum(x) != 0 else np.inf
        if cur_score < best_score:
            best_score = cur_score
            best_x = x.copy()

    # Final: normalize to have sum = 1, then round if appropriate
    # The task asks for "signed step heights" - let's return the optimized continuous values
    # normalized so that sum = 1
    final_x = best_x.copy()
    if np.sum(final_x) != 0:
        final_x = final_x / np.sum(final_x)

    return final_x
# EVOLVE-BLOCK-END
