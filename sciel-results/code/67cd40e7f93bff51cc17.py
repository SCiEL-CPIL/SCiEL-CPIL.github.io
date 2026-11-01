"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

N = 400
DEADLINE = 120.0
TIME_MARGIN = 10.0


def _init_R_positive(x, N):
    """Compute R_pos[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k] for k=0..N-1."""
    R_pos = np.zeros(N)
    for k in range(N):
        R_pos[k] = np.dot(x[:N-k], x[k:])
    return R_pos


def _compute_c3_from_R_positive(R_pos, N):
    """C3 = max_{k=1..N-1} |R_pos[k]| / R_pos[0]."""
    R0 = R_pos[0]
    if R0 == 0:
        return 1.0
    max_sidelobe = np.max(np.abs(R_pos[1:]))
    return max_sidelobe / R0


def _update_R_positive_for_flip(R_pos, x, j, N):
    """Update R_pos in place after flipping x[j]."""
    for k in range(1, N):
        delta = 0.0
        if j < N - k:
            delta += -2.0 * x[j] * x[j + k]
        if j >= k:
            delta += -2.0 * x[j - k] * x[j]
        R_pos[k] += delta


def _local_search(x, max_iter=1000000, time_limit=100.0):
    """Local search with random flips and incremental R updates."""
    start_time = time.time()
    N = len(x)
    x = x.copy()
    R_pos = _init_R_positive(x, N)
    current_c3 = _compute_c3_from_R_positive(R_pos, N)
    best_x = x.copy()
    best_c3 = current_c3
    no_improve = 0
    max_no_improve = 50000

    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break

        j = random.randint(0, N - 1)

        old_R = R_pos.copy()
        _update_R_positive_for_flip(R_pos, x, j, N)
        x[j] = -x[j]

        new_c3 = _compute_c3_from_R_positive(R_pos, N)

        if new_c3 < current_c3 - 1e-12:
            current_c3 = new_c3
            no_improve = 0
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_x = x.copy()
        else:
            R_pos[:] = old_R
            x[j] = -x[j]
            no_improve += 1
            if no_improve > max_no_improve:
                break

    return best_x, best_c3


def _simulated_annealing(x, max_iter=500000, T_start=1.0, T_end=0.001, time_limit=100.0):
    """Simulated annealing with incremental R updates."""
    start_time = time.time()
    N = len(x)
    x = x.copy()
    R_pos = _init_R_positive(x, N)
    current_c3 = _compute_c3_from_R_positive(R_pos, N)
    best_x = x.copy()
    best_c3 = current_c3

    for iteration in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if time.time() > DEADLINE - TIME_MARGIN:
            break

        T = T_start * (T_end / T_start) ** (iteration / max_iter)

        j = random.randint(0, N - 1)

        old_R = R_pos.copy()
        _update_R_positive_for_flip(R_pos, x, j, N)
        x[j] = -x[j]

        new_c3 = _compute_c3_from_R_positive(R_pos, N)

        delta = new_c3 - current_c3
        if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
            current_c3 = new_c3
            if current_c3 < best_c3:
                best_c3 = current_c3
                best_x = x.copy()
        else:
            R_pos[:] = old_R
            x[j] = -x[j]

    return best_x, best_c3


def construct():
    """Entry point: return a sequence of signed step heights (±1) of length N=400."""
    start_time = time.time()
    random.seed(42)
    np.random.seed(42)

    best_x = None
    best_c3 = float('inf')

    for trial in range(8):
        if time.time() > DEADLINE - TIME_MARGIN:
            break

        x_init = np.random.choice([-1, 1], size=N)

        remaining = DEADLINE - TIME_MARGIN - (time.time() - start_time)
        if remaining < 5:
            break

        ls_time = min(remaining * 0.35, 25)
        x1, c3_1 = _local_search(x_init, max_iter=500000, time_limit=ls_time)
        if c3_1 < best_c3:
            best_c3 = c3_1
            best_x = x1.copy()

        if time.time() > DEADLINE - TIME_MARGIN:
            break

        remaining = DEADLINE - TIME_MARGIN - (time.time() - start_time)
        if remaining < 5:
            break

        sa_time = min(remaining * 0.35, 25)
        x2, c3_2 = _simulated_annealing(x1, max_iter=300000, T_start=0.5, T_end=0.001, time_limit=sa_time)
        if c3_2 < best_c3:
            best_c3 = c3_2
            best_x = x2.copy()

    if best_x is None:
        best_x = np.ones(N)

    return np.sign(best_x).astype(int)
# EVOLVE-BLOCK-END
