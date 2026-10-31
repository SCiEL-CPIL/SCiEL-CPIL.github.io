import numpy as np
from scipy.optimize import linprog
import time

N = 26

def heuristic_sum(centers):
    x = centers[:, 0]
    y = centers[:, 1]
    B = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    d = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d, axis=1)
    r = np.minimum(B, min_d / 2.0)
    return np.sum(r)

def exact_sum(centers):
    n = len(centers)
    c = -np.ones(n)
    bounds = [(0, None) for _ in range(n)]
    x = centers[:, 0]
    y = centers[:, 1]
    B = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    n_boundary = n
    n_overlap = n * (n - 1) // 2
    total_constraints = n_boundary + n_overlap
    A_ub = np.zeros((total_constraints, n))
    b_ub = np.zeros(total_constraints)
    for i in range(n):
        A_ub[i, i] = 1.0
        b_ub[i] = B[i]
    idx = n_boundary
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx**2 + dy**2)
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun
        else:
            return 0.0
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    time_limit = 110.0
    best_sum = -1.0
    best_centers = None

    # Phase 1: Fast SA
    T_start = 0.01
    T_end = 0.0001
    max_steps = 20000
    num_restarts = 10

    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
        centers = np.random.uniform(0.1, 0.9, (N, 2))
        current_sum = heuristic_sum(centers)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
        T = T_start
        cool_rate = (T_end / T_start) ** (1.0 / max_steps)
        for step in range(max_steps):
            if time.time() - start_time > time_limit:
                break
            i = np.random.randint(N)
            step_size = 0.05
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            new_centers = centers.copy()
            new_centers[i, 0] = np.clip(centers[i, 0] + dx, 0, 1)
            new_centers[i, 1] = np.clip(centers[i, 1] + dy, 0, 1)
            new_sum = heuristic_sum(new_centers)
            delta = new_sum - current_sum
            if delta > 0 or (T > 0 and np.random.rand() < np.exp(delta / T)):
                centers = new_centers
                current_sum = new_sum
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
            T *= cool_rate

    # Phase 2: Polish
    if best_centers is not None:
        T_start_polish = 0.005
        T_end_polish = 0.0001
        max_steps_polish = 2000
        current_centers = best_centers.copy()
        current_sum = exact_sum(current_centers)
        best_sum_polish = current_sum
        best_centers_polish = current_centers.copy()
        T = T_start_polish
        cool_rate = (T_end_polish / T_start_polish) ** (1.0 / max_steps_polish)
        for step in range(max_steps_polish):
            if time.time() - start_time > time_limit:
                break
            i = np.random.randint(N)
            step_size = 0.02
            dx = np.random.normal(0, step_size)
            dy = np.random.normal(0, step_size)
            new_centers = current_centers.copy()
            new_centers[i, 0] = np.clip(current_centers[i, 0] + dx, 0, 1)
            new_centers[i, 1] = np.clip(current_centers[i, 1] + dy, 0, 1)
            new_sum = exact_sum(new_centers)
            delta = new_sum - current_sum
            if delta > 0 or (T > 0 and np.random.rand() < np.exp(delta / T)):
                current_centers = new_centers
                current_sum = new_sum
                if current_sum > best_sum_polish:
                    best_sum_polish = current_sum
                    best_centers_polish = current_centers.copy()
            T *= cool_rate
        best_centers = best_centers_polish

    return _format_result(best_centers)

def _format_result(centers):
    if centers is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    n = len(centers)
    x = centers[:, 0]
    y = centers[:, 1]
    B = np.minimum(np.minimum(x, 1-x), np.minimum(y, 1-y))
    n_boundary = n
    n_overlap = n * (n - 1) // 2
    total_constraints = n_boundary + n_overlap
    A_ub = np.zeros((total_constraints, n))
    b_ub = np.zeros(total_constraints)
    for i in range(n):
        A_ub[i, i] = 1.0
        b_ub[i] = B[i]
    idx = n_boundary
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            d = np.sqrt(dx**2 + dy**2)
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
    c = -np.ones(n)
    bounds = [(0, None) for _ in range(n)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        radii = res.x
    else:
        radii = np.zeros(n)
        for i in range(n):
            radii[i] = B[i]
            for j in range(n):
                if i != j:
                    dx = x[i] - x[j]
                    dy = y[i] - y[j]
                    d = np.sqrt(dx**2 + dy**2)
                    radii[i] = min(radii[i], d/2.0)
    result = np.column_stack((x, y, radii))
    return result

