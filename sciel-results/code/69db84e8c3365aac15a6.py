import numpy as np
from scipy.optimize import minimize
import time

N = 100

def compute_overlaps(h):
    overlaps = np.zeros(N - 1)
    for k in range(1, N):
        shifted = np.roll(h, -k)
        overlaps[k - 1] = np.sum(h * (1.0 - shifted)) * (2.0 / N)
    return overlaps

def max_overlap(h):
    return float(np.max(compute_overlaps(h)))

def objective_with_penalty(params):
    h = np.clip(params, 0.0, 1.0)
    sum_val = np.sum(h)
    target = N / 2.0
    penalty = 1e4 * (sum_val - target) ** 2
    return max_overlap(h) + penalty

def project_to_sum(h, target_sum):
    h = np.clip(h, 0.0, 1.0)
    residual = target_sum - np.sum(h)
    if abs(residual) < 1e-14:
        return h
    if residual > 0:
        idx = np.argsort(h)[::-1]
        for i in idx:
            add = min(residual, 1.0 - h[i])
            h[i] += add
            residual -= add
            if abs(residual) < 1e-14:
                break
    else:
        idx = np.argsort(h)
        for i in idx:
            sub = min(-residual, h[i])
            h[i] -= sub
            residual += sub
            if abs(residual) < 1e-14:
                break
    return h

def construct():
    start_time = time.time()
    deadline = 115.0

    best_h = np.full(N, 0.5)
    best_score = max_overlap(best_h)

    rng = np.random.default_rng(12345)
    inits = []

    # Alternating
    h = np.zeros(N); h[::2] = 1.0; inits.append(h)

    # Blocks of 10
    inits.append(np.tile([1.0] * 10 + [0.0] * 10, N // 20))

    # Random binaries
    for _ in range(15):
        h = (rng.random(N) < 0.5).astype(float)
        ones = int(np.sum(h))
        if ones > 50:
            idx = np.where(h == 1)[0]
            h[rng.choice(idx, size=ones - 50, replace=False)] = 0
        elif ones < 50:
            idx = np.where(h == 0)[0]
            h[rng.choice(idx, size=50 - ones, replace=False)] = 1
        inits.append(h)

    # Cosine threshold
    x = np.linspace(0, 2 * np.pi, N, endpoint=False)
    h = (np.sin(x) > 0).astype(float)
    ones = int(np.sum(h))
    if ones > 50:
        idx = np.where(h == 1)[0]
        h[rng.choice(idx, size=ones - 50, replace=False)] = 0
    elif ones < 50:
        idx = np.where(h == 0)[0]
        h[rng.choice(idx, size=50 - ones, replace=False)] = 1
    inits.append(h)

    for h0 in inits:
        if time.time() - start_time > deadline:
            break
        res = minimize(objective_with_penalty, h0, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-14})
        h_opt = project_to_sum(res.x, N / 2.0)
        s = max_overlap(h_opt)
        if s < best_score:
            best_score = s
            best_h = h_opt.copy()

    best_h = project_to_sum(best_h, N / 2.0)
    return best_h
