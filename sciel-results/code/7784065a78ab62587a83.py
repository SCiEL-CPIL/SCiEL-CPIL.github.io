"""Logarithmic energy minimization for 150 points on the sphere."""
import time
import numpy as np

N = 150
_DEADLINE = time.time() + 115.0  # safety margin before the 120s kill
_EPS = 1e-12


def _energy(x):
    """Sum_{i<j} log(1/|xi - xj|) with a small epsilon guard."""
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    return np.log(1.0 / d).sum() / 2.0


def _grad(x):
    """Gradient of the energy w.r.t. x (same shape as x)."""
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, 1.0)
    diff = x[None, :, :] - x[:, None, :]
    return (diff / d[:, :, None]).sum(axis=1)


def _project(x):
    """Project rows back onto the unit sphere."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, _EPS)


def _fibonacci():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random():
    v = np.random.randn(N, 3)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _momentum(x, lr, beta, iters, max_step=0.2):
    """Momentum gradient descent on the sphere with step clipping."""
    v = np.zeros_like(x)
    for _ in range(iters):
        g = _grad(x)
        v = beta * v - lr * g
        x = x + np.tanh(v)          # clipped, bounded update
        x = _project(x)
    return x


def construct():
    np.random.seed(12345)
    best = None
    best_e = np.inf

    starts = [_fibonacci()] + [_random() for _ in range(11)]

    for s in starts:
        if time.time() > _DEADLINE:
            break
        x = s.copy()
        # annealed learning-rate schedule over several momentum phases
        lr = 0.02
        for phase in range(6):
            if time.time() > _DEADLINE:
                break
            x = _momentum(x, lr, 0.9, 300)
            lr *= 0.5

        e = _energy(x)
        if e < best_e:
            best_e, best = e, x.copy()

    # final high-precision polish on the best configuration
    if best is not None and time.time() < _DEADLINE - 3.0:
        x = best.copy()
        lr = 0.002
        for _ in range(4):
            if time.time() > _DEADLINE:
                break
            x = _momentum(x, lr, 0.9, 200)
            lr *= 0.5
        e = _energy(x)
        if e < best_e:
            best_e, best = e, x

    return _project(best)
