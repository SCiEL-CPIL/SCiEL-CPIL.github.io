# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _build_idx():
    """Precompute integer indices for the four projections."""
    X, Y = np.meshgrid(np.arange(10), np.arange(10))  # X[y,x]=x, Y[y,x]=y
    idx_0 = X.astype(np.int64).ravel()                 # x
    idx_1 = (X + Y).astype(np.int64).ravel()           # x + y
    idx_2 = (X + 2 * Y).astype(np.int64).ravel()       # x + 2y
    idx_m1 = (X - Y + 9).astype(np.int64).ravel()      # x - y (shifted by 9 to be non-negative)
    return idx_0, idx_1, idx_2, idx_m1


def _softmax(z):
    """Numerically stable softmax."""
    z2 = z - np.max(z)
    e = np.exp(z2)
    return e / e.sum()


def _entropy_and_grad(q):
    """
    Compute Shannon entropy in bits and its gradient w.r.t. q.
    q is a 1D array of probabilities (may have zeros).
    Returns (H, grad_q).
    """
    mask = q > 1e-300
    H = -np.sum(q[mask] * np.log2(q[mask]))
    grad_q = np.zeros_like(q)
    # d/dq_i [-q_i log2(q_i)] = -(log2(q_i) + 1/ln(2))
    grad_q[mask] = -(np.log2(q[mask]) + 1.0 / np.log(2.0))
    return H, grad_q


def _make_objective(idx_0, idx_1, idx_2, idx_m1):
    """
    Build the objective function and its gradient for L-BFGS-B.
    We maximize D / M where:
      D = H_{-1} (entropy of x - y distribution)
      M = max(H_0, H_1, H_2)
    
    We minimize -D / M.
    """
    n0 = idx_0.max() + 1  # 10
    n1 = idx_1.max() + 1  # 19
    n2 = idx_2.max() + 1  # 27
    nm1 = idx_m1.max() + 1  # 19

    def _softmax_grad(p, dp):
        """
        Compute dp/dz @ dp where dp is dH/dp (shape 100).
        dp/dz = p * (I - p^T) in vector form:
        (dp/dz @ dp)_i = p_i * (dp_i - sum_j p_j * dp_j)
        """
        s = np.dot(p, dp)
        return p * (dp - s)

    def objective(z):
        """
        z: 100-dimensional vector of log-probabilities (unnormalized logits).
        Returns (value, grad_z).
        """
        p = _softmax(z)  # 100 probabilities, p[y, x] flattened

        # Compute projected distributions
        q0 = np.bincount(idx_0, weights=p, minlength=n0)
        q1 = np.bincount(idx_1, weights=p, minlength=n1)
        q2 = np.bincount(idx_2, weights=p, minlength=n2)
        qm1 = np.bincount(idx_m1, weights=p, minlength=nm1)

        H0, g0 = _entropy_and_grad(q0)
        H1, g1 = _entropy_and_grad(q1)
        H2, g2 = _entropy_and_grad(q2)
        Hm1, gm1 = _entropy_and_grad(qm1)

        D = Hm1
        M = max(H0, H1, H2)

        if M < 1e-12:
            return (0.0, np.zeros_like(z))

        ratio = D / M
        value = -ratio

        # Determine which H is the max
        Hs = np.array([H0, H1, H2])
        k_max = np.argmax(Hs)
        gHs = [g0, g1, g2]
        gM = gHs[k_max]

        # dD/dz: chain rule through softmax
        dHm1_dp = gm1[idx_m1]  # shape (100,)
        dHm1_dz = _softmax_grad(p, dHm1_dp)

        # dM/dz
        if k_max == 0:
            dM_dz = _softmax_grad(p, gM[idx_0])
        elif k_max == 1:
            dM_dz = _softmax_grad(p, gM[idx_1])
        else:
            dM_dz = _softmax_grad(p, gM[idx_2])

        grad_ratio = (1.0 / M) * dHm1_dz - (D / (M * M)) * dM_dz
        grad_z = -grad_ratio  # because we minimize -ratio

        return (value, grad_z)

    return objective


def _structured_initializations(n=10, n_struct=10):
    """
    Generate structured initializations that concentrate mass on diagonals
    or specific patterns to spread x-y while keeping other projections tight.
    """
    inits = []
    X, Y = np.meshgrid(np.arange(n), np.arange(n))
    
    for d in range(-9, 10):
        # Concentrate on diagonal x - y = d
        mask = (X - Y) == d
        z = np.full((n, n), -10.0)
        z[mask] = 0.0
        inits.append(z.ravel())
    
    # Also try anti-diagonals and other patterns
    for s in range(0, 18):
        mask = (X + Y) == s
        z = np.full((n, n), -10.0)
        z[mask] = 0.0
        inits.append(z.ravel())
    
    # Mixed: spread on x-y but concentrated on x
    for x_val in range(n):
        mask = (X == x_val)
        z = np.full((n, n), -10.0)
        z[mask] = 0.0
        inits.append(z.ravel())
    
    # Random structured: random diagonals
    rng = np.random.default_rng(123)
    for _ in range(n_struct):
        z = np.full((n, n), -10.0)
        # Pick a random diagonal offset
        d = rng.integers(-9, 10)
        mask = (X - Y) == d
        # Add some noise
        z[mask] = rng.normal(0, 0.5, size=n)
        # Add small mass elsewhere
        z[~mask] = -5.0 + rng.normal(0, 0.1, size=n*n)
        inits.append(z.ravel())
    
    return inits


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya entropy ratio on a 10x10 grid.
    
    Returns a 10x10 array P of finite weights, P[y, x] for integers x, y in 0..9.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # 20s safety margin before 600s

    idx_0, idx_1, idx_2, idx_m1 = _build_idx()
    objective = _make_objective(idx_0, idx_1, idx_2, idx_m1)

    n = 10
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    z_uniform = np.log(P_uniform.ravel())
    
    # Evaluate baseline
    val_base, _ = objective(z_uniform)
    best_val = val_base
    best_z = z_uniform.copy()

    # Generate structured initializations
    struct_inits = _structured_initializations()
    
    # Evaluate all structured initializations and keep top ones
    struct_scores = []
    for z_init in struct_inits:
        try:
            v, _ = objective(z_init)
            struct_scores.append((v, z_init))
        except Exception:
            pass
    
    # Sort by score (we're minimizing, so lower is better)
    struct_scores.sort(key=lambda x: x[0])
    top_struct = [z for _, z in struct_scores[:20]]  # Keep top 20

    # Multiple random restarts with L-BFGS-B
    rng = np.random.default_rng(42)
    n_restarts = 40
    max_iter_per_restart = 300

    all_starts = [z_uniform] + top_struct
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break

        # Choose initialization
        if i < len(all_starts):
            z0 = all_starts[i].copy()
        elif i % 2 == 0:
            # Perturb around best found so far
            z0 = best_z + rng.normal(0, 1.0, size=100)
        else:
            # Random initialization
            z0 = rng.normal(0, 2.0, size=100)

        try:
            res = minimize(
                objective,
                z0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter_per_restart, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_z = res.x.copy()
        except Exception:
            pass

    # Convert best_z back to probability distribution
    P_best = _softmax(best_z).reshape((n, n))

    # Ensure validity: clip negative, normalize
    P_best = np.maximum(P_best, 0.0)
    total = P_best.sum()
    if total > 1e-9:
        P_best = P_best / total
    else:
        P_best = P_uniform

    return P_best


# EVOLVE-BLOCK-END
