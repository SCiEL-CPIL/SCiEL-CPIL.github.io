# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: evaluator kills after 630s wall-clock. We use a generous safety margin.
_DEADLINE = 620.0


def _shannon_entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a 1D array of non-negative counts/probabilities.
    Zeros are ignored. If all are zero or sum is ~0, return 0."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 1e-15:
        return 0.0
    p = counts / total
    p = p[p > 1e-15]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def _compute_D_M(P):
    """Given a 10x10 array P[y, x] (rows y, cols x), compute D = H(x - y) and
    M = max(H(x), H(x+y), H(x+2y)). Returns (D, M)."""
    P = np.asarray(P, dtype=np.float64)
    # Clip negatives
    P = np.maximum(P, 0.0)
    total = P.sum()
    if total <= 1e-15:
        return 0.0, 0.0
    P = P / total

    n = P.shape[0]
    xs = np.arange(n)
    ys = np.arange(n)
    X, Y = np.meshgrid(xs, ys, indexing='xy')  # X[i,j] = i, Y[i,j] = j? 
    # Wait: P[y, x]. So row index is y, col index is x.
    # Let's be explicit:
    # P[y, x] where y in 0..9 (rows), x in 0..9 (cols)
    # So for a cell (y, x):
    #   value for slope r: x + r*y

    # Compute projections:
    # D: x - y
    # H0: x
    # H1: x + y
    # H2: x + 2y

    # Create arrays of values for each projection
    # For D: values = x - y, ranging from -9 to 9
    # For H0: values = x, ranging from 0 to 9
    # For H1: values = x + y, ranging from 0 to 18
    # For H2: values = x + 2y, ranging from 0 to 27

    # We can compute the marginal distribution for each projection by summing P over cells with same value.

    # D: x - y
    d_vals = np.zeros((n, n))
    for y in range(n):
        for x in range(n):
            d_vals[y, x] = x - y
    # Map x - y to index: x - y ranges from -9 to 9, so add 9 to get 0..18
    d_counts = np.zeros(19)
    for y in range(n):
        for x in range(n):
            idx = x - y + 9
            d_counts[idx] += P[y, x]
    D = _shannon_entropy_from_counts(d_counts)

    # H0: x
    h0_counts = np.zeros(n)
    for y in range(n):
        for x in range(n):
            h0_counts[x] += P[y, x]
    H0 = _shannon_entropy_from_counts(h0_counts)

    # H1: x + y
    h1_counts = np.zeros(2 * n - 1)
    for y in range(n):
        for x in range(n):
            h1_counts[x + y] += P[y, x]
    H1 = _shannon_entropy_from_counts(h1_counts)

    # H2: x + 2y
    h2_counts = np.zeros(2 * n + 1)  # max x + 2y = 9 + 18 = 27, so 28 bins
    for y in range(n):
        for x in range(n):
            h2_counts[x + 2 * y] += P[y, x]
    H2 = _shannon_entropy_from_counts(h2_counts)

    M = max(H0, H1, H2)
    return D, M


def _objective(params, n=10):
    """Objective function for optimization. We want to maximize D/M.
    We parameterize P via log-weights: params are log(p_yx) for all 100 cells.
    P = exp(params) / sum(exp(params))
    Objective to minimize: -D/M (since scipy minimizes)
    Also add a small penalty if M is too small (to avoid division by zero issues)."""
    # Convert params to P
    # Use log-sum-exp for numerical stability
    log_P = params - np.logaddexp.reduce(params)  # Actually, let's do it properly
    max_log = np.max(params)
    exp_params = np.exp(params - max_log)
    P = exp_params / exp_params.sum()
    P = P.reshape(n, n)

    D, M = _compute_D_M(P)
    if M < 1e-9:
        # Penalize heavily
        return 1e6
    return -D / M


def _objective_and_grad(params, n=10):
    """Objective and gradient via finite differences for L-BFGS-B.
    Actually, let's just use finite differences since analytic gradient is complex."""
    return _objective(params, n)


def _make_initial_params(n, structure='uniform'):
    """Create initial log-weight parameters."""
    if structure == 'uniform':
        return np.full(n * n, -np.log(n * n))
    elif structure == 'anti_diagonal':
        # Concentrate on cells where x - y is constant (say 0)
        params = np.full(n * n, -10.0)
        for y in range(n):
            x = y
            params[y * n + x] = 0.0
        return params
    elif structure == 'two_anti_diagonals':
        # Concentrate on two anti-diagonals
        params = np.full(n * n, -10.0)
        for y in range(n):
            for x in [y, (y + 1) % n]:
                params[y * n + x] = 0.0
        return params
    elif structure == 'corner':
        # Concentrate on corners
        params = np.full(n * n, -10.0)
        for y in [0, n - 1]:
            for x in [0, n - 1]:
                params[y * n + x] = 0.0
        return params
    else:
        return np.full(n * n, -np.log(n * n))


def _try_optimize(params0, n=10, maxiter=500, time_limit=60):
    """Run L-BFGS-B optimization from given initial params. Returns best P and score."""
    start_time = time.time()
    best_score = -1e10
    best_P = None

    try:
        result = minimize(
            _objective,
            params0,
            method='L-BFGS-B',
            args=(n,),
            options={'maxiter': maxiter, 'maxfun': 10000}
        )
        elapsed = time.time() - start_time
        if result.success or elapsed < time_limit:
            P = np.exp(result.x - np.max(result.x))
            P = P / P.sum()
            P = P.reshape(n, n)
            D, M = _compute_D_M(P)
            if M > 1e-9:
                score = D / M
                if score > best_score:
                    best_score = score
                    best_P = P
    except Exception:
        pass

    return best_P, best_score


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    n = 10
    global _DEADLINE

    # Start with uniform as baseline
    P_uniform = np.ones((n, n), dtype=np.float64) / (n * n)
    D_u, M_u = _compute_D_M(P_uniform)
    best_score = D_u / M_u if M_u > 1e-9 else 0.0
    best_P = P_uniform.copy()

    # Try multiple initializations
    initializations = [
        'uniform',
        'anti_diagonal',
        'two_anti_diagonals',
        'corner',
    ]

    # Add some random initializations
    rng = np.random.default_rng(42)
    for _ in range(5):
        random_params = rng.normal(0, 1, n * n)
        initializations.append(('random', random_params))

    for init in initializations:
        if time.time() > _DEADLINE - 60:
            break

        if isinstance(init, tuple):
            name, params0 = init
        else:
            name = init
            params0 = _make_initial_params(n, name)

        P_candidate, score = _try_optimize(params0, n=n, maxiter=300, time_limit=50)

        if P_candidate is not None and score > best_score:
            best_score = score
            best_P = P_candidate.copy()

    # Ensure validity: clip negatives, normalize
    best_P = np.maximum(best_P, 0.0)
    total = best_P.sum()
    if total > 1e-15:
        best_P = best_P / total
    else:
        best_P = P_uniform.copy()

    return best_P


# EVOLVE-BLOCK-END
