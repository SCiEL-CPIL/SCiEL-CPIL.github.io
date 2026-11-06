"""QAOA angle design with spectral initialization and BFGS local search."""
import time
import numpy as np
from scipy.optimize import minimize

# Deadline guard: must finish within 120s per evaluation call.
_DEADLINE = 120.0
_START_TIME = time.time()


def _build_adjacency(n, edges):
    """Build adjacency matrix from edge list."""
    A = np.zeros((n, n), dtype=np.float64)
    for u, v in edges:
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _expected_cut(n, edges, gammas, betas):
    """
    Compute the expected cut value for a 2-layer QAOA circuit.
    
    The circuit: |+>^n -> exp(-i*gamma1*C) -> exp(-i*beta1*X_total) -> exp(-i*gamma2*C) -> exp(-i*beta2*X_total)
    
    We compute E[cut] = sum_{(u,v) in E} P[u != v] where P is the final state probabilities.
    For efficiency with small n, we use the full state vector approach.
    """
    # Build cost matrix C (MaxCut Hamiltonian)
    # C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2  ... actually for MaxCut:
    # The standard QAOA MaxCut cost is C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # But the problem says C is unweighted MaxCut, and we apply exp(-i gamma C).
    # Let's use C = sum_{(u,v) in E} (1 - Z_u Z_v) / 2
    # Actually, let me re-read: "applies exp(-i gamma C), then exp(-i beta sum X), for each row; C is unweighted MaxCut"
    # Standard QAOA for MaxCut: H_C = sum_{(u,v) in E} (1 - Z_u Z_v)/2
    # H_B = sum_i X_i
    
    # For n up to 10, full state vector is 2^n = 1024, which is fine.
    
    # Build H_C as a matrix
    dim = 1 << n
    H_C = np.zeros((dim, dim), dtype=np.complex128)
    
    for u, v in edges:
        # (1 - Z_u Z_v) / 2
        # Z_u Z_v has eigenvalue +1 if bits u,v same, -1 if different
        # So (1 - Z_u Z_v)/2 has eigenvalue 0 if same, 1 if different
        # This is the projector onto u != v
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                H_C[state, state] += 0.5
    
    # Build H_B = sum_i X_i
    H_B = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_B[state, flipped] += 1.0
    
    # Initial state: |+>^n = (1/sqrt(2^n)) * ones
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Apply layers
    for gamma, beta in zip(gammas, betas):
        # Apply exp(-i * gamma * H_C)
        # Use matrix exponential
        from scipy.linalg import expm
        U_C = expm(-1j * gamma * H_C)
        psi = U_C @ psi
        # Apply exp(-i * beta * H_B)
        U_B = expm(-1j * beta * H_B)
        psi = U_B @ psi
    
    # Compute expected cut
    # E[cut] = sum_{(u,v) in E} P[u != v]
    # P[u != v] = sum_{state: bit_u != bit_v} |psi[state]|^2
    probs = np.abs(psi) ** 2
    expected = 0.0
    for u, v in edges:
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                expected += probs[state]
    return expected


def _expected_cut_fast(n, edges, gammas, betas, H_C, H_B):
    """Faster version using precomputed Hamiltonians."""
    from scipy.linalg import expm
    
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for gamma, beta in zip(gammas, betas):
        U_C = expm(-1j * gamma * H_C)
        psi = U_C @ psi
        U_B = expm(-1j * beta * H_B)
        psi = U_B @ psi
    
    probs = np.abs(psi) ** 2
    expected = 0.0
    for u, v in edges:
        mask = np.zeros(dim, dtype=bool)
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                mask[state] = True
        expected += np.sum(probs[mask])
    return expected


def _exact_maxcut(n, edges):
    """Compute exact MaxCut via brute force for small n."""
    best = 0
    for state in range(1 << n):
        cut = 0
        for u, v in edges:
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                cut += 1
        if cut > best:
            best = cut
    return best


def _spectral_init(n, edges):
    """
    Spectral initialization for QAOA angles.
    Use the Laplacian eigenvalues to suggest initial gamma/beta values.
    """
    A = _build_adjacency(n, edges)
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    # Eigenvalues of Laplacian
    try:
        eigvals = np.linalg.eigvalsh(L)
    except:
        eigvals = np.ones(n)
    
    # The largest eigenvalue gives a scale for the problem
    max_eig = max(np.max(eigvals), 1e-6)
    
    # Heuristic: gamma ~ 1/(2 * max_eig) * pi, beta ~ 0.5
    # These are rough starting points
    gamma_init = np.pi / (2 * max_eig)
    beta_init = 0.5
    
    return [gamma_init, beta_init], [gamma_init * 0.7, beta_init * 0.8]


def angles(n, edges, p):
    """
    Design p-layer QAOA angles for MaxCut on the given graph.
    
    Returns: (p, 2) array with rows (gamma, beta) in radians.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    deadline = _DEADLINE - 5.0  # 5 second safety margin
    elapsed = time.time() - _START_TIME
    remaining = deadline - elapsed
    
    if remaining <= 0:
        # Return a default
        return np.tile([0.35, 0.25], (p, 1))
    
    # Build Hamiltonians
    dim = 1 << n
    H_C = np.zeros((dim, dim), dtype=np.complex128)
    for u, v in edges:
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                H_C[state, state] += 0.5
    
    H_B = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        for state in range(dim):
            flipped = state ^ (1 << i)
            H_B[state, flipped] += 1.0
    
    # Precompute masks for expected cut
    cut_masks = []
    for u, v in edges:
        mask = np.zeros(dim, dtype=bool)
        for state in range(dim):
            bu = (state >> u) & 1
            bv = (state >> v) & 1
            if bu != bv:
                mask[state] = True
        cut_masks.append(mask)
    
    def objective(params):
        """Negative expected cut (to minimize)."""
        gammas = params[:p]
        betas = params[p:]
        return -_expected_cut_fast(n, edges, gammas, betas, H_C, H_B)
    
    # Spectral initialization
    gamma_init, beta_init = _spectral_init(n, edges)
    x0 = np.concatenate([gamma_init, beta_init])
    
    # Try multiple starting points
    best_result = None
    best_val = -np.inf
    
    # Starting points: spectral, plus some random perturbations
    starts = [x0.copy()]
    
    # Add a few random starts
    rng = np.random.RandomState(42)
    for _ in range(3):
        x_rand = rng.uniform(0, np.pi, size=2*p)
        # Scale to reasonable range
        x_rand[:p] = x_rand[:p] * 0.5  # gammas
        x_rand[p:] = x_rand[p:] * 0.5  # betas
        starts.append(x_rand)
    
    # Also try the seed program's values
    x_seed = np.tile([0.35, 0.25], (p, 1)).flatten()
    starts.append(x_seed)
    
    from scipy.linalg import expm
    
    for x_start in starts:
        elapsed = time.time() - _START_TIME
        if elapsed > deadline:
            break
        
        try:
            result = minimize(
                objective,
                x_start,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-6, 'disp': False}
            )
            
            if result.success or result.fun < best_val:
                val = -result.fun
                if val > best_val:
                    best_val = val
                    best_result = result.x.copy()
                    
        except Exception:
            continue
    
    if best_result is None:
        # Fallback
        return np.tile([0.35, 0.25], (p, 1))
    
    # Reshape to (p, 2)
    gammas = best_result[:p]
    betas = best_result[p:]
    return np.column_stack([gammas, betas])
