"""QAOA angle design for MaxCut with p=2 layers using precomputed unitaries and gradient-based optimization."""
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import expm
import time

# Deadline guard: 150s wall-clock kill, use 140s as safety margin
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0

def _remaining_time():
    return _DEADLINE - time.time()

def _exact_maxcut(n, edges):
    """Compute exact MaxCut via brute force for small n (n <= 10)."""
    m = len(edges)
    if m == 0:
        return 0.0
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if (mask >> u) & 1 != (mask >> v) & 1:
                cut += 1
        if cut > best:
            best = cut
    return float(best)

def _build_cost_hamiltonian(n, edges):
    """
    Build the cost Hamiltonian C = sum_{(u,v) in edges} (1 - Z_u Z_v)/2.
    C is diagonal in the computational basis.
    Returns a vector of diagonal entries (length 2^n).
    """
    dim = 1 << n
    c_diag = np.zeros(dim, dtype=np.float64)
    for u, v in edges:
        for x in range(dim):
            xu = (x >> u) & 1
            xv = (x >> v) & 1
            if xu != xv:
                c_diag[x] += 0.5  # (1 - (-1))/2 = 1
            # else: (1 - 1)/2 = 0
    return c_diag

def _build_mixer_hamiltonian(n):
    """
    Build the mixer Hamiltonian H_mix = sum_{j=0}^{n-1} X_j.
    Returns the full matrix (2^n x 2^n).
    """
    dim = 1 << n
    H_mix = np.zeros((dim, dim), dtype=np.complex128)
    for j in range(n):
        for x in range(dim):
            y = x ^ (1 << j)
            H_mix[x, y] += 1.0
    return H_mix

def _precompute_operators(n, edges):
    """
    Precompute all operators needed for QAOA simulation.
    Returns:
        c_diag: diagonal entries of cost Hamiltonian
        U_mix_cache: function to compute exp(-i*beta*H_mix) quickly
        H_mix: mixer Hamiltonian matrix
    """
    dim = 1 << n
    c_diag = _build_cost_hamiltonian(n, edges)
    H_mix = _build_mixer_hamiltonian(n)
    
    # Precompute eigen-decomposition of H_mix for fast mixer application
    # H_mix is Hermitian, so we can use eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(H_mix)
    
    return c_diag, eigvals, eigvecs, dim

def _expected_cut_precomputed(n, edges, gammas, betas, c_diag, eigvals, eigvecs, dim):
    """
    Compute expected cut using precomputed operators.
    State evolution:
      |psi_0> = |+>^n
      For i in range(p):
        |psi> = exp(-i*gamma_i*C) @ |psi>
        |psi> = exp(-i*beta_i*H_mix) @ |psi>
    """
    m = len(edges)
    if m == 0:
        return 0.0
    
    # Initial state: |+>^n
    state = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    for i in range(2):  # p=2
        # Apply cost: exp(-i*gamma*C) where C is diagonal
        gamma = gammas[i]
        # exp(-i*gamma*C) is diagonal with entries exp(-i*gamma*c_diag[x])
        cost_phase = np.exp(-1j * gamma * c_diag)
        state = cost_phase * state
        
        # Apply mixer: exp(-i*beta*H_mix)
        beta = betas[i]
        # Use eigen-decomposition: exp(-i*beta*H_mix) = V @ diag(exp(-i*beta*eigvals)) @ V^H
        mix_phase = np.exp(-1j * beta * eigvals)
        # state = V @ (mix_phase * (V^H @ state))
        state_in_eig = eigvecs.conj().T @ state
        state = eigvecs @ (mix_phase * state_in_eig)
    
    # Compute expected cut: <C> = sum_x |psi_x|^2 * c_diag[x]
    prob = np.abs(state) ** 2
    expected = np.sum(prob * c_diag)
    return float(expected)

def _objective_precomputed(params, c_diag, eigvals, eigvecs, dim, exact_opt):
    """Objective to minimize: negative of the ratio."""
    gammas = params[:2]
    betas = params[2:]
    expected = _expected_cut_precomputed(0, [], gammas, betas, c_diag, eigvals, eigvecs, dim)
    if exact_opt == 0:
        return 0.0
    return -expected / exact_opt

def _optimize_angles_for_graph(n, edges):
    """Optimize angles for a single graph."""
    exact_opt = _exact_maxcut(n, edges)
    if exact_opt == 0:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    # Precompute operators
    c_diag, eigvals, eigvecs, dim = _precompute_operators(n, edges)
    
    best_params = np.array([0.35, 0.35, 0.25, 0.25])
    best_val = _objective_precomputed(best_params, c_diag, eigvals, eigvecs, dim, exact_opt)
    
    # Try multiple initial points with gradient-based optimization
    init_points = [
        np.array([0.35, 0.35, 0.25, 0.25]),
        np.array([0.5, 0.5, 0.3, 0.3]),
        np.array([0.2, 0.2, 0.1, 0.1]),
        np.array([0.7, 0.7, 0.5, 0.5]),
        np.array([0.1, 0.4, 0.2, 0.1]),
        np.array([0.4, 0.1, 0.1, 0.4]),
        np.array([0.3, 0.6, 0.15, 0.35]),
        np.array([0.6, 0.3, 0.4, 0.15]),
    ]
    
    for init in init_points:
        if _remaining_time() < 5.0:
            break
        try:
            # Use L-BFGS-B for gradient-based optimization (faster convergence)
            res = minimize(_objective_precomputed, init, 
                          args=(c_diag, eigvals, eigvecs, dim, exact_opt),
                          method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            try:
                # Fallback to Nelder-Mead if gradient method fails
                res = minimize(_objective_precomputed, init,
                              args=(c_diag, eigvals, eigvecs, dim, exact_opt),
                              method='Nelder-Mead',
                              options={'maxiter': 80, 'xatol': 1e-4, 'fatol': 1e-5})
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Final polish with tighter tolerances
    if _remaining_time() > 10.0:
        try:
            res = minimize(_objective_precomputed, best_params,
                          args=(c_diag, eigvals, eigvecs, dim, exact_opt),
                          method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    return np.array([[best_params[0], best_params[2]],
                     [best_params[1], best_params[3]]])

def angles(n, edges, p):
    """
    Design graph-dependent two-layer QAOA angles.
    Returns array of shape (p, 2) with rows (gamma, beta) in radians.
    """
    if p != 2:
        return np.tile([0.35, 0.25], (p, 1))
    
    # Check time budget
    if _remaining_time() < 10.0:
        # Not enough time for optimization, return heuristic
        return np.array([[0.35, 0.25], [0.35, 0.25]])
    
    try:
        return _optimize_angles_for_graph(n, edges)
    except Exception:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
