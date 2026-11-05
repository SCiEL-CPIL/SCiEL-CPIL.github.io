"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
import random

def _exact_maxcut(n, edges):
    """Compute exact MaxCut value by brute force for small n."""
    if n > 20:
        raise ValueError("Exact MaxCut only for small n")
    best = 0
    for mask in range(1 << n):
        cut = 0
        for u, v in edges:
            if ((mask >> u) & 1) != ((mask >> v) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best

def _get_unitaries(n, edges):
    """
    Precompute the dense unitary matrices for the cost and mixer operators.
    Returns U_cost (2^n x 2^n) and U_mixer (2^n x 2^n).
    """
    dim = 1 << n
    
    # Cost Operator C = sum Z_u Z_v
    # Since C is diagonal, exp(-i gamma C) is diagonal.
    # But to use a single matrix multiplication approach for the whole layer,
    # we can construct the diagonal matrix.
    # However, the mixer is not diagonal.
    # The circuit is: U_mixer(gamma, beta) = exp(-i beta X_total) exp(-i gamma C)
    # We can precompute the diagonal matrix D = exp(-i gamma C) and the mixer M = exp(-i beta X_total).
    # But gamma and beta change. So we can't precompute the combined unitary for specific angles.
    # We CAN precompute:
    # 1. The diagonal values of C (c_vals).
    # 2. The matrix for X_total (X_mat).
    # Then for each step, we construct D (diagonal) and M (linear combo of I and X_mat).
    # Actually, exp(-i beta X_total) = cos(beta) I - i sin(beta) X_total.
    # So M is a linear combination of I and X_mat.
    # We can precompute X_mat.
    
    c_vals = np.zeros(dim, dtype=np.float64)
    for j in range(dim):
        c = 0
        for u, v in edges:
            if ((j >> u) & 1) != ((j >> v) & 1):
                c -= 1
            else:
                c += 1
        c_vals[j] = c
        
    # Precompute X_total matrix
    # X_total = sum_i X_i
    # X_i flips bit i.
    X_mat = np.zeros((dim, dim), dtype=np.complex128)
    for i in range(n):
        mask = 1 << i
        for j in range(dim):
            k = j ^ mask
            X_mat[j, k] += 1.0
            
    return c_vals, X_mat

def _expected_cut_fast(n, edges, gammas, betas, c_vals, X_mat):
    """
    Compute expected cut using precomputed matrices.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    
    # Precompute I
    I = np.eye(dim, dtype=np.complex128)
    
    for k in range(len(gammas)):
        # Apply Cost: exp(-i gamma_k C)
        # This is a diagonal matrix D_k
        # psi = D_k @ psi  =>  psi = (exp(-i gamma_k c_vals) * psi)
        phase = np.exp(-1j * gammas[k] * c_vals)
        psi = psi * phase
        
        # Apply Mixer: exp(-i beta_k X_total)
        # M_k = cos(beta_k) I - i sin(beta_k) X_mat
        # psi = M_k @ psi
        c_b = np.cos(betas[k])
        s_b = np.sin(betas[k])
        # M_k = c_b * I - 1j * s_b * X_mat
        # We can avoid constructing the full M_k matrix every time if we just do the math:
        # psi_new = c_b * psi - 1j * s_b * (X_mat @ psi)
        x_psi = X_mat @ psi
        psi = c_b * psi - 1j * s_b * x_psi
        
        # Normalize
        norm = np.linalg.norm(psi)
        if norm > 0:
            psi /= norm
            
    # Calculate expected cut
    probs = np.abs(psi) ** 2
    exp_cut = 0.0
    for u, v in edges:
        # E[Z_u Z_v]
        # We can precompute the vector for Z_u Z_v expectation?
        # Or just sum.
        # To be faster, we can precompute the Z_u Z_v vectors for all edges.
        # But for now, let's stick to the loop, it's small.
        exp_zu_zv = 0.0
        for j in range(dim):
            if ((j >> u) & 1) == ((j >> v) & 1):
                exp_zu_zv += probs[j]
            else:
                exp_zu_zv -= probs[j]
        exp_cut += exp_zu_zv
        
    return exp_cut

def _optimize_angles(n, edges, p):
    """
    Optimize angles using Nelder-Mead or similar, with many restarts.
    """
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    c_vals, X_mat = _get_unitaries(n, edges)
    
    best_params = None
    best_obj = -np.inf
    
    # Use scipy.optimize.minimize with Nelder-Mead
    from scipy.optimize import minimize
    
    def objective(params):
        g = params[0:2]
        b = params[2:4]
        # Ensure positive? No, angles can be anything, but periodic.
        # Let's just evaluate.
        return -_expected_cut_fast(n, edges, g, b, c_vals, X_mat)
        
    # Multi-start
    # Number of restarts depends on time.
    # Let's try 20 restarts.
    rng = np.random.default_rng(42)
    
    for restart in range(30):
        if time.time() > deadline:
            break
            
        # Random initialization
        x0 = rng.uniform(0, 2*np.pi, 4)
        
        # Add some heuristic starts
        if restart == 0:
            x0 = np.array([0.35, 0.35, 0.25, 0.25])
        elif restart == 1:
            x0 = np.array([np.pi/4, np.pi/4, np.pi/8, np.pi/8])
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True})
            
            if res.success or res.fun < -best_obj:
                obj = -res.fun
                if obj > best_obj:
                    best_obj = obj
                    best_params = res.x
        except Exception as e:
            pass
            
    if best_params is None:
        return np.array([[0.35, 0.25], [0.35, 0.25]])
        
    g = best_params[0:2]
    b = best_params[2:4]
    
    # Clean up angles to be in [0, 2*pi]
    g = g % (2*np.pi)
    b = b % (2*np.pi)
    
    return np.array([[g[0], b[0]], [g[1], b[1]]])

def angles(n, edges, p):
    """
    Entry point.
    """
    start_time = time.time()
    
    try:
        res = _optimize_angles(n, edges, p)
    except Exception:
        res = np.tile([0.35, 0.25], (p, 1))
        
    return res

# EVOLVE-BLOCK-END
