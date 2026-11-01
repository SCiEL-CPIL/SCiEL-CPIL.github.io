"""
Rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 4 matrix.

Strategy:
1. Default to Recursive Strassen (Rank 49).
2. Attempt a time-boxed numerical search for Rank 48 using SciPy.
3. If a near-exact solution is found, attempt rationalization to integers/half-integers.
4. Verify and return the best valid decomposition.
"""

import time
import numpy as np
import scipy.optimize as so

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START


def _make_strassen_2():
    """
    Strassen rank-7 factors for 2x2.
    Layout: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r].
    """
    U2 = np.array(
        [
            [1, 0, 1, 0, 1, -1, 0],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 1, 0, 0, 0, 1, 0],
            [1, 1, -1, 1, 0, 0, -1],
        ],
        dtype=float,
    )
    V2 = np.array(
        [
            [1, 1, 0, -1, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1],
            [1, 0, -1, 0, 1, 0, 1],
        ],
        dtype=float,
    )
    W2 = np.array(
        [
            [1, 0, 0, 1, -1, 0, 1],
            [0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [1, 0, -1, 0, 0, 1, -1],
        ],
        dtype=float,
    )
    return U2, V2, W2


def _kron_build(U2, V2, W2):
    """
    Assemble rank-49 factors for 4x4 from the 2x2 factors.
    Uses the standard Kronecker-product identity for block matrix multiplication.
    """
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    return U4, V4, W4


def _check_decomposition(U, V, W, tol=1e-10):
    """
    Verify that the three factor matrices reconstruct the matmul tensor.
    Returns True on exact match (within tight tolerance).
    """
    n, m, p = 4, 4, 4
    r = U.shape[1]
    Ua = U.reshape(n, m, r)
    Va = V.reshape(m, p, r)
    Wa = W.reshape(p, n, r)
    T = np.einsum("ijr,jkr,kir->ijk", Ua, Va, Wa)
    # The target tensor T_target[i,j,k] is 1 if the indices correspond to a valid multiplication term?
    # No, the tensor for matrix multiplication is T[i,j,k] = 1.
    # Wait, the standard tensor is T[i,j,k] = delta_{j,j} ? No.
    # C_ik = sum_j A_ij B_jk.
    # Tensor T_ijk such that C_ik = sum_{j} T_ijk A_ij B_jk.
    # So T_ijk = 1 for all i,j,k.
    # Let's verify the einsum:
    # Ua[i,j,r] * Va[j,k,r] * Wa[k,i,r] summed over r.
    # If U,V,W are correct, this sum should be 1 for all i,j,k.
    return bool(np.allclose(T, 1.0, atol=tol))


def _fallback_schoolbook():
    """
    Trivial rank-64 decomposition (always valid).
    """
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_find_rank_48(deadline):
    """
    Attempt to find a rank-48 decomposition using numerical optimization.
    Returns None if not found or time runs out.
    """
    n, m, p = 4, 4, 4
    target_rank = 48
    # Initial guess: Take Strassen rank-49 and drop the last column, then scale?
    # Or just random initialization? Random is better for escape local minima.
    # But we need to be close to a solution.
    # Let's try perturbing the Strassen rank-49 by removing one term and re-optimizing.
    
    U2, V2, W2 = _make_strassen_2()
    U4, V4, W4 = _kron_build(U2, V2, W2)
    
    # We want to reduce rank from 49 to 48.
    # This implies one linear dependency or a different structure.
    # We will optimize 3 matrices of size (16, 48).
    # Variables: x_U, x_V, x_W.
    # Residual: || T_reconstructed - 1 ||_F
    
    def objective(params):
        if time.time() > deadline:
            return 1e10 # Abort
        
        U = params[:16*target_rank].reshape(16, target_rank)
        V = params[16*target_rank:16*target_rank*2].reshape(16, target_rank)
        W = params[16*target_rank*2:].reshape(16, target_rank)
        
        Ua = U.reshape(4, 4, target_rank)
        Va = V.reshape(4, 4, target_rank)
        Wa = W.reshape(4, 4, target_rank)
        
        T = np.einsum("ijr,jkr,kir->ijk", Ua, Va, Wa)
        # Target is 1.0
        return np.sum((T - 1.0)**2)

    # Initialization: Use first 48 columns of Strassen, and zero the last?
    # This is a bad start because Strassen is minimal.
    # Better: Random small perturbation of a rank-49 solution projected?
    # Actually, let's just try a few random restarts with random initialization.
    # Given the difficulty, this is a long shot.
    
    best_params = None
    best_obj = float('inf')
    
    # Use multiple restarts
    for restart in range(5):
        if time.time() > deadline - 5:
            break
            
        # Random initialization
        np.random.seed(42 + restart)
        x0 = np.random.normal(0, 0.1, size=16*target_rank*3)
        
        # Use L-BFGS-B or Nelder-Mead? L-BFGS-B is better for large N.
        # But 2304 variables is a lot for L-BFGS-B in Python without Jacobian.
        # Let's try a simpler gradient-free method or just a few steps of L-BFGS-B.
        
        try:
            res = so.minimize(
                objective, 
                x0, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'maxls': 20}
            )
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                
            if best_obj < 1e-6:
                break
                
        except Exception:
            continue
            
    if best_params is not None and best_obj < 1e-6:
        # Rationalize
        U = best_params[:16*target_rank].reshape(16, target_rank)
        V = best_params[16*target_rank:16*target_rank*2].reshape(16, target_rank)
        W = best_params[16*target_rank*2:].reshape(16, target_rank)
        
        # Round to nearest 0.5
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        
        if _check_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
            
    return None


def construct():
    t_start = time.time()
    deadline = t_start + 115.0

    # 1. Try to find a better decomposition (Rank < 49)
    # Note: Rank 48 is the next integer below 49.
    # If we find it, score improves.
    better = _try_find_rank_48(deadline)
    
    if better is not None:
        U, V, W = better
        if time.time() < deadline and _check_decomposition(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Fallback to Strassen Rank 49
    U2, V2, W2 = _make_strassen_2()
    U4, V4, W4 = _kron_build(U2, V2, W2)

    if time.time() < deadline and _check_decomposition(U4, V4, W4):
        return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}

    # 3. Fallback to Schoolbook Rank 64
    Uf, Vf, Wf = _fallback_schoolbook()
    return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}


# EVOLVE-BLOCK-END
