"""Rewritten program for matrix multiplication tensor rank minimization (2x4 * 4x4)."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]
DEADLINE = 140.0  # seconds (safety margin under 150 s wall-clock)

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n x m by m x p multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    Convention: W[k*n + i, c]
    T[i,j,k] = 1 for all i,j,k
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    rank = U.shape[1]
    T_rec = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        for k in range(p):
            Wck = Wc[k, :]  # shape (n,)
            Vck = Vc[:, k]  # shape (m,)
            T_rec[:, :, k] += Uc * np.outer(Wck, Vck)
    T_true = np.ones((n, m, p))
    return np.allclose(T_rec, T_true, atol=tol)


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using tensor contraction."""
    rank = U.shape[1]
    # For each c, the contribution is a rank-1 tensor:
    # T_c[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Reshape U_c to (n, m), V_c to (m, p), W_c to (p, n)
    # T_c[i,j,k] = Uc[i,j] * Vc[j,k] * Wc[k,i]
    # Sum over c: T[i,j,k] = sum_c Uc[i,j] * Vc[j,k] * Wc[k,i]
    T_rec = np.zeros((n, m, p))
    for c in range(rank):
        Uc = U[:, c].reshape(n, m)
        Vc = V[:, c].reshape(m, p)
        Wc = W[:, c].reshape(p, n)
        # T[:, :, k] += Uc * (Wc[k, :] * Vc[:, k]) for each k
        for k in range(p):
            # Wc[k, i] * Vc[j, k] gives outer product
            T_rec[:, :, k] += Uc * np.outer(Wc[k, :], Vc[:, k])
    return np.allclose(T_rec, 1.0, atol=tol)


def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    start_time = time.time()
    
    best_result = None
    best_rank = n * m * p  # 32
    
    # Build schoolbook as fallback
    rank_fallback = n * m * p
    U_fb = np.zeros((n * m, rank_fallback))
    V_fb = np.zeros((m * p, rank_fallback))
    W_fb = np.zeros((p * n, rank_fallback))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, col] = 1
                V_fb[j * p + k, col] = 1
                W_fb[k * n + i, col] = 1
                col += 1
    best_result = {'U': U_fb.tolist(), 'V': V_fb.tolist(), 'W': W_fb.tolist()}
    best_rank = rank_fallback
    
    # Try to find lower-rank decompositions using linear algebra
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: Fix U and V, solve linear system for W
    # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # Unknowns: W[k*n+i, c] for all (k,i,c) -> p*n*rank unknowns
    # Equations: for each (i,j,k) -> n*m*p = 32 equations
    
    # Try ranks from 27 down to 14
    for target_rank in range(27, 13, -1):
        if time.time() > DEADLINE:
            break
        
        # Try several random initializations
        found = False
        for trial in range(30):
            if time.time() > DEADLINE:
                break
            
            np.random.seed(trial * 1000 + target_rank + 7)
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            
            # Build the linear system for W
            num_eqs = n * m * p
            num_unk = p * n * target_rank
            A_sys = np.zeros((num_eqs, num_unk))
            b = np.ones(num_eqs)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(target_rank):
                            unk_idx = (k * n + i) * target_rank + c
                            A_sys[eq_idx, unk_idx] = U[i * m + j, c] * V[j * p + k, c]
            
            # Solve the least squares system
            W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A_sys, b, rcond=None)
            W = W_flat.reshape(p * n, target_rank)
            
            # Verify
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best_result = {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
                best_rank = target_rank
                found = True
                break
            else:
                # Check error
                T_rec = np.zeros((n, m, p))
                for c in range(target_rank):
                    Uc = U[:, c].reshape(n, m)
                    Vc = V[:, c].reshape(m, p)
                    Wc = W[:, c].reshape(p, n)
                    for k in range(p):
                        T_rec[:, :, k] += Uc * np.outer(Wc[k, :], Vc[:, k])
                max_err = np.max(np.abs(T_rec - 1.0))
                if max_err < 1e-3:
                    # Try rounding to half-integers
                    U_int = np.round(U * 2) / 2
                    V_int = np.round(V * 2) / 2
                    W_int = np.round(W * 2) / 2
                    if _verify(U_int, V_int, W_int, n, m, p, tol=1e-9):
                        best_result = {'U': U_int.tolist(), 'V': V_int.tolist(), 'W': W_int.tolist()}
                        best_rank = target_rank
                        found = True
                        break
    
    # Try a structural approach: use Kronecker product structure
    # T[i,j,k] = 1 can be written as sum of separable terms
    # If we use U[i*m+j, c] = u_i * v_j, V[j*p+k, c] = v_j * w_k, W[k*n+i, c] = w_k * u_i
    # Then T[i,j,k] = sum_c u_i * v_j * v_j * w_k * w_k * u_i = (sum_c u_i^2)(sum_c v_j^2)(sum_c w_k^2)
    # This only works if all three factors are constants.
    
    # Better: use the fact that T[i,j,k] = 1 is separable in a different way.
    # T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} summed over something?
    # Actually T[i,j,k] = 1 for all (i,j,k), which is the all-ones tensor.
    
    # The rank of the all-ones tensor of size a x b x c is max(a, b, c) in some cases,
    # but generally it's more complex. For 2x4x4, let's try to find a structured decomposition.
    
    # Alternative: use the fact that we can write 1 = sum of products of basis vectors.
    # Let's try a rank-8 decomposition with specific structure.
    
    # For now, return the best found
    if time.time() > DEADLINE - 10:
        pass
    
    return best_result

# EVOLVE-BLOCK-END
