"""Tensor decomposition search for 3x4 * 4x7 matrix multiplication."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np
from itertools import product, permutations
from fractions import Fraction
from scipy.optimize import minimize
import random

# EVOLVE-BLOCK-START
def _schoolbook_decomposition(n, m, p):
    """Return the standard rank-n*m*p decomposition as a list of (u, v, w) index tuples."""
    decomp = []
    for i, j, k in product(range(n), range(m), range(p)):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        decomp.append((u_idx, v_idx, w_idx, 1))
    return decomp

def _decomposition_is_valid(U, V, W, n, m, p, tol=1e-8):
    """Check if the decomposition exactly represents the multiplication tensor."""
    rank = len(U[0])
    # Build the tensor from decomposition
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # T[i,j,k] = 1 if this contributes to C[i,k] through A[i,j]*B[j,k]
    # Actually the tensor for matmul is T[i,j,k] = delta(i,j) * delta(j,k) in a sense...
    # Let me be more careful. The multiplication tensor is defined by:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) since each term A[i,j]*B[j,k] appears.
    # Wait, that's not right either. The tensor has T[i,j,k] = 1 if the product A[i,j]*B[j,k] 
    # contributes to C[i,k]. It always does, so T[i,j,k] = 1 for all i,j,k.
    # No! The tensor is T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k in the sense that
    # <T, A⊗B> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # Hmm, let me think again. 
    # 
    # The multiplication tensor M has entries M[i,j,k] such that
    # sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} (AB)[i,k] * 1
    # But (AB)[i,k] = sum_j A[i,j] B[j,k]
    # So sum_{i,j,k} M[i,j,k] A[i,j] B[j,k] = sum_{i,k} sum_j A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # Therefore M[i,j,k] = 1 for all i,j,k.
    #
    # Wait, that can't be right because then the rank would just be n*m*p trivially.
    # Let me reconsider. Actually yes, for matrix multiplication the tensor is
    # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k, which indeed has rank n*m*p.
    # 
    # Hmm but that seems too simple. Let me double-check with a small example.
    # For 1x1 * 1x1: T[0,0,0] = 1, rank 1. Correct.
    # For 1x2 * 2x1: T[0,j,k] where j in {0,1}, k=0. T[0,0,0]=1, T[0,1,0]=1.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # So yes T[i,j,k] = 1 for all valid indices.
    
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    for i, j, k in product(range(n), range(m), range(p)):
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        val = 0.0
        for r in range(rank):
            val += U[u_row][r] * V[v_row][r] * W[w_row][r]
        if abs(val - 1.0) > tol:
            return False
    return True

def _try_round_half_integer(U, V, W, n, m, p):
    """Try to round U, V, W to half-integers while maintaining validity."""
    U_r = np.array(U, dtype=float)
    V_r = np.array(V, dtype=float)
    W_r = np.array(W, dtype=float)
    
    # Round to nearest half-integer
    U_half = np.round(U_r * 2) / 2
    V_half = np.round(V_r * 2) / 2
    W_half = np.round(W_r * 2) / 2
    
    # Check if still valid
    if _decomposition_is_valid(U_half.tolist(), V_half.tolist(), W_half.tolist(), n, m, p, tol=1e-6):
        return U_half.tolist(), V_half.tolist(), W_half.tolist()
    
    return None

def _random_init(n, m, p, rank, scale=1.0):
    """Random initialization for factor matrices."""
    U = np.random.randn(n * m, rank) * scale
    V = np.random.randn(m * p, rank) * scale
    W = np.random.randn(p * n, rank) * scale
    return U, V, W

def _objective(params, n, m, p, rank):
    """Objective function for optimization: sum of squared errors."""
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    # Compute errors for all (i,j,k)
    errors = 0.0
    for i, j, k in product(range(n), range(m), range(p)):
        u_row = i * m + j
        v_row = j * p + k
        w_row = k * n + i
        val = np.dot(U[u_row], V[v_row] * W[w_row])
        errors += (val - 1.0) ** 2
    return errors

def _refine_decomposition(n, m, p, rank, U0, V0, W0, max_iters=5000, tol=1e-12, deadline=None):
    """Use gradient-based optimization to refine the decomposition."""
    U_flat = U0.flatten()
    V_flat = V0.flatten()
    W_flat = W0.flatten()
    x0 = np.concatenate([U_flat, V_flat, W_flat])
    
    def obj_and_grad(x):
        U = x[:n*m*rank].reshape(n*m, rank)
        V = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        # Compute objective and gradient
        obj = 0.0
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for i, j, k in product(range(n), range(m), range(p)):
            u_row = i * m + j
            v_row = j * p + k
            w_row = k * n + i
            
            val = np.dot(U[u_row], V[v_row] * W[w_row])
            err = val - 1.0
            obj += err * err
            
            # Gradient with respect to U[u_row]
            gU[u_row] += 2 * err * V[v_row] * W[w_row]
            # Gradient with respect to V[v_row]
            gV[v_row] += 2 * err * U[u_row] * W[w_row]
            # Gradient with respect to W[w_row]
            gW[w_row] += 2 * err * U[u_row] * V[v_row]
        
        grad = np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
        return obj, grad
    
    # Use L-BFGS-B with gradient
    result = minimize(obj_and_grad, x0, method='L-BFGS-B', jac=True, 
                       options={'maxiter': max_iters, 'ftol': tol, 'gtol': 1e-10})
    
    if deadline is not None and time.time() > deadline:
        return None
    
    x_opt = result.x
    U_opt = x_opt[:n*m*rank].reshape(n*m, rank)
    V_opt = x_opt[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
    W_opt = x_opt[n*m*rank + m*p*rank:].reshape(p*n, rank)
    
    obj_val = result.fun
    if obj_val > 1e-10:
        return None
    
    return U_opt, V_opt, W_opt

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook decomposition
    schoolbook = _schoolbook_decomposition(n, m, p)
    
    def _schoolbook_to_matrices(decomp):
        rank = len(decomp)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for r, (u_idx, v_idx, w_idx, coeff) in enumerate(decomp):
            U[u_idx][r] = coeff
            V[v_idx][r] = coeff
            W[w_idx][r] = coeff
        return U, V, W
    
    U0, V0, W0 = _schoolbook_to_matrices(schoolbook)
    rank0 = len(schoolbook)
    
    # Try to reduce rank by removing zero or near-zero columns
    # First, try to find a lower rank decomposition
    
    best_U, best_V, best_W = U0, V0, W0
    best_rank = rank0
    
    # Attempt rank reduction through optimization
    # Try ranks from some lower value up to the schoolbook rank
    min_possible_rank = max(n * m, m * p, p * n)  # Lower bound
    
    # Try a few candidate ranks below schoolbook
    for candidate_rank in range(min_possible_rank, rank0):
        if time.time() > deadline - 10:
            break
        
        success = False
        for attempt in range(3):
            if time.time() > deadline - 5:
                break
            
            U_init, V_init, W_init = _random_init(n, m, p, candidate_rank, scale=1.0)
            
            result = _refine_decomposition(n, m, p, candidate_rank, U_init, V_init, W_init,
                                           max_iters=2000, deadline=deadline - 5)
            if result is not None:
                U_opt, V_opt, W_opt = result
                # Try to round to half-integers
                rounded = _try_round_half_integer(U_opt, V_opt, W_opt, n, m, p)
                if rounded is not None:
                    U_h, V_h, W_h = rounded
                    if _decomposition_is_valid(U_h, V_h, W_h, n, m, p, tol=1e-8):
                        best_U, best_V, best_W = U_h, V_h, W_h
                        best_rank = candidate_rank
                        success = True
                        break
                else:
                    # Use unrounded if valid
                    if _decomposition_is_valid(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-8):
                        best_U, best_V, best_W = U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
                        best_rank = candidate_rank
                        success = True
                        break
        
        if success:
            # Try even lower ranks
            continue
        else:
            # If this rank failed, lower ranks are unlikely to succeed easily
            # But try one more step down
            break
    
    # If we found a better rank, great. Otherwise return schoolbook.
    if best_rank < rank0:
        return {"U": best_U, "V": best_V, "W": best_W}
    else:
        # Return the schoolbook decomposition
        return {"U": U0, "V": V0, "W": W0}
# EVOLVE-BLOCK-END
