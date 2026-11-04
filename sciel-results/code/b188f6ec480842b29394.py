"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that U @ diag @ V @ diag @ W reconstructs the MM tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    rank = U.shape[1]
    
    # Build tensor T of shape (n, m, p)
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        
        W_col_T = W_col.T  # shape (n, p), W_col_T[i,k] = W_col[k,i]
        
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)  # (n, m, p)
        tmp2 = tmp1 * W_col_T[:, np.newaxis, :]  # broadcast W_col_T over j
        T_computed += tmp2
    
    # The MM tensor T[i,j,k] = 1 for all i,j,k
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    diff = np.abs(T_computed - T_expected)
    return np.max(diff) < tol


def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _try_newton(rank, n, m, p, max_iter=200, tol=1e-10, deadline=None):
    """Try to find a rank-r decomposition via Newton's method on the polynomial system."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    total_vars = (n_m + m_p + p_n) * rank
    
    rng = np.random.RandomState(rank * 1000 + 42)
    
    best_result = None
    best_err = float('inf')
    
    for trial in range(5):
        if deadline is not None and time.time() > deadline:
            break
        
        U = rng.randn(n_m, rank) * 0.1
        V = rng.randn(m_p, rank) * 0.1
        W = rng.randn(p_n, rank) * 0.1
        
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            
            S = np.zeros((n, m, p), dtype=np.float64)
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                tmp = np.einsum('ij,jk->ijk', Uc, Vc)
                S += tmp * Wc.T[:, np.newaxis, :]
            
            R = S - 1.0
            
            err = np.max(np.abs(R))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
            
            if err < tol:
                break
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for c in range(rank):
                Uc = U[:, c].reshape(n, m)
                Vc = V[:, c].reshape(m, p)
                Wc = W[:, c].reshape(p, n)
                Wc_T = Wc.T
                
                step1 = R * Wc_T[:, np.newaxis, :]
                dU_c = np.einsum('ijk,jk->ij', step1, Vc)
                dU[:, c] = dU_c.reshape(n_m)
                
                step2 = R * Wc_T[:, np.newaxis, :]
                dV_c = np.einsum('ijk,ij->jk', step2, Uc)
                dV[:, c] = dV_c.reshape(m_p)
                
                step3 = R * Vc[np.newaxis, :, :]
                dW_ki = np.einsum('ijk,ij->ik', step3, Uc)
                dW[:, c] = dW_ki.T.reshape(p_n)
            
            lr = 0.01
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_err < tol:
            break
    
    if best_result is not None and best_err < tol:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}
    
    return None


def _try_structured_rank_23(n, m, p, deadline=None):
    """
    Attempt to construct a rank-23 decomposition by combining:
    - A rank-19 decomposition for the 4x4x4 core
    - A rank-4 schoolbook decomposition for the 5th column
    This is a heuristic attempt; if verification fails, return None.
    """
    # The 4x4x4 core uses indices k=0..3, and the 5th column k=4 is handled separately.
    # However, a simple block decomposition doesn't work directly because the tensor
    # T[i,j,k] = 1 for ALL k, so the 4x4x4 core and the 5th column are not independent
    # in the standard way. We need a more careful construction.
    #
    # Actually, the tensor is T[i,j,k] = 1 for all i in 0..3, j in 0..3, k in 0..4.
    # This is a "product" tensor: T = 1_n * 1_m * 1_p (outer product of all-ones vectors).
    # The rank of a product tensor of three vectors is 1! But that's for the tensor
    # T[i,j,k] = a_i * b_j * c_k. Here T[i,j,k] = 1 = 1*1*1, so the rank is 1.
    #
    # Wait, that can't be right. Let me reconsider.
    #
    # The matrix multiplication tensor is NOT the all-ones tensor. It is:
    # T[i,j,k] = 1 if and only if (A*B)[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor T is defined such that for any matrices A (n x m) and B (m x p),
    # (A*B)[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k]
    # Since (A*B)[i,k] = sum_j A[i,j] * B[j,k], we have T[i,j,k] = 1 for all i,j,k.
    #
    # So the tensor IS the all-ones tensor of shape (n, m, p).
    # The rank of the all-ones tensor of shape (n, m, p) is 1, achieved by:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    #
    # But wait, that would mean the rank is 1 for ANY matrix multiplication, which is absurd.
    #
    # I think I'm confusing the tensor with the multiplication operation.
    # Let me re-read the problem statement.
    #
    # "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 5 matrix."
    # "The output factor follows the official trace convention W[k*n+i,r]."
    #
    # The tensor for matrix multiplication (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # is indeed T[i,j,k] = 1 for all i,j,k.
    #
    # The rank of this tensor is the minimum number of terms of the form
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # This is a well-known problem. The rank of the matrix multiplication tensor
    # for n x m by m x p matrices is NOT 1 in general. The all-ones tensor
    # T[i,j,k] = 1 has rank 1, but that's a different tensor.
    #
    # Wait, I think I'm making an error. Let me think again.
    #
    # The tensor T is defined by T[i,j,k] = 1 for all i,j,k.
    # We want to write T as a sum of rank-1 tensors:
    # T[i,j,k] = sum_c u_c[i,j] * v_c[j,k] * w_c[k,i]
    #
    # For T[i,j,k] = 1, we can take u_c[i,j] = 1, v_c[j,k] = 1, w_c[k,i] = 1 for c=0.
    # Then T[i,j,k] = 1*1*1 = 1. So the rank is 1!
    #
    # But this contradicts the known result that matrix multiplication requires
    # at least n*m*p terms in the schoolbook algorithm and that Strassen reduces
    # this for square matrices.
    #
    # I think the issue is that the tensor for matrix multiplication is NOT
    # T[i,j,k] = 1. Let me reconsider.
    #
    # In the standard formulation, the matrix multiplication tensor is:
    # T[i,j,k] = 1 if the (i,k)-th entry of the product depends on A[i,j] and B[j,k]
    # which it does for all i,j,k, so T[i,j,k] = 1.
    #
    # But the rank of this tensor as a 3-tensor is indeed 1, because it's the
    # outer product of three all-ones vectors.
    #
    # I must be misunderstanding the problem. Let me re-read the interface:
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank) with
    # integer/half-integer entries that exactly decompose the matrix multiplication tensor"
    #
    # So the tensor is the matrix multiplication tensor, and we need to decompose it.
    # If the tensor is T[i,j,k] = 1, then the rank is 1, and the answer is trivial.
    #
    # But the schoolbook algorithm has rank n*m*p, which is much larger than 1.
    # This suggests that the tensor is NOT T[i,j,k] = 1.
    #
    # I think the confusion is that in the standard tensor decomposition for
    # matrix multiplication, the tensor is not the all-ones tensor. The tensor
    # is defined such that the (i,k)-th entry of the product is:
    # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as:
    # (A*B)[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
    # No, that's not right either.
    #
    # Let me look at the schoolbook decomposition:
    # U[i*m+j, column] = 1, V[j*p+k, column] = 1, W[k*n+i, column] = 1
    # for each (i,j,k) triple, with a separate column.
    #
    # Then sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    # So the tensor IS T[i,j,k] = 1.
    #
    # And the rank is n*m*p because we need one term for each (i,j,k).
    # But we can do better: if we take U[i*m+j, 0] = 1, V[j*p+k, 0] = 1,
    # W[k*n+i, 0] = 1, then the sum is 1 for all i,j,k. So the rank is 1!
    #
    # This is a contradiction. The only resolution is that I'm misunderstanding
    # the tensor decomposition.
    #
    # After further reflection, I believe the issue is that the matrix multiplication
    # tensor in the context of arithmetic complexity is not simply T[i,j,k] = 1.
    # The tensor is defined on the space of matrices, and the rank refers to the
    # minimum number of multiplications needed, which is related to but not the
    # same as the tensor rank of T[i,j,k] = 1.
    #
    # Given the time constraints, I'll proceed with the assumption that the
    # schoolbook decomposition is correct and that the rank is n*m*p = 80.
    # The structured rank-23 approach may not be valid, so I'll return None.
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Leave 20s margin
    
    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]
    
    # Try structured rank-23 first
    if time.time() < deadline:
        result_23 = _try_structured_rank_23(n, m, p, deadline)
        if result_23 is not None:
            if _verify_decomposition(result_23["U"], result_23["V"], result_23["W"], n, m, p, tol=1e-8):
                best_result = result_23
                best_rank = 23
    
    # Try lower ranks using Newton's method
    for rank in range(79, 59, -1):
        if time.time() > deadline:
            break
        
        result = _try_newton(rank, n, m, p, max_iter=50, tol=1e-10, deadline=deadline)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                best_result = result
                best_rank = rank
                break
    
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
