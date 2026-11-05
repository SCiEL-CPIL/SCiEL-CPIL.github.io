"""Matrix multiplication 3,3,8 — rank-minimal exact tensor decomposition search."""
import time
import random
import itertools
import numpy as np

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 72

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
START_TIME = time.time()
DEADLINE = START_TIME + 130.0  # 20s safety margin


def _time_left():
    return DEADLINE - time.time()


def _schoolbook():
    """Return the schoolbook decomposition (rank 72)."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _verify(U, V, W, tol=1e-9):
    """
    Verify that U, V, W exactly decompose the matmul tensor.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    Convention: W[k*n+i, r]
    The tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    must equal 1 for all i,j,k.
    """
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    # We'll check by constructing the matmul: C = A @ B where A is (n,m), B is (m,p)
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition says: C[i,k] = sum_r (sum_j U[i*M+j,r] A[i,j]) * (sum_j V[j*P+k,r] B[j,k]) * W[k*N+i,r]
    # Actually the standard tensor decomposition:
    # C[i,k] = sum_r W[k*N+i, r] * (sum_j U[i*M+j, r] * A[i,j]) * (sum_j V[j*P+k, r] * B[j,k])
    # Wait, let me re-read. The convention is W[k*n+i, r].
    # Standard: C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor form: C[i,k] = sum_r U[i*M+j, r]... no.
    # The matmul tensor T[i,j,k] = delta (1 if the product contributes).
    # Actually the matmul tensor is T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]B[j,k]).
    # The decomposition is: T[i,j,k] = sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r]
    # So we need sum_r U[i*M+j, r] * V[j*P+k, r] * W[k*N+i, r] == 1 for all i,j,k.
    
    # Check all (i,j,k)
    for i in range(N):
        for j in range(M):
            for k in range(P):
                val = 0.0
                for r in range(r):
                    val += U[i * M + j, r] * V[j * P + k, r] * W[k * N + i, r]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _try_rank(r):
    """
    Try to find a rank-r decomposition using random initialization + iterative refinement.
    Returns (U, V, W) as lists if found, else None.
    """
    if r < 1:
        return None
    
    # Random initialization
    for attempt in range(20):
        if _time_left() < 5.0:
            return None
        
        U = np.random.randn(N * M, r) * 0.1
        V = np.random.randn(M * P, r) * 0.1
        W = np.random.randn(P * N, r) * 0.1
        
        # Iterative refinement: fix two factors, solve for the third.
        # T[i,j,k] = sum_r U[iM+j,r] V[jP+k,r] W[kN+i,r]
        # For fixed V,W: T[i,j,k] = sum_r U[iM+j,r] * (V[jP+k,r] W[kN+i,r])
        # This is a linear system for U.
        
        max_iter = 50
        for it in range(max_iter):
            if _time_left() < 5.0:
                return None
            
            # Solve for U given V, W
            # For each (i,j): T[i,j,k] = sum_r U[iM+j,r] * (V[jP+k,r] * W[kN+i,r]) for all k
            # This is: for each (i,j), we have P equations in r unknowns (U[iM+j, :])
            # A_{(i,j), k, r} = V[jP+k,r] * W[kN+i,r]
            # b_{(i,j), k} = 1
            # Solve: A_{(i,j)} @ U[iM+j,:] = b_{(i,j)}
            
            U_new = np.zeros((N * M, r))
            for i in range(N):
                for j in range(M):
                    # Build the P x r matrix
                    A_mat = np.zeros((P, r))
                    for k in range(P):
                        for rr in range(r):
                            A_mat[k, rr] = V[j * P + k, rr] * W[k * N + i, rr]
                    b_vec = np.ones(P)
                    # Least squares
                    try:
                        U_new[i * M + j, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Solve for V given U, W
            V_new = np.zeros((M * P, r))
            for j in range(M):
                for k in range(P):
                    A_mat = np.zeros((N, r))
                    for i in range(N):
                        for rr in range(r):
                            A_mat[i, rr] = U[i * M + j, rr] * W[k * N + i, rr]
                    b_vec = np.ones(N)
                    try:
                        V_new[j * P + k, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Solve for W given U, V
            W_new = np.zeros((P * N, r))
            for k in range(P):
                for i in range(N):
                    A_mat = np.zeros((M, r))
                    for j in range(M):
                        for rr in range(r):
                            A_mat[j, rr] = U[i * M + j, rr] * V[j * P + k, rr]
                    b_vec = np.ones(M)
                    try:
                        W_new[k * N + i, :] = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            diff = (np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V)) + np.max(np.abs(W_new - W)))
            U, V, W = U_new, V_new, W_new
            if diff < 1e-12:
                break
        
        # Verify
        if _verify(U, V, W):
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try to find lower rank decompositions
    # Known lower bounds: For 3x3 * 3x8, the rank is at least max(n*m, m*p, p*n) = max(9, 24, 24) = 24
    # But the actual minimum rank is likely higher.
    # Let's try ranks from a reasonable lower bound up to schoolbook.
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = 24
    # But we also need rank >= n*p (output size) = 24
    # So minimum possible rank is 24.
    
    # Try a few ranks near the lower bound
    for r in range(24, min(36, best_rank)):
        if _time_left() < 10.0:
            break
        result = _try_rank(r)
        if result is not None:
            U, V, W = result
            # Convert to integer/half-integer if possible
            # Round to nearest half-integer
            U = np.round(np.array(U) * 2) / 2
            V = np.round(np.array(V) * 2) / 2
            W = np.round(np.array(W) * 2) / 2
            # Verify after rounding
            if _verify(U, V, W):
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break
    
    return best
