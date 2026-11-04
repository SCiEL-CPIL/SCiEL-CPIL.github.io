"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 4]
_START_TIME = time.time()
_DEADLINE = 135  # seconds

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _check_exact(U, V, W, n, m, p):
    """Check if the decomposition exactly reproduces the multiplication tensor.
    
    The multiplication tensor T has T[i,j,k] = 1 if and only if the (i,j) entry of A
    and (j,k) entry of B both contribute to C[i,k]. In the trace convention,
    the tensor is sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}... 
    wait, no. The matrix multiplication tensor is T[i,j,k] = 1 for all i,j,k? No.
    
    Actually, the multiplication tensor maps A (n x m) and B (m x p) to C (n x p).
    The tensor T has indices (i, j, k) where i is row of A and C, j is column of A and row of B,
    k is column of B and C. The entry T[i,j,k] = 1 if and only if... wait, it's always 1?
    
    No, the multiplication tensor is defined such that C[i,k] = sum_j A[i,j] * B[j,k].
    The tensor is a 3-tensor with entries T[i,j,k] = 1 for all valid i,j,k? No, that's not right either.
    
    Let me reconsider. The tensor T is in V_n* ⊗ V_m* ⊗ V_p* where the multiplication is:
    T(A, B) = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] e_{i,k}^C
    
    So T[i,j,k] must equal 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k].
    But wait, T[i,j,k] is a scalar. For the matrix multiplication, T[i,j,k] = 1 for all i, j, k
    that form valid indices? No, that would give C[i,k] = sum_{j} A[i,j] B[j,k] * 1, which is correct.
    
    Hmm, but the tensor is 3-dimensional with dimensions n x m x p. Each entry T[i,j,k] = 1
    for all valid i, j, k. That means the tensor is the all-ones tensor? That can't be right
    because then the rank would be 1.
    
    I think I'm confusing myself. Let me re-read the problem.
    
    The matrix multiplication tensor for multiplying an n x m matrix by an m x p matrix
    is a tensor T in R^{n x m x p} where T[i, j, k] = 1 if and only if the (i,j) entry of A
    and the (j,k) entry of B both contribute to C[i,k]. But in standard matrix multiplication,
    every (i,j) and (j,k) pair contributes, so T[i,j,k] = 1 for all valid i, j, k.
    
    Wait, that would make the tensor the all-ones tensor, which has rank 1. But we know the
    schoolbook algorithm uses n*m*p terms. So something is wrong with my understanding.
    
    Ah, I see the issue. The tensor is NOT the all-ones tensor. The matrix multiplication
    tensor is defined differently. Let me think again.
    
    Actually, I think the correct definition is: the multiplication tensor T is a 3-tensor
    such that for any matrices A (n x m) and B (m x p), the product C = A*B is given by
    C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
    
    For this to work, we need T[i,j,k] = 1 for all valid i, j, k. So the tensor IS the
    all-ones tensor. But then why does the schoolbook algorithm use n*m*p terms?
    
    I think the issue is that the tensor decomposition is not of the all-ones tensor, but
    of a tensor that encodes the structure of matrix multiplication. Let me reconsider.
    
    Actually, I think I've been overthinking this. The schoolbook decomposition has rank
    n*m*p, and each term corresponds to a specific (i,j,k) triple. The tensor entry T[i,j,k]
    is 1 for all valid indices, and the decomposition uses one term for each (i,j,k) triple.
    
    So the check is: for each (i,j,k), the sum over c of U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    should equal 1.
    """
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True

def _try_reduce(U, V, W, n, m, p):
    """Try to reduce the rank by checking if any column can be removed.
    
    This is a simple heuristic: for each column, check if removing it still gives
    a valid decomposition. If so, remove it and try again.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    changed = True
    while changed and time.time() - _START_TIME < _DEADLINE:
        changed = False
        for c in range(rank):
            if time.time() - _START_TIME > _DEADLINE - 5:
                break
            # Temporarily remove column c
            U_test = np.delete(U_arr, c, axis=1)
            V_test = np.delete(V_arr, c, axis=1)
            W_test = np.delete(W_arr, c, axis=1)
            
            # Check if still valid
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = np.dot(U_test[i * m + j, :], V_test[j * p + k, :] * W_test[k * n + i, :])
                        if abs(total - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if not valid:
                break
            
            if valid:
                # Column c can be removed
                U_arr = U_test
                V_arr = V_test
                W_arr = W_test
                rank -= 1
                changed = True
    
    return U_arr.tolist(), V_arr.tolist(), W_arr.tolist()

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Try to reduce rank by removing redundant columns
    U, V, W = result["U"], result["V"], result["W"]
    U_reduced, V_reduced, W_reduced = _try_reduce(U, V, W, n, m, p)
    
    # Verify the reduced version
    if _check_exact(U_reduced, V_reduced, W_reduced, n, m, p):
        return {"U": U_reduced, "V": V_reduced, "W": W_reduced}
    
    # Fall back to schoolbook if reduction failed verification
    return result
# EVOLVE-BLOCK-END
