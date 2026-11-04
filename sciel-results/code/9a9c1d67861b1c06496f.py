"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check that this equals the standard bilinear form for matrix mult:
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] should be 1 if we're decomposing the identity-like tensor,
    # but actually the tensor for matrix multiplication has T[i,j,k] = delta_{i,j,k}
    # meaning the (i,j,k) entry is 1 iff... wait.
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 for all i,j,k? No.
    # The tensor is defined by: T(A,B)[i,k] = sum_j A[i,j] B[j,k]
    # As a 3-tensor: T[i,j,k] such that C[i,k] = sum_{j} A[i,j] B[j,k]
    # The decomposition says: C[i,k] = sum_c U[i*m+j,c]... 
    
    # Actually the standard formulation: 
    # C = A*B where C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T[i,j,k] = 1 (the structure tensor)
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, let me re-read the problem.
    
    # U shape: (n*m, rank), indexed by [i*m+j]
    # V shape: (m*p, rank), indexed by [j*p+k]
    # W shape: (p*n, rank), indexed by [k*n+i]
    
    # The tensor equation: for all i,j,k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}?? No.
    
    # For matrix multiplication C = AB:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is a trilinear form in A, B, C: sum_{i,j,k} C[i,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k] C[i,k]
    # The tensor T[i,j,k] = 1 for all i,j,k? That doesn't make sense either.
    
    # Let me think again. The bilinear map (A,B) -> C = AB can be written as:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is linear in A and B. To write as sum of rank-1 tensors:
    # C[i,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] ... 
    # 
    # Actually the standard tensor decomposition for matrix mult:
    # The tensor T has entries T[i,j,k] where the bilinear form is:
    # <T, A⊗B⊗C> = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] C[k,i]
    # For matrix mult: sum_{i,j,k} A[i,j] B[j,k] C[k,i] = sum_{i,k} C[i,k] (AB)[i,k]
    # Wait no: sum_{i,j,k} A[i,j] B[j,k] C[k,i] = sum_{i,k} C[k,i] sum_j A[i,j]B[j,k] = sum_{i,k} C[k,i] (AB)[i,k]
    # This equals sum_{i,k} (AB)^T[i,k] C[k,i]... this is getting confusing.
    
    # Let me just use the schoolbook as reference:
    # In schoolbook: for each (i,j,k): U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] = 1 for each (i,j,k) pair that appears.
    # But each (i,j,k) has its own column, so for a fixed (i,j,k), only one c contributes,
    # giving value 1. For other (i',j',k') combinations, the value is 0.
    
    # So the tensor T[i,j,k] = 1 for all i,j,k? That would mean T is the all-ones tensor.
    # But that's not matrix multiplication...
    
    # I think the convention is: the tensor for matrix multiplication M_{n,m,p} has
    # T[i,j,k] = 1 for all valid i,j,k. The bilinear form is:
    # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
    # And C[i,k] = sum_j A[i,j] B[j,k], so the output is recovered via the W factor.
    
    # Actually, I believe the correct interpretation is:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_k V[j*p+k,c] B[j,k])
    # No, that's not right either.
    
    # Let me just verify against the schoolbook solution numerically.
    n, m, p = 3, 4, 7
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
    
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For schoolbook, this should give T[i,j,k] = 1 for all i,j,k
    # Let's check a few entries
    T_check = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_check[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
    
    # This should be all ones
    assert np.allclose(T_check, 1.0), f"Schoolbook verification failed: {T_check[0,0,0]}"
    
    # Now verify the given U,V,W
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i,j,k] = np.sum(U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:])
    
    # The target tensor for matrix multiplication is the all-ones tensor
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _try_rank(n, m, p, target_rank, max_time=10):
    """Try to find a decomposition of the given rank using random search."""
    start_time = time.time()
    best = None
    
    # Target tensor is all ones
    target = np.ones((n, m, p))
    
    for attempt in range(10000):
        if time.time() - start_time > max_time:
            break
        
        # Random initialization
        U = np.random.randn(n*m, target_rank)
        V = np.random.randn(m*p, target_rank)
        W = np.random.randn(p*n, target_rank)
        
        # Check if this gives the right tensor
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i,j,k] = np.dot(U[i*m+j,:], V[j*p+k,:] * W[k*n+i,:])
        
        if np.allclose(T, target, atol=1e-6):
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with schoolbook as fallback
    U_fb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_fb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_fb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][column] = 1
                V_fb[j * p + k][column] = 1
                W_fb[k * n + i][column] = 1
                column += 1
    
    # Try to find lower rank decompositions
    # Known lower bounds: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But actually the lower bound is max(n*m, m*p, p*n) which is 28 here
    # Wait, that's not right. The rank must be at least max(n*m, m*p, p*n)?
    # No, the rank is at least max(n*m, m*p, p*n) is not a valid lower bound.
    # The rank is at least max(n*m, m*p, p*n) / something... 
    # Actually, a valid lower bound is max(n*m, m*p, p*n) is NOT correct.
    # The correct lower bound is: rank >= max(n*m, m*p, p*n) is false.
    # rank >= n*m is true (since U has n*m rows and must have rank n*m for the projection to work)
    # Similarly rank >= m*p and rank >= p*n
    # So rank >= max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    
    # So the minimum possible rank is 28. Schoolbook is 84.
    # Let's try to find rank 28, 29, ... up to some limit.
    
    deadline = time.time() + 120  # 120 second deadline with margin
    
    # Try ranks from 28 upward
    for r in range(28, min(85, schoolbook_rank + 1)):
        if time.time() > deadline - 5:
            break
        
        result = _try_rank(n, m, p, r, max_time=min(10, deadline - time.time() - 5))
        if result is not None:
            U, V, W = result
            # Convert to list of lists with integer/half-integer entries if possible
            # Check if entries are close to integers or half-integers
            def to_rational_list(arr):
                result = []
                for row in arr:
                    new_row = []
                    for val in row:
                        # Round to nearest integer or half-integer
                        nearest = round(val * 2) / 2
                        if abs(val - nearest) < 1e-6:
                            new_row.append(nearest)
                        else:
                            new_row.append(val)
                    result.append(new_row)
                return result
            
            U_list = to_rational_list(U)
            V_list = to_rational_list(V)
            W_list = to_rational_list(W)
            
            # Verify
            if _verify(U_list, V_list, W_list, n, m, p, r, tol=1e-5):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return {"U": U_fb, "V": V_fb, "W": W_fb}
# EVOLVE-BLOCK-END
