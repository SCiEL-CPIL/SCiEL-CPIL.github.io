"""Matrix multiplication 3,3,4 - rank-minimal tensor decomposition."""
import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.
    Uses a hybrid approach: start with schoolbook, then attempt local search to reduce rank.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    schoolbook_rank = n * m * p  # 36
    
    # Build the multiplication tensor T[i, j, k] = delta_{i,k} (for C[i,k] = sum_j A[i,j]*B[j,k])
    # Actually, T[i,j,k] = 1 if i==k else 0, where C[i,k] = sum_j A[i,j] B[j,k]
    # Wait, the tensor for matrix multiplication is: M[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = delta_{i,k}... no.
    # The matrix multiplication tensor has entries T[i,j,k] such that the bilinear form is sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,k} C[i,k]
    # For C[i,k] = sum_j A[i,j] B[j,k], we have T[i,j,k] = 1 if the output index matches.
    # Actually: C[i,k] = sum_j A[i,j] * B[j,k], so the tensor is T[i,j,k] = 1 for all i,j,k where the "output" is C[i,k].
    # The tensor M of shape (n, m, p) with M[i,j,k] = 1 if i == k? No.
    # Let me think again. The matrix multiplication tensor T has components T[i,j,k] where:
    # - i indexes rows of A (and C): 0..n-1
    # - j indexes columns of A / rows of B: 0..m-1  
    # - k indexes columns of B (and C): 0..p-1
    # T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k]. It always does.
    # So T[i,j,k] = 1 for all i,j,k? No, that's not right either.
    # 
    # The correct tensor: The matrix multiplication map is defined by:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # This can be written as a trilinear form: C[i,k] * D[k,i] = sum_{i,j,k} D[k,i] * A[i,j] * B[j,k]
    # The tensor T[i,j,k] = 1 for all valid i,j,k.
    # Wait no. Let me re-examine.
    #
    # For the tensor decomposition, we need:
    # U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank
    # such that for each output position (i,k) in C (indexed as k*n+i in W),
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_c U[i*m+j] * V[j*p+k] * W[k*n+i] (over j and c)
    # 
    # More precisely: the tensor T of shape (n*m, m*p, p*n) where:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # and all other entries are 0.
    #
    # A rank-1 decomposition is: T = sum_c U[:,c] outer V[:,c] outer W[:,c]
    # where U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    n, m, p = DIMENSIONS
    Nm = n * m  # 9
    Mp = m * p  # 12
    Pn = p * n  # 12
    
    # Build the target tensor as a 3D array
    # T[a, b, c] where a in [0,Nm), b in [0,Mp), c in [0,Pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((Nm, Mp, Pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook: rank = n*m*p = 36
    schoolbook_rank = n * m * p
    
    def make_schoolbook():
        rank = schoolbook_rank
        U = np.zeros((Nm, rank), dtype=np.int64)
        V = np.zeros((Mp, rank), dtype=np.int64)
        W = np.zeros((Pn, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-9):
        """Check if U, V, W exactly decompose T."""
        rank = U.shape[1]
        # Reconstruct tensor
        recon = np.zeros((Nm, Mp, Pn), dtype=np.float64)
        for c in range(rank):
            recon += np.outer(np.outer(U[:,c], V[:,c]), W[:,c])
        return np.allclose(recon, T, atol=tol, rtol=0)
    
    def verify_integer(U, V, W):
        """Check exact integer decomposition."""
        rank = U.shape[1]
        recon = np.zeros((Nm, Mp, Pn), dtype=np.int64)
        for c in range(rank):
            recon += np.outer(np.outer(U[:,c], V[:,c]), W[:,c])
        return np.array_equal(recon, T)
    
    # Start with schoolbook
    U, V, W = make_schoolbook()
    best_rank = schoolbook_rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if verify_integer(U, V, W):
        pass  # schoolbook is valid
    
    # Try to find lower rank using random initialization + local search
    # This is a research-level problem; for 3x3x4 the known minimal rank might be < 36
    # but finding it algorithmically is hard. Let's try a few approaches.
    
    # Approach: random rank-r decomposition with r < schoolbook_rank, then refine
    deadline = time.time() + 100  # 100s budget, leaving 20s safety margin
    
    for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 8, 20), -1):
        if time.time() > deadline:
            break
        # Try random rank-target_rank decomposition
        for attempt in range(20):
            if time.time() > deadline:
                break
            # Random sparse integer matrices
            U_try = np.zeros((Nm, target_rank), dtype=np.int64)
            V_try = np.zeros((Mp, target_rank), dtype=np.int64)
            W_try = np.zeros((Pn, target_rank), dtype=np.int64)
            
            for c in range(target_rank):
                # Each factor gets a sparse random pattern
                for _ in range(3):
                    idx_u = random.randint(0, Nm-1)
                    idx_v = random.randint(0, Mp-1)
                    idx_w = random.randint(0, Pn-1)
                    val = random.choice([-1, 1])
                    U_try[idx_u, c] = val
                    V_try[idx_v, c] = val
                    W_try[idx_w, c] = val
            
            if verify_integer(U_try, V_try, W_try):
                best_rank = target_rank
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                break
        
        if best_rank <= target_rank:
            break
    
    # Convert to lists for the interface
    result_U = best_U.tolist()
    result_V = best_V.tolist()
    result_W = best_W.tolist()
    
    return {"U": result_U, "V": result_V, "W": result_W}
# EVOLVE-BLOCK-END
