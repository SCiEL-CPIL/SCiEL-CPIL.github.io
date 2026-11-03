"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as baseline
    best_result = _schoolbook_decomposition(n, m, p)
    best_rank = n * m * p
    
    # Try Strassen-based hybrid: decompose 3x3x7 into 3x3x3 + 3x3x4
    if time.time() < deadline - 30:
        strassen_result = _strassen_hybrid(n, m, p)
        if strassen_result is not None:
            rank = len(strassen_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = strassen_result
    
    # Try DFT-based decomposition
    if time.time() < deadline - 20:
        dft_result = _dft_decomposition(n, m, p)
        if dft_result is not None:
            rank = len(dft_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = dft_result
    
    # Try structured low-rank search with perturbations
    if time.time() < deadline - 10:
        search_result = _structured_search(n, m, p, best_rank)
        if search_result is not None:
            rank = len(search_result['U'][0])
            if rank < best_rank:
                best_rank = rank
                best_result = search_result
    
    # Final verification
    if time.time() < deadline - 5:
        if not _verify_decomposition(best_result, n, m, p):
            # Fallback to schoolbook if verification fails
            best_result = _schoolbook_decomposition(n, m, p)
    
    return best_result

def _schoolbook_decomposition(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(result, n, m, p, num_trials=20):
    """Verify that U,V,W exactly decompose matrix multiplication."""
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    rank = U.shape[1]
    if V.shape[1] != rank or W.shape[1] != rank:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    
    for _ in range(num_trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Compute via tensor decomposition
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # But this is not how it works. The correct interpretation:
        # C[i,k] = sum_{j,c} A[i,j] * B[j,k] * (something)
        
        # Actually for matrix multiplication tensor:
        # T[i,j,k] = delta[i,j,k] where the multiplication is sum_j A[i,j]*B[j,k]
        # The tensor decomposition is: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, that's not right either.
        
        # The standard tensor decomposition for matrix multiplication:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # As a tensor: M[i,j,k] = 1 if the multiplication uses A[i,j]*B[j,k]
        # Decomposition: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Then: sum_j A[i,j]*B[j,k] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_k V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
        # This doesn't quite work. Let me think more carefully.
        
        # The correct formulation:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # We want: C[i,k] = sum_c alpha_c * beta_c * gamma_c
        # where alpha_c = sum_j U[i*m+j, c] * A[i,j]
        #       beta_c = sum_k V[j*p+k, c] * B[j,k]  -- wait, this mixes j
        # 
        # Actually the standard way:
        # C[i,k] = sum_c (sum_j U_c[i,j] * A[i,j]) * (sum_l V_c[j,l] * B[j,l]) * W_c[k,i]
        # But this sums over j in both U and V terms, which is wrong.
        
        # Let me use the correct tensor network:
        # The multiplication tensor T has T[i,j,k] = 1 (it's just the identity in some sense)
        # No wait. The matrix multiplication tensor is defined as:
        # <T, A ⊗ B> = C where C = A*B
        # T[i,j,k] such that C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's just the all-ones tensor in the right shape)
        # No, that's not right either.
        
        # Let me be very explicit:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # We can write this as: C[i,k] = sum_j (1) * A[i,j] * B[j,k]
        # So the tensor T[i,j,k] = 1 for all i,j,k in valid ranges.
        # 
        # A rank-r decomposition is: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # Wait, the indices need to match. Let me use the convention from the task:
        # U has shape (n*m, rank), indexed by (i*m+j)
        # V has shape (m*p, rank), indexed by (j*p+k)  
        # W has shape (p*n, rank), indexed by (k*n+i)
        # 
        # The decomposition is: for all i,j,k:
        # 1 = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # 
        # Then: C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
        # 
        # Hmm, this doesn't directly give us C[i,k]. The correct interpretation is:
        # We define bilinear forms:
        # f_c(A,B) = sum_{i,j} U[i*m+j, c] * A[i,j] * (something with B)
        # 
        # Actually, the standard way to verify:
        # Compute C_tensor[i,k] = sum_c (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k]) * W[k*n+i, c]
        # Wait no. Let me look at this differently.
        
        # The tensor T[i,j,k] = 1 satisfies:
        # sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = sum_{i,j,k} A[i,j] B[j,k]
        # But we want C[i,k] = sum_j A[i,j] B[j,k], so:
        # C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k] where T[i,j,k] = 1
        
        # For the decomposition T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]:
        # sum_j T[i,j,k] A[i,j] B[j,k] = sum_j sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] A[i,j] B[j,k]
        # = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] A[i,j]) * (sum_j V[j*p+k,c] B[j,k])
        
        # So: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c] * A[i,j]) * (sum_j V[j*p+k,c] * B[j,k])
        
        # Let me verify this formula
        C_computed = np.zeros((n, p))
        for c in range(rank):
            # alpha_c = sum_j U[i*m+j, c] * A[i,j] for each i
            # beta_c = sum_j V[j*p+k, c] * B[j,k] for each k
            # C[i,k] += W[k*n+i, c] * alpha_c[i] * beta_c[k]
            
            alpha = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    alpha[i] += U[i * m + j, c] * A[i, j]
            
            beta = np.zeros(p)
            for k in range(p):
                for j in range(m):
                    beta[k] += V[j * p + k, c] * B[j, k]
            
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += W[k * n + i, c] * alpha[i] * beta[k]
        
        if not np.allclose(C_computed, C, atol=1e-8):
            return False
    
    return True

def _strassen_3x3x3_decomposition():
    """Return Strassen's rank-7 decomposition for 3x3x3 matrix multiplication.
    Returns U, V, W as lists of lists with integer entries."""
    # Strassen's algorithm for 2x2x2 has rank 7. For 3x3x3, we can embed.
    # Actually, let me use the known rank-7 decomposition for 2x2x2 and extend.
    # 
    # For 2x2x2: C = A*B where A,B are 2x2
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A11)(B10+B11)
    # M7 = (A10-A00)(B00+B01)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For 3x3x3, I'll use a different approach. Let me just return None for now
    # and rely on other methods.
    return None

def _strassen_hybrid(n, m, p):
    """Try to use Strassen-like decomposition for the 3x3x3 block and schoolbook for rest."""
    # This is a placeholder - the 3x3x7 problem is complex
    # For now, return None to indicate this approach needs more work
    return None

def _dft_decomposition(n, m, p):
    """
    Try DFT-based decomposition.
    
    Idea: The 7-dimensional output can be analyzed via DFT.
    For each frequency component, we get a 3x3x3 sub-problem.
    If we can solve each sub-problem with rank r, total rank is 7*r.
    With Strassen (rank 7 for 3x3x3), total would be 49.
    
    However, the DFT introduces complex numbers, and we need integer/half-integer entries.
    Also, the 7 is prime, so DFT doesn't simplify things as nicely as for composite dimensions.
    
    For now, return None.
    """
    return None

def _structured_search(n, m, p, max_rank):
    """
    Try structured low-rank search.
    
    Start from schoolbook and try to merge/cancel terms.
    """
    # This is a placeholder for now
    return None

# EVOLVE-BLOCK-END
