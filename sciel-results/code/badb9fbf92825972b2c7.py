"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def _verify(U, V, W, n, m, p, rank):
    """Check if the decomposition is exact: C = sum_r U[:,r] * V[:,r] * W[:,r] (outer product structure)"""
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, T[i,j,k] should be 1 if this corresponds to a valid (i,j,k) triple,
    # but actually the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k (it's the identity 
    # in the sense that C[i,k] = sum_j A[i,j]B[j,k]).
    # Actually, the matrix multiplication tensor M has entries M[i,j,k] = delta_{i,j} * delta_{j,k}... 
    # No. Let me think again.
    # 
    # The matrix multiplication tensor is defined as:
    # M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (something)
    # Actually, for matrix multiplication C = AB where A is n×m, B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T = sum_{i,j,k} A_{i,j} * B_{j,k} * C_{i,k}
    # As a multilinear form, T(A,B,C) = sum_{i,j,k} A_{i,j} B_{j,k} C_{i,k}
    # The tensor indices are (i,j,k) and the structure is:
    # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the coefficient is 1
    # But wait, that's not right either. The tensor for matrix multiplication is:
    # T_{i,j,k} = 1 for all i,j,k? No.
    # 
    # Let me reconsider. The matrix multiplication tensor M_{n,m,p} is the tensor such that
    # <M, A ⊗ B ⊗ C> = trace(A B C) for appropriate dimensions, or more precisely
    # sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i}... 
    #
    # Actually, the standard definition: the matrix multiplication tensor has entries
    # T[i,j,k] = 1 if the triple (i,j,k) is valid, i.e., T is the tensor with all entries 1?
    # No, that would be the all-ones tensor.
    #
    # The correct definition: T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} 
    #   e_i ⊗ e_j ⊗ e_k
    # No, that's not right. The matrix multiplication tensor is:
    # T = sum_{i,j,k} (e_i ⊗ e_j ⊗ e_k) where the coefficient involves the structure of matmul.
    #
    # Let me use the schoolbook decomposition as reference:
    # In the schoolbook, for each (i,j,k) triple, we have a rank-1 term:
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1
    # So the reconstructed tensor is sum_c U[:,c] ⊗ V[:,c] ⊗ W[:,c]
    # where the tensor index (a,b,c) has value sum_r U[a,r]*V[b,r]*W[c,r]
    #
    # For the schoolbook, at position (a=i*m+j, b=j'*p+k', c=k''*n+i''):
    # sum_r U[a,r]*V[b,r]*W[c,r] = 1 if there exists r such that U[a,r]=V[b,r]=W[c,r]=1
    # In schoolbook, each r corresponds to a unique (i,j,k) triple, so:
    # U[i*m+j, r] = 1 iff r corresponds to (i,j,k) for some k
    # V[j*p+k, r] = 1 iff r corresponds to (i,j,k) for some i
    # W[k*n+i, r] = 1 iff r corresponds to (i,j,k) for some j
    #
    # So the reconstructed tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for the SAME (i,j,k)
    # i.e., T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k, and 0 otherwise.
    #
    # This is the matrix multiplication tensor! The entries are 1 at positions (i*m+j, j*p+k, k*n+i)
    # and 0 elsewhere. This makes sense because:
    # <T, A⊗B⊗C> = sum_{i,j,k} A[i*m+j] * B[j*p+k] * C[k*n+i]
    # If we reshape A to n×m, B to m×p, C to p×n, then this is sum_{i,j,k} A[i,j]*B[j,k]*C[k,i]
    # which is trace(ABC) when dimensions allow, or more generally the contraction.
    #
    # So the target tensor has T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k, else 0.

    # Build target tensor
    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    # Reconstruct from decomposition
    # T_reconstructed[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
    # This is a tensor product sum, which can be computed as:
    # T = sum_r outer(U[:,r], V[:,r]) * W[:,r] ... no
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is the element-wise product summed over r.
    # Efficiently: T = np.einsum('ar,br,cr->abc', U, V, W)
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check shape
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # Compute reconstructed tensor
    # For efficiency with small sizes, use einsum
    try:
        recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr, optimize=True)
    except Exception:
        return False
    
    # Check exact match (allowing small floating point error)
    if not np.allclose(recon, target, atol=1e-9, rtol=1e-9):
        return False
    
    return True

def _strassen_2x2_decomposition():
    """
    Use Strassen's algorithm for 2x2 block structure.
    Split 4x5 matrix into 2x2 blocks of sizes (2x2, 2x3), (2x2, 2x3)
    Split 5x6 matrix into 2x2 blocks of sizes (2x3, 2x3), (2x3, 3x3)... 
    
    Actually, let's think about this more carefully.
    4x5 times 5x6.
    Split 4 into 2+2, split 5 into 2+3, split 6 into 3+3.
    
    A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x3, A21 is 2x2, A22 is 2x3
    B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x3, B21 is 3x3, B22 is 3x3
    
    C = AB = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
              [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    This is getting complex with mixed sizes. Let me try a different approach.
    
    Alternative: Pad to 4x6 * 6x6 or use a different block structure.
    
    Actually, let me try: split 4=2+2, 5=3+2, 6=3+3
    A: 4x5 -> rows split 2+2, cols split 3+2
    A11: 2x3, A12: 2x2, A21: 2x3, A22: 2x2
    B: 5x6 -> rows split 3+2, cols split 3+3
    B11: 3x3, B12: 3x3, B21: 2x3, B22: 2x3
    
    C11 = A11*B11 + A12*B21  (2x3)
    C12 = A11*B12 + A12*B22  (2x3)
    C21 = A21*B11 + A22*B21  (2x3)
    C22 = A21*B12 + A22*B22  (2x3)
    
    Each block product is a small matrix multiplication. The total rank is the sum of ranks of each block product.
    - A11*B11: 2x3 * 3x3 -> rank at most 2*3*3=18, but can be lower
    - A12*B21: 2x2 * 2x3 -> rank at most 2*2*3=12
    - etc.
    
    This doesn't obviously help. Let me think about Strassen more carefully.
    
    Strassen works when we can group into 2x2 blocks of equal size. For 4x5*5x6, 
    we can't easily get equal-sized 2x2 blocks.
    
    Let me try a different approach: use the known result that matmul(4,5,6) has rank at most 108
    (from the knowledge: Strassen reduced 120 to 108).
    
    The idea: partition the 4x5 matrix into blocks and use Strassen on 2x2 subproblems.
    
    Split 4=2+2, 5=2+3, 6=2+4:
    A: 4x5, B: 5x6
    A11:2x2, A12:2x3, A21:2x2, A22:2x3
    B11:2x2, B12:2x4, B21:3x2, B22:3x4
    
    C11 = A11*B11 + A12*B21: 2x2*2x2 + 2x3*3x2 = 2x2
    C12 = A11*B12 + A12*B22: 2x2*2x4 + 2x3*3x4 = 2x4
    C21 = A21*B11 + A22*B21: 2x2*2x2 + 2x3*3x2 = 2x2
    C22 = A21*B12 + A22*B22: 2x2*2x4 + 2x3*3x4 = 2x4
    
    For C11: 2x2*2x2 has rank 8 (schoolbook) or 7 (Strassen). 2x3*3x2 has rank 18.
    Total for C11: 7+18=25? That's worse than schoolbook 8+18=26. Marginal improvement.
    
    This approach is getting complicated. Let me try a simpler strategy:
    Use Strassen on the 2x2 block structure where possible, and schoolbook elsewhere.
    
    Actually, let me just try to find a decomposition with rank < 120 using optimization.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # Leave 30s margin
    
    # Start with schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try to reduce rank using random restarts and local search
    # Strategy: start with schoolbook, randomly zero out some entries and try to repair
    # Or: use a greedy approach to merge rank-1 terms
    
    # Alternative: use the fact that we can sometimes combine terms
    # For now, let's try a simple random perturbation approach
    
    random.seed(42)
    
    # Try to find a lower rank decomposition by random search
    # This is a heuristic approach
    
    for trial in range(50):
        if time.time() > deadline - 30:
            break
        
        # Try random rank-1 eliminations
        current_U = [row[:] for row in best_U]
        current_V = [row[:] for row in best_V]
        current_W = [row[:] for row in best_W]
        current_rank = best_rank
        
        # Try to merge pairs of rank-1 terms
        merged = False
        for attempt in range(100):
            if time.time() > deadline - 30:
                break
            if current_rank <= 1:
                break
            
            # Pick two random columns
            c1 = random.randint(0, current_rank - 1)
            c2 = random.randint(0, current_rank - 1)
            if c1 == c2:
                continue
            
            # Try to express the sum of two rank-1 terms as a single rank-1 term
            # This is generally not possible, but let's try a different approach:
            # Randomly perturb and check
            
            # Actually, let's try a different strategy:
            # Randomly set some entries to 0 and see if the decomposition still works
            # after adjusting other entries
            
            # For now, let's just try random rank reductions
            new_rank = current_rank - 1
            if new_rank < 1:
                break
            
            # Create new U, V, W with one fewer column
            new_U = [row[:current_rank-1] for row in current_U]
            new_V = [row[:current_rank-1] for row in current_V]
            new_W = [row[:current_rank-1] for row in current_W]
            
            # Try to find values that work - this is complex, so skip for now
            # Instead, let's use a known algebraic structure
            
        if _verify(current_U, current_V, current_W, n, m, p, current_rank):
            if current_rank < best_rank:
                best_U, best_V, best_W = current_U, current_V, current_W
                best_rank = current_rank
    
    # Return the best decomposition found
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
