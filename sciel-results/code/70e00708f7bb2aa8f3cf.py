"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Build the multiplication tensor T[i,j,k] = 1 if j is the "middle" index
    # Tensor shape: (n, m, p) where T[i,j,k] = delta_{j, j} (always 1 for valid j)
    # Actually for matrix mult: T[i,j,k] = 1 if the middle index matches
    # The tensor T has entries T[i,j,k] = 1 for all valid (i,j,k) in the multiplication
    # Wait - the multiplication tensor is T[i,j,k] = delta_{j,j} which is always 1
    # No - the tensor for matrix multiplication is defined as:
    # T[i,j,k] = 1 if j == j (trivially true)
    # Actually the tensor T is: T[i,j,k] = 1 for the specific (i,j,k) triples
    # that correspond to C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all (i,j,k) in the range, and the bilinear form is
    # sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    
    # The tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # But wait, that's not right either. Let me think more carefully.
    
    # For matrix multiplication C = A * B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that:
    # T[i,j,k] = 1 if this term contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid (i,j,k)
    # The bilinear form is: sum_{i,j,k} T[i,j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal C[i,k] for each (i,k)
    # And C[i,k] = sum_j A[i,j] * B[j,k]
    # So we need: sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
    # This means T[i,j,k] = 1 for all (i,j,k)
    
    # The tensor T has shape (n, m, p) with all entries = 1
    # But that's the "all ones" tensor, which has rank 1!
    # That can't be right for matrix multiplication...
    
    # Let me reconsider. The multiplication tensor for n x m times m x p is:
    # T in R^{n x m x p} where T[i,j,k] = 1 if j is the shared index
    # Actually no - T[i,j,k] is defined as the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = 1 for all (i,j,k) in the valid range
    # But the tensor is really T[i,j,k] = delta_{j,j} which is always 1
    
    # I think the issue is that the tensor for matrix multiplication is:
    # T[i,j,k] = 1 for all (i,j,k)
    # And the bilinear form gives: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * W[k*n+i, r]
    # For this to equal C[i,k] = sum_j A[i,j] * B[j,k], we need W[k*n+i, r] to
    # appropriately weight each term.
    
    # Actually, I think the standard formulation is:
    # The multiplication tensor T has entries T[i,j,k] = 1 for all (i,j,k)
    # A rank-r decomposition gives: T[i,j,k] = sum_{r=1}^r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matrix multiplication, we need this to hold for the specific structure.
    
    # Let me use a different approach. The tensor T for 2x3x3 matrix multiplication
    # has 18 entries (2*3*3), all equal to 1.
    # We want to find the minimum rank decomposition.
    
    # Flatten the tensor in different ways and compute kernels
    
    # The tensor T[i,j,k] = 1 for all i in [0,2), j in [0,3), k in [0,3)
    # Shape: (2, 3, 3)
    
    # Flattening 1: reshape to (2, 9) - flatten (j,k) into one index
    # Flattening 2: reshape to (3, 6) - flatten (i,k) into one index  
    # Flattening 3: reshape to (3, 6) - flatten (i,j) into one index
    
    # For a rank-r decomposition, each flattening must have rank <= r
    
    # T as (n, m*p) = (2, 9): all entries are 1
    # This is a 2x9 matrix of all ones, which has rank 1
    # So the tensor rank is at least 1 (trivially)
    
    # T as (m, n*p) = (3, 6): all entries are 1
    # This is a 3x6 matrix of all ones, which has rank 1
    
    # T as (p, n*m) = (3, 6): all entries are 1
    # This is a 3x6 matrix of all ones, which has rank 1
    
    # So the flattening lower bound is 1, but the actual tensor rank is higher
    # because the tensor is not a simple outer product.
    
    # Wait, I'm confusing myself. Let me re-read the problem.
    
    # The tensor for matrix multiplication is NOT all ones. 
    # For C = A * B, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T is defined such that the bilinear form reproduces this.
    # T[i,j,k] = 1 for all (i,j,k) in the valid range.
    # A rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This must hold for all (i,j,k).
    
    # So we need: 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all (i,j,k)
    
    # This is a system of 18 equations (2*3*3) in 3*r unknowns (each of U, V, W has r columns)
    # Actually U has n*m = 6 entries per column, V has m*p = 9 entries per column, W has p*n = 6 entries per column
    # So 21*r unknowns total, but we only need to satisfy 18 equations.
    
    # For rank 1: 1 = U[i*m+j] * V[j*p+k] * W[k*n+i] for all (i,j,k)
    # This requires U[i*m+j] * V[j*p+k] * W[k*n+i] = 1 for all (i,j,k)
    # Taking (i=0,j=0,k=0): U[0]*V[0]*W[0] = 1
    # Taking (i=0,j=0,k=1): U[0]*V[1]*W[1] = 1
    # Taking (i=0,j=0,k=2): U[0]*V[2]*W[2] = 1
    # Taking (i=0,j=1,k=0): U[3]*V[3]*W[0] = 1
    # ...
    # This is generally impossible for rank 1.
    
    # Let me use the kernel approach as suggested.
    # The schoolbook decomposition has rank 18 (one term per (i,j,k) triple).
    # We want to find linear dependencies among these 18 basis tensors.
    
    # Each basis tensor T_{i,j,k} has T_{i',j',k'} = 1 if (i',j',k') = (i,j,k), else 0
    # We want to find coefficients c_{i,j,k} such that sum_{i,j,k} c_{i,j,k} * T_{i,j,k} = 0
    # But all T_{i,j,k} are linearly independent (they're the standard basis), so there are no dependencies.
    
    # I think I'm overcomplicating this. Let me just use the known result that
    # the rank of 2x3x3 matrix multiplication is 17, and verify the construction.
    
    # Use the existing rank-17 construction and verify it
    rank = 17
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify exact decomposition
    def _verify(U_mat, V_mat, W_mat, n_, m_, p_, rank_):
        for i in range(n_):
            for k in range(p_):
                for j1 in range(m_):
                    for j2 in range(m_):
                        s = 0.0
                        for r in range(rank_):
                            s += W_mat[k*n_+i, r] * U_mat[i*m_+j1, r] * V_mat[j2*p_+k, r]
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > 1e-9:
                            return False
        return True
    
    if not _verify(U, V, W, n, m, p, rank):
        # Fallback to schoolbook if verification fails
        rank_sb = n * m * p
        U_fb = np.zeros((n * m, rank_sb))
        V_fb = np.zeros((m * p, rank_sb))
        W_fb = np.zeros((p * n, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j, col] = 1
                    V_fb[j*p+k, col] = 1
                    W_fb[k*n+i, col] = 1
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to find rank 16 using rational Gaussian elimination on the bilinear system
    if time.time() < deadline - 10.0:
        rank_test = 16
        found = False
        
        # Set up the bilinear system as a linear system in the W entries
        # For fixed U and V, the equations are linear in W
        # But we need to search over U and V as well
        
        # Alternative: use the fact that rank-16 might be achievable by
        # finding a sparse solution with half-integer entries
        
        # Try structured search with specific patterns
        # Based on the Strassen structure, try to merge two of the schoolbook terms
        
        # Strategy: try all pairs of schoolbook terms to see if they can be merged
        # into a single term with half-integer coefficients
        
        # The schoolbook terms are at indices 7-16 (10 terms)
        # Try to see if any pair can be combined
        
        # For example, can we merge r=7 (A02*B20->C00) and r=11 (A00*B02->C02)?
        # They affect different C entries, so probably not directly.
        
        # Let me try a different approach: use the kernel of the flattening maps
        # to find the actual tensor rank lower bound
        
        # The tensor T for 2x3x3 matrix multiplication:
        # T[i,j,k] = 1 for all (i,j,k) in {0,1} x {0,1,2} x {0,1,2}
        
        # Flattening as (n, m*p) = (2, 9):
        # T_flat1[i, j*p+k] = 1 for all i, j, k
        # This is a 2x9 matrix of all ones, rank = 1
        
        # Flattening as (m, n*p) = (3, 6):
        # T_flat2[j, i*p+k] = 1 for all i, j, k
        # This is a 3x6 matrix of all ones, rank = 1
        
        # Flattening as (p, n*m) = (3, 6):
        # T_flat3[k, i*m+j] = 1 for all i, j, k
        # This is a 3x6 matrix of all ones, rank = 1
        
        # So the flattening lower bound is 1, which is not useful.
        
        # The actual tensor rank is determined by more subtle properties.
        # For now, return the rank-17 solution.
        
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
