import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    # Use Bini et al. (1979) rank-23 algorithm for 3x3 matrix multiplication.
    # The 3x3x6 multiplication is decomposed as two independent 3x3x3 multiplications
    # (columns 0-2 and 3-5 of B), giving rank 23+23 = 46.
    #
    # Bini et al. rank-23 algorithm for 3x3:
    # The 23 terms are derived from the following structure.
    # We use the standard formulation with indices 0..22.
    
    # Bini et al. (1979) 3x3x3 rank-23 algorithm
    # U, V, W are 9 x 23 matrices (flattened 3x3 indices)
    # The algorithm uses the following 23 terms:
    
    n, m, p = DIMENSIONS  # n=3, m=3, p=6
    rank = 46
    
    U = [[0] * rank for _ in range(n * m)]  # 9 x 46
    V = [[0] * rank for _ in range(m * p)]  # 18 x 46
    W = [[0] * rank for _ in range(p * n)]  # 18 x 46
    
    # Bini et al. rank-23 coefficients for 3x3x3
    # These are the 23 rank-1 terms of the matrix multiplication tensor T_{i,j,k} = delta_{i,k} (after proper indexing)
    # 
    # Standard Bini et al. (1979) algorithm:
    # The 23 terms are given by specific (U,V,W) triples.
    # Let me use the well-known formulation.
    
    # I'll use the explicit Bini et al. rank-23 algorithm.
    # Reference: Bini, Lotti, Piperno (1979), "Fast algorithms for matrix multiplication"
    
    # The 23 terms for 3x3 matrix multiplication:
    # Each term r has U_r (3x3), V_r (3x3), W_r (3x3) such that
    # sum_r U_r * A * V_r * W_r = C (with appropriate contractions)
    
    # Actually, the tensor decomposition is:
    # T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)} = delta_{i,k} (for matmul C = AB)
    # Wait, for C = AB where A is 3x3, B is 3x3, C is 3x3:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = 1 for all (i,j,k) in the appropriate sense...
    # No, the multiplication tensor for 3x3x3 is T = sum_{i,j,k} e_i^* ⊗ e_j^* ⊗ e_k^* where
    # the coefficient of e_i ⊗ e_j ⊗ e_k in A ⊗ B ⊗ C is A_{i,j} B_{j,k} C_{k,i}... 
    
    # Let me be more careful. The matrix multiplication tensor T_{3,3,3} is defined by:
    # T(e_i, e_j, e_k) = delta_{i,k} for the "multiplication" structure
    # Actually no. For C = AB:
    # T(A, B) = C where C_{i,k} = sum_j A_{i,j} B_{j,k}
    # As a trilinear form: sum_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = sum_{i,k} (AB)_{i,k} C_{k,i}
    # So T_{i,j,k} = 1 for all i,j,k? No...
    # T_{i,j,k} contributes to A_{i,j} B_{j,k} C_{k,i}.
    # We want sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k} C_{k,i} = sum_{i,k} C_{i,k} C_{k,i}... no that's not right either.
    
    # Let me think again. We want:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # This is equivalent to the tensor equation:
    # sum_r U^{(r)}_{i,j} V^{(r)}_{j,k} W^{(r)}_{k,i} = delta_{i,i'} delta_{k,k'} ... 
    
    # OK let me just use the known Bini et al. coefficients directly.
    # The rank-23 algorithm for 3x3x3 has the following form.
    # I'll encode it as a list of 23 tuples of (U_flat, V_flat, W_flat) where each is a 9-element vector.
    
    # Bini et al. (1979) rank-23 algorithm - explicit coefficients
    # Using the formulation from various references:
    
    # Let me use a different approach: I'll construct the algorithm from the known structure.
    # The 23 terms consist of:
    # - 9 diagonal terms (one for each (i,j,k) with i=k, j free... no)
    # 
    # Actually, let me just hardcode the known Bini et al. rank-23 algorithm.
    # From the literature, the 23 terms are:
    
    # I'll use the standard Bini et al. coefficients.
    # The algorithm can be described as follows (using 0-indexed):
    
    bini_terms = []
    
    # 9 terms of type (e_i, e_j, e_k) where i=k (diagonal in first and third indices)
    # These handle the "direct" contributions
    for j in range(3):
        for i in range(3):
            # Term: U has 1 at (i,j), V has 1 at (j,k=i), W has 1 at (k=i,i)
            # Wait, I need to be more careful about the tensor structure.
            pass
    
    # Let me use a completely different approach. I'll verify the schoolbook works
    # and try to find a rank-46 solution using two copies of a rank-23 algorithm.
    # 
    # For the rank-23 Bini algorithm, I'll use the following well-known formulation:
    # The 23 rank-1 tensors are:
    
    # From Bini, Lotti, Piperno (1979):
    # The algorithm uses variables:
    # x1..x9 for A, y1..y9 for B, z1..z9 for C
    # 
    # Let me just use the explicit matrix form from a reference implementation.
    
    # I'll define the 23 terms explicitly. Each term is (U_9, V_9, W_9) flattened.
    # Using the standard Bini et al. rank-23:
    
    # Term 0: U=(1,0,0, 0,0,0, 0,0,0), V=(0,0,0, 0,1,0, 0,0,0), W=(1,0,0, 0,0,0, 0,0,0)
    # This gives contribution to C[0,0] from A[0,0]*B[0,0]
    
    # Actually, I realize I should just verify the schoolbook is correct and then
    # try to find if there's a known rank-46 or lower algorithm for 3x3x6.
    
    # For now, let me implement the schoolbook correctly (rank 54) and then
    # attempt to use two rank-23 Bini algorithms for rank 46.
    
    # Bini et al. rank-23 algorithm - I'll use the explicit coefficients from the paper.
    # The 23 terms are (using 0-based indexing for 3x3 matrices flattened row-major):
    
    # I'll use the formulation where:
    # For each term r, we have U_r (3x3), V_r (3x3), W_r (3x3)
    # such that sum_r (U_r * A) * (V_r * B) * W_r = C in the appropriate sense
    
    # Let me use a verified set of Bini et al. coefficients.
    # From standard references, the rank-23 algorithm for 3x3 is:
    
    # I'll define it as a list of 23 (U, V, W) where each is a 3x3 matrix (flattened to 9 elements)
    
    # Using the Bini et al. (1979) algorithm as described in standard textbooks:
    
    # The 23 terms:
    # Types:
    # 1) 9 terms: U = E_{i,j}, V = E_{j,i}, W = E_{i,i} for all i,j (but this gives rank 9, not 23)
    # 
    # I think I'm overcomplicating this. Let me just use the schoolbook (rank 54)
    # since I can't reliably recall the exact Bini coefficients.
    
    # Schoolbook: rank = n*m*p = 3*3*6 = 54
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
# EVOLVE-BLOCK-END
