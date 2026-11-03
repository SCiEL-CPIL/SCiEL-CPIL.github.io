"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=8
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Build the multiplication tensor
    # T[i,j,k] = 1 if A[i] and B[j] contribute to C[k]
    # i: (r, k) in A, r in [0,n), k in [0,m)
    # j: (k, c) in B, k in [0,m), c in [0,p)
    # k: (c, r) in C, c in [0,p), r in [0,n)
    # T[i,j,k] = 1 if k_A = k_B and r_C = r_A and c_C = c_B
    
    # Schoolbook decomposition
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for r in range(n):
            for k in range(m):
                for c in range(p):
                    U[r * m + k, col] = 1
                    V[k * p + c, col] = 1
                    W[c * n + r, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a lower rank decomposition
    # Use the structure: for each output column c, C[:,c] = A @ B[:,c]
    # We can precompute linear combinations of A
    
    # Strategy: Use a rank-r decomposition where we precompute r linear combinations of A
    # and for each, compute products with B
    
    # Let's try to use the 3x3x3 Strassen decomposition (rank 7) as a building block
    # and extend it to handle 8 columns
    
    # For 3x3x3, the tensor T3[i,j,k] where:
    # i = (r,k), j = (k,c), k = (c,r)
    # T3[(r,k),(k',c),(c',r')] = 1 if k=k' and r=r' and c=c'
    
    # Strassen's rank-7 decomposition for 3x3x3:
    # We need to find 7 rank-1 tensors that sum to T3
    
    # Let me try a direct approach: construct the tensor and try to find a low-rank decomposition
    # using the fact that the problem has structure
    
    # Build the full tensor T of shape (n*m, m*p, p*n) = (9, 24, 24)
    T = np.zeros((n * m, m * p, p * n))
    for r in range(n):
        for k in range(m):
            for c in range(p):
                i = r * m + k
                j = k * p + c
                l = c * n + r
                T[i, j, l] = 1
    
    # Try to find a low-rank decomposition
    # Use the structure: T = sum_s U_s ⊗ V_s ⊗ W_s
    
    # Approach: Use the fact that for each pair (r,c), the contribution is:
    # C[r,c] = sum_k A[r,k] * B[k,c]
    # So T has a "block" structure
    
    # Let me try a construction with rank = n * p * (something)
    # Or use the 3x3x3 decomposition extended
    
    # For now, let's try to use a rank-24 decomposition
    # by precomputing 3 linear combinations of A for each of the 8 columns
    
    # Actually, let's try a more systematic approach
    # The tensor can be written as:
    # T = sum_{r,k,c} e_{(r,k)} ⊗ e_{(k,c)} ⊗ e_{(c,r)}
    
    # We can group by r and c:
    # T = sum_{r,c} (sum_k e_{(r,k)} ⊗ e_{(k,c)} ⊗ e_{(c,r)})
    
    # For fixed (r,c), the inner sum is a rank-1 tensor in the k dimension
    
    # Let me try to use a decomposition where we share the A-side factors
    
    # Construction: For each of the m=3 intermediate indices k, and for each output (r,c),
    # we have a term. That's 3*3*8 = 72 terms (schoolbook).
    
    # To reduce, we need to find linear dependencies or use a smarter decomposition.
    
    # Let me try a rank-18 decomposition:
    # Precompute 3 linear combinations of each row of A (total 9 precomputed values)
    # Then for each of the 8 columns, use 2 products... no, that doesn't work.
    
    # Let me try a different approach: use the known result that the tensor rank
    # of (n×m) × (m×p) is at most min(n*m*p, ...) and try to find a good bound.
    
    # For 3x3 x 3x8, let me try rank 24
    # Idea: For each of the 8 columns, use a rank-3 decomposition of the 3x3x3x1 tensor
    # But that's still 8*9 = 72 if done independently.
    
    # Better idea: Use the fact that A is shared. Precompute P = A @ X where X is 3x3
    # Then C[:,c] is a linear combination of columns of P.
    
    # Let me try: Precompute 3 matrices P_s = L_s(A) for s=1,2,3
    # Then C[:,c] = sum_s P_s @ M_s(c) where M_s(c) is 3x1
    
    # The total rank would be 3 * 8 = 24 if each P_s is used with all 8 columns.
    
    # But we need to verify this works. Let me construct it.
    
    # Actually, this is just the schoolbook in disguise if L_s are identity-like.
    
    # Let me try a more clever approach: use the 3x3x3 Strassen decomposition
    # and extend it to 8 columns.
    
    # For 3x3x3, rank-7 decomposition exists. Let me find it.
    
    # Strassen for 3x3x3 (from literature):
    # The 7 products are:
    # P1 = (A00 + A22) * (B00 + B22)
    # P2 = (A01 + A11 + A21) * B00
    # P3 = A00 * (B01 - B11)
    # P4 = A22 * (B11 - B00)
    # P5 = (A02 + A12 + A22) * B11
    # P6 = (A10 + A20) * (B00 + B10)
    # P7 = (A00 + A10) * (B10 - B20)
    # ... this is getting complex and I'm not sure of the exact form.
    
    # Let me try a simpler approach: use a rank-9 decomposition for the 3x3 part
    # (which is just the schoolbook for 3x3x3) and extend to 8 columns.
    
    # For 3x3x3, schoolbook is rank 9. For 3x3 x 3x8, if we use the same A-side
    # factors for all 8 columns, the rank would be 9 * 8 = 72.
    
    # To do better, we need to share more.
    
    # Let me try a construction with rank 24:
    # Use 3 "A-side" factors and 8 "B-side" factors per A-factor.
    
    # Actually, let me just try to find any valid decomposition with rank < 72
    # by using a structured approach.
    
    # Try rank 24: precompute 3 linear combinations of A, then multiply each by B
    rank = 24
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # For s in 0,1,2 (A-side factor index), c in 0..7 (column index):
    # U[:, s*8+c] = some linear combination of A
    # V[:, s*8+c] = B[:, c] (the c-th column of B)
    # W[:, s*8+c] = some linear combination to produce C[:, c]
    
    # This won't work directly because we need the products to combine correctly.
    
    # Let me try a different construction.
    
    # Use the fact that C[r,c] = sum_k A[r,k] * B[k,c]
    # = sum_k A[r,k] * B[k,c]
    
    # If I precompute for each r: A_r = A[r,:] (row r of A)
    # Then C[r,c] = A_r @ B[:,c]
    
    # The tensor for this is: for each (r,c), sum_k e_{(r,k)} ⊗ e_{(k,c)} ⊗ e_{(c,r)}
    
    # I can write this as: for each r, the tensor is sum_k e_{(r,k)} ⊗ e_{(k,:)} ⊗ e_{(:,r)}
    # where e_{(k,:)} is a vector in the B-space and e_{(:,r)} is in the C-space.
    
    # Let me try to use a decomposition where I share the k-index computation.
    
    # For each k, define:
    # U_k = sum_r e_{(r,k)} ⊗ (something)
    # This is getting too abstract.
    
    # Let me just return the schoolbook decomposition to ensure correctness,
    # and then try to improve it.
    
    U, V, W = schoolbook()
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition using a structured approach
    # Use the 3x3x3 tensor product with an 8-dimensional space
    
    # The 3x3x3 multiplication tensor has rank 9 (schoolbook) or 7 (Strassen)
    # The 3x8 part can be seen as 8 copies of the 3x1 case
    
    # Let me try to use a rank-7 decomposition for 3x3x3 and extend it.
    
    # For 3x3x3, I'll use a known rank-7 decomposition.
    # From the literature, one such decomposition uses the following:
    
    # Let me try to construct it directly.
    
    # For 3x3x3, the tensor T3 has T3[(r,k),(k',c),(c',r')] = 1 if k=k', r=r', c=c'
    
    # A rank-7 decomposition would be 7 rank-1 tensors.
    
    # Let me try to find one by solving a system of equations.
    
    # Actually, let me try a simpler approach: use a rank-9 decomposition (schoolbook)
    # for 3x3x3 and see if I can extend it to 3x3x8 with rank 9*8=72, which is the same
    # as the schoolbook for the full problem.
    
    # To do better, I need a non-trivial decomposition.
    
    # Let me try to use the following construction:
    # For 3x3 x 3x8, use rank 24 by precomputing 3 "combined" A matrices.
    
    # Define for s=0,1,2:
    # A_s = A @ S_s where S_s is a 3x3 matrix
    # Then C[:,c] = sum_s A_s @ T_s(c) where T_s(c) is 3x1
    
    # The total rank is 3 * 8 = 24.
    
    # But I need to verify that this works. Let me construct it.
    
    # For this to work, I need:
    # C[r,c] = sum_k A[r,k] * B[k,c] = sum_s (A @ S_s)[r,k'] * T_s(c)[k']
    # = sum_s sum_k' A[r,k'] * S_s[k',k''] * T_s(c)[k'']
    
    # This requires that for each (r,c):
    # sum_k A[r,k] * B[k,c] = sum_s sum_k' sum_k'' A[r,k'] * S_s[k',k''] * T_s(c)[k'']
    
    # This is a constraint on S_s and T_s.
    
    # Let me try S_0 = I, S_1 = 0, S_2 = 0, and T_0(c) = B[:,c], T_1(c) = 0, T_2(c) = 0.
    # Then C[r,c] = sum_k' A[r,k'] * I[k',k''] * B[k'',c] = sum_k A[r,k] * B[k,c]. ✓
    
    # But this is just the schoolbook with rank 9*8 = 72, not 24.
    
    # The issue is that T_s(c) depends on c, so we need a separate term for each c.
    
    # Let me try a different approach: use the tensor product structure.
    
    # The 3x3 x 3x8 tensor is the tensor product of the 3x3x3 tensor and an 8-dimensional
    # "identity" tensor, but that gives rank 9*8 = 72.
    
    # To reduce rank, I need to find a decomposition that shares computation.
    
    # Let me try to use a rank-18 decomposition:
    # Precompute 2 linear combinations of A (total 2*9 = 18 precomputed values)
    # Then for each of the 8 columns, use 2 products... no, that's 16 products, not 18.
    
    # I'm going in circles. Let me just return the schoolbook decomposition.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
