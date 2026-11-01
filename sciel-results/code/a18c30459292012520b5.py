"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Tensor T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    # U: (n*m) x rank = 6 x rank
    # V: (m*p) x rank = 18 x rank  
    # W: (p*n) x rank = 12 x rank
    # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 36
    
    # Use Kronecker product structure: 6 = 2*3
    # Decompose the 3x6 matrix multiplication as:
    # - Inner: 2x3 matrix (A) times 3x3 matrix (B_inner) -> 2x3 result
    # - Outer: 2x2 identity (I_2) times 2x2 matrix (B_outer) -> 2x2 result
    # Actually, let's think of it differently.
    # 
    # We can view the 3x6 matrix B as B = I_2 ⊗ B_3 where B_3 is 3x3
    # And the multiplication A * B where A is 2x3, B is 3x6
    # = A * (I_2 ⊗ B_3)
    # 
    # This doesn't directly decompose. Let's try a different approach.
    # 
    # The tensor for (2,3,6) matrix multiplication can be decomposed using
    # the known rank-7 decomposition for (2,3,3) and rank-2 for (1,1,2).
    # 
    # For (2,3,3): rank 7 is achievable.
    # For (1,1,2): rank 2 (just two scalar multiplications).
    # Combined rank: 7 * 2 = 14.
    
    # First, let's construct the rank-7 decomposition for (2,3,3) matrix multiplication.
    # The (2,3,3) tensor has entries T[i*3+j, j*3+k, k*2+i] = 1 for i in 0..1, j in 0..2, k in 0..2
    # U: (2*3) x 7 = 6 x 7
    # V: (3*3) x 7 = 9 x 7
    # W: (3*2) x 7 = 6 x 7
    
    # Known rank-7 decomposition for 2x3x3 matrix multiplication:
    # This can be derived from the 3x3x3 rank-7 (Strassen-like) by slicing.
    # 
    # Let's use a known construction. The 2x3x3 multiplication tensor
    # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1,2}, k in {0,1,2}
    # 
    # We can use the following approach:
    # Write A as [[a00, a01, a02], [a10, a11, a12]]
    # B as 3x3 matrix
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # One approach: use the fact that 2x3x3 is a sub-tensor of 3x3x3.
    # The rank-7 decomposition of 3x3x3 (from Strassen's algorithm) can be
    # specialized to the 2x3x3 case.
    
    # For now, let's try a direct construction for rank 14.
    # We'll use the tensor product of a (2,3,3) rank-7 decomposition and a (1,1,2) rank-2 decomposition.
    
    # (1,1,2) decomposition: rank 2
    # T[i*1+j, j*2+k, k*1+i] = 1 for i=0, j=0, k in {0,1}
    # This is just: C[k] = A[0,0] * B[0,k] for k in {0,1}
    # U: (1*1) x 2 = 1 x 2, V: (1*2) x 2 = 2 x 2, W: (2*1) x 2 = 2 x 2
    # U[0,0]=1, U[0,1]=1
    # V[0,0]=1, V[0,1]=0; V[1,0]=0, V[1,1]=1
    # W[0,0]=1, W[0,1]=0; W[1,0]=0, W[1,1]=1
    
    # Now for (2,3,3), let's construct a rank-7 decomposition.
    # Using the known result that rank(2,3,3) = 7.
    
    # One construction: 
    # Let A = [[a, b, c], [d, e, f]]
    # B = [[p, q, r], [s, t, u], [v, w, x]]
    # C[0,0] = a*p + b*s + c*v
    # C[0,1] = a*q + b*t + c*w
    # C[0,2] = a*r + b*u + c*x
    # C[1,0] = d*p + e*s + f*v
    # C[1,1] = d*q + e*t + f*w
    # C[1,2] = d*r + e*u + f*x
    
    # We need 7 rank-1 tensors. Let's use a known construction.
    # 
    # From the literature, one rank-7 decomposition for 2x3x3 is:
    # (This is a known result; we'll construct it explicitly)
    
    # Actually, let's use a simpler approach: try to find rank 14 directly
    # by exploiting the Kronecker structure.
    
    # The (2,3,6) multiplication can be seen as:
    # C[i, k*3+l] = sum_j A[i,j] * B[j, k*3+l] for i in {0,1}, j in {0,1,2}, k in {0,1}, l in {0,1,2}
    
    # If we write B[j, k*3+l] = sum over some decomposition...
    # 
    # Let's try a direct rank-14 construction using the Kronecker product idea.
    # 
    # We'll construct U, V, W of rank 14 directly.
    
    rank = 14
    U = np.zeros((n*m, rank))  # 6 x 14
    V = np.zeros((m*p, rank))  # 18 x 14
    W = np.zeros((p*n, rank))  # 12 x 14
    
    # We'll use the tensor product of (2,3,3) rank-7 and (1,1,2) rank-2.
    # For the (1,1,2) part:
    # U_1: 1 x 2 = [[1, 1]]
    # V_1: 2 x 2 = [[1, 0], [0, 1]]
    # W_1: 2 x 2 = [[1, 0], [0, 1]]
    
    # For the (2,3,3) part, we need a rank-7 decomposition.
    # Let's construct it.
    
    # (2,3,3) tensor: T[i*3+j, j*3+k, k*2+i] = 1
    # U_2: 6 x 7, V_2: 9 x 7, W_2: 6 x 7
    
    # Known rank-7 decomposition for 2x3x3:
    # We'll use a construction based on the 3x3x3 rank-7 (Strassen).
    
    # For 3x3x3, Strassen's algorithm gives rank 7:
    # M1 = (a00 + a11)(b00 + b11)
    # M2 = (a10 + a11)b00
    # M3 = a00(b10 - b11)
    # M4 = a11(b01 - b00)
    # M5 = (a00 + a01)b11
    # M6 = (a10 - a00)(b00 + b10)
    # M7 = (a01 - a11)(b11 + b01)
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M2 + M3 + M7
    
    # For 2x3x3, we need to adapt this. The 2x3x3 tensor is a sub-tensor of 3x3x3
    # (we only need the first 2 rows of A and all 3 columns/rows of B).
    
    # Let's construct the rank-7 decomposition for 2x3x3 explicitly.
    # We'll use the following approach:
    
    # A = [[a00, a01, a02], [a10, a11, a12]]
    # B = 3x3
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # We can use the following 7 products:
    # P1 = (a00 + a10 + a01 + a11) * (b00 + b10 + b01 + b11)  -- this is getting complicated
    
    # Let's use a different known construction.
    # From the literature, rank(2,3,3) = 7 can be achieved with:
    
    # Actually, let's just try to verify if rank 14 works by construction.
    # We'll use the Kronecker product of the (2,3,3) rank-7 and (1,1,2) rank-2.
    
    # For the (2,3,3) part, let's use a simple rank-9 construction first
    # (schoolbook for 2x3x3 is rank 9), then combine with rank-2 for (1,1,2)
    # to get rank 18. But we want rank 14.
    
    # Let's try a direct search for rank 14 using the Kronecker structure.
    
    # We'll parameterize U, V, W using the Kronecker product structure.
    # U = U_2 ⊗ U_1, V = V_2 ⊗ V_1, W = W_2 ⊗ W_1
    # where U_2 is 6 x 7, V_2 is 9 x 7, W_2 is 6 x 7 for (2,3,3)
    # and U_1 is 1 x 2, V_1 is 2 x 2, W_1 is 2 x 2 for (1,1,2)
    
    # But the Kronecker product of 6x7 and 1x2 gives 6x14, which is correct for U.
    # V: 9x7 ⊗ 2x2 gives 18x14, correct.
    # W: 6x7 ⊗ 2x2 gives 12x14, correct.
    
    # So we need a rank-7 decomposition for (2,3,3).
    
    # Let's construct it. We'll use the following known rank-7 decomposition:
    # (This is a standard construction from the literature)
    
    # For 2x3x3 matrix multiplication:
    # A is 2x3, B is 3x3, C = A*B is 2x3
    
    # We'll use the following 7 bilinear forms:
    # (Using a construction from the theory of tensor rank)
    
    # Let's define the 7 rank-1 tensors for (2,3,3):
    # u_r (6-dim), v_r (9-dim), w_r (6-dim) for r = 1..7
    
    # Construction:
    # We'll use the fact that 2x3x3 can be embedded in 3x3x3.
    # The 3x3x3 rank-7 (Strassen) gives us 7 products.
    # We need to restrict to the 2x3x3 sub-tensor.
    
    # For 3x3x3 with Strassen:
    # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
    # But we only need the first 2 rows of A.
    
    # Let's use a direct construction for 2x3x3 rank 7.
    
    # One known construction:
    # P1 = (a00 + a10) * (b00 + b10)
    # P2 = (a01 + a11) * (b01 + b11)
    # P3 = (a02 + a12) * (b02 + b12)
    # P4 = a00 * b11
    # P5 = a11 * b00
    # P6 = (a00 - a10) * (b01 - b11)
    # P7 = (a01 - a11) * (b00 - b10)
    
    # This might not work. Let's use a verified construction.
    
    # Actually, let's just use the schoolbook rank for (2,3,3) which is 9,
    # and combine with rank-2 for (1,1,2) to get rank 18.
    # But we want to try rank 14.
    
    # Let's try a different approach: use the known fact that rank(2,3,6) <= 14
    # and construct it directly.
    
    # We'll use the following construction based on the Kronecker product:
    # The (2,3,6) tensor = (2,3,3) tensor ⊗ (1,1,2) tensor
    
    # For (1,1,2):
    # U_1 = [[1, 1]]  (1x2)
    # V_1 = [[1, 0], [0, 1]]  (2x2)
    # W_1 = [[1, 0], [0, 1]]  (2x2)
    
    # For (2,3,3), we need rank 7. Let's use a known construction.
    
    # From the literature, one rank-7 decomposition for 2x3x3 is:
    # (I'll use a construction from the theory of matrix multiplication tensors)
    
    # Let's define the 7 rank-1 tensors for (2,3,3):
    # We'll use the following explicit construction:
    
    # u1 = [1, 0, 0, 1, 0, 0]  (a00 and a10)
    # v1 = [1, 0, 0, 1, 0, 0, 0, 0, 0]  (b00 and b10)
    # w1 = [1, 0, 0, 1, 0, 0]  (c00 and c10)
    
    # u2 = [0, 1, 0, 0, 1, 0]  (a01 and a11)
    # v2 = [0, 1, 0, 0, 1, 0, 0, 0, 0]  (b01 and b11)
    # w2 = [0, 1, 0, 0, 1, 0]  (c01 and c11)
    
    # u3 = [0, 0, 1, 0, 0, 1]  (a02 and a12)
    # v3 = [0, 0, 1, 0, 0, 1, 0, 0, 0]  (b02 and b12)
    # w3 = [0, 0, 1, 0, 0, 1]  (c02 and c12)
    
    # This gives rank 6, but we need rank 7. The issue is that this doesn't capture
    # all the cross terms.
    
    # Let's use a different approach. We'll use the known rank-7 decomposition
    # from the Strassen algorithm adapted for 2x3x3.
    
    # For now, let's just return the schoolbook rank 36 to ensure correctness,
    # and then try to improve.
    
    # Actually, let's try rank 18 first (schoolbook for 2x3x3 is 9, times 2 for 1x1x2).
    
    # (2,3,3) schoolbook rank 9:
    # U_2: 6 x 9, V_2: 9 x 9, W_2: 6 x 9
    # For each (i,j,k) in {0,1}x{0,1,2}x{0,1,2}:
    # U_2[i*3+j, r] = 1 where r = i*3 + j*1 + k... no, this is getting complicated.
    
    # Let's just use the schoolbook for (2,3,6) directly: rank 36.
    # But we want to try lower ranks.
    
    # Let's try rank 18 using the Kronecker product of (2,3,3) rank 9 and (1,1,2) rank 2.
    
    # (2,3,3) schoolbook rank 9:
    # U_2[i*3+j, r] = 1 if r = i*3 + j (for i in 0..1, j in 0..2)
    # Wait, that's not right. Let's think more carefully.
    
    # For (2,3,3): T[i*3+j, j*3+k, k*2+i] = 1 for i in 0..1, j in 0..2, k in 0..2
    # Schoolbook: rank = 2*3*3 = 18? No, that's not right either.
    # The schoolbook rank for (n,m,p) is n*m*p. So for (2,3,3), rank = 18.
    # But that's not optimal. The optimal is 7.
    
    # Let's just use the schoolbook for (2,3,6): rank 36.
    
    rank = schoolbook_rank
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
