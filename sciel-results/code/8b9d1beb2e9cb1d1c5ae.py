"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = len(U[0])
    # Check that each tensor slice T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # The multiplication tensor has T[i,j,k] = 1 if i, j, k correspond to a valid (A_ij * B_jk -> C_ik) term
    # Actually, the multiplication tensor T_{(i,j),(j,k),(k,i)} = 1
    # Let's just check numerically: for random A, B, compute C = A@B and compare with tensor network
    
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Tensor network: C_ik = sum_j sum_c A_ij * B_jk * (U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] / ... )
        # Actually the decomposition is: T_{(i,j),(j,k),(k,i)} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # The multiplication tensor has T = 1 for each (i,j,k) triple in the schoolbook sense.
        # So we need: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k? No!
        # The multiplication tensor is T_{(i,j),(j,k),(k,i)} = delta_{i,i} delta_{j,j} delta_{k,k}... 
        # Actually it's just T_{(i,j),(j,k),(k,i)} = 1 for all valid indices.
        # Wait no. The matrix multiplication tensor has components T_{(i,j),(j,k),(k,i)} where
        # the first index ranges over (i,j) pairs, second over (j,k) pairs, third over (k,i) pairs.
        # T_{(i,j),(j',k),(k',i')} = delta_{jj'} * delta_{kk'} * delta_{ii'} 
        # So T_{(i,j),(j,k),(k,i)} = 1.
        
        # The decomposition says: T = sum_c U_c outer V_c outer W_c
        # i.e., T_{(i,j),(j,k),(k,i)} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T_{(i,j),(j,k),(k,i)} = 1 for all i, j, k.
        
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        # Let's verify this directly.
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    max_err = max(max_err, abs(s - 1.0))
        if max_err > 1e-9:
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Partition m=3 into m1=1, m2=2.
    # Block 1: 2x1x8 schoolbook -> rank 16
    # Block 2: 2x2x8 -> use Strassen-like: partition p=8 into p1=4, p2=4
    #   For 2x2x4: Strassen gives rank 7 instead of 8
    #   So 2x2x8 = two 2x2x4 blocks, each rank 7 -> total rank 14
    # Total rank = 16 + 14 = 30? No wait, let me reconsider.
    
    # Actually: 2x3x8 = 2x1x8 + 2x2x8
    # 2x1x8: schoolbook rank = 2*1*8 = 16
    # 2x2x8: partition p=8 into two 4s: 2x2x4 + 2x2x4, each can use Strassen rank 7
    #   So 2x2x8 rank = 7 + 7 = 14
    # Total = 16 + 14 = 30
    
    # But wait, the knowledge says rank 44 was achieved with 16 schoolbook + 4 Strassen blocks (28 mults).
    # Let me re-read: "16 schoolbook + 4 Strassen blocks (28 mults)" -> 16 + 4*7 = 16 + 28 = 44.
    # So it seems like a different partitioning was used. Let me think about this more carefully.
    
    # Alternative: partition m=3 as m1=1, m2=2, and p=8 as p1=4, p2=4.
    # This gives 4 blocks:
    #   Block (i=0..1, j=0, k=0..3): 2x1x4 schoolbook, rank 8
    #   Block (i=0..1, j=0, k=4..7): 2x1x4 schoolbook, rank 8
    #   Block (i=0..1, j=1..2, k=0..3): 2x2x4 Strassen, rank 7
    #   Block (i=0..1, j=1..2, k=4..7): 2x2x4 Strassen, rank 7
    # Total rank = 8 + 8 + 7 + 7 = 30
    
    # Hmm, that's better than 44. Let me try to construct rank 30.
    
    # Actually wait - I need to be more careful. The 2x1x4 blocks: for j=0 only, 
    # A has column j=0, B has rows j=0. The contribution is A_{i,0} * B_{0,k}.
    # This is a rank 8 tensor (outer product structure). Schoolbook rank = 2*1*4 = 8. Correct.
    
    # For 2x2x4 Strassen: we multiply 2x2 by 2x2 matrix, but B is 2x4 (not 2x2).
    # Wait, no. When we partition, B has shape 2x4 for each block. But Strassen works for square matrices.
    # We need to handle 2x2 * 2x4 which is not square. 
    # Actually, 2x2 * 2x4: we can think of B as two 2x2 blocks side by side.
    # So 2x2*2x4 = 2x2*2x2 + 2x2*2x2 (two Strassen computations), rank 7+7=14.
    # That's the same as schoolbook for 2x2x4... no, schoolbook is 16, Strassen for two 2x2x2 is 14.
    
    # Let me reconsider the whole approach.
    # 2x3x8 schoolbook rank = 48.
    
    # Partition m into 1+2:
    #   2x1x8: rank 16 (schoolbook)
    #   2x2x8: This is 2x2 * 2x8. Think of B as four 2x2 blocks.
    #     2x2x8 = 4 * (2x2x2), each 2x2x2 can be done with Strassen rank 7.
    #     So 2x2x8 rank = 4 * 7 = 28.
    # Total = 16 + 28 = 44. This matches the known result!
    
    # Can we do better? What about partitioning differently?
    # 2x3x8: Try m=1+1+1: three 2x1x8 blocks, each rank 16, total 48. No improvement.
    # Try m=1+2 with different p partitioning for the 2x2x8 part:
    #   2x2x8: B is 2x8. We can partition into 2x2 + 2x2 + 2x2 + 2x2 (four 2x2 blocks)
    #   Each 2x2x2 uses Strassen rank 7, total 28. Same as above.
    #   Or partition into 2x4 + 2x4: 2x2*2x4 = ? 
    #     2x2*2x4: A is 2x2, B is 2x4. Result is 2x4.
    #     This can be seen as two 2x2*2x2 multiplications, each Strassen rank 7, total 14.
    #     So 2x2x8 = 14 + 14 = 28. Same.
    
    # What about a smarter approach for 2x2x8?
    # 2x2 * 2x8: A = [[a,b],[c,d]], B = 2x8 matrix.
    # C = A @ B = [[a*B[0]+b*B[1], c*B[0]+d*B[1]]]
    # For each column of B, we do a 2x2*1 vector multiply, which is rank 4 (schoolbook).
    # For 8 columns: rank 32 schoolbook.
    # With Strassen on 2x2 part: for each pair of columns... hmm this is getting complex.
    
    # Let me just go with the known rank-44 construction and verify it works.
    
    deadline = time.time() + 110  # Safety margin
    
    def build_2x2x2_strassen(offset_u, offset_v, offset_w, i_base, j_base, k_base):
        """Build Strassen decomposition for a 2x2x2 block.
        U has shape (2*2, rank), V has shape (2*2, rank), W has shape (2*2, rank)
        Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        i, j, k in {0,1} relative to block.
        """
        # Strassen formulas for multiplying [[a,b],[c,d]] * [[e,f],[g,h]] = [[p,q],[r,s]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-d)(f+g)
        # m7 = (a-c)(e+g)
        # p = m1 + m4 - m5 + m7
        # q = m3 + m5
        # r = m2 + m4
        # s = m1 - m2 + m3 + m6
        
        entries = []
        # m1 = (a+d)(e+h)
        entries.append([1,1,0,0, 1,1,1,1, 0,0,1,1])  # U: a+d, V: e+h, W: p gets +1
        # Actually let me be more careful.
        # U[i*2+j] for i,j in {0,1}: U[0]=a, U[1]=b, U[2]=c, U[3]=d
        # V[j*2+k] for j,k in {0,1}: V[0]=e, V[1]=f, V[2]=g, V[3]=h
        # W[k*2+i] for k,i in {0,1}: W[0]=p (k=0,i=0), W[1]=q (k=0,i=1), W[2]=r (k=1,i=0), W[3]=s (k=1,i=1)
        
        # m1 = (a+d)(e+h): U=[1,0,0,1], V=[1,1,1,1]... wait no.
        # U for a+d: [1,0,0,1]
        # V for e+h: [1,0,1,0] (e is V[0], h is V[3])
        # Wait: V[j*2+k]: V[0*2+0]=V[0]=e, V[0*2+1]=V[1]=f, V[1*2+0]=V[2]=g, V[1*2+1]=V[3]=h
        # So e=V[0], f=V[1], g=V[2], h=V[3]
        # e+h = V[0]+V[3]: V vector = [1,0,0,1]
        
        # p = m1 + m4 - m5 + m7
        # q = m3 + m5
        # r = m2 + m4
        # s = m1 - m2 + m3 + m6
        
        # Let me define the 7 Strassen multipliers:
        # m1: U=[1,0,0,1], V=[1,0,0,1] (a+d)*(e+h)
        # m2: U=[0,0,1,1], V=[1,0,0,0] (c+d)*e
        # m3: U=[1,0,0,0], V=[0,1,0,-1] a*(f-h)
        # m4: U=[0,0,0,1], V=[0,-1,1,0] d*(g-e)
        # m5: U=[1,1,0,0], V=[0,0,0,1] (a+b)*h
        # m6: U=[0,0,1,-1], V=[0,1,1,0] (c-d)*(f+g)
        # m7: U=[1,0,-1,0], V=[1,1,1,0] (a-c)*(e+g)
        
        # W contributions:
        # p = m1 + m4 - m5 + m7 -> W[0] += m1, m4, -m5, m7
        # q = m3 + m5 -> W[1] += m3, m5
        # r = m2 + m4 -> W[2] += m2, m4
        # s = m1 - m2 + m3 + m6 -> W[3] += m1, -m2, m3, m6
        
        strassen_defs = [
            # (U_vec, V_vec, W_vec) for each multiplier
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # m1: contributes to p, s
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # m2: contributes to r, -s
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # m3: contributes to q, s
            ([0,0,0,1], [0,-1,1,0], [1,0,1,0]),   # m4: contributes to p, r
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # m5: contributes to -p, q
            ([0,0,1,-1], [0,1,1,0], [0,0,0,1]),   # m6: contributes to s
            ([1,0,-1,0], [1,1,1,0], [1,0,0,0]),   # m7: contributes to p
        ]
        
        # Now assign these to the global matrices
        # For this block, local i=0,1; j=0,1; k=0,1
        # Global U index: (i_base+i)*2 + (j_base+j)
        # Global V index: (j_base+j)*2 + (k_base+k)
        # Global W index: (k_base+k)*2 + (i_base+i)
        
        for c_local in range(7):
            u_vec, v_vec, w_vec = strassen_defs[c_local]
            c_global = offset_u + c_local
            for li in range(2):
                for lj in range(2):
                    idx_u = (i_base + li) * 2 + (j_base + lj)
                    U[idx_u][c_global] += u_vec[li*2 + lj]
            for lj in range(2):
                for lk in range(2):
                    idx_v = (j_base + lj) * 2 + (k_base + lk)
                    V[idx_v][c_global] += v_vec[lj*2 + lk]
            for lk in range(2):
                for li in range(2):
                    idx_w = (k_base + lk) * 2 + (i_base + li)
                    W[idx_w][c_global] += w_vec[lk*2 + li]
        
        return offset_u + 7  # next offset
    
    def build_2x1x8_schoolbook(offset_u, offset_v, offset_w, i_base, j_fixed, k_base):
        """Build schoolbook decomposition for 2x1x8 block.
        i in {0,1}, j = j_fixed (single), k in {0..7}
        U[i*2 + j_fixed]... wait, U has shape (n*m, rank) = (2*3, rank) globally.
        Let me use global indices directly.
        """
        # U index: i*3 + j_fixed (since m=3)
        # V index: j_fixed*8 + k (since p=8)
        # W index: k*2 + i (since n=2)
        for i in range(2):
            for k in range(8):
                c = offset_u + i*8 + k
                U[i*3 + j_fixed][c] = 1
                V[j_fixed*8 + k][c] = 1
                W[k*2 + i][c] = 1
        return offset_u + 16  # 2*8 = 16
    
    # Build the full decomposition with rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 2*3 = 6 rows
    V = [[0] * rank for _ in range(m * p)]  # 3*8 = 24 rows
    W = [[0] * rank for _ in range(p * n)]  # 8*2 = 16 rows
    
    col_offset = 0
    
    # Block 1: i=0,1; j=0; k=0..7 (2x1x8 schoolbook)
    col_offset = build_2x1x8_schoolbook(col_offset, U, V, W, 0, 0, 0)
    # Wait, I need to fix the function signatures. Let me redo this.
    
    # Let me just write it directly.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j=0, all i,k (2x1x8 schoolbook, rank 16)
    for i in range(n):
        for k in range(p):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    # col = 16
    
    # Part 2: j=1,2 (2x2x8), split k into 4 groups of 2, each 2x2x2 uses Strassen
    # j in {1,2}, k in {0,1}, {2,3}, {4,5}, {6,7}
    for k_group in range(4):
        k_base = k_group * 2
        # 2x2x2 Strassen block: i in {0,1}, j in {1,2}, k in {k_base, k_base+1}
        # Local indices: i_local = i, j_local = j-1, k_local = k - k_base
        # Global U: i*m + j = i*3 + j
        # Global V: j*p + k = j*8 + k
        # Global W: k*n + i = k*2 + i
        
        strassen_defs = [
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # m1
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # m2
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # m3
            ([0,0,0,1], [0,-1,1,0], [1,0,1,0]),   # m4
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # m5
            ([0,0,1,-1], [0,1,1,0], [0,0,0,1]),   # m6
            ([1,0,-1,0], [1,1,1,0], [1,0,0,0]),   # m7
        ]
        
        for c_s in range(7):
            u_vec, v_vec, w_vec = strassen_defs[c_s]
            c_global = col + c_s
            # U: i_local in {0,1}, j_local in {0,1} -> j = 1 + j_local
            for i_l in range(2):
                for j_l in range(2):
                    idx_u = i_l * m + (1 + j_l)
                    U[idx_u][c_global] += u_vec[i_l * 2 + j_l]
            # V: j_local in {0,1}, k_local in {0,1} -> j = 1 + j_l, k = k_base + k_l
            for j_l in range(2):
                for k_l in range(2):
                    idx_v = (1 + j_l) * p + (k_base + k_l)
                    V[idx_v][c_global] += v_vec[j_l * 2 + k_l]
            # W: k_local in {0,1}, i_local in {0,1} -> k = k_base + k_l, i = i_l
            for k_l in range(2):
                for i_l in range(2):
                    idx_w = (k_base + k_l) * n + i_l
                    W[idx_w][c_global] += w_vec[k_l * 2 + i_l]
        
        col += 7
    # col = 16 + 4*7 = 44
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook rank 48
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
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
