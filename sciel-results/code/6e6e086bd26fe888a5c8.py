<<<<<<< SEARCH
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] should be 1 if (i,j,k) corresponds to a product term
    # Actually, the tensor for matrix multiplication A(4x4) * B(4x7) -> C(4x7) is:
    # T[i,j,k] = delta_{i,j,k} in the sense that C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all valid i,j,k (since each product term contributes)
    # Wait, no. The multiplication tensor is T_{ijk} = delta_{ij} * delta_{jk} in Einstein notation
    # More precisely, for A (n x m) and B (m x p), C = A B means C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T is defined such that T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # and the decomposition is T = sum_c U_c ⊗ V_c ⊗ W_c where (U_c)_i = U[i,c], (V_c)_j = V[j,c], (W_c)_k = W[k,c]
    # Actually, the correct tensor for matrix multiplication is:
    # T[i,j,k] = 1 if the indices form a valid product term, i.e., T[i,j,k] = 1 for all valid i,j,k
    # No wait, that's not right either. Let me think more carefully.
    # The multiplication tensor M_{n,m,p} is in U* ⊗ V* ⊗ W* where dim U = nm, dim V = mp, dim W = pn
    # M_{(i,j),(j',k),(k',i')} = delta_{j,j'} * delta_{k,k'}
    # So M[(i*m+j), (j'*p+k), (k'*n+i')] = 1 iff j == j' and k == k'
    # In the decomposition, we have M = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{nm}, v_c is in R^{mp}, w_c is in R^{pn}
    # So M[(i*m+j), (j'*p+k), (k'*n+i')] = sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c]
    # This should equal delta_{j,j'} * delta_{k,k'}
    # Let's verify the schoolbook decomposition
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            lhs = 0
                            for c in range(rank):
                                lhs += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i2][c]
                            rhs = (1 if j == j2 and k == k2 else 0)
                            if abs(lhs - rhs) > 1e-10:
                                return False
    return True

def _strassen_2x2_2x2():
    # Strassen for 2x2 * 2x2 -> 2x2, rank 7
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C = [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    # U is (2*2, 7) = (4, 7), V is (2*2, 7) = (4, 7), W is (2*2, 7) = (4, 7)
    # Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    U = [
        [1, 0, 0, 0, 1, 0, 0],  # a: row 0
        [0, 0, 1, 0, 0, 1, 1],  # b: row 1
        [0, 1, 0, 0, 0, 1, -1], # c: row 2
        [1, 1, 0, 1, 0, -1, 0]  # d: row 3
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],  # e: row 0
        [0, 1, 1, 1, 0, 0, 0],  # f: row 1
        [0, 0, -1, 0, 0, 1, 1], # g: row 2
        [1, 0, 0, -1, 1, 0, 0]  # h: row 3
    ]
    W = [
        [1, 0, 1, 0, 1, 0, 1],  # i=0, k=0: M1-M2+M3+M6 -> wait, C[0,0] = M1+M4-M5+M7
        # Let me redo this more carefully
        # C[0,0] = M1 + M4 - M5 + M7
        # C[0,1] = M3 + M5
        # C[1,0] = M2 + M4
        # C[1,1] = M1 - M2 + M3 + M6
        # W[k*2+i, c] where k is col index of C, i is row index of C
        # So W[0, c] is for C[0,0], W[1, c] for C[0,1], W[2, c] for C[1,0], W[3, c] for C[1,1]
        [1, 0, 1, 1, -1, 0, 1],  # C[0,0]
        [0, 0, 1, 0, 1, 0, 0],   # C[0,1]
        [0, 1, 0, 1, 0, 0, 0],   # C[1,0]
        [1, -1, 1, 0, 0, 1, 0]   # C[1,1]
    ]
    return {"U": U, "V": V, "W": W}

def _combine_blocks(U1, V1, W1, U2, V2, W2):
    # Combine two block decompositions
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    U = [[0] * (rank1 + rank2) for _ in range(len(U1))]
    V = [[0] * (rank1 + rank2) for _ in range(len(V1))]
    W = [[0] * (rank1 + rank2) for _ in range(len(W1))]
    for r in range(rank1):
        for i in range(len(U1)):
            U[i][r] = U1[i][r]
        for i in range(len(V1)):
            V[i][r] = V1[i][r]
        for i in range(len(W1)):
            W[i][r] = W1[i][r]
    for r in range(rank2):
        for i in range(len(U2)):
            U[i][rank1 + r] = U2[i][r]
        for i in range(len(V2)):
            V[i][rank1 + r] = V2[i][r]
        for i in range(len(W2)):
            W[i][rank1 + r] = W2[i][r]
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Leave margin before 150s kill
    
    n, m, p = DIMENSIONS
    
    # Start with schoolbook
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    # Try Strassen-based approaches
    # For 4x4 * 4x7, we can try to use Strassen recursively
    # 4x4 * 4x4 uses Strassen with rank 49 (7^2) if we split into 2x2 blocks
    # But we have 4x4 * 4x7
    
    # Let's try a direct approach using the structure of the problem
    # The key insight is that for matrix multiplication A(n,m) * B(m,p),
    # the tensor rank is at most the product of the ranks of smaller multiplications
    # when using block decomposition
    
    # For 4x4 * 4x7:
    # We can view this as two 4x4 * 4x4 multiplications plus a 4x4 * 4x3 multiplication
    # But that's not quite right. Let's think differently.
    
    # Actually, let's try to use the fact that 4 = 2*2 and 7 = 4+3
    # Split A into 2x2 blocks: A = [[A00, A01], [A10, A11]] where each is 2x2
    # Split B into 2x2 blocks: B = [[B00, B01, B02], [B10, B11, B12]] where each is 2x2 or 2x1
    # This is getting complicated. Let me try a simpler approach.
    
    # Try to find a lower rank decomposition using random search with verification
    import random
    random.seed(42)
    
    # We know the schoolbook rank is 4*4*7 = 112
    # The theoretical lower bound is at least max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But in practice, for these dimensions, the rank is likely higher
    
    # Let's try to use the Strassen algorithm recursively
    # 2x2 * 2x2 has rank 7 (Strassen)
    # 4x4 * 4x4 can be done with rank 49 using Strassen recursively
    # For 4x4 * 4x7, we can pad B to 4x8 and use 4x4 * 4x8, then take the first 7 columns
    # 4x4 * 4x8: split into 2x2 blocks
    # This gives rank 7^3 = 343, which is worse than schoolbook 112
    
    # So Strassen recursion doesn't help for these small dimensions
    # The schoolbook is actually competitive here
    
    # Let's try a different approach: use the structure of the 7 dimension
    # 7 is prime, so we can't easily factor it
    
    # For now, return the schoolbook decomposition
    if time.time() > deadline:
        return best
    
    # Try to verify the schoolbook decomposition
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # If verification fails, something is wrong, but this shouldn't happen
    return best
=======
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for j2 in range(m):
                for k in range(p):
                    for k2 in range(p):
                        for i2 in range(n):
                            lhs = 0
                            for c in range(rank):
                                lhs += U[i * m + j][c] * V[j2 * p + k][c] * W[k2 * n + i2][c]
                            rhs = (1 if j == j2 and k == k2 else 0)
                            if abs(lhs - rhs) > 1e-10:
                                return False
    return True

def _strassen_2x2_2x2():
    # Strassen for 2x2 * 2x2 -> 2x2, rank 7
    U = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, -1, 0]
    ]
    V = [
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [0, 0, -1, 0, 0, 1, 1],
        [1, 0, 0, -1, 1, 0, 0]
    ]
    W = [
        [1, 0, 1, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2x4():
    # Multiply 2x2 by 2x4 -> 2x4, rank 14 (two 2x2*2x2 Strassen)
    # B is split into two 2x2 blocks: B = [B_left, B_right]
    # C = [C_left, C_right] where each is 2x2
    s22 = _strassen_2x2_2x2()
    U22, V22, W22 = s22["U"], s22["V"], s22["W"]
    
    # U for 2x2*2x4: same as 2x2*2x2 (A is 2x2)
    U = [list(row) for row in U22]
    
    # V for 2x2*2x4: V[j*4+k, c] for j in 0..1, k in 0..3
    # B = [[b00, b01, b02, b03], [b10, b11, b12, b13]]
    # Split into B_left = [[b00,b01],[b10,b11]] and B_right = [[b02,b03],[b12,b13]]
    # V for left block: V[j*4+0, c] = V22[j*2+0, c], V[j*4+1, c] = V22[j*2+1, c]
    # V for right block: V[j*4+2, c] = V22[j*2+0, c], V[j*4+3, c] = V22[j*2+1, c]
    # But we need separate columns for left and right, so rank doubles
    rank = 14
    V = [[0] * rank for _ in range(8)]  # 2*4 = 8 rows
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j * 4 + k][c] = V22[j * 2 + k][c]
                V[j * 4 + 2 + k][7 + c] = V22[j * 2 + k][c]
    
    # W for 2x2*2x4: W[k*2+i, c] for k in 0..3, i in 0..1
    # C = [C_left, C_right] where C_left is 2x2, C_right is 2x2
    # W for left: W[k*2+i, c] = W22[k*2+i, c] for k in 0..1
    # W for right: W[(2+k)*2+i, c] = W22[k*2+i, 7+c] for k in 0..1
    W = [[0] * rank for _ in range(8)]  # 4*2 = 8 rows
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k * 2 + i][c] = W22[k * 2 + i][c]
                W[(2 + k) * 2 + i][7 + c] = W22[k * 2 + i][c]
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x4_4x4():
    # Multiply 2x4 by 4x4 -> 2x4, rank 14 (two 2x2*2x2 Strassen)
    # A is split into two 2x2 blocks: A = [A_left, A_right]
    # B is split into two 2x2 blocks vertically: B = [[B_top], [B_bottom]]
    # C = [C_left, C_right] where each is 2x2
    s22 = _strassen_2x2_2x2()
    U22, V22, W22 = s22["U"], s22["V"], s22["W"]
    
    # U for 2x4*4x4: U[i*4+j, c] for i in 0..1, j in 0..3
    # A = [[a00, a01, a02, a03], [a10, a11, a12, a13]]
    # A_left = [[a00,a01],[a10,a11]], A_right = [[a02,a03],[a12,a13]]
    rank = 14
    U = [[0] * rank for _ in range(8)]  # 2*4 = 8 rows
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * 4 + j][c] = U22[i * 2 + j][c]
                U[i * 4 + 2 + j][7 + c] = U22[i * 2 + j][c]
    
    # V for 2x4*4x4: V[j*4+k, c] for j in 0..3, k in 0..3
    # B = [[b00,b01,b02,b03], [b10,b11,b12,b13], [b20,b21,b22,b23], [b30,b31,b32,b33]]
    # B_top = [[b00,b01],[b10,b11]], B_bottom = [[b20,b21],[b30,b31]]
    # For left product: A_left * B_top -> C_left
    # For right product: A_right * B_bottom -> C_right
    # Wait, that's not right. A(2x4) * B(4x4) = C(2x4)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # If we split A = [A_left, A_right] (each 2x2) and B = [[B_top], [B_bottom]] (each 4x2... no)
    # Actually B is 4x4. Let's think of it as B = [[B00, B01], [B10, B11]] where each is 2x2
    # A = [A0, A1] where A0, A1 are 2x2
    # C = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    # This requires 4 multiplications of 2x2*2x2, so rank 28
    
    # Let me reconsider. For 2x4 * 4x4:
    # A is 2x4, B is 4x4, C is 2x4
    # We can write A = [A0, A1] where A0, A1 are 2x2
    # B = [[B0], [B1]] where B0, B1 are 2x4... no, B is 4x4
    # B = [[B00, B01], [B10, B11]] where each is 2x2
    # C = [A0*B00 + A1*B10, A0*B01 + A1*B11]
    # This needs 4 2x2*2x2 multiplications = rank 28
    
    # Alternative: treat as two 2x2*4x4? No, that doesn't help.
    # Let's just do rank 28 for now
    rank = 28
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]
    
    col = 0
    # A0 * B00 -> contributes to C[:, 0:2]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * 4 + j][col + c] = U22[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j * 4 + k][col + c] = V22[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k * 2 + i][col + c] = W22[k * 2 + i][c]
    col += 7
    
    # A0 * B01 -> contributes to C[:, 2:4]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * 4 + j][col + c] = U22[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[j * 4 + 2 + k][col + c] = V22[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[(2 + k) * 2 + i][col + c] = W22[k * 2 + i][c]
    col += 7
    
    # A1 * B10 -> contributes to C[:, 0:2]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * 4 + 2 + j][col + c] = U22[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[(2 + j) * 4 + k][col + c] = V22[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[k * 2 + i][col + c] = W22[k * 2 + i][c]
    col += 7
    
    # A1 * B11 -> contributes to C[:, 2:4]
    for i in range(2):
        for j in range(2):
            for c in range(7):
                U[i * 4 + 2 + j][col + c] = U22[i * 2 + j][c]
    for j in range(2):
        for k in range(2):
            for c in range(7):
                V[(2 + j) * 4 + 2 + k][col + c] = V22[j * 2 + k][c]
    for k in range(2):
        for i in range(2):
            for c in range(7):
                W[(2 + k) * 2 + i][col + c] = W22[k * 2 + i][c]
    col += 7
    
    return {"U": U, "V": V, "W": W}

def _block_decompose_4x4_4x7():
    # Multiply 4x4 by 4x7 -> 4x7
    # Split A = [[A00, A01], [A10, A11]] (each 2x2)
    # Split B = [[B00, B01, B02], [B10, B11, B12]] where B0*,B1* are 2x2 or 2x1
    # Actually B is 4x7, so split into 2x2 blocks with remainder
    # B = [[B00, B01, B02], [B10, B11, B12]] where B00,B01,B10,B11 are 2x2, B02,B12 are 2x1
    # C = [[A00*B00+A10*B10, A00*B01+A10*B11, A00*B02+A10*B12],
    #      [A01*B00+A11*B10, A01*B01+A11*B11, A01*B02+A11*B12]]
    # Each 2x2*2x2 uses rank 7, each 2x2*2x1 needs rank 4 (or use Strassen-like)
    # Total: 8 * 7 + 4 * 4 = 56 + 16 = 72... not great
    # 
    # Better approach: pad B to 4x8, use 4x4*4x8 with rank 49, then discard last column
    # 4x4 * 4x8: split into 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01, B02], [B10, B11, B12]] (each 2x2)
    # C = [[A00*B00+A10*B10, A00*B01+A10*B11, A00*B02+A10*B12],
    #      [A01*B00+A11*B10, A01*B01+A11*B11, A01*B02+A11*B12]]
    # 9 products of 2x2*2x2, each rank 7, total rank 63
    # Still worse than schoolbook 112? No, 63 < 112, so this is better!
    
    # Let's build 4x4 * 4x8 decomposition with rank 63
    s22 = _strassen_2x2_2x2()
    U22, V22, W22 = s22["U"], s22["V"], s22["W"]
    
    n, m, p = 4, 4, 8
    rank = 9 * 7  # 9 block products, each rank 7
    U = [[0] * rank for _ in range(n * m)]  # 16 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 32 rows
    
    # A block (i_block, j_block) at position i_block*2+i, j_block*2+j
    # B block (j_block, k_block) at position j_block*2+j, k_block*2+k
    # C block (i_block, k_block) at position k_block*2+k, i_block*2+i
    
    block_idx = 0
    for i_block in range(2):
        for j_block in range(2):
            for k_block in range(4):  # 4 blocks in the 7->8 dimension
                base = block_idx * 7
                # U: A[i_block*2+i, j_block*2+j] for i,j in 0..1
                for i in range(2):
                    for j in range(2):
                        for c in range(7):
                            U[(i_block * 2 + i) * m + (j_block * 2 + j)][base + c] = U22[i * 2 + j][c]
                
                # V: B[j_block*2+j, k_block*2+k] for j,k in 0..1
                for j in range(2):
                    for k in range(2):
                        for c in range(7):
                            V[(j_block * 2 + j) * p + (k_block * 2 + k)][base + c] = V22[j * 2 + k][c]
                
                # W: C[k_block*2+k, i_block*2+i] for k,i in 0..1
                for k in range(2):
                    for i in range(2):
                        for c in range(7):
                            W[(k_block * 2 + k) * n + (i_block * 2 + i)][base + c] = W22[k * 2 + i][c]
                
                block_idx += 1
    
    # Now we have 4x4 * 4x8. We need 4x4 * 4x7.
    # The 8th column of B is zero, so we can just ignore the last block column.
    # But our decomposition already assumes B has 8 columns. If B only has 7 columns,
    # the 8th column is implicitly zero, so the decomposition still works for 4x4*4x7.
    # We just need to truncate V to have 4*7=28 rows instead of 32, and W to have 7*4=28 rows.
    
    # Actually, we should keep the full 4x8 decomposition since the evaluator expects
    # specific shapes. Let me re-read the task.
    # "factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # So for 4x4*4x7: U is (16, rank), V is (28, rank), W is (28, rank)
    # We need to adapt our 4x8 decomposition to 4x7.
    
    # Since the 8th column of B is zero in a 4x7 matrix, the terms involving B[:,7] 
    # don't contribute. We can simply truncate V and W.
    
    V_trunc = [[0] * rank for _ in range(m * 7)]  # 28 rows
    for i in range(m * 7):
        if i < len(V):
            V_trunc[i] = list(V[i])
    
    W_trunc = [[0] * rank for _ in range(7 * n)]  # 28 rows
    for i in range(7 * n):
        if i < len(W):
            W_trunc[i] = list(W[i])
    
    return {"U": U, "V": V_trunc, "W": W_trunc}

def _combine_blocks(U1, V1, W1, U2, V2, W2):
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    U = [[0] * (rank1 + rank2) for _ in range(len(U1))]
    V = [[0] * (rank1 + rank2) for _ in range(len(V1))]
    W = [[0] * (rank1 + rank2) for _ in range(len(W1))]
    for r in range(rank1):
        for i in range(len(U1)):
            U[i][r] = U1[i][r]
        for i in range(len(V1)):
            V[i][r] = V1[i][r]
        for i in range(len(W1)):
            W[i][r] = W1[i][r]
    for r in range(rank2):
        for i in range(len(U2)):
            U[i][rank1 + r] = U2[i][r]
        for i in range(len(V2)):
            V[i][rank1 + r] = V2[i][r]
        for i in range(len(W2)):
            W[i][rank1 + r] = W2[i][r]
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Try block Strassen decomposition (rank 63)
    best = _block_decompose_4x4_4x7()
    best_rank = len(best["U"][0])
    
    # Verify
    if _verify(best["U"], best["V"], best["W"]):
        return best
    
    # Fallback to schoolbook
    schoolbook = _schoolbook()
    if _verify(schoolbook["U"], schoolbook["V"], schoolbook["W"]):
        return schoolbook
    
    return best
>>>>>>> REPLACE
