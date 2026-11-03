import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=6
    # Rank 68 decomposition
    
    # 1. A1 (4x4) * B1L (4x4) -> C1L (4x4)
    #    Uses 4 Strassen blocks. Rank 56.
    # 2. A1 (4x4) * B1R (4x2) -> C1R (4x2)
    #    Uses 2 Rank-4 blocks. Rank 8.
    # 3. A2 (4x1) * B2 (1x6) -> C2 (4x6)
    #    Uses 1 Rank-4 block. Rank 4.
    
    rank = 68
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for 2x2x2
    def add_strassen(i_off, j_off, k_off, col_start):
        # i_off: row offset in A (0 or 2)
        # j_off: col offset in A (0 or 2)
        # k_off: col offset in B (0 or 2)
        # Local indices:
        # A: i in 0..1, j in 0..1
        # B: j in 0..1, k in 0..1
        # C: i in 0..1, k in 0..1
        
        # Strassen products:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B11 - B12)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to U, V, W.
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        
        # For each of the 7 products, we add 1 to the rank count.
        # But wait, the Strassen decomposition of 2x2x2 has rank 7.
        # Each "product" in Strassen is a bilinear form.
        # We add 7 columns to U, V, W.
        
        # Let's define the 7 bilinear forms.
        # Each form is a sum of products of linear combinations of A and B entries.
        # Actually, the standard rank 7 decomposition is:
        # t1 = (a11+a22)(b11+b22)
        # t2 = (a21+a22)b11
        # t3 = a11(b21-b22)
        # t4 = a22(b11-b12)
        # t5 = (a11+a12)b22
        # t6 = (a21-a11)(b11+b12)
        # t7 = (a12-a22)(b21+b22)
        
        # We need to express these as sums of U*V*W.
        # Since they are bilinear, each term like (a11+a22)(b11+b22) can be written as:
        # (a11+a22) * (b11+b22) = 1 * 1 * (a11+a22)(b11+b22)?
        # No, U and V are matrices.
        # For a single term $L_A \cdot L_B$, we can use rank 1:
        # $U_{idx, c} = L_A$, $V_{idx, c} = L_B$, $W_{idx, c} = 1$?
        # No, $W$ is also involved.
        # The form is $C_{ik} = \sum U_{ij} V_{jk} W_{ki}$.
        # For a bilinear form $f(A) g(B)$ contributing to $C_{ik}$,
        # we can set $U_{ij, c} = f(A)_{ij}$? No, $U$ is constant.
        
        # Correct interpretation:
        # The tensor $T$ is decomposed into rank 1 tensors $u \otimes v \otimes w$.
        # $T_{ijk} = \sum_r U_{ij,r} V_{jk,r} W_{ki,r}$.
        # For Strassen, the 7 terms are linear combinations of the $T_{ijk}$ entries.
        # Actually, we just need to add the rank 7 decomposition of the $2 \times 2 \times 2$ tensor.
        
        # The rank 7 decomposition of $2 \times 2 \times 2$ multiplication:
        # The 7 columns correspond to the 7 Strassen products.
        # For each product $t_r$, it contributes to specific $C_{ik}$ entries.
        # $t_r$ is a scalar computed from $A$ and $B$.
        # $C_{ik} = \sum_r \alpha_{ik,r} t_r$.
        # We can represent this by setting:
        # $U_{ij,r} = 1$ if the $A$ part of $t_r$ involves $A_{ij}$?
        # No, $t_r$ is a product of a linear form in $A$ and a linear form in $B$.
        # $t_r = L_A^{(r)}(A) L_B^{(r)}(B)$.
        # $L_A^{(r)}$ is a linear combination of $A_{ij}$.
        # $L_B^{(r)}$ is a linear combination of $B_{jk}$.
        
        # We can set:
        # $U_{ij,r} = \text{coefficient of } A_{ij} \text{ in } L_A^{(r)}$.
        # $V_{jk,r} = \text{coefficient of } B_{jk} \text{ in } L_B^{(r)}$.
        # $W_{ki,r} = \alpha_{ik,r}$.
        
        # Let's define the 7 forms.
        # A indices: 00, 01, 10, 11
        # B indices: 00, 01, 10, 11
        
        # t1: L_A = A00+A11, L_B = B00+B11
        # t2: L_A = A10+A11, L_B = B00
        # t3: L_A = A00, L_B = B10-B11
        # t4: L_A = A11, L_B = B00-B01
        # t5: L_A = A00+A01, L_B = B11
        # t6: L_A = A10-A00, L_B = B00+B01
        # t7: L_A = A01-A11, L_B = B10+B11
        
        # Contributions to C:
        # C00 = t1 + t4 - t5 + t7
        # C01 = t3 + t5
        # C10 = t2 + t4
        # C11 = t1 - t2 + t3 + t6
        
        # We will add 7 columns for this block.
        for r in range(7):
            c = col_start + r
            # Set U
            if r == 0:
                U[(i_off + 0)*m + (j_off + 0)][c] = 1
                U[(i_off + 1)*m + (j_off + 1)][c] = 1
                V[(j_off + 0)*p + (k_off + 0)][c] = 1
                V[(j_off + 1)*p + (k_off + 1)][c] = 1
            elif r == 1:
                U[(i_off + 1)*m + (j_off + 0)][c] = 1
                U[(i_off + 1)*m + (j_off + 1)][c] = 1
                V[(j_off + 0)*p + (k_off + 0)][c] = 1
            elif r == 2:
                U[(i_off + 0)*m + (j_off + 0)][c] = 1
                V[(j_off + 1)*p + (k_off + 0)][c] = 1
                V[(j_off + 1)*p + (k_off + 1)][c] = -1
            elif r == 3:
                U[(i_off + 1)*m + (j_off + 1)][c] = 1
                V[(j_off + 0)*p + (k_off + 0)][c] = 1
                V[(j_off + 0)*p + (k_off + 1)][c] = -1
            elif r == 4:
                U[(i_off + 0)*m + (j_off + 0)][c] = 1
                U[(i_off + 0)*m + (j_off + 1)][c] = 1
                V[(j_off + 1)*p + (k_off + 1)][c] = 1
            elif r == 5:
                U[(i_off + 1)*m + (j_off + 0)][c] = 1
                U[(i_off + 0)*m + (j_off + 0)][c] = -1
                V[(j_off + 0)*p + (k_off + 0)][c] = 1
                V[(j_off + 0)*p + (k_off + 1)][c] = 1
            elif r == 6:
                U[(i_off + 0)*m + (j_off + 1)][c] = 1
                U[(i_off + 1)*m + (j_off + 1)][c] = -1
                V[(j_off + 1)*p + (k_off + 0)][c] = 1
                V[(j_off + 1)*p + (k_off + 1)][c] = 1
            
            # Set W
            # C00
            if r == 0: W[(k_off + 0)*n + (i_off + 0)][c] = 1
            if r == 3: W[(k_off + 0)*n + (i_off + 0)][c] = 1
            if r == 4: W[(k_off + 0)*n + (i_off + 0)][c] = -1
            if r == 6: W[(k_off + 0)*n + (i_off + 0)][c] = 1
            
            # C01
            if r == 2: W[(k_off + 1)*n + (i_off + 0)][c] = 1
            if r == 4: W[(k_off + 1)*n + (i_off + 0)][c] = 1
            
            # C10
            if r == 1: W[(k_off + 0)*n + (i_off + 1)][c] = 1
            if r == 3: W[(k_off + 0)*n + (i_off + 1)][c] = 1
            
            # C11
            if r == 0: W[(k_off + 1)*n + (i_off + 1)][c] = 1
            if r == 1: W[(k_off + 1)*n + (i_off + 1)][c] = -1
            if r == 2: W[(k_off + 1)*n + (i_off + 1)][c] = 1
            if r == 5: W[(k_off + 1)*n + (i_off + 1)][c] = 1
            
        return col_start + 7

    # Block 1: A1 (4x4) * B1L (4x4)
    # i in 0..3, j in 0..3, k in 0..3
    # 4 Strassen blocks
    col = add_strassen(0, 0, 0, col)
    col = add_strassen(0, 2, 0, col)
    col = add_strassen(2, 0, 0, col)
    col = add_strassen(2, 2, 0, col)
    
    # Block 2: A1 (4x4) * B1R (4x2)
    # i in 0..3, j in 0..3, k in 4..5
    # We can use 2 Rank-4 blocks (one for each col of B1R)
    # Rank-4 for 4x4x1
    def add_rank4_4x4x1(i_off, j_off, k_idx, col_start):
        # i_off: 0 or 2 (row offset in A)
        # j_off: 0 or 2 (col offset in A)
        # k_idx: global k index (4 or 5)
        # A is 2x2 block, B is 2x1 block (col k_idx)
        # C is 2x1 block
        # C_i = sum_j A_ij B_jk
        # Rank 4.
        for r in range(4):
            c = col_start + r
            # U
            U[(i_off + r)*m + (j_off + r)][c] = 1
            # V
            V[(j_off + r)*p + k_idx][c] = 1
            # W
            W[k_idx*n + (i_off + r)][c] = 1
        return col_start + 4

    # B1R has columns 4 and 5.
    # j ranges 0..3.
    # We need to cover j=0,1 and j=2,3.
    col = add_rank4_4x4x1(0, 0, 4, col)
    col = add_rank4_4x4x1(0, 2, 4, col)
    col = add_rank4_4x4x1(2, 0, 4, col)
    col = add_rank4_4x4x1(2, 2, 4, col)
    
    col = add_rank4_4x4x1(0, 0, 5, col)
    col = add_rank4_4x4x1(0, 2, 5, col)
    col = add_rank4_4x4x1(2, 0, 5, col)
    col = add_rank4_4x4x1(2, 2, 5, col)
    
    # Block 3: A2 (4x1) * B2 (1x6)
    # i in 0..3, j=4, k in 0..5
    # C_i = A_i4 B_4k
    # Rank 4.
    def add_rank4_4x1x6(col_start):
        for r in range(4):
            c = col_start + r
            # U
            U[r*m + 4][c] = 1
            # V
            for k in range(6):
                V[4*p + k][c] = 1
            # W
            for k in range(6):
                W[k*n + r][c] = 1
        return col_start + 4

    col = add_rank4_4x1x6(col)
    
    return {"U": U, "V": V, "W": W}
