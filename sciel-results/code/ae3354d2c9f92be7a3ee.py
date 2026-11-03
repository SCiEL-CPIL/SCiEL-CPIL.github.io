"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will build the decomposition recursively.
    # The function returns U, V, W for a given n, m, p.
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank)
    
    def get_decomposition(n, m, p):
        # Base case: 1x1x1
        if n == 1 and m == 1 and p == 1:
            return [[1]], [[1]], [[1]]
        
        # Base case: 1xm x px1 or similar simple cases
        # 1 x m x p: C_k = sum_j A_j B_jk. Rank m.
        if n == 1:
            rank = m
            U = [[0]*rank for _ in range(m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p)]
            for j in range(m):
                U[j][j] = 1
                for k in range(p):
                    V[j*p+k][j] = 1
                W[k][j] = 1 # Wait, W index is k*n+i = k*1+0 = k.
            # Correction: W[k][j] = 1?
            # C_k = sum_j A_j B_jk.
            # Term j: A_j * B_jk * 1.
            # U[j][j]=1, V[j*p+k][j]=1, W[k][j]=1.
            # Sum over j: sum_j 1 * B_jk * 1 = C_k. Correct.
            return U, V, W
            
        if p == 1:
            rank = n*m
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m)]
            W = [[0]*rank for _ in range(n)]
            col = 0
            for i in range(n):
                for j in range(m):
                    U[i*m+j][col] = 1
                    V[j][col] = 1
                    W[i][col] = 1
                    col += 1
            return U, V, W

        if m == 1:
            rank = n*p
            U = [[0]*rank for _ in range(n)]
            V = [[0]*rank for _ in range(p)]
            W = [[0]*rank for _ in range(n*p)]
            col = 0
            for i in range(n):
                for k in range(p):
                    U[i][col] = 1
                    V[k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
            return U, V, W

        # Recursive case: Strassen-like decomposition
        # Only works if n, m, p are even (or can be split)
        # For simplicity, we assume we can split into halves.
        # If n, m, p are not even, we might need padding or fallback to schoolbook.
        # In our specific case 4,4,9, we split 9 into 4+5, then 5 into 4+1.
        # So we need to handle non-even dimensions by splitting off 1s or using schoolbook.
        
        # Strategy: If any dimension is odd, split off a 1.
        # C = A B.
        # If p is odd, B = [B1 (m x (p-1)) | B2 (m x 1)].
        # C = [C1 (n x (p-1)) | C2 (n x 1)].
        # C1 = A B1. C2 = A B2.
        # Recurse on (n, m, p-1) and (n, m, 1).
        
        if p % 2 != 0 and p > 1:
            U1, V1, W1 = get_decomposition(n, m, p-1)
            U2, V2, W2 = get_decomposition(n, m, 1)
            # Combine
            # U is same for both? No, A is the same, but the rank columns are different.
            # We need to concatenate the ranks.
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            # Fill part 1 (for B1, p-1 cols)
            for i in range(n*m):
                for c in range(rank1):
                    U[i][c] = U1[i][c]
            for i in range(m*(p-1)):
                for c in range(rank1):
                    V[i][c] = V1[i][c]
            # V for B2 (m x 1) is at indices m*(p-1) .. m*p-1
            for i in range(m):
                for c in range(rank2):
                    V[m*(p-1) + i][rank1 + c] = V2[i][c]
            
            for i in range((p-1)*n):
                for c in range(rank1):
                    W[i][c] = W1[i][c]
            # W for C2 (1 x n) is at indices (p-1)*n .. p*n-1
            for i in range(n):
                for c in range(rank2):
                    W[(p-1)*n + i][rank1 + c] = W2[i][c]
                    
            return U, V, W

        if n % 2 != 0 and n > 1:
            # Split A into A1 (n-1 x m) and A2 (1 x m)
            # C = A1 B + A2 B
            # This is not a direct tensor sum because C is n x p.
            # C1 (n-1 x p) = A1 B. C2 (1 x p) = A2 B.
            U1, V1, W1 = get_decomposition(n-1, m, p)
            U2, V2, W2 = get_decomposition(1, m, p)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            # U1 is (n-1)*m x rank1. U2 is 1*m x rank2.
            for i in range((n-1)*m):
                for c in range(rank1):
                    U[i][c] = U1[i][c]
            for i in range(m):
                for c in range(rank2):
                    U[(n-1)*m + i][rank1 + c] = U2[i][c]
            
            for i in range(m*p):
                for c in range(rank1):
                    V[i][c] = V1[i][c]
                for c in range(rank2):
                    V[i][rank1 + c] = V2[i][c]
            
            for i in range(p*(n-1)):
                for c in range(rank1):
                    W[i][c] = W1[i][c]
            for i in range(p):
                for c in range(rank2):
                    W[p*(n-1) + i][rank1 + c] = W2[i][c]
                    
            return U, V, W

        if m % 2 != 0 and m > 1:
            # Split B into B1 (m-1 x p) and A into A1 (n x m-1), A2 (n x 1)
            # C = A1 B1 + A2 B2
            U1, V1, W1 = get_decomposition(n, m-1, p)
            U2, V2, W2 = get_decomposition(n, 1, p)
            
            rank1 = len(U1[0])
            rank2 = len(U2[0])
            total_rank = rank1 + rank2
            
            U = [[0]*total_rank for _ in range(n*m)]
            V = [[0]*total_rank for _ in range(m*p)]
            W = [[0]*total_rank for _ in range(p*n)]
            
            # U1 is n*(m-1) x rank1. U2 is n*1 x rank2.
            for i in range(n*(m-1)):
                for c in range(rank1):
                    U[i][c] = U1[i][c]
            for i in range(n):
                for c in range(rank2):
                    U[n*(m-1) + i][rank1 + c] = U2[i][c]
            
            for i in range((m-1)*p):
                for c in range(rank1):
                    V[i][c] = V1[i][c]
            for i in range(p):
                for c in range(rank2):
                    V[(m-1)*p + i][rank1 + c] = V2[i][c]
            
            for i in range(p*n):
                for c in range(rank1):
                    W[i][c] = W1[i][c]
                for c in range(rank2):
                    W[i][rank1 + c] = W2[i][c]
                    
            return U, V, W

        # All n, m, p are even and > 1. Use Strassen.
        # Split into 2x2 blocks.
        n2, m2, p2 = n//2, m//2, p//2
        
        # 7 sub-problems of size n2 x m2 x p2
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        U1, V1, W1 = get_decomposition(n2, m2, p2)
        U2, V2, W2 = get_decomposition(n2, m2, p2)
        U3, V3, W3 = get_decomposition(n2, m2, p2)
        U4, V4, W4 = get_decomposition(n2, m2, p2)
        U5, V5, W5 = get_decomposition(n2, m2, p2)
        U6, V6, W6 = get_decomposition(n2, m2, p2)
        U7, V7, W7 = get_decomposition(n2, m2, p2)
        
        rank_sub = len(U1[0])
        total_rank = 7 * rank_sub
        
        U = [[0]*total_rank for _ in range(n*m)]
        V = [[0]*total_rank for _ in range(m*p)]
        W = [[0]*total_rank for _ in range(p*n)]
        
        def add_block(U_target, U_src, row_offset, col_offset, scale=1):
            for r in range(len(U_src)):
                for c in range(len(U_src[0])):
                    U_target[row_offset + r][col_offset + c] += scale * U_src[r][c]
                    
        def add_block_V(V_target, V_src, row_offset, col_offset, scale=1):
            for r in range(len(V_src)):
                for c in range(len(V_src[0])):
                    V_target[row_offset + r][col_offset + c] += scale * V_src[r][c]
                    
        def add_block_W(W_target, W_src, row_offset, col_offset, scale=1):
            for r in range(len(W_src)):
                for c in range(len(W_src[0])):
                    W_target[row_offset + r][col_offset + c] += scale * W_src[r][c]

        # M1: (A11+A22)(B11+B22)
        # A11 is at 0, A22 is at n2*m2 (in A's flattened index i*m+j)
        # A11 rows: 0..n2*m2-1. A22 rows: n2*m2 .. n*m-1 (offset n2*m2)
        # B11 cols: 0..p2-1. B22 cols: p2..2*p2-1 (offset p2)
        # In V, B11 is at 0, B22 is at m2*p2 (offset m2*p2)
        
        # U for M1: A11 + A22
        for r in range(n2*m2):
            for c in range(rank_sub):
                U[r][c] = U1[r][c]
                U[n2*m2 + r][c] = U1[r][c]
        # V for M1: B11 + B22
        for r in range(m2*p2):
            for c in range(rank_sub):
                V[r][c] = V1[r][c]
                V[m2*p2 + r][c] = V1[r][c]
        # W for M1: C11 + C22 (but we need to add to C11, C22, etc later)
        # Wait, Strassen combines the results.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will accumulate W contributions.
        # W1 contributes to C11 and C22.
        # W index for C11: 0..p2*n2-1.
        # W index for C22: (p2+n2)*n2 .. (2*p2+2*n2)*n2-1?
        # W index formula: k*n + i.
        # C11: i in 0..n2-1, k in 0..p2-1. Index k*n + i.
        # C22: i in n2..2*n2-1, k in p2..2*p2-1. Index k*n + i.
        
        # This indexing is tricky. It's better to just store the W blocks and combine at the end?
        # No, W is the output factor. We need to map the sub-problem outputs to the main output.
        
        # Let's just use the schoolbook for the combination to be safe and simple, 
        # but that defeats the purpose.
        # Instead, we construct U, V, W for the 7 multiplications, 
        # and then we need to express C as a linear combination of the 7 results.
        # But the results M1..M7 are matrices of size n2 x p2.
        # The final C is n x p.
        # C11 = M1 + M4 - M5 + M7.
        # This means the output tensor is a sum of tensors.
        # Each Mk has a tensor decomposition Uk, Vk, Wk.
        # The total tensor is sum_k (coeffs) * (Uk, Vk, Wk).
        # But the coefficients apply to the output C.
        # This means we can just add the W matrices with the appropriate signs and shifts.
        
        # Let's implement the shifts for W.
        # W1 (M1) adds to C11 (+1) and C22 (+1).
        # C11 W offset: 0.
        # C22 W offset: (p2 + n2) * n2? No.
        # W index: k*n + i.
        # C22: i = i2 + n2, k = k2 + p2.
        # Index = (k2+p2)*n + (i2+n2) = k2*n + i2 + p2*n + n2.
        # Offset = p2*n + n2.
        
        offset_C11 = 0
        offset_C12 = 0 # k in p2..2p2-1? No.
        # C12: i in 0..n2-1, k in p2..2p2-1.
        # Index = k*n + i.
        # k = k2 + p2.
        # Index = (k2+p2)*n + i2 = k2*n + i2 + p2*n.
        # Offset = p2*n.
        
        offset_C21 = n2 # i in n2..2n2-1, k in 0..p2-1.
        # Index = k*n + (i2+n2) = k*n + i2 + n2.
        # Offset = n2.
        
        offset_C22 = p2*n + n2
        
        # M1: C11 +1, C22 +1
        add_block_W(W, W1, offset_C11, 0, 1)
        add_block_W(W, W1, offset_C22, 0, 1)
        
        # M2: C21 +1, C22 -1
        add_block_W(W, W2, offset_C21, rank_sub, 1)
        add_block_W(W, W2, offset_C22, rank_sub, -1)
        
        # M3: C12 +1, C22 +1
        add_block_W(W, W3, offset_C12, 2*rank_sub, 1)
        add_block_W(W, W3, offset_C22, 2*rank_sub, 1)
        
        # M4: C11 +1, C21 +1
        add_block_W(W, W4, offset_C11, 3*rank_sub, 1)
        add_block_W(W, W4, offset_C21, 3*rank_sub, 1)
        
        # M5: C11 -1, C12 +1
        add_block_W(W, W5, offset_C11, 4*rank_sub, -1)
        add_block_W(W, W5, offset_C12, 4*rank_sub, 1)
        
        # M6: C22 +1
        add_block_W(W, W6, offset_C22, 5*rank_sub, 1)
        
        # M7: C11 +1
        add_block_W(W, W7, offset_C11, 6*rank_sub, 1)
        
        # Now fill U and V for the 7 blocks.
        # M1: A11+A22, B11+B22
        add_block(U, U1, 0, 0, 1) # A11
        add_block(U, U1, n2*m2, 0, 1) # A22
        add_block(V, V1, 0, 0, 1) # B11
        add_block(V, V1, m2*p2, 0, 1) # B22
        
        # M2: A21+A22, B11
        add_block(U, U2, n2*m2, rank_sub, 1) # A22 (row offset n2*m2)
        add_block(U, U2, n2*m2 + n2*m2, rank_sub, 1) # A21 (row offset 2*n2*m2? No)
        # A21 is at i in n2..2n2-1, j in 0..m2-1.
        # Index i*m + j.
        # i = i2+n2. Index (i2+n2)*m + j2 = i2*m + j2 + n2*m.
        # A22 is at i in n2..2n2-1, j in m2..2m2-1.
        # Index i*m + j.
        # i = i2+n2, j = j2+m2.
        # Index (i2+n2)*m + (j2+m2) = i2*m + j2 + n2*m + m2.
        # So A21 offset is n2*m. A22 offset is n2*m + m2.
        
        # My previous offset for A22 was n2*m2. That was wrong.
        # A is n x m. Flattened i*m+j.
        # A11: i<n2, j<m2. Offset 0.
        # A12: i<n2, j>=m2. Offset m2.
        # A21: i>=n2, j<m2. Offset n2*m.
        # A22: i>=n2, j>=m2. Offset n2*m + m2.
        
        # B is m x p. Flattened j*p+k.
        # B11: j<m2, k<p2. Offset 0.
        # B12: j<m2, k>=p2. Offset p2.
        # B21: j>=m2, k<p2. Offset m2*p.
        # B22: j>=m2, k>=p2. Offset m2*p + p2.
        
        # Let's re-do U and V with correct offsets.
        
        # M1: A11+A22, B11+B22
        add_block(U, U1, 0, 0, 1) # A11
        add_block(U, U1, n2*m + m2, 0, 1) # A22
        add_block(V, V1, 0, 0, 1) # B11
        add_block(V, V1, m2*p + p2, 0, 1) # B22
        
        # M2: A21+A22, B11
        add_block(U, U2, n2*m, rank_sub, 1) # A21
        add_block(U, U2, n2*m + m2, rank_sub, 1) # A22
        add_block(V, V2, 0, rank_sub, 1) # B11
        
        # M3: A11, B12-B22
        add_block(U, U3, 0, 2*rank_sub, 1) # A11
        add_block(V, V3, p2, 2*rank_sub, 1) # B12
        add_block(V, V3, m2*p + p2, 2*rank_sub, -1) # B22
        
        # M4: A22, B21-B11
        add_block(U, U4, n2*m + m2, 3*rank_sub, 1) # A22
        add_block(V, V4, m2*p, 3*rank_sub, 1) # B21
        add_block(V, V4, 0, 3*rank_sub, -1) # B11
        
        # M5: A11+A12, B22
        add_block(U, U5, 0, 4*rank_sub, 1) # A11
        add_block(U, U5, m2, 4*rank_sub, 1) # A12
        add_block(V, V5, m2*p + p2, 4*rank_sub, 1) # B22
        
        # M6: A21-A11, B11+B12
        add_block(U, U6, n2*m, 5*rank_sub, 1) # A21
        add_block(U, U6, 0, 5*rank_sub, -1) # A11
        add_block(V, V6, 0, 5*rank_sub, 1) # B11
        add_block(V, V6, p2, 5*rank_sub, 1) # B12
        
        # M7: A12-A22, B21+B22
        add_block(U, U7, m2, 6*rank_sub, 1) # A12
        add_block(U, U7, n2*m + m2, 6*rank_sub, -1) # A22
        add_block(V, V7, m2*p, 6*rank_sub, 1) # B21
        add_block(V, V7, m2*p + p2, 6*rank_sub, 1) # B22
        
        return U, V, W

    U, V, W = get_decomposition(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
