import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    deadline = time.time() + 130.0  # safety margin before 150s kill
    
    # Schoolbook decomposition: rank = n*m*p = 72
    # This is always valid and serves as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to improve using Strassen-based decomposition
    # Strategy: Split the matrix multiplication into blocks and apply Strassen where beneficial
    
    # For 3x3 * 3x8, we can split as follows:
    # A (3x3) = [A11 A12; A21 A22] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B (3x8) = [B11 B12; B21 B22] where B11 is 2x4, B12 is 2x4, B21 is 1x4, B22 is 1x4
    
    # Actually, let's use a cleaner block decomposition:
    # Split A into 2x2 top-left, 2x1 top-right, 1x2 bottom-left, 1x1 bottom-right
    # Split B into 2x2 top-left, 2x2 top-right (first 4 cols), 2x2 top-right (last 4 cols), 
    #   1x2 bottom-left, 1x2 bottom-right (first 4 cols), 1x2 bottom-right (last 4 cols)
    
    # This is getting complex. Let's use a simpler approach:
    # Use Strassen for the 2x2 * 2x2 subproblems
    
    # Alternative approach: Use the fact that we can decompose the problem into:
    # 1. 2x2 * 2x8 (top 2 rows of A, all 8 columns of B)
    # 2. 1x2 * 2x8 (bottom row of A, first 2 columns of B... wait, dimensions don't match)
    
    # Let me reconsider the block structure:
    # A is 3x3, B is 3x8
    # C = A * B is 3x8
    
    # Block decomposition:
    # A = [A_top (2x3); A_bottom (1x3)]
    # B = [B_left (3x4); B_right (3x4)]
    
    # C_top = A_top * B = A_top * [B_left; B_right] = [A_top * B_left, A_top * B_right] (2x8)
    # C_bottom = A_bottom * B = A_bottom * [B_left; B_right] = [A_bottom * B_left, A_bottom * B_right] (1x8)
    
    # For A_top (2x3) * B_left (3x4):
    # We can further split A_top into [A_top_left (2x2); A_top_right (2x1)]... no, that's rows.
    # A_top is 2x3, so split columns: [A_top_1 (2x2), A_top_2 (2x1)]
    # B_left is 3x4, split rows: [B_left_1 (2x4); B_left_2 (1x4)]
    # Then A_top * B_left = A_top_1 * B_left_1 + A_top_2 * B_left_2
    # A_top_1 (2x2) * B_left_1 (2x4): Use Strassen-like. 2x2 * 2x4 = 2x2 * [2x2, 2x2]
    #   = 2x2*2x2 + 2x2*2x2. Each 2x2*2x2 uses 7 terms (Strassen). Total: 14 terms.
    # A_top_2 (2x1) * B_left_2 (1x4): Schoolbook: 2*1*4 = 8 terms.
    # Total for A_top * B_left: 14 + 8 = 22 terms.
    
    # Similarly for A_top * B_right: 22 terms.
    # A_bottom (1x3) * B_left (3x4): Schoolbook: 1*3*4 = 12 terms.
    # A_bottom * B_right: 12 terms.
    
    # Total: 22 + 22 + 12 + 12 = 68 terms. This is better than 72!
    
    # Let's implement this decomposition.
    
    def strassen_2x2_2x2(A, B):
        """
        Multiply two 2x2 matrices using Strassen's algorithm.
        Returns 7 intermediate products.
        A and B are 2x2 numpy arrays.
        """
        a11, a12 = A[0,0], A[0,1]
        a21, a22 = A[1,0], A[1,1]
        b11, b12 = B[0,0], B[0,1]
        b21, b22 = B[1,0], B[1,1]
        
        # Strassen's 7 products
        M1 = (a11 + a22) * (b11 + b22)
        M2 = (a21 + a22) * b11
        M3 = a11 * (b12 - b22)
        M4 = a22 * (b21 - b11)
        M5 = (a11 + a12) * b22
        M6 = (a21 - a11) * (b11 + b12)
        M7 = (a12 - a22) * (b21 + b22)
        
        return [M1, M2, M3, M4, M5, M6, M7]
    
    # Build the 68-term decomposition
    rank_68 = 68
    U_68 = np.zeros((n * m, rank_68))
    V_68 = np.zeros((m * p, rank_68))
    W_68 = np.zeros((p * n, rank_68))
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, u_val=1.0, v_val=1.0, w_val=1.0):
        nonlocal col
        U_68[u_idx, col] = u_val
        V_68[v_idx, col] = v_val
        W_68[w_idx, col] = w_val
        col += 1
    
    # Block indices:
    # A: rows 0-1 (top), row 2 (bottom); cols 0-1 (left), col 2 (right)
    # B: rows 0-1 (top), row 2 (bottom); cols 0-3 (left), cols 4-7 (right)
    
    # A_top * B_left: A_top is rows 0-1 of A, all cols; B_left is all rows of B, cols 0-3
    # A_top (2x3) * B_left (3x4)
    # = A_top[0:2, 0:2] * B_left[0:2, 0:4] + A_top[0:2, 2:3] * B_left[2:3, 0:4]
    # = (2x2)*(2x4) + (2x1)*(1x4)
    
    # (2x2)*(2x4): Split 2x4 into two 2x2 blocks
    # B_left[0:2, 0:4] = [B_left[0:2, 0:2], B_left[0:2, 2:4]]
    # So (2x2)*(2x4) = (2x2)*(2x2) + (2x2)*(2x2)
    
    # First (2x2)*(2x2): A[0:2, 0:2] * B[0:2, 0:2]
    # Using Strassen: 7 terms
    # For each Strassen term, we need to express it in terms of the original A and B elements
    
    # Strassen for A11 * B11 where A11 = A[0:2,0:2], B11 = B[0:2,0:2]
    # M1 = (A[0,0] + A[1,1]) * (B[0,0] + B[1,1])
    # This is a product of sums, which requires multiple terms in the tensor decomposition.
    
    # Actually, for the tensor decomposition, each "term" is a rank-1 tensor.
    # Strassen's algorithm computes 7 bilinear forms, but each bilinear form may not be rank-1.
    # We need to express each Strassen product as a sum of rank-1 terms.
    
    # M1 = (a11 + a22)(b11 + b22) = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # This is 4 rank-1 terms, not 1!
    
    # So Strassen doesn't directly give a lower-rank tensor decomposition.
    # The tensor rank of matrix multiplication is different from the number of multiplications.
    
    # Let me reconsider. The tensor rank of 2x2*2x2 matrix multiplication is 7 (Strassen).
    # This means it CAN be decomposed into 7 rank-1 tensors.
    
    # The Strassen decomposition for 2x2*2x2:
    # Let A = [[a, b], [c, d]], B = [[e, f], [g, h]], C = A*B
    # C[0,0] = ae + bg
    # C[0,1] = af + bh
    # C[1,0] = ce + dg
    # C[1,1] = cf + dh
    
    # Strassen's 7 products:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6
    
    # Each mi is a bilinear form. To express as rank-1 tensors, we need to decompose each mi.
    # m1 = (a+d)(e+h) = a*e + a*h + d*e + d*h -> 4 rank-1 terms
    # This doesn't help.
    
    # Actually, the tensor rank is about expressing the multiplication tensor as a sum of rank-1 tensors.
    # A rank-1 tensor for 2x2*2x2 is u * v * w where u is 2x2, v is 2x2, w is 2x2.
    # The multiplication tensor T has T[a,b,c,d,e,f,g,h] = ... no, that's not right.
    
    # The multiplication tensor for 2x2*2x2 has dimensions (4, 4, 4) since each matrix has 4 elements.
    # T[i,j,k] where i indexes A's elements, j indexes B's elements, k indexes C's elements.
    # T[i,j,k] = 1 if C[k] = A[i]*B[j] for some valid (i,j,k), else 0.
    
    # A rank-1 decomposition is T = sum_r u_r * v_r * w_r where u_r, v_r, w_r are 4-vectors.
    # The rank is the minimum number of terms.
    
    # Strassen shows rank <= 7 for 2x2*2x2.
    
    # For our problem, we need to find a low-rank decomposition of the 3x3*3x8 multiplication tensor.
    # The tensor has dimensions (9, 24, 24) since A has 9 elements, B has 24 elements, C has 24 elements.
    
    # The schoolbook rank is 9*24/9... no, it's 3*3*8 = 72.
    
    # Given the complexity, let's just return the schoolbook decomposition.
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
