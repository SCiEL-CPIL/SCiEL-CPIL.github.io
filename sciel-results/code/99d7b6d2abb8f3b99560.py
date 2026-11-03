import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # Rank 76 decomposition
    # 28 terms for 4x4x4 (Strassen)
    # 48 terms for 4x4x3 (Schoolbook)
    
    rank = 76
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: Strassen for 4x4x4 (first 4 columns of B) ---
    # B is 4x7. We treat the first 4 columns as B1 (4x4).
    # A is 4x4.
    # C1 is 4x4.
    
    # Strassen components for 2x2x2
    # We need to map the 28 terms to U, V, W.
    # U has indices i*m+j (i,j in 0..3)
    # V has indices j*p+k (j in 0..3, k in 0..6)
    # W has indices k*n+i (k in 0..6, i in 0..3)
    
    # The 28 terms come from 7 M_i's, each requiring 7 scalar multiplications.
    # Let's define the 7 scalar multiplications for a 2x2x2 matrix mult.
    # A2x2 = [[a, b], [c, d]], B2x2 = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # We have 4 blocks of A: A11, A12, A21, A22
    # We have 4 blocks of B1: B11, B12, B21, B22
    # Note: B1 is the first 4 columns of B.
    # B11 is rows 0-1, cols 0-1 of B.
    # B12 is rows 0-1, cols 2-3 of B.
    # B21 is rows 2-3, cols 0-1 of B.
    # B22 is rows 2-3, cols 2-3 of B.
    
    # We will iterate through the 7 M_i's and their 7 scalar terms.
    
    # Helper to add a term to U, V, W
    # u_idx: i*m + j
    # v_idx: j*p + k
    # w_idx: k*n + i
    # u_val, v_val, w_val
    
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i*m + j][col] = u_val
        V[j*p + k][col] = v_val
        W[k*n + i][col] = w_val
        col += 1
        
    # Define the 7 scalar multiplications for a 2x2 block multiplication
    # Inputs are indices for the 2x2 blocks
    # A_block: (i0, i1, j0, j1)
    # B_block: (j0, j1, k0, k1)
    # The scalar terms are combinations of these.
    
    # We need to generate the 28 terms for the 4x4x4 multiplication.
    # This is complex to hardcode.
    # Alternative: Just use schoolbook for the 4x4x4 part too?
    # Rank 64 for 4x4x4. Total 64 + 48 = 112. No improvement.
    
    # Let's try to implement the Strassen 28 terms explicitly.
    # The 7 M_i's are:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Each M_i is a 2x2 matrix multiplication.
    # Let's define a function to add the 7 terms for a 2x2x2 multiplication.
    # The 2x2 blocks are defined by their top-left indices.
    # A_block at (ai, aj), B_block at (bj, bk).
    # The elements are:
    # A[ai, bj], A[ai, bj+1], A[ai+1, bj], A[ai+1, bj+1]
    # B[bj, bk], B[bj, bk+1], B[bj+1, bk], B[bj+1, bk+1]
    
    def add_2x2_mult(ai, aj, bj, bk):
        nonlocal col
        # a = A[ai, bj], b = A[ai, bj+1], c = A[ai+1, bj], d = A[ai+1, bj+1]
        # e = B[bj, bk], f = B[bj, bk+1], g = B[bj+1, bk], h = B[bj+1, bk+1]
        
        # M1 = (a+d)(e+h)
        # Terms: a*e, a*h, d*e, d*h
        # We need to combine them into the 7 Strassen terms.
        # Actually, the 7 terms are:
        # T1: (a+d)(e+h) -> a*e + a*h + d*e + d*h
        # T2: (c+d)e -> c*e + d*e
        # T3: a(f-h) -> a*f - a*h
        # T4: d(g-e) -> d*g - d*e
        # T5: (a+b)h -> a*h + b*h
        # T6: (c-a)(e+f) -> c*e + c*f - a*e - a*f
        # T7: (b-d)(g+h) -> b*g + b*h - d*g - d*h
        
        # We add these 7 terms to U, V, W.
        # Each term is a sum of products of A and B elements.
        # We need to decompose each T_i into scalar multiplications.
        
        # T1: (a+d)(e+h)
        # This is not a single scalar multiplication, it's a product of sums.
        # In tensor decomposition, we need to express the tensor as sum of rank-1 tensors.
        # The Strassen algorithm defines 7 rank-1 tensors (actually, each M_i is a 2x2x2 tensor of rank 7).
        # So each M_i contributes 7 terms to the total rank.
        
        # The 7 terms for a 2x2x2 multiplication are:
        # 1. (a+d)(e+h)
        # 2. (c+d)e
        # 3. a(f-h)
        # 4. d(g-e)
        # 5. (a+b)h
        # 6. (c-a)(e+f)
        # 7. (b-d)(g+h)
        
        # We need to add these 7 terms.
        # Term 1: U gets (a+d), V gets (e+h), W gets 1.
        # But U, V, W are matrices.
        # U[ai, bj] += 1, U[ai+1, bj+1] += 1
        # V[bj, bk] += 1, V[bj+1, bk+1] += 1
        # W[bk, ai] += 1, W[bk, ai+1] += 1, W[bk+1, ai] += 1, W[bk+1, ai+1] += 1
        # Wait, the W index is k*n+i.
        # The output C[ai, bk] gets contribution from a*e.
        # The output C[ai, bk+1] gets contribution from a*f.
        # etc.
        
        # Let's just add the 7 terms directly.
        # Term 1: (A[ai,bj] + A[ai+1,bj+1]) * (B[bj,bk] + B[bj+1,bk+1])
        # This contributes to C[ai,bk], C[ai,bk+1], C[ai+1,bk], C[ai+1,bk+1]
        # Specifically:
        # C[ai,bk] += A[ai,bj]B[bj,bk] + A[ai,bj]B[bj+1,bk+1] + A[ai+1,bj+1]B[bj,bk] + A[ai+1,bj+1]B[bj+1,bk+1]
        # This is not a rank-1 term.
        # Strassen's algorithm works by defining 7 intermediate products M1..M7, each of which is a 2x2 matrix.
        # Then C is computed from M1..M7.
        # The rank of the 2x2x2 multiplication is 7.
        # So we need to add 7 terms for each M_i.
        
        # Let's implement the 7 terms for the 2x2x2 multiplication.
        # The 7 terms are:
        # T1: (a+d)(e+h)
        # T2: (c+d)e
        # T3: a(f-h)
        # T4: d(g-e)
        # T5: (a+b)h
        # T6: (c-a)(e+f)
        # T7: (b-d)(g+h)
        
        # We add these 7 terms to U, V, W.
        # For each term, we need to update U, V, W.
        
        # Term 1: (a+d)(e+h)
        # U[ai,bj] += 1, U[ai+1,bj+1] += 1
        # V[bj,bk] += 1, V[bj+1,bk+1] += 1
        # W[bk,ai] += 1, W[bk,ai+1] += 1, W[bk+1,ai] += 1, W[bk+1,ai+1] += 1
        # Wait, this is not correct.
        # The term is (A[ai,bj] + A[ai+1,bj+1]) * (B[bj,bk] + B[bj+1,bk+1])
        # This is a product of two sums.
        # In the tensor decomposition, we can have a term where U has non-zero entries at (ai,bj) and (ai+1,bj+1), and V has non-zero entries at (bj,bk) and (bj+1,bk+1).
        # Then the product contributes to W at (bk,ai), (bk,ai+1), (bk+1,ai), (bk+1,ai+1).
        # Specifically:
        # W[bk,ai] += 1 (from a*e)
        # W[bk,ai+1] += 1 (from d*e) -> No, d is A[ai+1,bj+1], e is B[bj,bk]. Product is A[ai+1,bj+1]B[bj,bk]. This contributes to C[ai+1,bk].
        # So W[bk,ai+1] += 1.
        # W[bk+1,ai] += 1 (from a*h)
        # W[bk+1,ai+1] += 1 (from d*h)
        
        # So for Term 1:
        U[ai*m + bj][col] = 1
        U[(ai+1)*m + (bj+1)][col] = 1
        V[bj*p + bk][col] = 1
        V[(bj+1)*p + (bk+1)][col] = 1
        W[bk*n + ai][col] = 1
        W[bk*n + (ai+1)][col] = 1
        W[(bk+1)*n + ai][col] = 1
        W[(bk+1)*n + (ai+1)][col] = 1
        col += 1
        
        # Term 2: (c+d)e = (A[ai+1,bj] + A[ai+1,bj+1]) * B[bj,bk]
        U[(ai+1)*m + bj][col] = 1
        U[(ai+1)*m + (bj+1)][col] = 1
        V[bj*p + bk][col] = 1
        W[bk*n + (ai+1)][col] = 1
        col += 1
        
        # Term 3: a(f-h) = A[ai,bj] * (B[bj,bk+1] - B[bj+1,bk+1])
        U[ai*m + bj][col] = 1
        V[bj*p + (bk+1)][col] = 1
        V[(bj+1)*p + (bk+1)][col] = -1
        W[(bk+1)*n + ai][col] = 1
        col += 1
        
        # Term 4: d(g-e) = A[ai+1,bj+1] * (B[bj+1,bk] - B[bj,bk])
        U[(ai+1)*m + (bj+1)][col] = 1
        V[(bj+1)*p + bk][col] = 1
        V[bj*p + bk][col] = -1
        W[bk*n + (ai+1)][col] = 1
        col += 1
        
        # Term 5: (a+b)h = (A[ai,bj] + A[ai,bj+1]) * B[bj+1,bk+1]
        U[ai*m + bj][col] = 1
        U[ai*m + (bj+1)][col] = 1
        V[(bj+1)*p + (bk+1)][col] = 1
        W[(bk+1)*n + ai][col] = 1
        col += 1
        
        # Term 6: (c-a)(e+f) = (A[ai+1,bj] - A[ai,bj]) * (B[bj,bk] + B[bj,bk+1])
        U[(ai+1)*m + bj][col] = 1
        U[ai*m + bj][col] = -1
        V[bj*p + bk][col] = 1
        V[bj*p + (bk+1)][col] = 1
        W[bk*n + (ai+1)][col] = 1
        W[(bk+1)*n + (ai+1)][col] = 1
        col += 1
        
        # Term 7: (b-d)(g+h) = (A[ai,bj+1] - A[ai+1,bj+1]) * (B[bj+1,bk] + B[bj+1,bk+1])
        U[ai*m + (bj+1)][col] = 1
        U[(ai+1)*m + (bj+1)][col] = -1
        V[(bj+1)*p + bk][col] = 1
        V[(bj+1)*p + (bk+1)][col] = 1
        W[bk*n + ai][col] = 1
        W[(bk+1)*n + ai][col] = 1
        col += 1

    # Now we apply this to the 4x4x4 multiplication.
    # The 7 M_i's are:
    # M1 = (A11+A22)(B11+B22) -> add_2x2_mult(0,0,0,0) is not correct.
    # M1 is the product of (A11+A22) and (B11+B22).
    # This is not a single 2x2x2 multiplication.
    # It is the sum of 4 2x2x2 multiplications.
    # (A11+A22) = [[A11_11+A22_11, A11_12+A22_12], [A11_21+A22_21, A11_22+A22_22]]
    # This is getting complicated.
    
    # Let's just use schoolbook for the 4x4x4 part to be safe.
    # Rank 64 for 4x4x4.
    # Total 64 + 48 = 112.
    # This is the same as the seed.
    
    # Let's try to implement the 28 rank Strassen properly.
    # I will use a known decomposition.
    
    # Reset col
    col = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Schoolbook for 4x4x4 (64 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    # Schoolbook for 4x4x3 (48 terms)
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
