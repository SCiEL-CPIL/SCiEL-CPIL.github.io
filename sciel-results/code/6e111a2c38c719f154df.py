import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # A: n x m (2x3), B: m x p (3x9), C: n x p (2x9)
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Tensor indices:
    # U index: i*m + j  (0..n*m-1 = 0..5)
    # V index: j*p + k  (0..m*p-1 = 0..26)
    # W index: k*n + i  (0..p*n-1 = 0..17)
    
    # The multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = 1
    # Rank-r decomposition: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # Start with the schoolbook (naive) decomposition of rank n*m*p = 54
    # Then try to reduce rank by merging terms.
    
    # Strategy: Use the structure of the problem.
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    # For fixed i and k, the output is a linear combination of 3 products.
    # We can potentially share some of these products across different (i,k) pairs.
    
    # Let's think about this more carefully.
    # The tensor is: T[i*m+j, j*p+k, k*n+i] = 1
    # 
    # This means: for each (i,j,k), we have a term U[i*m+j]*V[j*p+k]*W[k*n+i]
    # 
    # Total terms: n*m*p = 2*3*9 = 54
    
    # Can we do better?
    # 
    # Key observation: For a fixed j, the products A[i,j]*B[j,k] appear in
    # multiple outputs C[i,k]. Specifically, for fixed j, we have n*p = 2*9 = 18
    # such products.
    #
    # But these products are shared! For example, A[0,0]*B[0,k] appears in C[0,k]
    # and A[1,0]*B[0,k] appears in C[1,k]. They're different products though.
    #
    # Actually, the sharing is across different k values for the same (i,j) pair,
    # and across different i values for the same (j,k) pair.
    
    # Let me think about a different approach.
    # 
    # The matrix multiplication M_{n,m,p} has rank at most n*m*p (schoolbook).
    # For M_{2,3,9}, the schoolbook rank is 54.
    #
    # Can we use Strassen-like tricks?
    # 
    # Strassen works for square matrices. For rectangular, we can still apply
    # block decomposition.
    #
    # Let's partition:
    # A = [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # B = [B1; B2] where B1 is 2x9 and B2 is 1x9
    #
    # C = A1*B1 + A2*B2
    #
    # A1*B1 is 2x2 * 2x9. We can use Strassen on the 2x2 part of A1.
    # A2*B2 is 2x1 * 1x9, which is just 2*9 = 18 scalar products (rank 18).
    
    # For A1*B1 (2x2 * 2x9):
    # Let B1 = [B1a | B1b] where B1a is 2x4 and B1b is 2x5
    # Or better, process column by column or in blocks.
    
    # Actually, let's think about it differently.
    # 2x2 * 2x9 can be done as:
    # - Split the 9 columns into 4 groups of 2 + 1 group of 1
    # - For each 2x2 block of B, use Strassen (rank 7)
    # - For the remaining 2x1 block, use schoolbook (rank 4)
    # Total: 4*7 + 4 = 32
    
    # Then A2*B2 (2x1 * 1x9): rank 18
    # Total: 32 + 18 = 50
    
    # That's what the current solution does! Rank 50.
    
    # Can we do better?
    
    # Alternative: Don't split A into 2x2 + 2x1. Instead, think of the whole thing.
    # 
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    #
    # For each k, we're computing a 2-vector as a linear combination of 3 vectors
    # (columns of B). This is like a 2x3 * 3x1 multiplication for each k.
    #
    # For a single k, M_{2,3,1} has rank at most 2*3*1 = 6 (schoolbook).
    # Can we do better? M_{2,3,1} is just computing 2 linear combinations of 3 values.
    # The rank of this tensor is actually 6 (it's essentially two independent
    # inner products of length 3, but they share the B values).
    #
    # Wait, for fixed k, C[:,k] = A * B[:,k]. This is a 2x3 matrix times a 3x1 vector.
    # The tensor for this is T[i, j, 0] = delta_{i,j}... no.
    #
    # Actually, for the full problem, the key question is whether we can share
    # computations across different k values.
    
    # Let me try a different approach: use the fact that n=2 is small.
    #
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k]
    #
    # Define:
    # x_k = B[0,k], y_k = B[1,k], z_k = B[2,k]
    # a0 = A[0,0], a1 = A[0,1], a2 = A[0,2]
    # b0 = A[1,0], b1 = A[1,1], b2 = A[1,2]
    #
    # C[0,k] = a0*x_k + a1*y_k + a2*z_k
    # C[1,k] = b0*x_k + b1*y_k + b2*z_k
    #
    # This is like: for each k, compute [a0,a1,a2] dot [x_k,y_k,z_k]
    # and [b0,b1,b2] dot [x_k,y_k,z_k]
    #
    # The tensor structure: T depends on (i,j,k) where i in {0,1}, j in {0,1,2}, k in {0,...,8}
    # T[i*m+j, j*p+k, k*n+i] = 1
    #
    # For i=0: T[j, j*9+k, k] = 1 for j=0,1,2 and k=0,...,8
    # For i=1: T[3+j, j*9+k, 9+k] = 1 for j=0,1,2 and k=0,...,8
    #
    # Total 54 non-zero entries.
    
    # The question is: what is the minimum rank of this tensor?
    #
    # Lower bound: The border rank or tensor rank.
    # For M_{n,m,p}, the rank is at least max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
    # But this is a very weak lower bound.
    #
    # A better lower bound: The rank is at least the number of non-zero entries in
    # any "slice" of the tensor.
    #
    # Actually, for matrix multiplication, the rank is known to be at least
    # (n*m*p) / max(n,m,p) in some cases, but this isn't tight.
    
    # Let's just try to find a good decomposition.
    #
    # Idea: Use the structure to share computations.
    #
    # For each k, we need:
    # C[0,k] = a0*x_k + a1*y_k + a2*z_k
    # C[1,k] = b0*x_k + b1*y_k + b2*z_k
    #
    # Define intermediate quantities:
    # P_k = x_k * (a0 + b0)  -- not quite right
    #
    # Let's try:
    # s0_k = a0*x_k
    # s1_k = a1*y_k  
    # s2_k = a2*z_k
    # t0_k = b0*x_k
    # t1_k = b1*y_k
    # t2_k = b2*z_k
    #
    # C[0,k] = s0_k + s1_k + s2_k
    # C[1,k] = t0_k + t1_k + t2_k
    #
    # This is 6 products per k, total 54. No savings.
    
    # Can we share? Note that s0_k and t0_k both involve x_k.
    # s0_k = a0*x_k, t0_k = b0*x_k
    #
    # If we compute u_k = x_k * (a0 + b0) and v_k = x_k * (a0 - b0), then:
    # a0*x_k = (u_k + v_k)/2
    # b0*x_k = (u_k - v_k)/2
    #
    # But this uses 2 products instead of 2, so no savings in rank.
    # However, if we can share the (a0+b0) and (a0-b0) across k values...
    #
    # Wait, a0 and b0 are fixed (they're entries of A). So (a0+b0) is a constant.
    # The product is (a0+b0) * x_k, which is one product.
    # Then we need to combine with other terms.
    
    # Let me think about this more carefully using tensor decomposition language.
    #
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i]
    #
    # For the schoolbook: each (i,j,k) gets its own term r, with
    # U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    #
    # To reduce rank, we need to find linear combinations.
    #
    # Key insight: If two terms differ only in one index, we might be able to
    # combine them using a Strassen-like trick.
    
    # Let's try a systematic approach.
    #
    # Group by k. For each k, we have a 2x3 tensor slice:
    # T[i, j, k] for i in {0,1}, j in {0,1,2}
    #
    # This slice has rank at most 6 (schoolbook for 2x3*3x1).
    # But the slices for different k are not independent because they share
    # the A and B structure.
    
    # Actually, I think the rank-50 solution is close to optimal for this
    # blocking strategy. Let me try a different blocking.
    
    # Alternative: Split B into 3 blocks of 3 columns each.
    # B = [B1 | B2 | B3] where each B_i is 3x3
    # C = A * B = A*B1 + A*B2 + A*B3
    #
    # Each A*B_i is 2x3 * 3x3.
    #
    # For 2x3 * 3x3:
    # Schoolbook rank: 2*3*3 = 18
    # Can we do better?
    #
    # 2x3 * 3x3: Let A be 2x3, B be 3x3, C be 2x3.
    # C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # This is like computing 2 linear combinations of 3 vectors (columns of B).
    #
    # I don't see an obvious Strassen-like reduction for this shape.
    
    # Let me try yet another approach: use the fact that the problem is small
    # enough to potentially find a better decomposition through optimization.
    
    # But that's complex. Let me stick with the known rank-50 solution
    # but implement it more carefully to ensure correctness.
    
    # Actually, let me reconsider. The current solution has rank 50.
    # Let me verify: 4 blocks of 2x2*2x2 (Strassen, rank 7 each) = 28
    # 1 block of 2x2*2x1 (schoolbook, rank 4) = 4
    # 1 block of 2x1*1x9 (schoolbook, rank 18) = 18
    # Total: 28 + 4 + 18 = 50. Yes.
    
    # Can we improve the 2x2*2x1 part?
    # 2x2 * 2x1: C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
    # This is 4 scalar products. Can we do it in 3?
    # 
    # C[0,0] = a*e + b*f
    # C[1,0] = c*e + d*f
    #
    # This is M_{2,2,1} which has rank 4 (I believe). No savings.
    
    # What about the 2x1*1x9 part?
    # C[i,k] = A[i,2]*B[2,k] for i=0,1 and k=0,...,8
    # This is 18 scalar products. No sharing possible since each product
    # involves a unique (i,k) pair.
    
    # So rank 50 seems to be the best for this blocking strategy.
    
    # Let me try a completely different blocking.
    #
    # Split A into columns: A = [A_col0 | A_col1 | A_col2]
    # where A_colj is 2x1.
    #
    # C = A_col0 * B_row0 + A_col1 * B_row1 + A_col2 * B_row2
    #
    # Each term is 2x1 * 1x9, which has rank 18.
    # Total: 3*18 = 54. Worse than 50.
    
    # Split B into rows: B = [B_row0; B_row1; B_row2]
    # Same as above.
    
    # I think rank 50 is the best achievable with simple blocking.
    #
    # Let me just implement the rank-50 solution carefully.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A[0:2, 0:2] * B[0:2, :]
    # A_L is 2x2, B_T is 2x9
    # Split B_T into 4 blocks of 2 columns + 1 block of 1 column
    
    # Blocks of 2x2 * 2x2 using Strassen
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # B indices: B[0,k0], B[0,k1], B[1,k0], B[1,k1]
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # A indices: A[0,0], A[0,1], A[1,0], A[1,1]
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        # Strassen products (7 multiplications)
        # M1 = (a+d)(e+h)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # M2 = (c+d)e
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # M3 = a(f-h)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # M4 = d(g-e)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # M5 = (a+b)h
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # Output: C[0,k0], C[0,k1], C[1,k0], C[1,k1]
        # W indices: k*n+i
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,k0] = M1 + M4 - M5 + M7
        W[w_00][col] += 1
        W[w_00][col+3] += 1
        W[w_00][col+4] -= 1
        W[w_00][col+6] += 1
        
        # C[0,k1] = M3 + M5
        W[w_01][col+2] += 1
        W[w_01][col+4] += 1
        
        # C[1,k0] = M2 + M4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+2] += 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # Last block: 2x2 * 2x1 (column k=8)
    k8 = 8
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    
    # Schoolbook: 4 products
    # P1 = a*e -> C[0,k8]
    U[u_a][col] += 1
    V[v_e][col] += 1
    W[k8 * n + 0][col] += 1
    
    # P2 = b*g -> C[0,k8]
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    W[k8 * n + 0][col+1] += 1
    
    # P3 = c*e -> C[1,k8]
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    W[k8 * n + 1][col+2] += 1
    
    # P4 = d*g -> C[1,k8]
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    W[k8 * n + 1][col+3] += 1
    
    col += 4
    
    # Part 2: A[0:2, 2:3] * B[2:3, :]
    # A_R is 2x1, B_B is 1x9
    # Schoolbook: 18 products
    
    u_ar0 = 0 * m + 2  # A[0,2]
    u_ar1 = 1 * m + 2  # A[1,2]
    
    for k in range(9):
        v_b = 2 * p + k  # B[2,k]
        
        # Q_k = A[0,2] * B[2,k] -> C[0,k]
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        
        # R_k = A[1,2] * B[2,k] -> C[1,k]
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        
        col += 2
    
    # Verify: col should be 50
    assert col == rank, f"Expected col={rank}, got col={col}"
    
    return {"U": U, "V": V, "W": W}
