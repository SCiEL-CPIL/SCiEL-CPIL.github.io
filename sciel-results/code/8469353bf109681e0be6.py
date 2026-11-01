"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = 2, 3, 6
    # We decompose into 3 blocks of 2x3x2, each using rank 8 (Strassen 7 + outer 1)
    # Total rank = 24
    
    rank = 24
    U = [[0] * rank for _ in range(n * m)]  # 6 x 24
    V = [[0] * rank for _ in range(m * p)]  # 18 x 24
    W = [[0] * rank for _ in range(p * n)]  # 12 x 24
    
    # For each block t = 0, 1, 2 (corresponding to B columns 2t, 2t+1):
    # We use 8 terms per block.
    # Block t uses columns 2t and 2t+1 of B (which are rows of B in V).
    # A has columns 0,1,2.
    
    # For block t, the sub-problem is:
    # A (2x3) times B_sub (3x2) where B_sub rows are 2t, 2t+1 of original B.
    # Let's index A as A[i,j] for i in 0..1, j in 0..2
    # B_sub as B_sub[j,k] for j in 0..2, k in 0..1 (k=0 -> col 2t, k=1 -> col 2t+1)
    # Output C[i,k] = sum_j A[i,j] * B_sub[j,k]
    
    # In our global indexing:
    # U index: i*m + j = 3*i + j (i in 0..1, j in 0..2)
    # V index: j*p + k' where k' is the global column of B. For block t, k' = 2t + k.
    #          So V index = 3*j + (2t + k) = 3j + 2t + k
    # W index: k'*n + i = (2t + k)*2 + i = 4t + 2k + i
    
    # Strassen for A_L (2x2, cols 0,1) times B_L (2x2, rows 0,1 of B_sub):
    # A_L[i,l] for i in 0..1, l in 0..1
    # B_L[j,k] for j in 0..1, k in 0..1
    # Strassen terms M1..M7:
    # M1 = (A_L[0,0] + A_L[1,1]) * (B_L[0,0] + B_L[1,1])
    # M2 = (A_L[1,0] + A_L[1,1]) * B_L[0,0]
    # M3 = A_L[0,0] * (B_L[0,1] - B_L[1,1])
    # M4 = A_L[1,1] * (B_L[1,0] - B_L[0,0])
    # M5 = (A_L[0,0] + A_L[0,1]) * B_L[1,1]
    # M6 = (A_L[1,0] - A_L[0,0]) * (B_L[0,0] + B_L[0,1])
    # M7 = (A_L[0,1] - A_L[1,1]) * (B_L[1,0] + B_L[1,1])
    #
    # C_L[i,k] = sum of appropriate M's:
    # C_L[0,0] = M1 + M4 - M5 + M7
    # C_L[0,1] = M3 + M5
    # C_L[1,0] = M2 + M4
    # C_L[1,1] = M1 - M2 + M3 + M6
    
    # Outer product term for a (col 2 of A) times b (row 2 of B_sub):
    # C_outer[i,k] = A[0,2] * B_sub[2,k] for i=0? No.
    # C[i,k] += A[i,2] * B_sub[2,k]
    # This is rank 1: U[i, term] = A[i,2] (but U must be constant!), V[j, term] = B_sub[2,j] (constant!), W[k*n+i, term] = 1
    # Wait, the decomposition is: sum_r U[i,r] * V[j,r] * W[k*n+i, r] = C[i,k]
    # For the outer product term r_outer:
    # We want sum_r U[i,r] * V[j,r] * W[k*n+i, r] to contribute A[i,2] * B_sub[2,k] to C[i,k] when j=2.
    # So set: U[i, r_outer] = A[i,2]... but U[i,r] must be a constant (the coefficient of x_{i,j} in the r-th factor).
    # Actually, U[i*m+j, r] is the coefficient. For the term r_outer, we need it to pick out A[i,2] from U and B_sub[2,k] from V.
    # So: U[3*i + 2, r_outer] = 1 for i=0,1. All other U entries for this r are 0.
    # V[3*2 + (2t+k), r_outer] = 1 for k=0,1. All other V entries for this r are 0.
    # W[(2t+k)*2 + i, r_outer] = 1 for all i,k.
    # Then the contribution is sum_{i,k} (1 * A[i,2]) * (1 * B_sub[2,k]) * 1 * ... wait.
    # The term is: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For fixed r = r_outer:
    # U[3i+2, r] = 1, others 0. So we need j=2.
    # V[6 + 2t + k, r] = 1, others 0. So we need the V index to be 6 + 2t + k.
    # W[4t + 2k + i, r] = 1.
    # Product: 1 * (x_{i,2}) * (y_{2, 2t+k}) * 1 = x_{i,2} * y_{2, 2t+k}
    # This is exactly A[i,2] * B_sub[2,k]. 
    
    # Strassen terms: each M_s is a product of a linear combo of A's and a linear combo of B's.
    # M_s = (sum_l c^A_{s,l} * A[i,l]) * (sum_j c^B_{s,j} * B_sub[j,k])
    # Wait, M1 = (A_L[0,0] + A_L[1,1]) * (B_L[0,0] + B_L[1,1])
    # This involves A[0,0], A[1,1] and B_sub[0,0], B_sub[1,1].
    # In the tensor: we need a term r such that:
    # U[3*0+0, r] = 1, U[3*1+1, r] = 1 (for A[0,0] and A[1,1])
    # V[3*0+2t+0, r] = 1, V[3*1+2t+1, r] = 1 (for B_sub[0,0] and B_sub[1,1])
    # W[4t+2*0+0, r] = 1, W[4t+2*1+1, r] = 1, etc.
    # Actually, M1 contributes to C_L[0,0] and C_L[1,1].
    # C_L[0,0] gets +M1, C_L[1,1] gets +M1.
    # So W[4t+0+0, r] += 1, W[4t+2+1, r] += 1.
    
    # Let's define the Strassen coefficients.
    # Each M_s has a specific pattern of U, V, W coefficients.
    # There are 7 M's per block, so 21 terms from Strassen + 3 outer terms = 24 terms.
    
    # Let's build the U, V, W matrices by filling in the entries.
    
    # We'll use a helper to add a term.
    # term_index: which of the 24 terms (0-23)
    # For block t (0,1,2), the terms are at indices 8*t .. 8*t+7.
    # Strassen terms: 8*t + 0 .. 8*t + 6
    # Outer term: 8*t + 7
    
    # Strassen term definitions (U indices, V indices, W indices with signs):
    # U index = 3*i + l (i in 0..1, l in 0..1 for A_L)
    # V index = 3*j + 2t + k (j in 0..1 for B_L, k in 0..1)
    # W index = 4t + 2*k + i (k in 0..1, i in 0..1)
    
    # M1: U: (0,0)=1, (1,1)=1 -> U[0, r]=1, U[4, r]=1
    #      V: (0,0)=1, (1,1)=1 -> V[2t, r]=1, V[3+2t+1, r]=1
    #      W: (0,0)=+1, (1,1)=+1 -> W[4t, r]+=1, W[4t+2+1, r]+=1
    # M2: U: (1,0)=1, (1,1)=1 -> U[3, r]=1, U[4, r]=1
    #      V: (0,0)=1 -> V[2t, r]=1
    #      W: (1,0)=+1 -> W[4t+0+1, r]+=1
    # M3: U: (0,0)=1 -> U[0, r]=1
    #      V: (0,1)=1, (1,1)=-1 -> V[3+2t+1, r]=1, V[3+2t+1, r]... wait
    #      B_L[0,1] is V[3*0 + 2t + 1, r] = V[2t+1, r]
    #      B_L[1,1] is V[3*1 + 2t + 1, r] = V[3+2t+1, r]
    #      So V[2t+1, r]=1, V[3+2t+1, r]=-1
    #      W: (0,1)=+1 -> W[4t+2+0, r]+=1
    # M4: U: (1,1)=1 -> U[4, r]=1
    #      V: (1,0)=1, (0,0)=-1 -> V[3+2t, r]=1, V[2t, r]=-1
    #      W: (0,0)=+1, (1,0)=+1 -> W[4t, r]+=1, W[4t+1, r]+=1
    # M5: U: (0,0)=1, (0,1)=1 -> U[0, r]=1, U[1, r]=1
    #      V: (1,1)=1 -> V[3+2t+1, r]=1
    #      W: (0,0)=-1, (1,1)=-1 -> W[4t, r]-=1, W[4t+3, r]-=1
    # M6: U: (1,0)=1, (0,0)=-1 -> U[3, r]=1, U[0, r]=-1
    #      V: (0,0)=1, (0,1)=1 -> V[2t, r]=1, V[2t+1, r]=1
    #      W: (1,1)=+1 -> W[4t+3, r]+=1
    # M7: U: (0,1)=1, (1,1)=-1 -> U[1, r]=1, U[4, r]=-1
    #      V: (1,0)=1, (1,1)=1 -> V[3+2t, r]=1, V[3+2t+1, r]=1
    #      W: (0,0)=+1, (0,1)=+1 -> W[4t, r]+=1, W[4t+2, r]+=1
    
    # Let's double check W indices:
    # W index = k'*n + i = (2t+k)*2 + i = 4t + 2k + i
    # For C_L[0,0]: i=0, k=0 -> W[4t + 0 + 0] = W[4t]
    # For C_L[0,1]: i=0, k=1 -> W[4t + 2 + 0] = W[4t+2]
    # For C_L[1,0]: i=1, k=0 -> W[4t + 0 + 1] = W[4t+1]
    # For C_L[1,1]: i=1, k=1 -> W[4t + 2 + 1] = W[4t+3]
    
    # Corrected Strassen W contributions:
    # M1: C_L[0,0]+=M1, C_L[1,1]+=M1 -> W[4t]+=1, W[4t+3]+=1
    # M2: C_L[1,0]+=M2 -> W[4t+1]+=1
    # M3: C_L[0,1]+=M3 -> W[4t+2]+=1
    # M4: C_L[0,0]+=M4, C_L[1,0]+=M4 -> W[4t]+=1, W[4t+1]+=1
    # M5: C_L[0,0]-=M5, C_L[0,1]+=M5 -> W[4t]-=1, W[4t+2]+=1
    # M6: C_L[1,1]+=M6 -> W[4t+3]+=1
    # M7: C_L[0,0]+=M7, C_L[1,1]+=M7 -> W[4t]+=1, W[4t+3]+=1
    
    # Let's verify C_L combinations:
    # C_L[0,0] = M1 + M4 - M5 + M7 ✓
    # C_L[0,1] = M3 + M5 ✓
    # C_L[1,0] = M2 + M4 ✓
    # C_L[1,1] = M1 - M2 + M3 + M6... wait, original was M1 - M2 + M3 + M6?
    # Standard Strassen: C11 = M1 - M2 + M3 + M6. Let me recheck.
    # Actually, standard Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B12-B22)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # My earlier listing had C00 = M1 + M4 - M5 + M7 which is wrong. Let me fix.
    
    # Correct Strassen:
    # C[0,0] = M1 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    # So W contributions:
    # M1: W[4t]+=1, W[4t+3]+=1
    # M2: W[4t+1]+=1
    # M3: W[4t+2]+=1, W[4t+3]+=1
    # M4: W[4t+1]+=1
    # M5: W[4t]-=1, W[4t+2]+=1
    # M6: W[4t+3]+=1
    # M7: W[4t]+=1, W[4t+3]+=1
    
    # Verify:
    # C[0,0] = M1 - M5 + M7 -> W[4t]: +1 - (-1) + 1 = +3? No, W[4t] gets M1(+1), M5(-1), M7(+1). Sum = 1-1+1=1. ✓
    # C[0,1] = M3 + M5 -> W[4t+2]: M3(+1) + M5(+1) = 2? No, M3 contributes to C[0,1] with +1, M5 contributes to C[0,1] with +1. W[4t+2] gets M3(+1) and M5(+1). ✓
    # C[1,0] = M2 + M4 -> W[4t+1]: M2(+1) + M4(+1) = 2. ✓
    # C[1,1] = M1 - M2 + M3 + M6 -> W[4t+3]: M1(+1) - M2(0) + M3(+1) + M6(+1) = 3? No, M2 doesn't contribute to C[1,1]. W[4t+3] gets M1(+1), M3(+1), M6(+1). Sum = 3. ✓
    
    # Now let's code this up.
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(3):
        base = 8 * t
        # M1: base + 0
        r = base
        U[0][r] += 1
        U[4][r] += 1
        V[2*t][r] += 1
        V[3 + 2*t + 1][r] += 1
        W[4*t][r] += 1
        W[4*t + 3][r] += 1
        # M2: base + 1
        r = base + 1
        U[3][r] += 1
        U[4][r] += 1
        V[2*t][r] += 1
        W[4*t + 1][r] += 1
        # M3: base + 2
        r = base + 2
        U[0][r] += 1
        V[2*t + 1][r] += 1
        V[3 + 2*t + 1][r] -= 1
        W[4*t + 2][r] += 1
        W[4*t + 3][r] += 1
        # M4: base + 3
        r = base + 3
        U[4][r] += 1
        V[3 + 2*t][r] += 1
        V[2*t][r] -= 1
        W[4*t + 1][r] += 1
        # M5: base + 4
        r = base + 4
        U[0][r] += 1
        U[1][r] += 1
        V[3 + 2*t + 1][r] += 1
        W[4*t][r] -= 1
        W[4*t + 2][r] += 1
        # M6: base + 5
        r = base + 5
        U[3][r] += 1
        U[0][r] -= 1
        V[2*t][r] += 1
        V[2*t + 1][r] += 1
        W[4*t + 3][r] += 1
        # M7: base + 6
        r = base + 6
        U[1][r] += 1
        U[4][r] -= 1
        V[3 + 2*t][r] += 1
        V[3 + 2*t + 1][r] += 1
        W[4*t][r] += 1
        W[4*t + 3][r] += 1
        # Outer: base + 7
        r = base + 7
        U[3*0 + 2][r] += 1  # i=0, j=2
        U[3*1 + 2][r] += 1  # i=1, j=2
        V[3*2 + 2*t + 0][r] += 1  # j=2, k=0
        V[3*2 + 2*t + 1][r] += 1  # j=2, k=1
        W[4*t + 2*0 + 0][r] += 1  # k=0, i=0
        W[4*t + 2*0 + 1][r] += 1  # k=0, i=1
        W[4*t + 2*1 + 0][r] += 1  # k=1, i=0
        W[4*t + 2*1 + 1][r] += 1  # k=1, i=1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

