"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy import linalg

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # First, provide the schoolbook decomposition as a safe fallback
    def schoolbook():
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
    
    # Try to construct a lower-rank decomposition using block Strassen
    # For 6x6 matrices, we can partition into 3x3 blocks of 2x2 matrices
    # Each 2x2 block multiplication can be done with Strassen (rank 7)
    # Total rank = 3^3 * 7 = 189 (same as schoolbook 216? No, 216 vs 189)
    
    # Actually schoolbook is 6*6*6 = 216
    # Block Strassen: 3x3 blocks, each 2x2*2x2 uses 7 multiplications
    # Total: 27 * 7 = 189
    
    # Let's try to construct the block Strassen decomposition
    # The idea: A = [A_00 A_01 A_02; A_10 A_11 A_12; A_20 A_21 A_22] where each A_ij is 2x2
    # C = A * B, where B similarly partitioned
    # Each C_ij is a 2x2 matrix computed from products of 2x2 blocks
    
    # For 3x3 block matrix multiplication, we can use Strassen-like algorithms
    # But the standard approach is to just do block matrix multiplication
    # where each block product A_ij * B_jk is a 2x2 * 2x2 = 2x2 multiplication
    
    # Using Strassen for each 2x2*2x2:
    # Given X, Y (2x2), compute Z = X*Y using 7 multiplications:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # For block multiplication C = A*B where A, B are 3x3 blocks of 2x2:
    # C_ij = sum_k A_ik * B_kj
    # Each A_ik * B_kj is a 2x2*2x2 product, which we can do with 7 scalar multiplications
    # But wait, we need to be careful: the "scalar" multiplications in Strassen are actually
    # 2x2 matrix multiplications? No, Strassen works on entries.
    
    # Actually, let me reconsider. The tensor decomposition view:
    # We have a 6x6x6x6x6x6 tensor (or rather, the bilinear map from (R^6xR^6) x (R^6xR^6) -> R^6xR^6)
    # The rank is the minimum number of rank-1 tensors needed.
    
    # A rank-1 term in the decomposition corresponds to:
    # U[:, c] * V[:, c] * W[:, c] where these are vectors of length 36, 36, 36
    # More precisely, the matrix multiplication tensor T has entries T[i,j,k,l,m,n] = 1 if j==l and k==m else 0
    # Wait, let me think about this more carefully.
    
    # The matrix multiplication map: C = A * B
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has indices (i,j) for A, (j,k) for B, and (i,k) for C
    # T[(i,j),(j,k),(i,k)] = 1
    
    # A rank-1 decomposition term is: u[(i,j)] * v[(j,k)] * w[(i,k)]
    # So we need: sum_c u_c[(i,j)] * v_c[(j,k)] * w_c[(i,k)] = sum_j A[i,j]*B[j,k] = C[i,k]
    
    # This is equivalent to: the coefficient of A[i,j]*B[j,k] in the product is 1 for each (i,j,k)
    # and 0 otherwise.
    
    # For the schoolbook: each term c corresponds to a unique (i,j,k) triple
    # u_c[(i',j')] = 1 if (i',j')==(i,j) else 0
    # v_c[(j',k')] = 1 if (j',k')==(j,k) else 0
    # w_c[(i',k')] = 1 if (i',k')==(i,k) else 0
    
    # For Strassen on 2x2 blocks:
    # We partition indices into 3 groups of 2: {0,1}, {2,3}, {4,5}
    # Let's say group a contains indices 2a, 2a+1
    
    # A[i,j] where i in group a, j in group b -> A_ab
    # B[j,k] where j in group b, k in group c -> B_bc
    # C[i,k] where i in group a, k in group c -> C_ac
    
    # C_ac = sum_b A_ab * B_bc (where * is 2x2 matrix multiplication)
    
    # Each 2x2*2x2 multiplication can be done with 7 "scalar" multiplications
    # But here, the "scalars" are actually the entries of the 2x2 blocks
    # So A_ab is a 2x2 matrix with entries A[2a+r, 2b+s] for r,s in {0,1}
    
    # For the tensor decomposition, each Strassen term for a specific block pair (a,b,c)
    # and a specific Strassen multiplier index t (0..6) gives one rank-1 term
    # Total: 3*3*3 * 7 = 189 terms
    
    # Let's construct this:
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # For each block position (a,b,c) where a,b,c in {0,1,2}:
    # We need to compute A_ab * B_bc and add to C_ac
    # Using Strassen, we have 7 multiplications M1..M7
    
    # The U factor: for term (a,b,c,t), U[i,j] gets the coefficient from X = A_ab
    # The V factor: for term (a,b,c,t), V[j,k] gets the coefficient from Y = B_bc
    # The W factor: for term (a,b,c,t), W[k,i] gets the coefficient from Z = C_ac contribution
    
    # Note: W[k*n+i] corresponds to C[i,k]
    
    # Let's define the Strassen terms more carefully.
    # For X = A_ab (2x2) and Y = B_bc (2x2):
    # X entries: X[0,0]=A[2a,2b], X[0,1]=A[2a,2b+1], X[1,0]=A[2a+1,2b], X[1,1]=A[2a+1,2b+1]
    # Y entries: Y[0,0]=B[2b,2c], Y[0,1]=B[2b,2c+1], Y[1,0]=B[2b+1,2c], Y[1,1]=B[2b+1,2c+1]
    
    # Strassen multipliers:
    # M1: X factors: [1,0,0,1] (X00+X11), Y factors: [1,0,0,1] (Y00+Y11)
    # M2: X factors: [0,0,1,1] (X10+X11), Y factors: [1,0,0,0] (Y00)
    # M3: X factors: [1,0,0,0] (X00), Y factors: [0,1,-1,0] (Y01-Y11)
    # M4: X factors: [0,0,0,1] (X11), Y factors: [0,0,1,-1] (Y10-Y00)
    # M5: X factors: [1,1,0,0] (X00+X01), Y factors: [0,0,0,1] (Y11)
    # M6: X factors: [-1,0,1,0] (X10-X00), Y factors: [1,1,0,0] (Y00+Y01)
    # M7: X factors: [0,1,-1,0] (X01-X11), Y factors: [0,0,1,1] (Y10+Y11)
    
    # Output contributions:
    # Z00 = M1+M4-M5+M7
    # Z01 = M3+M5
    # Z10 = M2+M4
    # Z11 = M1-M2+M3+M6
    
    # For the tensor decomposition, each (a,b,c,t) gives one rank-1 term.
    # The U vector for this term: for index (i,j) where i=2a+r, j=2b+s, U[i*m+j] = X_factor[r,s]
    # The V vector for this term: for index (j,k) where j=2b+r, k=2c+s, V[j*p+k] = Y_factor[r,s]
    # The W vector for this term: for index (k,i) where k=2c+r, i=2a+s, W[k*n+i] = Z_contribution[r,s]
    
    # Wait, I need to be careful with the indexing. Let me re-examine.
    # C[i,k] where i is in group a, k is in group c
    # i = 2a + r, k = 2c + s for r,s in {0,1}
    # C[i,k] = sum_b (A_ab * B_bc)[r,s]
    
    # For a fixed (a,b,c), the contribution to C[2a+r, 2c+s] from A_ab * B_bc is:
    # sum_{u,v} A[2a+u, 2b+v] * B[2b+v, 2c+w] ... no wait.
    # (A_ab * B_bc)[r,s] = sum_{v=0}^{1} A[2a+r, 2b+v] * B[2b+v, 2c+s]
    
    # So for the Strassen decomposition of this 2x2 multiplication:
    # We express (A_ab * B_bc)[r,s] as a linear combination of the 7 Strassen multipliers M_t
    # Each M_t is a product of a linear combination of A entries and a linear combination of B entries.
    
    # Let X_factor_t be the vector of coefficients for A entries in M_t
    # Let Y_factor_t be the vector of coefficients for B entries in M_t
    # Let Z_coeff_t[r,s] be the coefficient of M_t in (A_ab * B_bc)[r,s]
    
    # Then the rank-1 term for (a,b,c,t) has:
    # U[2a+u, 2b+v] = X_factor_t[u,v] for u,v in {0,1}
    # V[2b+u, 2c+v] = Y_factor_t[u,v] for u,v in {0,1}
    # W[2c+v, 2a+u] = Z_coeff_t[u,v] for u,v in {0,1}
    
    # Note: W[k*n+i] where k=2c+v, i=2a+u, so W[2c+v, 2a+u] = Z_coeff_t[v,u]? 
    # No, W is indexed as W[k*n+i], so for a given (k,i), the value is W[k*n+i].
    # In our term, k=2c+v, i=2a+u, so W[2c+v, 2a+u] should be Z_coeff_t[u,v]
    # where Z_coeff_t[u,v] is the coefficient for C[2a+u, 2c+v]
    
    # Let me define the Strassen data:
    # X_factors[t][u,v] for t in 0..6, u,v in 0..1
    # Y_factors[t][u,v] for t in 0..6, u,v in 0..1
    # Z_coeffs[t][u,v] for t in 0..6, u,v in 0..1
    
    X_factors = [
        [[1, 0], [0, 1]],   # M1: X00+X11
        [[0, 0], [1, 1]],   # M2: X10+X11
        [[1, 0], [0, 0]],   # M3: X00
        [[0, 0], [0, 1]],   # M4: X11
        [[1, 1], [0, 0]],   # M5: X00+X01
        [[-1, 0], [1, 0]],  # M6: X10-X00
        [[0, 1], [-1, 0]],  # M7: X01-X11
    ]
    
    Y_factors = [
        [[1, 0], [0, 1]],   # M1: Y00+Y11
        [[1, 0], [0, 0]],   # M2: Y00
        [[0, 1], [-1, 0]],  # M3: Y01-Y11
        [[0, 0], [1, -1]],  # M4: Y10-Y00
        [[0, 0], [0, 1]],   # M5: Y11
        [[1, 1], [0, 0]],   # M6: Y00+Y01
        [[0, 0], [1, 1]],   # M7: Y10+Y11
    ]
    
    Z_coeffs = [
        [[1, 0], [0, 0]],   # M1 contributes to Z00
        [[0, 0], [0, 0]],   # M2 doesn't contribute directly? Wait...
        # Let me re-derive:
        # Z00 = M1+M4-M5+M7
        # Z01 = M3+M5
        # Z10 = M2+M4
        # Z11 = M1-M2+M3+M6
        # So:
        # M1: Z00+=1, Z11+=1
        # M2: Z10+=1, Z11-=1
        # M3: Z01+=1, Z11+=1
        # M4: Z00+=1, Z10+=1
        # M5: Z00-=1, Z01+=1
        # M6: Z11+=1
        # M7: Z00+=1
    ]
    
    # Let me redo Z_coeffs properly:
    Z_coeffs = [
        # M1: Z00=1, Z11=1
        [[1, 0], [0, 1]],
        # M2: Z10=1, Z11=-1
        [[0, 0], [1, -1]],
        # M3: Z01=1, Z11=1
        [[0, 1], [0, 1]],
        # M4: Z00=1, Z10=1
        [[1, 0], [1, 0]],
        # M5: Z00=-1, Z01=1
        [[-1, 1], [0, 0]],
        # M6: Z11=1
        [[0, 0], [0, 1]],
        # M7: Z00=1
        [[1, 0], [0, 0]],
    ]
    
    # Now construct the decomposition
    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for t in range(7):
                    # U factor
                    for u in range(2):
                        for v in range(2):
                            i = 2*a + u
                            j = 2*b + v
                            U[i * m + j][col] = X_factors[t][u][v]
                    
                    # V factor
                    for u in range(2):
                        for v in range(2):
                            j = 2*b + u
                            k = 2*c + v
                            V[j * p + k][col] = Y_factors[t][u][v]
                    
                    # W factor
                    for u in range(2):
                        for v in range(2):
                            i = 2*a + u
                            k = 2*c + v
                            W[k * n + i][col] = Z_coeffs[t][u][v]
                    
                    col += 1
    
    # Verify the decomposition
    def verify(U, V, W, tol=1e-10):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Check: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should equal delta_{j,k} * delta_{i,i}... 
        # Actually, we need: for all i,j,k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if this is a valid term
        # The matrix multiplication tensor: T[i,j,k] = 1 for each (i,j,k) where C[i,k] = sum_j A[i,j]*B[j,k]
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # Let's check a few entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False, val
        return True, None
    
    # Verify
    if time.time() < deadline:
        valid, val = verify(U, V, W)
        if valid:
            return {"U": U, "V": V, "W": W}
        else:
            # If verification fails, fall back to schoolbook
            return schoolbook()
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
