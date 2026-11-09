"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    deadline = time.time() + 130  # safety margin before 150s kill

    def schoolbook():
        """Standard rank-128 decomposition."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for l in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + l][col] = 1
                    W[l * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2():
        """
        Return the 7 Strassen products as a list of (u, v, w) index triples
        for computing C = A * B where A is 2x2 and B is 2x2.
        
        A = [[a, b], [c, d]], B = [[e, f], [g, h]]
        C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
        
        Strassen products:
        M1 = (a+d)(e+h)
        M2 = (c+d)e
        M3 = a(f-h)
        M4 = d(g-e)
        M5 = (a+b)h
        M6 = (c-a)(e+f)
        M7 = (b-d)(g+h)
        
        C[0,0] = M1 + M4 - M5 + M7
        C[0,1] = M3 + M5
        C[1,0] = M2 + M4
        C[1,1] = M1 - M2 + M3 + M6
        
        Returns list of (u_indices, v_indices, w_indices) where each is a dict
        mapping local index -> coefficient
        """
        # Each product is defined by which elements of A and B it uses
        # A local indices: 0=A[0,0], 1=A[0,1], 2=A[1,0], 3=A[1,1]
        # B local indices: 0=B[0,0], 1=B[0,1], 2=B[1,0], 3=B[1,1]
        # C local indices: 0=C[0,0], 1=C[0,1], 2=C[1,0], 3=C[1,1]
        
        # M1 = (a+d)(e+h): A{0:1, 3:1}, B{0:1, 3:1}, C{0:1, 3:1}
        # M2 = (c+d)e: A{2:1, 3:1}, B{0:1}, C{2:1}
        # M3 = a(f-h): A{0:1}, B{1:1, 3:-1}, C{1:1}
        # M4 = d(g-e): A{3:1}, B{2:1, 0:-1}, C{0:1, 2:1}
        # M5 = (a+b)h: A{0:1, 1:1}, B{3:1}, C{0:-1, 1:1}
        # M6 = (c-a)(e+f): A{2:1, 0:-1}, B{0:1, 1:1}, C{3:1}
        # M7 = (b-d)(g+h): A{1:1, 3:-1}, B{2:1, 3:1}, C{0:1}
        
        products = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
            ({2: 1, 3: 1}, {0: 1}, {2: 1}),                 # M2
            ({0: 1}, {1: 1, 3: -1}, {1: 1}),                # M3
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),          # M4
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),          # M5
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),          # M6
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),          # M7
        ]
        return products

    def strassen_block_decomposition():
        """
        Use Strassen's algorithm to decompose 4x4 * 4x8 into 16 blocks of 2x2 * 2x4.
        Each 2x2 * 2x4 block can be split into two 2x2 * 2x2 blocks.
        Each 2x2 * 2x2 uses 7 Strassen products.
        Total: 16 * 2 * 7 = 224? No, that's worse.
        
        Actually, let me think about this differently.
        
        4x4 * 4x8:
        Split A into 2x2 blocks (4 blocks), B into 2x4 blocks (4 blocks).
        C[i,j] = sum_k A[i,k] * B[k,j]
        
        For block (I, J) where I is a 2x2 block of A (rows 2I to 2I+1, cols 2J to 2J+1)
        and J is a 2x4 block of B (rows 2J to 2J+1, cols 4L to 4L+3 for some L)
        
        Wait, B is 4x8. Let me index B as B[k, l] where k in [0,4), l in [0,8).
        C[i, l] = sum_k A[i, k] * B[k, l]
        
        Split into 2x2 blocks:
        A has 2x2 = 4 blocks (rows 0-1,2-3; cols 0-1,2-3)
        B has 2x4 = 4 blocks (rows 0-1,2-3; cols 0-3,4-7)
        C has 2x4 = 4 blocks (rows 0-1,2-3; cols 0-3,4-7)
        
        C_block(I, L) = sum_J A_block(I, J) * B_block(J, L)
        where I in {0,1}, J in {0,1}, L in {0,1}
        
        A_block(I, J) is 2x2, B_block(J, L) is 2x4, C_block(I, L) is 2x4
        
        For each (I, J, L), we compute a 2x2 * 2x4 product.
        Split B_block(J, L) into two 2x2 blocks: B_block(J, L, 0) and B_block(J, L, 1)
        Then C_block(I, L) = A_block(I, J) * B_block(J, L, 0) + A_block(I, J) * B_block(J, L, 1)
        
        Each 2x2 * 2x2 uses 7 Strassen products.
        So each 2x2 * 2x4 uses 14 products.
        Total: 4 * 4 * 14 = 224. That's worse than 128!
        
        Hmm, that's not helpful. Let me think differently.
        
        Actually, for the 2x2 * 2x4 case, we can do better.
        C (2x4) = A (2x2) * B (2x4)
        C[:, 0:2] = A * B[:, 0:2]  (2x2 * 2x2)
        C[:, 2:4] = A * B[:, 2:4]  (2x2 * 2x2)
        
        So we need 2 * 7 = 14 products per (I, J, L) block.
        Total: 2 * 2 * 2 * 14 = 112 products.
        
        That's 112 < 128! Let me implement this.
        """
        rank = 112
        U = [[0] * rank for _ in range(n * m)]  # 16 x 112
        V = [[0] * rank for _ in range(m * p)]  # 32 x 112
        W = [[0] * rank for _ in range(p * n)]  # 32 x 112
        
        strassen_products = strassen_2x2()
        
        col = 0
        # I: row block of A and C (0 or 1), corresponds to rows 2*I to 2*I+1
        # J: col block of A and row block of B (0 or 1), corresponds to cols/rows 2*J to 2*J+1
        # L: col block of B and C (0 or 1), corresponds to cols 4*L to 4*L+3
        for I in range(2):
            for J in range(2):
                for L in range(2):
                    # A_block(I, J): rows 2*I to 2*I+1, cols 2*J to 2*J+1
                    # B_block(J, L): rows 2*J to 2*J+1, cols 4*L to 4*L+3
                    # C_block(I, L): rows 2*I to 2*I+1, cols 4*L to 4*L+3
                    
                    # Split B_block into two 2x2 blocks
                    for half in range(2):
                        # B_sub: rows 2*J to 2*J+1, cols 4*L+2*half to 4*L+2*half+1
                        # C_sub: rows 2*I to 2*I+1, cols 4*L+2*half to 4*L+2*half+1
                        
                        for prod_idx in range(7):
                            u_dict, v_dict, w_dict = strassen_products[prod_idx]
                            
                            for a_local, a_coeff in u_dict.items():
                                # Global A index: row 2*I + a_local//2, col 2*J + a_local%2
                                a_row = 2 * I + a_local // 2
                                a_col = 2 * J + a_local % 2
                                U_idx = a_row * m + a_col
                                U[U_idx][col] += a_coeff
                            
                            for b_local, b_coeff in v_dict.items():
                                # Global B index: row 2*J + b_local//2, col 4*L + 2*half + b_local%2
                                b_row = 2 * J + b_local // 2
                                b_col = 4 * L + 2 * half + b_local % 2
                                V_idx = b_row * p + b_col
                                V[V_idx][col] += b_coeff
                            
                            for c_local, c_coeff in w_dict.items():
                                # Global C index: row 2*I + c_local//2, col 4*L + 2*half + c_local%2
                                c_row = 2 * I + c_local // 2
                                c_col = 4 * L + 2 * half + c_local % 2
                                W_idx = c_col * n + c_row
                                W[W_idx][col] += c_coeff
                            
                            col += 1
        
        # Verify col == rank
        assert col == rank, f"Expected {rank} columns, got {col}"
        
        return {"U": U, "V": V, "W": W}

    def verify_decomposition(result):
        """Verify that the decomposition is correct."""
        U = np.array(result["U"], dtype=float)
        V = np.array(result["V"], dtype=float)
        W = np.array(result["W"], dtype=float)
        
        n, m, p = DIMENSIONS
        rank = U.shape[1]
        
        # Build the tensor T
        # T[a, b, c] where a = i*m+j, b = j*p+l, c = l*n+i
        # T should be 1 for all valid (i, j, l)
        
        # Compute sum_s U[a,s] * V[b,s] * W[c,s] for all (a,b,c)
        # This is equivalent to: for each s, form outer product U[:,s] * V[:,s] * W[:,s]
        # and sum them up
        
        # More efficiently: for each s, we can think of it as
        # M_s = U[:,s] * V[:,s]^T  (n*m x m*p matrix)
        # Then the contribution to W is M_s * W[:,s]... no, that's not right.
        
        # The tensor is T[a,b,c] = sum_s U[a,s] * V[b,s] * W[c,s]
        # Let's compute this for a few random (a,b,c) and check
        
        # Actually, let's verify by checking that for random A and B,
        # the computed C matches A*B
        
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from the decomposition
        # C[i,l] = sum_j sum_s U[i*m+j, s] * V[j*p+l, s] * W[l*n+i, s]
        # = sum_s W[l*n+i, s] * sum_j U[i*m+j, s] * V[j*p+l, s]
        
        C_computed = np.zeros((n, p))
        for s in range(rank):
            # For each s, compute the contribution
            # contrib[i,l] = W[l*n+i, s] * sum_j U[i*m+j, s] * V[j*p+l, s]
            for i in range(n):
                for l in range(p):
                    inner = 0
                    for j in range(m):
                        inner += U[i * m + j, s] * V[j * p + l, s]
                    C_computed[i, l] += W[l * n + i, s] * inner
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            diff = np.max(np.abs(C_computed - C_expected))
            return False, diff
        return True, 0.0

    # Try the Strassen block decomposition
    try:
        result = strassen_block_decomposition()
        valid, diff = verify_decomposition(result)
        if valid:
            return result
        else:
            # If Strassen decomposition is incorrect, fall back to schoolbook
            return schoolbook()
    except Exception as e:
        return schoolbook()
# EVOLVE-BLOCK-END
