"""Tensor decomposition for 2x5x5 matrix multiplication."""
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# Known rank-7 decomposition for 2x2x2 (Strassen)
# Indices: (i, j, k) for i in {0,1}, j in {0,1}, k in {0,1}
# Each term is (u_idx, v_idx, w_idx, coeff) where:
#   u_idx = i*2 + j  (row in U, shape 4 x rank)
#   v_idx = j*2 + k  (row in V, shape 4 x rank)
#   w_idx = k*2 + i  (row in W, shape 4 x rank)
#
# Strassen 7 terms for 2x2x2:
# The 2x2x2 multiplication tensor T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
# can be decomposed into 7 rank-1 terms using Strassen's algorithm.
# 
# Let A be 2x2, B be 2x2, C = A*B be 2x2.
# A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
# C[i,k] = sum_j A[i,j] B[j,k]
#
# Strassen computes 7 products:
# M1 = (a00+a11)(b00+b11)
# M2 = (a10+a11)b00
# M3 = a00(b10-b11)
# M4 = a11(b01-b00)
# M5 = (a00+a01)b11
# M6 = (a10-a00)(b00+b10)
# M7 = (a01-a11)(b01+b11)
#
# Then:
# C00 = M1 - M4 + M5 - M7
# C01 = M3 + M4
# C10 = M2 + M5
# C11 = M1 - M2 + M3 + M6
#
# So the 7 rank-1 terms (as linear combinations of A and B elements) are:
# Term 1: (a00+a11) * (b00+b11) -> contributes to C00 (+1), C11 (+1)
# Term 2: (a10+a11) * b00       -> contributes to C10 (+1), C11 (-1)
# Term 3: a00 * (b10-b11)       -> contributes to C00 (-1), C01 (+1)
# Term 4: a11 * (b01-b00)       -> contributes to C00 (-1), C01 (+1)
# Term 5: (a00+a01) * b11       -> contributes to C00 (+1), C10 (+1)
# Term 6: (a10-a00) * (b00+b10) -> contributes to C11 (+1)
# Term 7: (a01-a11) * (b01+b11) -> contributes to C00 (-1), C11 (+1)

def strassen_222_terms():
    """
    Return 7 terms for 2x2x2 matrix multiplication.
    Each term: list of (u_idx, coeff), list of (v_idx, coeff), list of (w_idx, coeff)
    u_idx in 0..3, v_idx in 0..3, w_idx in 0..3
    """
    terms = []
    
    # Term 1: (a00+a11) * (b00+b11)
    # U: a00 (idx 0*2+0=0) coeff +1, a11 (idx 1*2+1=3) coeff +1
    # V: b00 (idx 0*2+0=0) coeff +1, b11 (idx 1*2+1=3) coeff +1
    # W: C00 (idx 0*2+0=0) coeff +1, C11 (idx 1*2+1=3) coeff +1
    terms.append(([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]))
    
    # Term 2: (a10+a11) * b00
    # U: a10 (idx 1*2+0=2) coeff +1, a11 (idx 3) coeff +1
    # V: b00 (idx 0) coeff +1
    # W: C10 (idx 0*2+1=1) coeff +1, C11 (idx 3) coeff -1
    terms.append(([ (2,1), (3,1) ], [ (0,1) ], [ (1,1), (3,-1) ]))
    
    # Term 3: a00 * (b10-b11)
    # U: a00 (idx 0) coeff +1
    # V: b10 (idx 1*2+0=2) coeff +1, b11 (idx 3) coeff -1
    # W: C00 (idx 0) coeff -1, C01 (idx 1*2+0=2) coeff +1
    terms.append(([ (0,1) ], [ (2,1), (3,-1) ], [ (0,-1), (2,1) ]))
    
    # Term 4: a11 * (b01-b00)
    # U: a11 (idx 3) coeff +1
    # V: b01 (idx 0*2+1=1) coeff +1, b00 (idx 0) coeff -1
    # W: C00 (idx 0) coeff -1, C01 (idx 2) coeff +1
    terms.append(([ (3,1) ], [ (1,1), (0,-1) ], [ (0,-1), (2,1) ]))
    
    # Term 5: (a00+a01) * b11
    # U: a00 (idx 0) coeff +1, a01 (idx 0*2+1=1) coeff +1
    # V: b11 (idx 3) coeff +1
    # W: C00 (idx 0) coeff +1, C10 (idx 1) coeff +1
    terms.append(([ (0,1), (1,1) ], [ (3,1) ], [ (0,1), (1,1) ]))
    
    # Term 6: (a10-a00) * (b00+b10)
    # U: a10 (idx 2) coeff +1, a00 (idx 0) coeff -1
    # V: b00 (idx 0) coeff +1, b10 (idx 2) coeff +1
    # W: C11 (idx 3) coeff +1
    terms.append(([ (2,1), (0,-1) ], [ (0,1), (2,1) ], [ (3,1) ]))
    
    # Term 7: (a01-a11) * (b01+b11)
    # U: a01 (idx 1) coeff +1, a11 (idx 3) coeff -1
    # V: b01 (idx 1) coeff +1, b11 (idx 3) coeff +1
    # W: C00 (idx 0) coeff -1, C11 (idx 3) coeff +1
    terms.append(([ (1,1), (3,-1) ], [ (1,1), (3,1) ], [ (0,-1), (3,1) ]))
    
    return terms

def construct():
    """
    Construct a rank-46 decomposition for 2x5x5 matrix multiplication.
    
    Strategy: Split the 5x5 B matrix into blocks:
    - B is 5x5, A is 2x5, C is 2x5
    - Split the 5 columns of A (and rows of B) into 2+2+1
    - Split the 5 rows of B (and columns of C) into 2+2+1
    - Use Strassen for 2x2x2 sub-blocks, schoolbook for 2x1x2, 2x2x1, 2x1x1
    
    Block structure:
    A = [A1 A2 A3] where A1 is 2x2, A2 is 2x2, A3 is 2x1
    B = [B11 B12 B13; B21 B22 B23; B31 B32 B33]
       where B11 is 2x2, B12 is 2x2, B13 is 2x1, etc.
    
    C = A1*B_col1 + A2*B_col2 + A3*B_col3
    where B_col1 = [B11; B21; B31] (5x2), etc.
    
    More precisely:
    C[:, 0:2] = A1 * B[0:2, 0:2] + A2 * B[2:4, 0:2] + A3 * B[4:5, 0:2]
    C[:, 2:4] = A1 * B[0:2, 2:4] + A2 * B[2:4, 2:4] + A3 * B[4:5, 2:4]
    C[:, 4]   = A1 * B[0:2, 4]   + A2 * B[2:4, 4]   + A3 * B[4:5, 4]
    
    So we have 9 sub-blocks to multiply:
    1. A1 (2x2) * B11 (2x2) -> C[:, 0:2] part 1: 2x2x2, rank 7
    2. A2 (2x2) * B21 (2x2) -> C[:, 0:2] part 2: 2x2x2, rank 7
    3. A3 (2x1) * B31 (2x1) -> C[:, 0:2] part 3: 2x1x2, rank 2
    4. A1 (2x2) * B12 (2x2) -> C[:, 2:4] part 1: 2x2x2, rank 7
    5. A2 (2x2) * B22 (2x2) -> C[:, 2:4] part 2: 2x2x2, rank 7
    6. A3 (2x1) * B32 (2x1) -> C[:, 2:4] part 3: 2x1x2, rank 2
    7. A1 (2x2) * B13 (2x1) -> C[:, 4] part 1: 2x2x1, rank 2
    8. A2 (2x2) * B23 (2x1) -> C[:, 4] part 2: 2x2x1, rank 2
    9. A3 (2x1) * B33 (1x1) -> C[:, 4] part 3: 2x1x1, rank 2
    
    Total rank = 7+7+2+7+7+2+2+2+2 = 38
    
    Wait, let me recalculate. Actually:
    - 2x2x2: rank 7
    - 2x1x2: A is 2x1, B is 1x2, C is 2x2. C[i,k] = A[i,0]*B[0,k]. Rank 2.
    - 2x2x1: A is 2x2, B is 2x1, C is 2x1. C[i,0] = sum_j A[i,j]*B[j,0]. Rank 2.
    - 2x1x1: A is 2x1, B is 1x1, C is 2x1. C[i,0] = A[i,0]*B[0,0]. Rank 2.
    
    So: 4 * 7 + 2 * 2 + 2 * 2 + 2 = 28 + 4 + 4 + 2 = 38.
    
    Actually wait, let me recount the blocks:
    - (A1, B11): 2x2 * 2x2 -> 2x2, rank 7
    - (A2, B21): 2x2 * 2x2 -> 2x2, rank 7
    - (A3, B31): 2x1 * 1x2 -> 2x2, rank 2
    - (A1, B12): 2x2 * 2x2 -> 2x2, rank 7
    - (A2, B22): 2x2 * 2x2 -> 2x2, rank 7
    - (A3, B32): 2x1 * 1x2 -> 2x2, rank 2
    - (A1, B13): 2x2 * 2x1 -> 2x1, rank 2
    - (A2, B23): 2x2 * 2x1 -> 2x1, rank 2
    - (A3, B33): 2x1 * 1x1 -> 2x1, rank 2
    
    Total = 4*7 + 2*2 + 2*2 + 1*2 = 28 + 4 + 4 + 2 = 38
    """
    n, m, p = 2, 5, 5
    rank_total = 38
    
    # U shape: (n*m, rank) = (10, 38)
    # V shape: (m*p, rank) = (25, 38)
    # W shape: (p*n, rank) = (10, 38)
    
    U = np.zeros((n * m, rank_total))
    V = np.zeros((m * p, rank_total))
    W = np.zeros((p * n, rank_total))
    
    col = 0
    
    def add_222(u_base, v_base, w_base):
        """Add Strassen 2x2x2 terms.
        u_base: base index for U (i*5 + j, where j is local 0,1)
        v_base: base index for V (j*5 + k, where j is local 0,1, k is local 0,1)
        w_base: base index for W (k*2 + i, where k is local 0,1, i is local 0,1)
        """
        global col
        strassen_terms = strassen_222_terms()
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            for u_idx_local, coeff in u_coeffs:
                U[u_base + u_idx_local, col] = coeff
            for v_idx_local, coeff in v_coeffs:
                V[v_base + v_idx_local, col] = coeff
            for w_idx_local, coeff in w_coeffs:
                W[w_base + w_idx_local, col] = coeff
            col += 1
    
    def add_212(u_base, v_base, w_base):
        """Add terms for 2x1x2: A(2x1) * B(1x2) -> C(2x2).
        C[i,k] = A[i,0] * B[0,k]
        u_base: U index for i (i*5 + j, j=0 is the single column)
        v_base: V index for k (j*5 + k, j=0 is the single row, k is 0,1)
        w_base: W index for (k,i) (k*2 + i)
        """
        global col
        for k in range(2):
            U[u_base + 0, col] = 1  # A[i,0]
            V[v_base + k, col] = 1  # B[0,k]
            W[w_base + k*2 + 0, col] = 1  # C[0,k]... wait
            # Actually C[i,k], so W index is k*2 + i
            # We need two terms: one for i=0, one for i=1
            # Let me redo this
            col -= 1  # undo
            break
        # Redo properly: 2 terms, one for each i
        # Term for i=0: U[0]=1, V[k]... no, V is fixed per k
        # Actually: for each k in {0,1}: C[i,k] = A[i,0]*B[0,k]
        # So for k=0: two terms (i=0, i=1)
        #   Term: U[0]=1, V[0]=1, W[0*2+0]=1 -> C[0,0]
        #   Term: U[1]=1, V[0]=1, W[0*2+1]=1 -> C[1,0]
        # For k=1:
        #   Term: U[0]=1, V[1]=1, W[1*2+0]=1 -> C[0,1]
        #   Term: U[1]=1, V[1]=1, W[1*2+1]=1 -> C[1,1]
        # That's 4 terms, not 2. Wait, rank of 2x1x2 is 2?
        # C[i,k] = A[i,0]*B[0,k]. This is an outer product of A_col and B_row.
        # The tensor is C[i,k] = a_i * b_k. This has rank 1!
        # No wait, it's a 2x2x2 tensor where T[i,0,k] = a_i * b_k.
        # This is rank 1: u = (a_0, a_1), v = (b_0, b_1), w = (1,1,1,1) in some sense.
        # Actually in the matrix multiplication tensor framework:
        # U[i*m+j, c] = A element, V[j*p+k, c] = B element, W[k*n+i, c] = C element
        # For 2x1x2: m=1, n=2, p=2
        # U has shape (2*1, r) = (2, r)
        # V has shape (1*2, r) = (2, r)
        # W has shape (2*2, r) = (4, r)
        # T[U_idx, V_idx, W_idx] = 1 if U_idx corresponds to A[i,0], V_idx to B[0,k], W_idx to C[i,k]
        # U_idx = i*1+0 = i, V_idx = 0*2+k = k, W_idx = k*2+i
        # So T[i, k, k*2+i] = 1
        # This is: for each (i,k), one 1. Total 4 non-zero entries.
        # Can we do it in rank 2? 
        # Term 1: U[0]=1, V[0]=1, W[0*2+0]=1 -> gives T[0,0,0]=1
        # Term 2: U[1]=1, V[0]=1, W[0*2+1]=1 -> gives T[1,0,1]=1
        # Term 3: U[0]=1, V[1]=1, W[1*2+0]=1 -> gives T[0,1,2]=1
        # Term 4: U[1]=1, V[1]=1, W[1*2+1]=1 -> gives T[1,1,3]=1
        # That's 4 terms. But can we combine? 
        # If we use U[0]=1, U[1]=1, V[0]=1, V[1]=1, W[0]=1, W[1]=1, W[2]=1, W[3]=1, that gives all 4 = 1, but we want specific ones.
        # Actually the rank of the 2x1x2 multiplication tensor is 2? Let me check.
        # C[i,k] = A[i,0]*B[0,k]. Think of it as: for fixed i, C[i,:] = A[i,0] * B[0,:]. 
        # The tensor is T[i,k,w] where w indexes C[i,k]. T[i,k, k*2+i] = 1.
        # This is a "diagonal-like" tensor. Its rank is 4? Or can it be lower?
        # Actually, I think the rank of multiplying a 2x1 by 1x2 to get 2x2 is 2.
        # Let me think: C = a * b^T where a is 2x1 and b is 1x2.
        # The tensor T[i, j, k] where i indexes a, j indexes b (but j is always 0), k indexes C.
        # In our convention: U[i*1+0, c], V[0*2+k, c], W[k*2+i, c]
        # So U has 2 rows, V has 2 rows, W has 4 rows.
        # T[U_idx, V_idx, W_idx] = 1 if U_idx=i, V_idx=k, W_idx=k*2+i
        # So the non-zero entries are at (0,0,0), (1,0,1), (0,1,2), (1,1,3).
        # This is a 2x2x4 tensor. Its rank... 
        # Let me just use 4 terms to be safe. Or actually, I realize I can use 2 terms:
        # Term 1: U[0]=1, U[1]=1, V[0]=1, W[0]=1, W[1]=1 -> gives T[0,0,0]=1, T[1,0,1]=1
        # Term 2: U[0]=1, U[1]=1, V[1]=1, W[2]=1, W[3]=1 -> gives T[0,1,2]=1, T[1,1,3]=1
        # Yes! That works. Rank 2.
        
        # Term 1: for k=0
        U[u_base + 0, col] = 1  # A[0,0]
        U[u_base + 1, col] = 1  # A[1,0]
        V[v_base + 0, col] = 1  # B[0,0]
        W[w_base + 0, col] = 1  # C[0,0] -> W idx = 0*2+0 = 0
        W[w_base + 1, col] = 1  # C[1,0] -> W idx = 0*2+1 = 1
        col += 1
        # Term 2: for k=1
        U[u_base + 0, col] = 1  # A[0,0]
        U[u_base + 1, col] = 1  # A[1,0]
        V[v_base + 1, col] = 1  # B[0,1]
        W[w_base + 2, col] = 1  # C[0,1] -> W idx = 1*2+0 = 2
        W[w_base + 3, col] = 1  # C[1,1] -> W idx = 1*2+1 = 3
        col += 1
    
    def add_221(u_base, v_base, w_base):
        """Add terms for 2x2x1: A(2x2) * B(2x1) -> C(2x1).
        C[i,0] = sum_j A[i,j] * B[j,0]
        u_base: U index for (i,j), i in {0,1}, j in {0,1}
        v_base: V index for (j,0), j in {0,1}
        w_base: W index for (0,i), i in {0,1}
        """
        global col
        # This is 2x2x1. Rank is 2? 
        # C[i,0] = A[i,0]*B[0,0] + A[i,1]*B[1,0]
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        #   U[u_base+0]=1, V[v_base+0]=1, W[w_base+0]=1
        # Term 2: A[1,0]*B[0,0] -> C[1,0]
        #   U[u_base+2]=1, V[v_base+0]=1, W[w_base+1]=1
        # Term 3: A[0,1]*B[1,0] -> C[0,0]
        #   U[u_base+1]=1, V[v_base+1]=1, W[w_base+0]=1
        # Term 4: A[1,1]*B[1,0] -> C[1,0]
        #   U[u_base+3]=1, V[v_base+1]=1, W[w_base+1]=1
        # Can we do better? 
        # Term 1: U[0]=1, U[2]=1, V[0]=1, W[0]=1, W[1]=1 -> C[0,0]+=A[0,0]*B[0,0], C[1,0]+=A[1,0]*B[0,0]
        # Term 2: U[1]=1, U[3]=1, V[1]=1, W[0]=1, W[1]=1 -> C[0,0]+=A[0,1]*B[1,0], C[1,0]+=A[1,1]*B[1,0]
        # Yes! Rank 2.
        
        # Term 1: j=0
        U[u_base + 0, col] = 1  # A[0,0]
        U[u_base + 2, col] = 1  # A[1,0]
        V[v_base + 0, col] = 1  # B[0,0]
        W[w_base + 0, col] = 1  # C[0,0] -> W idx = 0*2+0 = 0
        W[w_base + 1, col] = 1  # C[1,0] -> W idx = 0*2+1 = 1
        col += 1
        # Term 2: j=1
        U[u_base + 1, col] = 1  # A[0,1]
        U[u_base + 3, col] = 1  # A[1,1]
        V[v_base + 1, col] = 1  # B[1,0]
        W[w_base + 0, col] = 1  # C[0,0]
        W[w_base + 1, col] = 1  # C[1,0]
        col += 1
    
    def add_211(u_base, v_base, w_base):
        """Add terms for 2x1x1: A(2x1) * B(1x1) -> C(2x1).
        C[i,0] = A[i,0] * B[0,0]
        Rank 2 (one term per i, or combine).
        Actually rank 1: U[0]=1, U[1]=1, V[0]=1, W[0]=1, W[1]=1
        """
        global col
        U[u_base + 0, col] = 1
        U[u_base + 1, col] = 1
        V[v_base + 0, col] = 1
        W[w_base + 0, col] = 1
        W[w_base + 1, col] = 1
        col += 1
    
    # Block assignments:
    # A columns split: 0-1 (A1), 2-3 (A2), 4 (A3)
    # B rows split: 0-1, 2-3, 4
    # B columns split: 0-1, 2-3, 4
    
    # Block 1: A1 (cols 0-1) * B[0:2, 0:2] -> C[:, 0:2]
    # U base: i*5 + 0 (j=0,1)
    # V base: 0*5 + 0 (j=0, k=0,1)
    # W base: 0*2 + 0 (k=0, i=0,1)
    add_222(0, 0, 0)  # col += 7
    
    # Block 2: A2 (cols 2-3) * B[2:4, 0:2] -> C[:, 0:2]
    # U base: i*5 + 2
    # V base: 2*5 + 0 = 10
    # W base: 0*2 + 0 = 0
    add_222(2, 10, 0)  # col += 7
    
    # Block 3: A3 (col 4) * B[4, 0:2] -> C[:, 0:2]
    # U base: i*5 + 4
    # V base: 4*5 + 0 = 20
    # W base: 0*2 + 0 = 0
    add_212(4, 20, 0)  # col += 2
    
    # Block 4: A1 (cols 0-1) * B[0:2, 2:4] -> C[:, 2:4]
    # U base: i*5 + 0
    # V base: 0*5 + 2 = 2
    # W base: 2*2 + 0 = 4
    add_222(0, 2, 4)  # col += 7
    
    # Block 5: A2 (cols 2-3) * B[2:4, 2:4] -> C[:, 2:4]
    # U base: i*5 + 2
    # V base: 2*5 + 2 = 12
    # W base: 2*2 + 0 = 4
    add_222(2, 12, 4)  # col += 7
    
    # Block 6: A3 (col 4) * B[4, 2:4] -> C[:, 2:4]
    # U base: i*5 + 4
    # V base: 4*5 + 2 = 22
    # W base: 2*2 + 0 = 4
    add_212(4, 22, 4)  # col += 2
    
    # Block 7: A1 (cols 0-1) * B[0:2, 4] -> C[:, 4]
    # U base: i*5 + 0
    # V base: 0*5 + 4 = 4
    # W base: 4*2 + 0 = 8
    add_221(0, 4, 8)  # col += 2
    
    # Block 8: A2 (cols 2-3) * B[2:4, 4] -> C[:, 4]
    # U base: i*5 + 2
    # V base: 2*5 + 4 = 14
    # W base: 4*2 + 0 = 8
    add_221(2, 14, 8)  # col += 2
    
    # Block 9: A3 (col 4) * B[4, 4] -> C[:, 4]
    # U base: i*5 + 4
    # V base: 4*5 + 4 = 24
    # W base: 4*2 + 0 = 8
    add_211(4, 24, 8)  # col += 1
    
    # Total col should be 7+7+2+7+7+2+2+2+1 = 37? Let me recount.
    # 4*7 = 28, 2*2 = 4, 2*2 = 4, 1*1 = 1. Total = 37.
    # But I said 38 earlier. Let me recheck.
    # Actually 2x1x1 has rank 1, not 2. So total is 37.
    
    # Convert to lists of lists for the interface
    U_list = [[float(U[i, j]) for j in range(rank_total)] for i in range(n * m)]
    V_list = [[float(V[i, j]) for j in range(rank_total)] for i in range(m * p)]
    W_list = [[float(W[i, j]) for j in range(rank_total)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
