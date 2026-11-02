"""
Rank-minimal exact tensor decomposition for 4x4 * 4x6 matrix multiplication.

Strategy:
  1. Build a rank-7 Winograd decomposition for 2x2 * 2x2 (the 7-term bilinear algorithm).
  2. Apply it to the 4x4 * 4x4 sub-block (split A and B into 2x2 blocks).
     This gives rank 7*4 = 28 for the 4x4 * 4x4 part.
  3. For the remaining 4x4 * 4x2 part, use a rank-16 schoolbook decomposition
     (each of the 4x2 outputs needs 4 terms, but we can share: actually 4*4*2/2... 
     let's just do schoolbook for 4x4*4x2 which is 4*4*2 = 32... no wait.
     4x4 * 4x2: C[i,k] = sum_j A[i,j] B[j,k] for i in 0..3, k in 0..1.
     That's 4*2 = 8 outputs, each a sum of 4 products. Schoolbook rank = 4*4*2 = 32.
     
     Hmm, but we can also try to be smarter. Actually let me reconsider the whole approach.

  Alternative structural approach:
  - The tensor for 4x4*4x6 has dimensions (4*4, 4*6, 6*4) = (16, 24, 24) in the factor matrices.
  - The known best for 4x4*4x4 is 49 (Strassen recursive).
  - For 4x4*4x6, we can think of it as 4x4*4x4 + 4x4*4x2.
  - If we use rank 49 for the 4x4*4x4 part and rank 32 for the 4x4*4x2 part, total = 81.
  - But we can do better by not separating them so rigidly.

  Let me try a different structural approach:
  - Use the fact that 4x4*4x6 can be embedded in a larger problem.
  - Actually, let me just try to construct a concrete lower-rank decomposition.

  Key insight for a STRUCTURAL change:
  Instead of trying to hand-craft a decomposition, use a systematic approach:
  1. Start with the schoolbook (rank 96).
  2. Use linear algebra to find dependencies among the rank-1 tensors.
  3. Specifically, the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k
     (in the appropriate indexing). We want to write T as a sum of rank-1 tensors.
     
  Actually, the most promising structural approach for this specific size:
  - Use Winograd for 2x2*2x2 (rank 7) applied to the 4x4*4x4 subproblem.
  - For the 4x4*4x2 part, note that we can split it into two 4x4*4x1 problems.
  - A 4x4*4x1 matrix-vector product has rank at most 16 (schoolbook).
  - But we can potentially share structure.

  Let me just build the Winograd-based 4x4*4x4 (rank 49... actually 7*4=28 if we use
  Winograd directly on 2x2 blocks, but that's not quite right either).

  Actually, the cleanest structural change: use a KNOWN decomposition.
  
  For 2x2*2x2, Winograd gives rank 7. The 7 terms are:
    m1 = (a00+a11)(b00+b11)
    m2 = (a10+a11)b00
    m3 = a00(b10-b11)
    m4 = a11(b01-b00)
    m5 = (a00+a01)b11
    m6 = (a10-a01)(b00+b01)
    m7 = (a01+a11)(b10+b11)
  
  Then:
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6

  For 4x4*4x4, split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]]  (each 2x2)
    B = [[B00, B01], [B10, B11]]  (each 2x2)
    C = [[C00, C01], [C10, C11]]  (each 2x2)
    
    C00 = A00*B00 + A01*B10  -> 2 Winograd multiplications: 14 terms
    C01 = A00*B01 + A01*B11  -> 2 Winograd multiplications: 14 terms
    C10 = A10*B00 + A11*B10  -> 2 Winograd multiplications: 14 terms
    C11 = A10*B01 + A11*B11  -> 2 Winograd multiplications: 14 terms
    
    Total: 56 terms for 4x4*4x4.

  For 4x4*4x2:
    A is 4x4, B is 4x2. Split A into 2x2 blocks, B into 2x1 blocks.
    B = [[B00, B01], [B10, B11]] where B00,B10 are 2x1, B01,B11 are 2x1.
    C = [[C00, C01], [C10, C11]] where each Ci is 2x1.
    
    C00 = A00*B00 + A01*B10  (2x2 * 2x1 + 2x2 * 2x1)
    C01 = A00*B01 + A01*B11
    C10 = A10*B00 + A11*B10
    C11 = A10*B01 + A11*B11
    
    For 2x2 * 2x1: C[i,0] = a[i,0]*b[0,0] + a[i,1]*b[1,0]
    Schoolbook: 4 terms. Can we do better? Probably not for such small size.
    So each 2x2*2x1 needs rank 4, and we have 8 such multiplications: 32 terms.
    
    Total: 56 + 32 = 88. That's worse than 96? No, 88 < 96. Good.

  Wait, I can do better on the 2x2*2x1 part. Actually no, 2x2*2x1 is just a
  matrix-vector product of a 2x2 matrix with a 2-vector. The tensor is:
  T[i,j,k] where i in {0,1}, j in {0,1}, k in {0} (only one output column).
  T[i,j,0] = 1 for all i,j. Rank is 4 (schoolbook) and it's known to be 4.

  So total rank = 56 + 32 = 88.

  Let me verify this is correct and build it.
"""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-88 decomposition for 4x4 * 4x6 matrix multiplication.
    Uses Winograd (rank 7) for 2x2*2x2 subproblems and schoolbook for 2x2*2x1.
    """
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # We'll build the decomposition in terms of (i, j, k) indices.
    # U has shape (n*m, rank) = (16, rank), indexed by [i*m + j]
    # V has shape (m*p, rank) = (24, rank), indexed by [j*p + k]
    # W has shape (p*n, rank) = (24, rank), indexed by [k*n + i]
    
    # We'll accumulate contributions from each sub-problem.
    # Each sub-problem contributes a set of rank-1 tensors.
    
    # Start with empty lists for U, V, W entries.
    # Each rank-1 tensor is a triple (u, v, w) where u is a 16-vector,
    # v is a 24-vector, w is a 24-vector.
    
    # Actually, let's build U, V, W as lists of lists, adding columns as we go.
    
    U_entries = []  # list of (index, value) pairs for U
    V_entries = []  # list of (index, value) pairs for V
    W_entries = []  # list of (index, value) pairs for W
    
    rank = 0
    
    def add_tensor(u_dict, v_dict, w_dict):
        """Add a rank-1 tensor with sparse entries."""
        nonlocal rank
        col = rank
        for idx, val in u_dict.items():
            U_entries.append((idx, val, col))
        for idx, val in v_dict.items():
            V_entries.append((idx, val, col))
        for idx, val in w_dict.items():
            W_entries.append((idx, val, col))
        rank += 1
    
    # Winograd 2x2*2x2: 7 terms
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # 
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a01)(b00+b01)
    # m7 = (a01+a11)(b10+b11)
    #
    # c00 = m1 + m4 - m5 + m7
    # c01 = m3 + m5
    # c10 = m2 + m4
    # c11 = m1 - m2 + m3 + m6
    #
    # In terms of the tensor factors:
    # For each m_t, we have a rank-1 tensor where:
    # U factors pick out the A entries, V factors pick out the B entries,
    # W factors pick out the C entries (with appropriate signs).
    
    def winograd_2x2(a_block, b_block, c_block, sign=1):
        """
        Add 7 rank-1 tensors for a 2x2*2x2 multiplication.
        a_block: dict mapping (i,j) to the global U index for A[i,j]
        b_block: dict mapping (j,k) to the global V index for B[j,k]
        c_block: dict mapping (i,k) to the global W index for C[i,k]
        sign: global sign multiplier
        """
        # m1 = (a00+a11)(b00+b11)
        u = {a_block[(0,0)]: sign, a_block[(1,1)]: sign}
        v = {b_block[(0,0)]: sign, b_block[(1,1)]: sign}
        w = {c_block[(0,0)]: sign, c_block[(1,1)]: sign}
        add_tensor(u, v, w)
        
        # m2 = (a10+a11)b00
        u = {a_block[(1,0)]: sign, a_block[(1,1)]: sign}
        v = {b_block[(0,0)]: sign}
        w = {c_block[(1,0)]: sign, c_block[(1,1)]: -sign}
        add_tensor(u, v, w)
        
        # m3 = a00(b10-b11)
        u = {a_block[(0,0)]: sign}
        v = {b_block[(1,0)]: sign, b_block[(1,1)]: -sign}
        w = {c_block[(0,1)]: sign, c_block[(1,1)]: sign}
        add_tensor(u, v, w)
        
        # m4 = a11(b01-b00)
        u = {a_block[(1,1)]: sign}
        v = {b_block[(0,1)]: sign, b_block[(0,0)]: -sign}
        w = {c_block[(0,0)]: sign, c_block[(1,0)]: sign}
        add_tensor(u, v, w)
        
        # m5 = (a00+a01)b11
        u = {a_block[(0,0)]: sign, a_block[(0,1)]: sign}
        v = {b_block[(1,1)]: sign}
        w = {c_block[(0,0)]: -sign, c_block[(0,1)]: sign}
        add_tensor(u, v, w)
        
        # m6 = (a10-a01)(b00+b01)
        u = {a_block[(1,0)]: sign, a_block[(0,1)]: -sign}
        v = {b_block[(0,0)]: sign, b_block[(0,1)]: sign}
        w = {c_block[(1,1)]: sign}
        add_tensor(u, v, w)
        
        # m7 = (a01+a11)(b10+b11)
        u = {a_block[(0,1)]: sign, a_block[(1,1)]: sign}
        v = {b_block[(1,0)]: sign, b_block[(1,1)]: sign}
        w = {c_block[(0,0)]: sign}
        add_tensor(u, v, w)
    
    # Schoolbook 2x2*2x1: 4 terms
    # A is 2x2, B is 2x1 (column vector), C is 2x1
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    # 
    # m1 = a00*b0 -> c0 += m1
    # m2 = a01*b1 -> c0 += m2
    # m3 = a10*b0 -> c1 += m3
    # m4 = a11*b1 -> c1 += m4
    
    def schoolbook_2x2_2x1(a_block, b_block, c_block, sign=1):
        """
        Add 4 rank-1 tensors for a 2x2*2x1 multiplication.
        a_block: dict mapping (i,j) to the global U index for A[i,j]
        b_block: dict mapping (j,0) to the global V index for B[j,0]
        c_block: dict mapping (i,0) to the global W index for C[i,0]
        """
        # m1 = a00*b0
        u = {a_block[(0,0)]: sign}
        v = {b_block[(0,0)]: sign}
        w = {c_block[(0,0)]: sign}
        add_tensor(u, v, w)
        
        # m2 = a01*b1
        u = {a_block[(0,1)]: sign}
        v = {b_block[(1,0)]: sign}
        w = {c_block[(0,0)]: sign}
        add_tensor(u, v, w)
        
        # m3 = a10*b0
        u = {a_block[(1,0)]: sign}
        v = {b_block[(0,0)]: sign}
        w = {c_block[(1,0)]: sign}
        add_tensor(u, v, w)
        
        # m4 = a11*b1
        u = {a_block[(1,1)]: sign}
        v = {b_block[(1,0)]: sign}
        w = {c_block[(1,0)]: sign}
        add_tensor(u, v, w)
    
    # Now define the global indexing:
    # A is 4x4: A[i,j] -> U index i*4 + j
    # B is 4x6: B[j,k] -> V index j*6 + k
    # C is 4x6: C[i,k] -> W index k*4 + i  (trace convention: W[k*n+i])
    
    # For the 4x4*4x4 part:
    # A blocks: A00 = A[0:2,0:2], A01 = A[0:2,2:4], A10 = A[2:4,0:2], A11 = A[2:4,2:4]
    # B blocks: B00 = B[0:2,0:2], B01 = B[0:2,2:4], B10 = B[2:4,0:2], B11 = B[2:4,2:4]
    # C blocks: C00 = C[0:2,0:2], C01 = C[0:2,2:4], C10 = C[2:4,0:2], C11 = C[2:4,2:4]
    
    # For the 4x4*4x2 part:
    # A blocks: same as above
    # B blocks: B00 = B[0:2,4:6], B01 = B[0:2,4:6]... wait, B is 4x6, columns 4,5
    # B is 4x6: rows 0-3, cols 0-5.
    # For 4x4*4x2, we use B[:, 4:6] which is 4x2.
    # B00 = B[0:2, 4:6] (2x2), B10 = B[2:4, 4:6] (2x2)
    # Wait, B is 4x2 in this subproblem: rows 0-3, cols 4-5.
    # Split into 2x2 blocks: B_top = B[0:2, 4:6], B_bot = B[2:4, 4:6]
    # C is 4x2: C[:, 4:6]. Split into 2x2 blocks: C00 = C[0:2, 4:6], C10 = C[2:4, 4:6]
    
    # Let me redefine more carefully.
    
    # Global indices:
    # U[i*4+j] for A[i,j], i in 0..3, j in 0..3
    # V[j*6+k] for B[j,k], j in 0..3, k in 0..5
    # W[k*4+i] for C[i,k], i in 0..3, k in 0..5
    
    # 4x4*4x4 part (k in 0..3):
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] for k in 0..3
    
    # Split into 2x2 blocks:
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4]
    # C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]
    
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each term is a 2x2*2x2 multiplication, using Winograd (rank 7).
    # 8 such multiplications * 7 = 56 terms.
    
    # Define helper to get global indices for 2x2 blocks
    
    def a_idx(i, j):
        return i * 4 + j
    
    def b_idx(j, k):
        return j * 6 + k
    
    def w_idx(i, k):
        return k * 4 + i
    
    # 4x4*4x4 part
    # C00 = A00*B00 + A01*B10
    a_block_00 = {(0,0): a_idx(0,0), (0,1): a_idx(0,1), (1,0): a_idx(1,0), (1,1): a_idx(1,1)}
    b_block_00 = {(0,0): b_idx(0,0), (0,1): b_idx(0,1), (1,0): b_idx(1,0), (1,1): b_idx(1,1)}
    c_block_00 = {(0,0): w_idx(0,0), (0,1): w_idx(0,1), (1,0): w_idx(1,0), (1,1): w_idx(1,1)}
    
    a_block_01 = {(0,0): a_idx(0,2), (0,1): a_idx(0,3), (1,0): a_idx(1,2), (1,1): a_idx(1,3)}
    b_block_10 = {(0,0): b_idx(2,0), (0,1): b_idx(2,1), (1,0): b_idx(3,0), (1,1): b_idx(3,1)}
    
    winograd_2x2(a_block_00, b_block_00, c_block_00)
    winograd_2x2(a_block_01, b_block_10, c_block_00)
    
    # C01 = A00*B01 + A01*B11
    b_block_01 = {(0,0): b_idx(0,2), (0,1): b_idx(0,3), (1,0): b_idx(1,2), (1,1): b_idx(1,3)}
    c_block_01 = {(0,0): w_idx(0,2), (0,1): w_idx(0,3), (1,0): w_idx(1,2), (1,1): w_idx(1,3)}
    b_block_11 = {(0,0): b_idx(2,2), (0,1): b_idx(2,3), (1,0): b_idx(3,2), (1,1): b_idx(3,3)}
    
    winograd_2x2(a_block_00, b_block_01, c_block_01)
    winograd_2x2(a_block_01, b_block_11, c_block_01)
    
    # C10 = A10*B00 + A11*B10
    a_block_10 = {(0,0): a_idx(2,0), (0,1): a_idx(2,1), (1,0): a_idx(3,0), (1,1): a_idx(3,1)}
    a_block_11 = {(0,0): a_idx(2,2), (0,1): a_idx(2,3), (1,0): a_idx(3,2), (1,1): a_idx(3,3)}
    c_block_10 = {(0,0): w_idx(2,0), (0,1): w_idx(2,1), (1,0): w_idx(3,0), (1,1): w_idx(3,1)}
    
    winograd_2x2(a_block_10, b_block_00, c_block_10)
    winograd_2x2(a_block_11, b_block_10, c_block_10)
    
    # C11 = A10*B01 + A11*B11
    c_block_11 = {(0,0): w_idx(2,2), (0,1): w_idx(2,3), (1,0): w_idx(3,2), (1,1): w_idx(3,3)}
    
    winograd_2x2(a_block_10, b_block_01, c_block_11)
    winograd_2x2(a_block_11, b_block_11, c_block_11)
    
    # 4x4*4x2 part (k in 4..5):
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k] for k in 4..5
    # B columns 4,5: B[j,4], B[j,5]
    # Split A into 2x2 blocks as before.
    # B_top = B[0:2, 4:6] (2x2), B_bot = B[2:4, 4:6] (2x2)
    # C_top = C[0:2, 4:6] (2x2), C_bot = C[2:4, 4:6] (2x2)
    #
    # C_top = A00*B_top + A01*B_bot
    # C_bot = A10*B_top + A11*B_bot
    #
    # Each is a sum of two 2x2*2x2 multiplications. Use Winograd for each.
    # 4 such multiplications * 7 = 28 terms.
    
    b_block_top = {(0,0): b_idx(0,4), (0,1): b_idx(0,5), (1,0): b_idx(1,4), (1,1): b_idx(1,5)}
    b_block_bot = {(0,0): b_idx(2,4), (0,1): b_idx(2,5), (1,0): b_idx(3,4), (1,1): b_idx(3,5)}
    c_block_top = {(0,0): w_idx(0,4), (0,1): w_idx(0,5), (1,0): w_idx(1,4), (1,1): w_idx(1,5)}
    c_block_bot = {(0,0): w_idx(2,4), (0,1): w_idx(2,5), (1,0): w_idx(3,4), (1,1): w_idx(3,5)}
    
    # C_top = A00*B_top + A01*B_bot
    winograd_2x2(a_block_00, b_block_top, c_block_top)
    winograd_2x2(a_block_01, b_block_bot, c_block_top)
    
    # C_bot = A10*B_top + A11*B_bot
    winograd_2x2(a_block_10, b_block_top, c_block_bot)
    winograd_2x2(a_block_11, b_block_bot, c_block_bot)
    
    # Total rank = 8 * 7 + 4 * 7 = 56 + 28 = 84
    
    # Now build the U, V, W matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for idx, val, col in U_entries:
        U[idx][col] = val
    for idx, val, col in V_entries:
        V[idx][col] = val
    for idx, val, col in W_entries:
        W[idx][col] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
