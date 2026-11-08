"""Rewritten program for matrix multiplication 2x4 x 4x4 with rank-minimal search."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

# Deadline guard: evaluator kills after 150s, we use 120s limit
DEADLINE = 120.0
START_TIME = time.time()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def construct():
    """
    Construct a low-rank decomposition for (2x4) * (4x4) matrix multiplication.
    
    Strategy: Use Strassen's 2x2 block algorithm on the top-left 2x2 block of B (which is 4x4),
    combined with direct scalar multiplications for the remaining columns.
    
    The 4x4 matrix B can be partitioned as:
    B = [[B11, B12],
         [B21, B22]]
    where each Bi is 2x2.
    
    A is 2x4, partitioned as:
    A = [A1, A2]
    where A1 and A2 are 2x2.
    
    Result C = A * B = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    
    For the 2x2 * 2x2 blocks, we can use Strassen's algorithm (7 multiplications instead of 8).
    
    However, let's think more carefully. We have:
    C_left = A1 * B11 + A2 * B21  (2x2)
    C_right = A1 * B12 + A2 * B22  (2x2)
    
    Each of these is a sum of two 2x2*2x2 products.
    
    Using Strassen for each 2x2*2x2: 7 mults each, total 14 mults.
    But wait, we can potentially share some computations?
    
    Actually, let's just use the straightforward approach first:
    - 8 scalar mults for A1*B11
    - 8 scalar mults for A2*B21
    - 8 scalar mults for A1*B12
    - 8 scalar mults for A2*B22
    Total: 32 mults (schoolbook)
    
    With Strassen on each 2x2*2x2:
    - 7 mults for A1*B11
    - 7 mults for A2*B21
    - 7 mults for A1*B12
    - 7 mults for A2*B22
    Total: 28 mults
    
    But we can do better by recognizing structure. Let me think about the tensor rank.
    
    Actually, the known lower bound for 2x4 * 4x4 is not well-known, but the schoolbook is 32.
    Strassen-based gives 28. Let's aim for a rank <= 28 decomposition.
    
    Let's construct a rank-28 decomposition using Strassen's 7-mult algorithm for each of the four 2x2*2x2 block products.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    
    # We'll build U, V, W matrices.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (16, rank)
    # W has shape (p*n, rank) = (8, rank)
    
    # Indexing:
    # A is 2x4: A[i, j] where i in {0,1}, j in {0,1,2,3}
    # B is 4x4: B[j, k] where j in {0,1,2,3}, k in {0,1,2,3}
    # C is 2x4: C[i, k] where i in {0,1}, k in {0,1,2,3}
    #
    # U index: i*m + j = i*4 + j
    # V index: j*p + k = j*4 + k
    # W index: k*n + i = k*2 + i (trace convention W[k*n+i, r])
    
    # Partition A into A1 = A[:, 0:2], A2 = A[:, 2:4]
    # Partition B into B11 = B[0:2, 0:2], B12 = B[0:2, 2:4], B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # C[:, 0:2] = A1 * B11 + A2 * B21
    # C[:, 2:4] = A1 * B12 + A2 * B22
    
    # For each 2x2 * 2x2 product, use Strassen with 7 multiplications.
    
    def strassen_2x2_mults():
        """
        Return the 7 Strassen intermediate expressions for X*Y where X,Y are 2x2.
        X = [[x00, x01], [x10, x11]]
        Y = [[y00, y01], [y10, y11]]
        
        M1 = (x00+x11)*(y00+y11)
        M2 = (x10+x11)*y00
        M3 = x00*(y01-y11)
        M4 = x11*(y10-y00)
        M5 = (x00+x01)*y11
        M6 = (x10-x00)*(y00+y10)
        M7 = (x01-x11)*(y01+y11)
        
        Result:
        Z00 = M1 + M4 - M5 + M7
        Z01 = M3 + M5
        Z10 = M2 + M4
        Z11 = M1 - M2 + M3 + M6
        """
        return 7  # number of multiplications
    
    # Build the decomposition term by term.
    # Each term r corresponds to one "multiplication" in the tensor decomposition.
    # U[row_u, r] * V[row_v, r] * W[row_w, r] contributes to C.
    
    # We'll collect (u_idx, v_idx, w_idx, coeff) tuples.
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1.0):
        terms.append((u_idx, v_idx, w_idx, coeff))
    
    # Helper: Strassen for product of X (2x2) and Y (2x2)
    # X indices in A: (i, j) where i in {0,1}, j depends on which block
    # Y indices in B: (j, k) where j,k depend on which block
    
    def add_strassen_block(A_base_j, B_base_j, B_base_k, C_base_k):
        """
        Add Strassen decomposition for A_block * B_block where:
        A_block is A[:, A_base_j:A_base_j+2] (2x2)
        B_block is B[B_base_j:B_base_j+2, B_base_k:B_base_k+2] (2x2)
        C_block is C[:, C_base_k:C_base_k+2] (2x2)
        
        A indices: i in {0,1}, j in {A_base_j, A_base_j+1}
        B indices: j in {B_base_j, B_base_j+1}, k in {B_base_k, B_base_k+1}
        C indices: i in {0,1}, k in {C_base_k, C_base_k+1}
        """
        # X = A_block, Y = B_block
        # x00 = A[0, A_base_j], x01 = A[0, A_base_j+1]
        # x10 = A[1, A_base_j], x11 = A[1, A_base_j+1]
        # y00 = B[B_base_j, B_base_k], y01 = B[B_base_j, B_base_k+1]
        # y10 = B[B_base_j+1, B_base_k], y11 = B[B_base_j+1, B_base_k+1]
        
        # M1 = (x00+x11)*(y00+y11)
        # Contribution to Z00 and Z11
        # Z00 += M1, Z11 += M1
        # x00+x11: U rows (0*4+A_base_j, 1*4+A_base_j+1) with coeff 1
        # y00+y11: V rows (B_base_j*4+B_base_k, (B_base_j+1)*4+(B_base_k+1)) with coeff 1
        # Z00: W row C_base_k*2+0, Z11: W row (C_base_k+1)*2+1
        
        # We need to add two terms for M1 (one for Z00, one for Z11)? No, M1 is one multiplication.
        # In tensor decomposition, each multiplication is one term.
        # M1 contributes to both Z00 and Z11, so W row for Z00 gets +1 and W row for Z11 gets +1? 
        # No, in the tensor decomposition, each term is a rank-1 component.
        # A single multiplication M1 = (x00+x11)*(y00+y11) contributes to multiple output entries.
        # So we have ONE term with U having entries for x00 and x11, V having entries for y00 and y11,
        # and W having entries for the output positions that M1 contributes to.
        
        # For M1:
        # U: row 0*4+A_base_j -> 1, row 1*4+A_base_j+1 -> 1
        # V: row B_base_j*4+B_base_k -> 1, row (B_base_j+1)*4+(B_base_k+1) -> 1
        # W: row C_base_k*2+0 -> 1 (Z00), row (C_base_k+1)*2+1 -> 1 (Z11)
        
        u_idx_1 = 0 * 4 + A_base_j
        u_idx_2 = 1 * 4 + (A_base_j + 1)
        v_idx_1 = B_base_j * 4 + B_base_k
        v_idx_2 = (B_base_j + 1) * 4 + (B_base_k + 1)
        w_idx_1 = C_base_k * 2 + 0  # Z00
        w_idx_2 = (C_base_k + 1) * 2 + 1  # Z11
        
        # Term M1
        add_term(u_idx_1, v_idx_1, w_idx_1, 1.0)
        # But this only captures one part. We need the full rank-1 tensor for M1.
        # Actually, in the tensor decomposition, each term is a product of three vectors.
        # For M1 = (x00+x11)*(y00+y11), the U vector has 1 at positions x00 and x11,
        # V vector has 1 at positions y00 and y11, and W vector has 1 at positions Z00 and Z11.
        # So it's ONE term, not multiple.
        
        # Let me restart the term collection with proper rank-1 tensors.
        pass
    
    # Let me rebuild this more carefully.
    # A rank-1 term is defined by three vectors u, v, w such that
    # T[i,j,k] += u[i*m+j] * v[j*p+k] * w[k*n+i]
    # where T is the multiplication tensor.
    
    # For Strassen M1 = (x00+x11)*(y00+y11):
    # u has 1 at (i=0, j=A_base_j) and (i=1, j=A_base_j+1)
    # v has 1 at (j=B_base_j, k=B_base_k) and (j=B_base_j+1, k=B_base_k+1)
    # w has 1 at (k=C_base_k, i=0) and (k=C_base_k+1, i=1)
    
    # This is one rank-1 term. Let me collect these properly.
    
    terms.clear()
    
    def add_rank1_term(u_positions, v_positions, w_positions, coeff=1.0):
        """
        u_positions: list of (u_idx, coeff_u)
        v_positions: list of (v_idx, coeff_v)  
        w_positions: list of (w_idx, coeff_w)
        This adds one rank-1 term where U[u_idx] = coeff_u*coeff, etc.
        """
        for u_idx, cu in u_positions:
            for v_idx, cv in v_positions:
                for w_idx, cw in w_positions:
                    terms.append((u_idx, v_idx, w_idx, coeff * cu * cv * cw))
    
    # Actually, that's not right either. A rank-1 term is a single outer product.
    # U vector: sparse, with values at certain indices
    # V vector: sparse
    # W vector: sparse
    # The term contributes U[i]*V[j]*W[k] to T[i,j,k]
    
    # So for one rank-1 term, I specify the full U, V, W vectors (sparse).
    # Let me use a different approach: collect (u_dict, v_dict, w_dict) for each term.
    
    rank1_terms = []
    
    def add_rank1(u_dict, v_dict, w_dict):
        rank1_terms.append((u_dict, v_dict, w_dict))
    
    # Strassen for 2x2 * 2x2:
    def add_strassen(A_j0, B_j0, B_k0, C_k0):
        """
        A_block = A[:, A_j0:A_j0+2], B_block = B[B_j0:B_j0+2, B_k0:B_k0+2]
        C_block = C[:, C_k0:C_k0+2]
        """
        # x00 = A[0, A_j0], x01 = A[0, A_j0+1]
        # x10 = A[1, A_j0], x11 = A[1, A_j0+1]
        # y00 = B[B_j0, B_k0], y01 = B[B_j0, B_k0+1]
        # y10 = B[B_j0+1, B_k0], y11 = B[B_j0+1, B_k0+1]
        
        # M1 = (x00+x11)*(y00+y11)
        u1 = {0*4+A_j0: 1, 1*4+(A_j0+1): 1}
        v1 = {B_j0*4+B_k0: 1, (B_j0+1)*4+(B_k0+1): 1}
        w1 = {C_k0*2+0: 1, (C_k0+1)*2+1: 1}
        add_rank1(u1, v1, w1)
        
        # M2 = (x10+x11)*y00
        u2 = {1*4+A_j0: 1, 1*4+(A_j0+1): 1}
        v2 = {B_j0*4+B_k0: 1}
        w2 = {(C_k0+1)*2+0: 1}  # Z10
        add_rank1(u2, v2, w2)
        
        # M3 = x00*(y01-y11)
        u3 = {0*4+A_j0: 1}
        v3 = {B_j0*4+(B_k0+1): 1, (B_j0+1)*4+B_k0: -1}
        w3 = {C_k0*2+1: 1, (C_k0+1)*2+1: 1}  # Z01 and Z11
        add_rank1(u3, v3, w3)
        
        # M4 = x11*(y10-y00)
        u4 = {1*4+(A_j0+1): 1}
        v4 = {(B_j0+1)*4+B_k0: 1, B_j0*4+B_k0: -1}
        w4 = {C_k0*2+0: 1, (C_k0+1)*2+0: 1}  # Z00 and Z10
        add_rank1(u4, v4, w4)
        
        # M5 = (x00+x01)*y11
        u5 = {0*4+A_j0: 1, 0*4+(A_j0+1): 1}
        v5 = {(B_j0+1)*4+(B_k0+1): 1}
        w5 = {C_k0*2+0: -1, (C_k0+1)*2+1: 1}  # -Z00, +Z11
        add_rank1(u5, v5, w5)
        
        # M6 = (x10-x00)*(y00+y10)
        u6 = {1*4+A_j0: 1, 0*4+A_j0: -1}
        v6 = {B_j0*4+B_k0: 1, (B_j0+1)*4+B_k0: 1}
        w6 = {(C_k0+1)*2+1: 1}  # Z11
        add_rank1(u6, v6, w6)
        
        # M7 = (x01-x11)*(y01+y11)
        u7 = {0*4+(A_j0+1): 1, 1*4+(A_j0+1): -1}
        v7 = {B_j0*4+(B_k0+1): 1, (B_j0+1)*4+(B_k0+1): 1}
        w7 = {C_k0*2+0: 1, C_k0*2+1: 1}  # Z00 and Z01
        add_rank1(u7, v7, w7)
    
    # C[:, 0:2] = A1*B11 + A2*B21
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B11 = B[0:2, 0:2], B21 = B[2:4, 0:2]
    add_strassen(A_j0=0, B_j0=0, B_k0=0, C_k0=0)  # A1*B11
    add_strassen(A_j0=2, B_j0=2, B_k0=0, C_k0=0)  # A2*B21
    
    # C[:, 2:4] = A1*B12 + A2*B22
    # B12 = B[0:2, 2:4], B22 = B[2:4, 2:4]
    add_strassen(A_j0=0, B_j0=0, B_k0=2, C_k0=2)  # A1*B12
    add_strassen(A_j0=2, B_j0=2, B_k0=2, C_k0=2)  # A2*B22
    
    # Now convert rank1_terms to U, V, W matrices.
    rank = len(rank1_terms)
    n_rows_u = n * m  # 8
    n_rows_v = m * p  # 16
    n_rows_w = p * n  # 8
    
    U = np.zeros((n_rows_u, rank))
    V = np.zeros((n_rows_v, rank))
    W = np.zeros((n_rows_w, rank))
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank1_terms):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
