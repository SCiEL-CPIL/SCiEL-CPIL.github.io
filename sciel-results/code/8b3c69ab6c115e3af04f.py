"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from collections import defaultdict

DIMENSIONS = [4, 4, 5]

def _schoolbook(n, m, p):
    """Return (U, V, W) as numpy arrays for schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _strassen_2x2x2():
    """Return the 7 Strassen products for 2x2x2 matrix multiplication.
    Each product is (a_mask, b_mask, w_terms) where:
    - a_mask: list of (i,j) indices for A submatrix (with +1/-1 coefficients)
    - b_mask: list of (j,k) indices for B submatrix (with +1/-1 coefficients)
    - w_terms: list of (k,i,coeff) for the output contributions
    """
    # Strassen's 7 products for C = A * B where A is 2x2, B is 2x2, C is 2x2
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C00 = M1 - M4 + M5 + M6
    # C01 = M2 + M4
    # C10 = M3 + M5
    # C11 = M1 - M2 + M3 + M7
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10+B11)
    # M4 = (A00+A01)B11
    # M5 = (A10+A11)(B01-B11)
    # M6 = (A00-A10)(B00+B10)
    # M7 = (A01-A11)(B10+B11)
    
    products = [
        # M1
        {
            'a': [(0,0,1.0), (1,1,1.0)],
            'b': [(0,0,1.0), (1,1,1.0)],
            'w': [(0,0,1.0), (0,1,-1.0), (1,0,1.0), (1,1,1.0)]
        },
        # M2
        {
            'a': [(1,0,1.0), (1,1,1.0)],
            'b': [(0,0,1.0), (0,1,0.0)],  # B00 only, but we need B00 which is (0,0) in local
            'w': [(0,1,1.0), (1,0,0.0), (1,1,-1.0)]
        },
        # M3
        {
            'a': [(0,0,1.0)],
            'b': [(1,0,1.0), (1,1,1.0)],
            'w': [(1,0,1.0), (1,1,1.0)]
        },
        # M4
        {
            'a': [(0,0,1.0), (0,1,1.0)],
            'b': [(1,1,1.0)],
            'w': [(0,0,-1.0), (1,1,1.0)]
        },
        # M5
        {
            'a': [(1,0,1.0), (1,1,1.0)],
            'b': [(0,1,1.0), (1,1,-1.0)],
            'w': [(0,0,1.0), (1,0,1.0)]
        },
        # M6
        {
            'a': [(0,0,1.0), (1,0,-1.0)],
            'b': [(0,0,1.0), (1,0,1.0)],
            'w': [(0,0,1.0)]
        },
        # M7
        {
            'a': [(0,1,1.0), (1,1,-1.0)],
            'b': [(1,0,1.0), (1,1,1.0)],
            'w': [(1,1,1.0)]
        },
    ]
    return products

def _build_block_decomposition(n, m, p, deadline=None):
    """Build a block decomposition using Strassen where possible.
    
    For 4x4x5:
    - Split A into 2x2 blocks of 2x2: A11, A12, A21, A22 (each 2x2)
    - Split B into blocks: B11, B12, B21, B22 where B11 is 2x2, B12 is 2x3, etc.
    - Actually for 4x4 times 4x5:
      A is 4x4, B is 4x5
      Split A into 2x2 blocks: A11(2x2), A12(2x2), A21(2x2), A22(2x2)
      Split B into 2x2 and 2x3 blocks based on row split:
        B_top (2x5), B_bottom (2x5)
      Then C = A * B is 4x5, split into 2x2 blocks:
        C_top(2x5), C_bottom(2x5)
    
    Let me think more carefully.
    
    A (4x4) = [[A11, A12], [A21, A22]] where each is 2x2
    B (4x5) = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x3, B21 is 2x2, B22 is 2x3
    
    C (4x5) = [[C11, C12], [C21, C22]]
    C11 (2x2) = A11*B11 + A12*B21
    C12 (2x3) = A11*B12 + A12*B22
    C21 (2x2) = A21*B11 + A22*B21
    C22 (2x3) = A21*B12 + A22*B22
    
    For the 2x2x2 products (C11 and C21), we can use Strassen (rank 7 each).
    For the 2x2x3 products (C12 and C22), we need a decomposition.
    
    For 2x2x3: schoolbook rank is 12. Can we do better?
    Actually, for 2x2x3, the known minimal rank is 9 (Landsberg).
    
    So total rank = 7 + 7 + 9 + 9 = 32? Wait, that seems too good.
    
    Actually wait. The blocks don't share factors in a simple way. Let me reconsider.
    
    The full tensor is a sum of block tensors. Each block tensor (e.g., the C11 part)
    is itself a tensor that can be decomposed. The total decomposition is the union
    of the decompositions of each block.
    
    So:
    - C11 = A11*B11 + A12*B21: This is a 2x2x2 + 2x2x2 structure.
      Actually, this is the product of a 2x4 matrix [A11 A12] with a 4x2 matrix [[B11],[B21]].
      So it's a 2x4x2 matrix multiplication.
      Schoolbook rank for 2x4x2 is 2*4*2 = 16.
      But we can split: [A11 A12] * [[B11],[B21]] = A11*B11 + A12*B21
      Each term is 2x2x2 with rank 8 schoolbook, or 7 with Strassen.
      So C11 block has rank at most 7+7 = 14.
    
    Similarly for C21: rank at most 14.
    C12 = [A11 A12] * [[B12],[B22]] where B12 is 2x3, B22 is 2x3.
      This is 2x4x3. Schoolbook rank 2*4*3 = 24.
      Split: A11*B12 + A12*B22, each 2x2x3 with rank 12 schoolbook or 9 with optimal.
      So C12 block has rank at most 9+9 = 18.
    C22 similarly: rank at most 18.
    
    Total: 14 + 14 + 18 + 18 = 64. That matches the knowledge!
    """
    n, m, p = DIMENSIONS  # 4, 4, 5
    
    # We'll build the decomposition as a list of (U_col, V_col, W_col) tuples
    # where each is a vector (with sparse representation)
    
    # For efficiency, let's use the block approach
    # The idea: enumerate all the "atomic" products and combine them.
    
    # Let's define the structure more carefully.
    # A is 4x4, B is 4x5, C is 4x5 (output C[k,i] = sum_j A[i,j] * B[j,k])
    
    # Block structure:
    # A rows: 0,1 (top), 2,3 (bottom)
    # A cols: 0,1 (left), 2,3 (right)
    # B rows: 0,1 (top), 2,3 (bottom)
    # B cols: 0,1,2 (left, 3 cols), 3,4 (right, 2 cols)
    
    # Wait, p=5, so B cols are 0,1,2,3,4.
    # Let's split B cols into 2+3 or 3+2.
    # For the 2x2x2 blocks, we need B to have 2 cols in the relevant subblock.
    # For 2x2x3, B has 3 cols.
    
    # Let's split:
    # A: 2x2 blocks (2x2 each)
    # B: rows split 2+2, cols split 2+3
    #   B_top_left: 2x2, B_top_right: 2x3
    #   B_bot_left: 2x2, B_bot_right: 2x3
    
    # C: 2x2 blocks in rows, cols same as B
    #   C_top_left: 2x2 = A11*B11 + A12*B21
    #   C_top_right: 2x3 = A11*B12 + A12*B22
    #   C_bot_left: 2x2 = A21*B11 + A22*B21
    #   C_bot_right: 2x3 = A21*B12 + A22*B22
    
    # For 2x2x2 products, use Strassen (rank 7)
    # For 2x2x3 products, use optimal (rank 9)
    
    # Let's build each block's decomposition.
    
    all_U = []  # list of dicts: {row_index: coeff}
    all_V = []
    all_W = []
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term."""
        all_U.append(u_dict)
        all_V.append(v_dict)
        all_W.append(w_dict)
    
    # Helper to create sparse dict from list of (index, coeff)
    def make_sparse(pairs):
        d = {}
        for idx, coeff in pairs:
            if idx in d:
                d[idx] += coeff
            else:
                d[idx] = coeff
        # Remove zeros
        return {k: v for k, v in d.items() if v != 0}
    
    # Strassen 2x2x2: returns list of (a_sparse, b_sparse, w_sparse)
    # where a_sparse is for the 2x2 A block (local indices 0-3 mapped to global)
    def strassen_2x2x2(a_offset_i, a_offset_j, b_offset_j, b_offset_k):
        """Generate rank-1 terms for 2x2x2 Strassen.
        a_offset_i: global row offset for A block
        a_offset_j: global col offset for A block (also B row offset)
        b_offset_k: global col offset for B block
        """
        terms = []
        # Local to global mappings
        # A local (i,j) -> global (a_offset_i + i, a_offset_j + j)
        # B local (j,k) -> global (a_offset_j + j, b_offset_k + k)
        # C local (k,i) -> global (b_offset_k + k, a_offset_i + i)
        
        # Strassen's 7 products
        # M1 = (A00+A11)(B00+B11)
        # A part: local (0,0)=1, (1,1)=1
        # B part: local (0,0)=1, (1,1)=1
        # C contributions:
        #   C00 += 1, C01 -= 1, C10 += 1, C11 += 1
        
        M1_a = [(0,0,1.0), (1,1,1.0)]
        M1_b = [(0,0,1.0), (1,1,1.0)]
        M1_w = [(0,0,1.0), (0,1,-1.0), (1,0,1.0), (1,1,1.0)]
        
        # M2 = (A10+A11)B00
        # A part: local (1,0)=1, (1,1)=1
        # B part: local (0,0)=1
        # C contributions:
        #   C01 += 1, C11 -= 1
        
        M2_a = [(1,0,1.0), (1,1,1.0)]
        M2_b = [(0,0,1.0)]
        M2_w = [(0,1,1.0), (1,1,-1.0)]
        
        # M3 = A00(B10+B11)
        # A part: local (0,0)=1
        # B part: local (1,0)=1, (1,1)=1
        # C contributions:
        #   C10 += 1, C11 += 1
        
        M3_a = [(0,0,1.0)]
        M3_b = [(1,0,1.0), (1,1,1.0)]
        M3_w = [(1,0,1.0), (1,1,1.0)]
        
        # M4 = (A00+A01)B11
        # A part: local (0,0)=1, (0,1)=1
        # B part: local (1,1)=1
        # C contributions:
        #   C00 -= 1, C11 += 1
        
        M4_a = [(0,0,1.0), (0,1,1.0)]
        M4_b = [(1,1,1.0)]
        M4_w = [(0,0,-1.0), (1,1,1.0)]
        
        # M5 = (A10+A11)(B01-B11)
        # A part: local (1,0)=1, (1,1)=1
        # B part: local (0,1)=1, (1,1)=-1
        # C contributions:
        #   C00 += 1, C10 += 1
        
        M5_a = [(1,0,1.0), (1,1,1.0)]
        M5_b = [(0,1,1.0), (1,1,-1.0)]
        M5_w = [(0,0,1.0), (1,0,1.0)]
        
        # M6 = (A00-A10)(B00+B10)
        # A part: local (0,0)=1, (1,0)=-1
        # B part: local (0,0)=1, (1,0)=1
        # C contributions:
        #   C00 += 1
        
        M6_a = [(0,0,1.0), (1,0,-1.0)]
        M6_b = [(0,0,1.0), (1,0,1.0)]
        M6_w = [(0,0,1.0)]
        
        # M7 = (A01-A11)(B10+B11)
        # A part: local (0,1)=1, (1,1)=-1
        # B part: local (1,0)=1, (1,1)=1
        # C contributions:
        #   C11 += 1
        
        M7_a = [(0,1,1.0), (1,1,-1.0)]
        M7_b = [(1,0,1.0), (1,1,1.0)]
        M7_w = [(1,1,1.0)]
        
        products = [
            (M1_a, M1_b, M1_w),
            (M2_a, M2_b, M2_w),
            (M3_a, M3_b, M3_w),
            (M4_a, M4_b, M4_w),
            (M5_a, M5_b, M5_w),
            (M6_a, M6_b, M6_w),
            (M7_a, M7_b, M7_w),
        ]
        
        for a_local, b_local, w_local in products:
            # Convert to global indices
            a_global = make_sparse([(a_offset_i + i, a_offset_j + j, c) for i, j, c in a_local])
            b_global = make_sparse([(a_offset_j + j, b_offset_k + k, c) for j, k, c in b_local])
            w_global = make_sparse([(b_offset_k + k, a_offset_i + i, c) for k, i, c in w_local])
            terms.append((a_global, b_global, w_global))
        
        return terms
    
    # For 2x2x3, we need a rank-9 decomposition.
    # This is more complex. Let's use a known decomposition.
    # Actually, for simplicity and correctness, let's use schoolbook for 2x2x3 (rank 12)
    # and see if we can improve later.
    # Schoolbook for 2x2x3: rank = 2*2*3 = 12
    
    def schoolbook_2x2x3(a_offset_i, a_offset_j, b_offset_j, b_offset_k):
        """Generate rank-1 terms for 2x2x3 schoolbook."""
        terms = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    a_global = {(a_offset_i + i, a_offset_j + j): 1.0}
                    b_global = {(a_offset_j + j, b_offset_k + k): 1.0}
                    w_global = {(b_offset_k + k, a_offset_i + i): 1.0}
                    terms.append((a_global, b_global, w_global))
        return terms
    
    # Now build the full decomposition:
    
    # C11 (2x2) = A11*B11 + A12*B21
    # A11: rows 0-1, cols 0-1. offset_i=0, offset_j=0
    # B11: rows 0-1, cols 0-1. offset_j=0, offset_k=0
    # A12: rows 0-1, cols 2-3. offset_i=0, offset_j=2
    # B21: rows 2-3, cols 0-1. offset_j=2, offset_k=0
    
    terms_c11 = strassen_2x2x2(0, 0, 0, 0) + strassen_2x2x2(0, 2, 2, 0)
    for a, b, w in terms_c11:
        add_term(a, b, w)
    
    # C21 (2x2) = A21*B11 + A22*B21
    # A21: rows 2-3, cols 0-1. offset_i=2, offset_j=0
    # B11: rows 0-1, cols 0-1. offset_j=0, offset_k=0
    # A22: rows 2-3, cols 2-3. offset_i=2, offset_j=2
    # B21: rows 2-3, cols 0-1. offset_j=2, offset_k=0
    
    terms_c21 = strassen_2x2x2(2, 0, 0, 0) + strassen_2x2x2(2, 2, 2, 0)
    for a, b, w in terms_c21:
        add_term(a, b, w)
    
    # C12 (2x3) = A11*B12 + A12*B22
    # A11: rows 0-1, cols 0-1. offset_i=0, offset_j=0
    # B12: rows 0-1, cols 2-4. offset_j=0, offset_k=2
    # A12: rows 0-1, cols 2-3. offset_i=0, offset_j=2
    # B22: rows 2-3, cols 2-4. offset_j=2, offset_k=2
    
    terms_c12 = schoolbook_2x2x3(0, 0, 0, 2) + schoolbook_2x2x3(0, 2, 2, 2)
    for a, b, w in terms_c12:
        add_term(a, b, w)
    
    # C22 (2x3) = A21*B12 + A22*B22
    # A21: rows 2-3, cols 0-1. offset_i=2, offset_j=0
    # B12: rows 0-1, cols 2-4. offset_j=0, offset_k=2
    # A22: rows 2-3, cols 2-3. offset_i=2, offset_j=2
    # B22: rows 2-3, cols 2-4. offset_j=2, offset_k=2
    
    terms_c22 = schoolbook_2x2x3(2, 0, 0, 2) + schoolbook_2x2x3(2, 2, 2, 2)
    for a, b, w in terms_c22:
        add_term(a, b, w)
    
    # Now convert to dense matrices
    n, m, p = DIMENSIONS
    rank = len(all_U)
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for col in range(rank):
        for idx, coeff in all_U[col].items():
            i, j = idx
            U[i * m + j, col] = coeff
        for idx, coeff in all_V[col].items():
            j, k = idx
            V[j * p + k, col] = coeff
        for idx, coeff in all_W[col].items():
            k, i = idx
            W[k * n + i, col] = coeff
    
    return U, V, W, rank

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    # Check that sum_r U[:,r] * V[:,r] * W[:,r]^T = T where T is the matrix multiplication tensor
    # T[i,j,k] = delta_{i,j} * delta_{j,k} ... no
    # Actually T[i,j,k] = 1 if the indices form a valid product term
    # For C = A * B, C[k,i] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all i,j,k (it's the standard matrix multiplication tensor)
    
    # Let's verify by checking a few random inputs
    import random
    random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        
        # Compute C from decomposition
        C_computed = np.zeros((p, n))
        for r in range(U.shape[1]):
            u_vec = U[:, r]  # (n*m,)
            v_vec = V[:, r]  # (m*p,)
            w_vec = W[:, r]  # (p*n,)
            
            # Reshape u_vec to (n, m), v_vec to (m, p)
            u_mat = u_vec.reshape(n, m)
            v_mat = v_vec.reshape(m, p)
            
            # Product of u_mat and v_mat gives an (n, p) matrix
            prod = u_mat @ v_mat  # (n, p)
            
            # Add to C_computed
            # w_vec is (p*n,), index k*n+i
            for k in range(p):
                for i in range(n):
                    C_computed[k, i] += prod[i, k] * w_vec[k * n + i]
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    return True

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Try the block decomposition with Strassen for 2x2x2 and schoolbook for 2x2x3
    try:
        U, V, W, rank = _build_block_decomposition(n, m, p, deadline)
        
        # Verify correctness
        if _verify(U, V, W, n, m, p):
            # Convert to lists for the return format
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
        else:
            # Fall back to schoolbook
            U, V, W = _schoolbook(n, m, p)
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    except Exception as e:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
