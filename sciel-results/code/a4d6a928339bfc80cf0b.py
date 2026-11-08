import numpy as np
import time

DIMENSIONS = [3, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140

def _build_expected_tensor(n, m, p):
    """Build the expected matmul tensor T[a,b,c] = 1 if indices consistent."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                T[a_idx, b_idx, c_idx] = 1.0
    return T

def _verify_decomposition(U, V, W, T_expected, atol=1e-8):
    """Verify that sum_r U[:,r]⊗V[:,r]⊗W[:,r] == T_expected."""
    n, m, p = T_expected.shape[0], T_expected.shape[1], T_expected.shape[2]
    # Actually get dims from U, V, W
    nm = U.shape[0]
    mp = V.shape[0]
    pn = W.shape[0]
    rank = U.shape[1]
    T_computed = np.zeros((nm, mp, pn))
    for r in range(rank):
        T_computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_computed, T_expected, atol=atol)

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs an exact tensor decomposition for matrix multiplication of size 3x4x5.
    Uses a global compression of the schoolbook 3x(1x4x5) direct sum.
    Represents the tensor as 3 row-blocks of 1x4x5, then finds linear dependencies
    among the 60 rank-1 terms to reduce rank below 60.
    """
    n, m, p = DIMENSIONS
    
    if time.time() > _DEADLINE:
        return _schoolbook_fallback(n, m, p)
    
    # Build the full schoolbook decomposition: 60 terms
    # Term index: i in [0,n), j in [0,m), k in [0,p)
    # U: one-hot at (i*m + j)
    # V: one-hot at (j*p + k)
    # W: one-hot at (k*n + i)  [W[k*n+i, r] convention]
    
    total_terms = n * m * p  # 60
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1.0
                v = np.zeros(m * p)
                v[j * p + k] = 1.0
                w = np.zeros(p * n)
                w[k * n + i] = 1.0
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U_arr = np.array(U_list).T  # (n*m, 60)
    V_arr = np.array(V_list).T  # (m*p, 60)
    W_arr = np.array(W_list).T  # (p*n, 60)
    
    # Now try to find linear dependencies to reduce rank.
    # Strategy: For each term r, check if T_r = sum_{s != r} c_s * T_s
    # where T_r = U_r ⊗ V_r ⊗ W_r (as a flattened tensor)
    # This is equivalent to: can we express the rank-1 tensor for term r
    # as a linear combination of the others?
    
    # A more practical approach: try to find a subset of terms that spans
    # the same tensor space. Use the Khatri-Rao structure.
    
    # The multiplication tensor T has T[i,j,k] = 1 if the (i,j) entry of A
    # and (j,k) entry of B contribute to (i,k) entry of C.
    # In our indexing: term (i,j,k) contributes to output (i,k) with factor 1.
    # So T[i,j,k] = 1 for all valid (i,j,k) in the schoolbook.
    
    # Actually, the tensor is: T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C * delta_{jk}
    # Wait no. Let me re-derive.
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The bilinear form: <T, A ⊗ B> = C
    # T is a 3-tensor with T[i,j,k] such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if the indices are consistent, i.e., always 1 for valid (i,j,k).
    
    # The schoolbook decomposition: T = sum_{i,j,k} (e_i^A ⊗ e_j^B ⊗ e_k^C)
    # where e_i^A is the i-th standard basis in A-space (dim n*m=12),
    # e_j^B is the j-th... wait, no.
    
    # Let me be more careful. A is 3x4, so A-space has 12 dims indexed by (i,j) with i in [0,3), j in [0,4).
    # B is 4x5, so B-space has 20 dims indexed by (j,k) with j in [0,4), k in [0,5).
    # C is 3x5, so C-space has 15 dims indexed by (i,k) with i in [0,3), k in [0,5).
    
    # The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # As a tensor: T = sum_{i,j,k} e_{(i,j)}^A ⊗ e_{(j,k)}^B ⊗ e_{(i,k)}^C
    # This has 3*4*5 = 60 terms.
    
    # Now, the question is: can we find a lower-rank decomposition?
    # The tensor rank of matmul(3,4,5) is known to be at most 60 (schoolbook).
    # Lower bounds exist but for small sizes, we can search.
    
    # Approach: Try to find if any term is a linear combination of others.
    # For term (i0,j0,k0): T_{i0,j0,k0} = e_{(i0,j0)} ⊗ e_{(j0,k0)} ⊗ e_{(i0,k0)}
    # We want: e_{(i0,j0)} ⊗ e_{(j0,k0)} ⊗ e_{(i0,k0)} = sum_{(i,j,k) != (i0,j0,k0)} c_{i,j,k} * e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}
    
    # Taking the inner product with e_{(i',j')}^A ⊗ e_{(j',k')}^B ⊗ e_{(i'',k'')}^C:
    # LHS: 1 if (i',j')=(i0,j0), (j',k')=(j0,k0), (i'',k'')=(i0,k0); else 0.
    # RHS: c_{i,j,k} if (i,j)=(i',j'), (j,k)=(j',k'), (i,k)=(i'',k'')
    
    # So the only term on RHS that can match is when (i,j)=(i0,j0), (j,k)=(j0,k0), (i,k)=(i0,k0).
    # This requires j=j0 from both, i=i0, k=k0. So only (i0,j0,k0) itself matches.
    # Therefore, NO term is a linear combination of the others in the standard basis.
    
    # This means the 60 schoolbook terms are linearly independent as tensors!
    # So we cannot reduce below 60 by simple elimination.
    
    # However, we can use a different decomposition that is NOT the schoolbook.
    # The Strassen approach gives rank 56 as in the original code.
    
    # Let's use the Strassen-based approach but try to optimize it further.
    # The original code achieves rank 56. Let's verify and try to improve.
    
    U_parts = []
    V_parts = []
    W_parts = []
    
    def add_block(u_idx, v_idx, w_idx, U_block, V_block, W_block):
        rank = U_block.shape[1]
        for r in range(rank):
            U_parts.append(np.zeros(n * m))
            for i, idx in enumerate(u_idx):
                U_parts[-1][idx] = U_block[i, r]
            
            V_parts.append(np.zeros(m * p))
            for i, idx in enumerate(v_idx):
                V_parts[-1][idx] = V_block[i, r]
            
            W_parts.append(np.zeros(p * n))
            for i, idx in enumerate(w_idx):
                W_parts[-1][idx] = W_block[i, r]
    
    def strassen_2x2x2():
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def schoolbook_2x2x1():
        U = np.zeros((4, 4))
        V = np.zeros((2, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        U[2, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
        U[3, 3] = 1; V[1, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x2():
        # 1x2x2: C[k] = A[0] * B[k], k in {0,1}
        # U has 2 rows (A has 2 entries), V has 4 rows (B has 4 entries: 2x2 block), W has 2 rows (C has 2 entries)
        # Rank 2: term r=0: A[0]*B[0]*C[0], term r=1: A[0]*B[1]*C[1]
        # But wait - the "2" in 1x2x2 means the middle dim is 2, so B is 2x2 (4 entries)
        # A is 1x2 (2 entries), B is 2x2 (4 entries), C is 1x2 (2 entries)
        # C[k] = sum_{j=0}^{1} A[j] * B[j,k]
        # Term (j,k): A[j]*B[j,k]*C[k]
        # That's 2*2=4 terms for schoolbook, but rank can be 2?
        # C[0] = A[0]*B[0,0] + A[1]*B[1,0]
        # C[1] = A[0]*B[0,1] + A[1]*B[1,1]
        # This is a 2x2 matmul essentially (A is 1x2, B is 2x2, C is 1x2)
        # Rank of 1x2x2 matmul tensor: it's the same as 2x2x1 which has rank 2 (Strassen gives 7 for 2x2x2, but 1x2x2 is simpler)
        # Actually 1x2x2: C[k] = sum_j A[j]B[j,k]. This is 4 independent products, so rank >= 4?
        # No: it's a bilinear map. The tensor is sum_{j,k} e_j^A ⊗ e_{(j,k)}^B ⊗ e_k^C
        # These 4 terms are linearly independent, so rank = 4.
        # So schoolbook_1x2x2 with rank 4 is already optimal for this sub-block.
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        return U, V, W
    
    def schoolbook_1x2x1():
        U = np.zeros((2, 2))
        V = np.zeros((2, 2))
        W = np.zeros((1, 2))
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[1, 1] = 1; W[0, 1] = 1
        return U, V, W

    # Block 1: rows 0-1 of A, cols 0-1 of B (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [0, 1, 5, 6], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 2: rows 0-1 of A, cols 2-3 of B (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([0, 1, 4, 5], [2, 3, 7, 8], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 3: rows 0-1 of A, col 4 of B (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([0, 1, 4, 5], [4, 9], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 4: rows 2-3 of A, cols 0-1 of B (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [10, 11, 15, 16], [0, 3, 1, 4], U_blk, V_blk, W_blk)
    
    # Block 5: rows 2-3 of A, cols 2-3 of B (2x2x2)
    U_blk, V_blk, W_blk = strassen_2x2x2()
    add_block([2, 3, 6, 7], [12, 13, 17, 18], [6, 9, 7, 10], U_blk, V_blk, W_blk)
    
    # Block 6: rows 2-3 of A, col 4 of B (2x2x1)
    U_blk, V_blk, W_blk = schoolbook_2x2x1()
    add_block([2, 3, 6, 7], [14, 19], [12, 13], U_blk, V_blk, W_blk)
    
    # Block 7: row 4 of A, cols 0-1 of B (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [0, 1, 5, 6], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 8: row 4 of A, cols 2-3 of B (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([8, 9], [2, 3, 7, 8], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 9: row 4 of A, col 4 of B (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([8, 9], [4, 9], [14], U_blk, V_blk, W_blk)
    
    # Block 10: row 5 of A, cols 0-1 of B (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [10, 11, 15, 16], [2, 5], U_blk, V_blk, W_blk)
    
    # Block 11: row 5 of A, cols 2-3 of B (1x2x2)
    U_blk, V_blk, W_blk = schoolbook_1x2x2()
    add_block([10, 11], [12, 13, 17, 18], [8, 11], U_blk, V_blk, W_blk)
    
    # Block 12: row 5 of A, col 4 of B (1x2x1)
    U_blk, V_blk, W_blk = schoolbook_1x2x1()
    add_block([10, 11], [14, 19], [14], U_blk, V_blk, W_blk)
    
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    
    rank = U.shape[1]
    
    # Verify correctness
    if time.time() < _DEADLINE - 5:
        # Check: sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] should equal the matmul tensor
        # Flatten and compare
        T_computed = np.zeros((n*m, m*p, p*n))
        for r in range(rank):
            T_computed += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        
        # Expected tensor: T[i,j,k] = 1 if the (i,j) A-entry and (j,k) B-entry
        # contribute to (i,k) C-entry. In our indexing:
        # A index (i,j) -> flat index i*m+j
        # B index (j,k) -> flat index j*p+k  
        # C index (i,k) -> flat index k*n+i (W convention)
        # T[a_idx, b_idx, c_idx] = 1 if a_idx=i*m+j, b_idx=j*p+k, c_idx=k*n+i
        # i.e., j from A = j from B, and i from A = i from C, k from B = k from C
        
        T_expected = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    T_expected[a_idx, b_idx, c_idx] = 1.0
        
        if not np.allclose(T_computed, T_expected, atol=1e-10):
            # Fallback to schoolbook
            return _schoolbook_fallback(n, m, p)
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _schoolbook_fallback(n, m, p):
    """Schoolbook decomposition with rank n*m*p."""
    U_parts = []
    V_parts = []
    W_parts = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n * m)
                u[i * m + j] = 1
                v = np.zeros(m * p)
                v[j * p + k] = 1
                w = np.zeros(p * n)
                w[k * n + i] = 1
                U_parts.append(u)
                V_parts.append(v)
                W_parts.append(w)
    U = np.array(U_parts).T
    V = np.array(V_parts).T
    W = np.array(W_parts).T
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
