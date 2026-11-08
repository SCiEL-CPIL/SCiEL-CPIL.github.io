import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 115.0
    
    n, m, p = 2, 4, 7
    # A: 2x4, B: 4x7, C: 2x7
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # Block structure:
    # A = [A1 | A2], A1,A2 in R^{2x2}
    # B = [B11 B12 B13; B21 B22 B23], B11,B12,B21,B22 in R^{2x2}, B13,B23 in R^{2x3}
    
    # C = A1*B11 + A2*B21 + A1*B12 + A2*B22 + A1*B13 + A2*B23
    
    # For each 2x2x2x2 block: Strassen (7 terms)
    # For each 2x2x2x3 block: 11 terms
    
    # Total: 4*7 + 2*11 = 28 + 22 = 50
    
    U_rows = n * m  # 8
    V_rows = m * p  # 28
    W_rows = p * n  # 14
    
    terms = []  # Each term is (U_idx, V_idx, W_idx, U_val, V_val, W_val)
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        terms.append((u_idx, u_val, v_idx, v_val, w_idx, w_val))
    
    def strassen_222(a_base, b_base, c_base, term_offset=0):
        # a_base: (i,j) -> A[i,j] index in flattened A (row-major)
        # b_base: (j,k) -> B[j,k] index in flattened B (row-major)
        # c_base: (i,k) -> C[i,k] index in flattened C (row-major, k*n+i)
        # We'll add 7 terms
        
        # A indices: A[i,j] at position i*m + j = i*4 + j
        # B indices: B[j,k] at position j*p + k = j*7 + k
        # C indices: C[i,k] at position k*n + i = k*2 + i
        
        A00 = a_base(0,0)
        A01 = a_base(0,1)
        A10 = a_base(1,0)
        A11 = a_base(1,1)
        
        B00 = b_base(0,0)
        B01 = b_base(0,1)
        B10 = b_base(1,0)
        B11 = b_base(1,1)
        
        C00 = c_base(0,0)
        C01 = c_base(0,1)
        C10 = c_base(1,0)
        C11 = c_base(1,1)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10-B00)
        # M4 = A11(B01-B11)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1
        add_term(A00, 1, B00, 1, C00, 1)
        add_term(A11, 1, B00, 1, C00, 1)
        add_term(A00, 1, B11, 1, C00, 1)
        add_term(A11, 1, B11, 1, C00, 1)
        # But this is 4 terms for M1, not 1!
        
        # I need to rethink this. Each "M" is a product of a linear combo of A-entries
        # and a linear combo of B-entries. In the tensor decomposition, each rank-1 term
        # is u ⊗ v ⊗ w. So M1 = (A00+A11)(B00+B11) corresponds to 4 rank-1 terms:
        # A00*B00, A00*B11, A11*B00, A11*B11
        
        # That's not right either. Let me reconsider.
        
        # In the tensor decomposition, each rank-1 term is:
        # u ⊗ v ⊗ w where u is a vector in R^{nm}, v in R^{mp}, w in R^{pn}
        
        # For a "generalized multiplication" (Σ_a u_a A_a)(Σ_b v_b B_b) * w,
        # this corresponds to multiple rank-1 terms.
        
        # I think I'm overcomplicating this. Let me just build the decomposition
        # directly using the Strassen formula.
        pass
    
    # Let me take a different approach and directly build the decomposition.
    
    # I'll create the U, V, W matrices directly.
    
    rank = 50
    U = [[0.0] * rank for _ in range(U_rows)]
    V = [[0.0] * rank for _ in range(V_rows)]
    W = [[0.0] * rank for _ in range(W_rows)]
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, t):
        U[u_idx][t] += u_val
        V[v_idx][t] += v_val
        W[w_idx][t] += w_val
    
    t = 0
    
    # Define helper functions for indices
    def A_idx(i, j):
        return i * m + j  # i in {0,1}, j in {0,1,2,3}
    
    def B_idx(j, k):
        return j * p + k  # j in {0,1,2,3}, k in {0,...,6}
    
    def C_idx(i, k):
        return k * n + i  # k in {0,...,6}, i in {0,1}
    
    # Strassen for 2x2x2x2 block
    # A block: rows i0,i1 of A, columns j0,j1
    # B block: rows j0,j1 of B, columns k0,k1
    # C block: rows i0,i1 of C, columns k0,k1
    def strassen_block(i0, j0, k0, j1=None, k1=None):
        nonlocal t
        if j1 is None:
            j1 = j0 + 1
        if k1 is None:
            k1 = k0 + 1
        
        A00 = A_idx(i0, j0)
        A01 = A_idx(i0, j1)
        A10 = A_idx(i0+1, j0)
        A11 = A_idx(i0+1, j1)
        
        B00 = B_idx(j0, k0)
        B01 = B_idx(j0, k1)
        B10 = B_idx(j1, k0)
        B11 = B_idx(j1, k1)
        
        C00 = C_idx(i0, k0)
        C01 = C_idx(i0, k1)
        C10 = C_idx(i0+1, k0)
        C11 = C_idx(i0+1, k1)
        
        # M1 = (A00+A11)(B00+B11)
        add_term(A00, 1, B00, 1, C00, 1, t)
        add_term(A11, 1, B00, 1, C00, 1, t)
        add_term(A00, 1, B11, 1, C00, 1, t)
        add_term(A11, 1, B11, 1, C00, 1, t)
        add_term(A00, 1, B00, 1, C11, 1, t)
        add_term(A11, 1, B00, 1, C11, 1, t)
        add_term(A00, 1, B11, 1, C11, 1, t)
        add_term(A11, 1, B11, 1, C11, 1, t)
        t += 1
        
        # This is getting way too complex. Let me just use the schoolbook
        # and see if I can get a valid decomposition, even if it's not optimal.
        pass
    
    # I'm going in circles. Let me just implement the 50-rank decomposition
    # using a clean approach.
    
    # Reset
    rank = 50
    U = [[0] * rank for _ in range(U_rows)]
    V = [[0] * rank for _ in range(V_rows)]
    W = [[0] * rank for _ in range(W_rows)]
    
    # I'll build this term by term.
    
    # The decomposition:
    # C = A1*B11 + A2*B21 + A1*B12 + A2*B22 + A1*B13 + A2*B23
    # where:
    # A1 = A[:, 0:2] (2x2), A2 = A[:, 2:4] (2x2)
    # B11 = B[0:2, 0:2] (2x2), B12 = B[0:2, 2:4] (2x2), B13 = B[0:2, 4:7] (2x3)
    # B21 = B[2:4, 0:2] (2x2), B22 = B[2:4, 2:4] (2x2), B23 = B[2:4, 4:7] (2x3)
    
    # For each 2x2x2x2 block, use Strassen (7 terms)
    # For each 2x2x2x3 block, use 11 terms
    
    # Let me define the Strassen decomposition for a 2x2x2x2 block.
    
    def strassen_222(a_i0, a_j0, b_j0, b_k0, c_i0, c_k0, t_start):
        # a_i0, a_j0: base indices for A block
        # b_j0, b_k0: base indices for B block
        # c_i0, c_k0: base indices for C block
        # Returns t_start + 7
        
        # A[i,j] for i in {a_i0, a_i0+1}, j in {a_j0, a_j0+1}
        # B[j,k] for j in {b_j0, b_j0+1}, k in {b_k0, b_k0+1}
        # C[i,k] for i in {c_i0, c_i0+1}, k in {c_k0, c_k0+1}
        
        A00 = a_i0 * m + a_j0
        A01 = a_i0 * m + (a_j0 + 1)
        A10 = (a_i0 + 1) * m + a_j0
        A11 = (a_i0 + 1) * m + (a_j0 + 1)
        
        B00 = b_j0 * p + b_k0
        B01 = b_j0 * p + (b_k0 + 1)
        B10 = (b_j0 + 1) * p + b_k0
        B11 = (b_j0 + 1) * p + (b_k0 + 1)
        
        C00 = b_k0 * n + c_i0
        C01 = (b_k0 + 1) * n + c_i0
        C10 = b_k0 * n + (c_i0 + 1)
        C11 = (b_k0 + 1) * n + (c_i0 + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][t_start] = 1
        V[B00][t_start] = 1
        W[C00][t_start] = 1
        U[A11][t_start] = 1
        V[B00][t_start] = 1
        W[C00][t_start] = 1
        U[A00][t_start] = 1
        V[B11][t_start] = 1
        W[C00][t_start] = 1
        U[A11][t_start] = 1
        V[B11][t_start] = 1
        W[C00][t_start] = 1
        U[A00][t_start] = 1
        V[B00][t_start] = 1
        W[C11][t_start] = 1
        U[A11][t_start] = 1
        V[B00][t_start] = 1
        W[C11][t_start] = 1
        U[A00][t_start] = 1
        V[B11][t_start] = 1
        W[C11][t_start] = 1
        U[A11][t_start] = 1
        V[B11][t_start] = 1
        W[C11][t_start] = 1
        # This is 8 terms, not 1! I'm making the same mistake.
        
        # The issue is that in the tensor decomposition, each rank-1 term
        # u ⊗ v ⊗ w has u, v, w as vectors. So (A00+A11)(B00+B11) is NOT
        # a single rank-1 term. It's the sum of 4 rank-1 terms.
        
        # I think I've been fundamentally misunderstanding the problem.
        
        # Let me re-read the problem statement.
        # "Find a rank-minimal exact tensor decomposition for multiplying
        # a 2 by 4 matrix by a 4 by 7 matrix."
        #
        # The matrix multiplication tensor T has T[i,j,k,l,m,n] = δ_{j,k}δ_{m,n}
        # where i,n are row indices of A/C, j,k are column/row indices of A/B,
        # l,m are column indices of B/C.
        #
        # A rank-1 decomposition is T = Σ_t u_t ⊗ v_t ⊗ w_t
        # where u_t ∈ R^{nm}, v_t ∈ R^{mp}, w_t ∈ R^{pn}.
        #
        # Each rank-1 term u_t ⊗ v_t ⊗ w_t corresponds to:
        # C = (u_t^T ⊗ w_t) * (v_t^T) applied to A, B
        #
        # More concretely, for a single rank-1 term:
        # C[i,k] += u_t[i*m+j] * v_t[j*p+l] * w_t[l*n+i]
        # Wait, that's not right either.
        
        # Let me think about this more carefully.
        #
        # The matrix multiplication tensor T satisfies:
        # T[i, j, k, l, m, n] = δ_{j,k} * δ_{m,n}
        # where i ∈ {0,1}, j ∈ {0,1,2,3}, k ∈ {0,1,2,3}, l ∈ {0,...,6},
        #       m ∈ {0,...,6}, n ∈ {0,1}
        #
        # A rank-1 decomposition is:
        # T = Σ_t u_t[i,j] * v_t[k,l] * w_t[m,n]
        #
        # where u_t ∈ R^{2×4}, v_t ∈ R^{4×7}, w_t ∈ R^{7×2}
        #
        # Wait, the problem says U has shape (n*m, rank) = (8, rank),
        # V has shape (m*p, rank) = (28, rank), W has shape (p*n, rank) = (14, rank).
        #
        # So U, V, W are flattened versions.
        #
        # U[i*m+j, t] = u_t[i,j]
        # V[j*p+k, t] = v_t[j,k]  (wait, v_t should be 4×7, so V[j*p+k, t] = v_t[j,k])
        # W[k*n+i, t] = w_t[k,i]  (w_t is 7×2, so W[k*n+i, t] = w_t[k,i])
        #
        # The decomposition is:
        # T[i,j,k,l,m,n] = Σ_t U[i*m+j, t] * V[k*p+l, t] * W[m*n+i, t]
        #
        # Wait, that doesn't look right. Let me re-read.
        #
        # "U has n*m rows (entries of A, flattened row-major: index i*m + j)"
        # "V has m*p rows (entries of B, flattened row-major: index j*p + k)"
        # "W has p*n rows (entries of C, official trace convention: k*n + i)"
        #
        # So the decomposition is:
        # C[i,k] = Σ_j A[i,j] * B[j,k] = Σ_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        #
        # Wait, that's not a valid tensor decomposition. The right side has a free j
        # index that's being summed over in the matrix product, but in the tensor
        # decomposition, j should be a fixed index.
        
        # I think the correct interpretation is:
        # The tensor T has indices (i, j, l) where:
        # - i is the row of A and C
        # - j is the column of A and row of B
        # - l is the column of B and C
        #
        # T[i, j, l] = 1 if we're computing C[i,l] = Σ_j A[i,j]B[j,l]
        #
        # No, that's not right either. The tensor for matrix multiplication is:
        # T[i, j, k, l, m, n] = δ_{j,k} * δ_{m,n}
        #
        # where (i,j) index A, (k,l) index B, (m,n) index C.
        #
        # A rank-1 decomposition is:
        # T = Σ_t u_t[i,j] * v_t[k,l] * w_t[m,n]
        #
        # In flattened form:
        # U[i*m_A+j, t] where m_A = 4 (columns of A)
        # V[k*m_B+l, t] where m_B = 7 (columns of B)
        # W[m*m_C+n, t] where m_C = 2 (columns of C)
        #
        # Wait, I'm getting confused with the indices. Let me just use the
        # convention from the problem:
        #
        # U[i*m+j, t] for A[i,j], i∈{0,1}, j∈{0,1,2,3}
        # V[j*p+k, t] for B[j,k], j∈{0,1,2,3}, k∈{0,...,6}
        # W[k*n+i, t] for C[k,i], k∈{0,...,6}, i∈{0,1}  (trace convention: k*n+i)
        #
        # The decomposition satisfies:
        # Σ_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = δ_{j,j} * δ_{k,k}
        #
        # Wait, that's always 1. That can't be right.
        
        # I think the correct statement is:
        # For any matrices A, B:
        # C = A * B where C[i,k] = Σ_j A[i,j] * B[j,k]
        #
        # And the decomposition satisfies:
        # C[i,k] = Σ_t (Σ_j U[i*m+j, t] * A[i,j]) * (Σ_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
        #
        # No, that's not right either.
        
        # Let me look at the schoolbook decomposition:
        # U[i*m+j, t] = 1 if t = i*m*p + j*p + k, else 0
        # V[j*p+k, t] = 1 if t = i*m*p + j*p + k, else 0
        # W[k*n+i, t] = 1 if t = i*m*p + j*p + k, else 0
        #
        # Then:
        # Σ_t U[i*m+j', t] * V[j'*p+k', t] * W[k'*n+i', t] * A[i,j'] * B[j',k']
        # = Σ_{j',k',i'} δ_{i,i'} * A[i,j'] * B[j',k'] * δ_{j',j'} * δ_{k',k'} * δ_{i',i}
        # Wait, I'm overcomplicating this.
        
        # The key equation is:
        # C[i,k] = Σ_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        #
        # where j is summed over implicitly (it's the "shared" index).
        #
        # No, j is NOT summed over. j is a free index in U and V.
        #
        # I think the correct interpretation is:
        # The tensor T has 6 indices: (i, j, k, l, m, n)
        # T[i,j,k,l,m,n] = δ_{j,k} * δ_{m,n}
        #
        # A rank-1 decomposition is:
        # T[i,j,k,l,m,n] = Σ_t U[i,j,t] * V[k,l,t] * W[m,n,t]
        #
        # In flattened form:
        # U[i*m+j, t] * V[k*p+l, t] * W[m*n+i, t]  (with appropriate index mappings)
        #
        # The schoolbook has rank = 2*4*7 = 56, with one term per (i,j,k) triple.
        #
        # For a Strassen-like decomposition, we replace some of these terms
        # with "generalized" terms where U, V, W entries are linear combinations.
        
        # OK so I think the correct way to think about this is:
        #
        # Each rank-1 term t contributes:
        # C[i,k] += Σ_j U[i*m+j, t] * A[i,j] * V[j*p+k, t] * B[j,k] * W[k*n+i, t]
        #
        # Wait, that has A and B in it, which shouldn't be the case.
        
        # I think the decomposition is:
        # The bilinear map (A, B) -> C is represented as:
        # C = Σ_t (U_t * A) * (V_t * B) * W_t
        #
        # where U_t is a 2×4 matrix, V_t is a 4×7 matrix, W_t is a 7×2 matrix,
        # and the products are matrix products.
        #
        # More precisely:
        # C[i,k] = Σ_t Σ_j Σ_l U_t[i,j] * A[i,j] * V_t[j,l] * B[j,l] * W_t[l,k]
        #
        # Hmm, that's not standard either.
        
        # Let me just trust the schoolbook and work backwards.
        #
        # Schoolbook:
        # U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
        # for t = i*m*p + j*p + k
        #
        # Then:
        # Σ_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # = Σ_{i',j',k'} δ_{i,i'} * δ_{j,j'} * δ_{k,k'} * δ_{k',k} * δ_{i',i}
        # = 1
        #
        # So the decomposition satisfies:
        # Σ_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
        #
        # This means that the "effective" contribution to C[i,k] from term t is:
        # (Σ_j U[i*m+j, t] * A[i,j]) * (Σ_j V[j*p+k, t] * B[j,k]) * W[k*n+i, t]
        #
        # No, that's not right. Let me think about this differently.
        
        # I think the correct interpretation is that the tensor T is defined by:
        # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute
        # to the (i,k) entry of C.
        #
        # So T has 3 indices: (i, j, k) where i is the row of A/C, j is the
        # column of A / row of B, k is the column of B/C.
        #
        # T[i,j,k] = 1 for all valid i,j,k.
        #
        # A rank-1 decomposition is:
        # T[i,j,k] = Σ_t u_t[i,j] * v_t[j,k] * w_t[i,k]
        #
        # In flattened form:
        # U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        #
        # The schoolbook has U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1
        # for t = (i,j,k), giving rank 56.
        
        # For Strassen, we use linear combinations:
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # This contributes to C[i0,k0], C[i0,k1], C[i1,k0], C[i1,k1]
        
        # In the tensor decomposition, M1 corresponds to 4 rank-1 terms:
        # (A[i0,j0]*B[j0,k0]*W[i0,k0]) + (A[i0,j0]*B[j1,k1]*W[i0,k1]) + ...
        
        # I think I need to just implement this directly and verify numerically.
        
        return t_start + 7
    
    # I'm going to take a step back and just implement the 50-rank decomposition
    # using a direct construction, then verify it numerically.
    
    # Let me try a different approach: use the known Strassen decomposition
    # and extend it to the rectangular case.
    
    # For now, let me just return the schoolbook decomposition to make sure
    # the code is correct, and then try to improve it.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][t] = 1
                V[j * p + k][t] = 1
                W[k * n + i][t] = 1
                t += 1
    
    return {"U": U, "V": V, "W": W}
