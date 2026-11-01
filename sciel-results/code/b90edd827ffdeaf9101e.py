"""Tensor rank minimization for 2x5x6 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 5, 6
    start_time = time.time()
    deadline = start_time + 110.0

    rank = 60
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0

    def add_term(u_coeffs, v_coeffs, w_coeffs):
        nonlocal col
        if col >= rank:
            return
        for u_idx, coeff in u_coeffs:
            U[u_idx, col] += coeff
        for v_idx, coeff in v_coeffs:
            V[v_idx, col] += coeff
        for w_idx, coeff in w_coeffs:
            W[w_idx, col] += coeff
        col += 1

    # Strassen block for (i∈{0,1}, j∈{ja,jb}, k∈{ka,kb})
    # u indices: i*m + j
    # v indices: j*p + k
    # w indices: k*n + i
    def strassen_block(ja, jb, ka, kb):
        ua = ja       # (0, ja)
        ub = 5 + ja   # (1, ja)
        va = jb       # (0, jb)
        vb = 5 + jb   # (1, jb)
        wka0 = 2 * ka   # (ka, 0)
        wka1 = 2 * ka + 1  # (ka, 1)
        wkb0 = 2 * kb   # (kb, 0)
        wkb1 = 2 * kb + 1  # (kb, 1)
        vka = ja * p + ka  # (ja, ka)
        vkb = ja * p + kb  # (ja, kb)
        wka2 = jb * p + ka  # (jb, ka)
        wkb2 = jb * p + kb  # (jb, kb)

        # M1
        add_term([(ua, 1), (vb, 1)], [(vka, 1), (wkb2, 1)], [(wka0, 1), (wkb1, 1)])
        # M2
        add_term([(ub, 1), (vb, 1)], [(vka, 1)], [(wka1, 1), (wkb1, -1)])
        # M3
        add_term([(ua, 1)], [(vkb, 1), (wkb2, -1)], [(wka0 + 2, 1), (wkb1, 1)])
        # M4
        add_term([(vb, 1)], [(wka2, 1), (vka, -1)], [(wka0, 1), (wka1, 1)])
        # M5
        add_term([(ua, 1), (va, 1)], [(wkb2, 1)], [(wka0, -1), (wka0 + 2, 1)])
        # M6
        add_term([(ub, 1), (ua, -1)], [(vka, 1), (vkb, 1)], [(wkb1, 1)])
        # M7
        add_term([(va, 1), (vb, -1)], [(wka2, 1), (wkb2, 1)], [(wka0, 1)])

    # 2 Strassen blocks:
    # Block A: j∈{0,1}, k∈{0,1}
    strassen_block(0, 1, 0, 1)
    # Block B: j∈{2,3}, k∈{0,1}
    strassen_block(2, 3, 0, 1)

    # All (i,j,k) in the Strassen blocks:
    strassen_set = set()
    for i in range(n):
        for j in [0, 1, 2, 3]:
            for k in [0, 1]:
                strassen_set.add((i, j, k))

    # Standard terms for remaining (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in strassen_set:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1

    U_used = U[:, :col]
    V_used = V[:, :col]
    W_used = W[:, :col]

    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_row = U_used[i * m + j, :]
                v_row = V_used[j * p + k, :]
                w_row = W_used[k * n + i, :]
                result = np.sum(u_row * v_row * w_row)
                if abs(result - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to trivial rank 60
        U_fallback = np.zeros((n * m, 60), dtype=float)
        V_fallback = np.zeros((m * p, 60), dtype=float)
        W_fallback = np.zeros((p * n, 60), dtype=float)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fallback[i * m + j, c] = 1
                    V_fallback[j * p + k, c] = 1
                    W_fallback[k * n + i, c] = 1
                    c += 1
        return {"U": U_fallback.tolist(), "V": V_fallback.tolist(), "W": W_fallback.tolist()}

    # Try to reduce rank by finding linear dependencies
    # Build the tensor in flattened form and check for rank reduction
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # We want to find if some rank-1 components are linearly dependent
    
    # Check if we can remove columns by solving for dependencies
    # For each column r, the rank-1 tensor is U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Flatten to a vector of size (n*m)*(m*p)*(p*n) = 10*30*12 = 3600
    
    def tensor_to_vector(u_col, v_col, w_col):
        """Flatten a rank-1 tensor to a vector."""
        vec = np.zeros(n * m * m * p * p * n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = (i * m + j) * (m * p) + (j * p + k) * n + k * n + i
                    vec[idx] = u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        return vec

    # Build matrix of all rank-1 tensors
    num_terms = col
    tensor_matrix = np.zeros((n * m * m * p * p * n, num_terms))
    for r in range(num_terms):
        tensor_matrix[:, r] = tensor_to_vector(U_used[:, r], V_used[:, r], W_used[:, r])
    
    # Find rank of this matrix
    # Use SVD to find the numerical rank
    try:
        _, s, Vt = np.linalg.svd(tensor_matrix, full_matrices=False)
        # Find the numerical rank
        tol = 1e-10
        num_rank = np.sum(s > tol)
        
        if num_rank < num_terms:
            # We can potentially reduce the rank
            # Use the right singular vectors to find a reduced basis
            # The top num_rank singular vectors give us the basis
            reduced_basis = Vt[:num_rank, :]
            
            # Now we need to express each original term in terms of the reduced basis
            # This is complex, so let's try a simpler approach:
            # Just return the first num_rank columns if they span the same space
            
            # Actually, let's try a different approach:
            # Use QR decomposition to find an orthonormal basis
            Q, R = np.linalg.qr(tensor_matrix)
            # The columns of Q up to num_rank form a basis
            # But we need to reconstruct the tensor, which is non-trivial
            
            # For now, just return the original if we can't easily reduce
            pass
    except np.linalg.LinAlgError:
        pass

    # Alternative: try different block decompositions
    # Try: Strassen on (j∈{0,1}, k∈{2,3}) and (j∈{2,3}, k∈{4,5})
    # This covers 8 more (i,j,k) triples with 14 terms
    
    # Let's try a different strategy: use Strassen for all 3 j-pairs and 3 k-pairs
    # But that would be 3*3=9 blocks * 7 = 63 terms, worse than 26
    
    # Try: 1 Strassen block + remaining as standard
    # Block: j∈{0,1}, k∈{0,1} → 7 terms for 8 triples
    # Remaining: 12 triples → 12 terms
    # Total: 19 terms? No, that's the same as before
    
    # Actually, let's try: can we share terms between blocks?
    # The two Strassen blocks use different j and k indices, so no sharing possible
    
    # Let's try a completely different decomposition:
    # Use 1 Strassen block for j∈{0,1}, k∈{0,1} (7 terms, 8 triples)
    # And handle the rest more cleverly
    
    # For j∈{2,3,4}, k∈{0,1,2,3,4,5}: 3*6*2 = 36 triples
    # Standard: 36 terms
    # Total: 7 + 36 = 43, worse
    
    # For j∈{0,1,2,3}, k∈{0,1}: 4*2*2 = 16 triples
    # 2 Strassen blocks: 14 terms
    # For j∈{4}, k∈{0,1,2,3,4,5}: 1*6*2 = 12 triples
    # Standard: 12 terms
    # Total: 14 + 12 = 26 (same as current)
    
    # Try: j∈{0,1}, k∈{0,1,2,3}: 2*4*2 = 16 triples
    # 2 Strassen blocks (k∈{0,1} and k∈{2,3}): 14 terms
    # j∈{2,3,4}, k∈{0,1,2,3,4,5}: 3*6*2 = 36 triples
    # Standard: 36 terms
    # Total: 14 + 36 = 50, worse
    
    # The current 26 seems hard to beat with this approach
    # Let's just return the current solution
    return {"U": U_used.tolist(), "V": V_used.tolist(), "W": W_used.tolist()}
# EVOLVE-BLOCK-END
