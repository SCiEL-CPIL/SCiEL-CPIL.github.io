"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Combines block decomposition with Strassen and peeling algorithm for rank reduction.
"""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    Uses block decomposition with Strassen, then applies peeling to reduce rank.
    """
    start_time = time.time()
    deadline = 135.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 3, 4, 7
    nm, mp, pn = n*m, m*p, p*n

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        """Add a single rank-1 tensor term to the decomposition."""
        u = [0.0] * nm
        v = [0.0] * mp
        w = [0.0] * pn

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        Adds 7 rank-1 terms.
        """
        i1 = i0 + 1
        j1 = j0 + 1
        k1 = k0 + 1

        # M1: (A11+A22)*(B11+B22)
        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])

        # M2: (A21+A22)*B11
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])

        # M3: A11*(B12-B22)
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])

        # M4: A22*(B21-B11)
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])

        # M5: (A11+A12)*B22
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])

        # M6: (A21-A11)*(B11+B12)
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])

        # M7: (A12-A22)*(B21+B22)
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Split p=7 as 2+2+2+1: Strassen x3 + schoolbook = 25
        """
        strassen_222(i0, j0, k0)
        strassen_222(i0, j0, k0 + 2)
        strassen_222(i0, j0, k0 + 4)
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Schoolbook = rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m327_decomp(i0, j0, k0):
        """
        Decomposition for 3x2 * 2x7 block.
        Split n=3 as 2+1: M_{2,2,7}=25 + M_{1,2,7}=14 = 39
        """
        m227_decomp(i0, j0, k0)
        m127_decomp(i0 + 2, j0, k0)

    def m347_decomp(i0, j0, k0):
        """
        Decomposition for 3x4 * 4x7 block.
        Split m=4 as 2+2: 2 * M_{3,2,7} = 78
        """
        m327_decomp(i0, j0, k0)
        m327_decomp(i0, j0 + 2, k0)

    # Build the initial decomposition
    if time.time() - start_time < deadline:
        m347_decomp(0, 0, 0)

    rank = len(U_vecs)
    U = np.array(U_vecs).T  # shape (nm, rank)
    V = np.array(V_vecs).T  # shape (mp, rank)
    W = np.array(W_vecs).T  # shape (pn, rank)

    # Peeling algorithm: try to reduce rank by combining terms
    if time.time() - start_time < deadline - 10.0:
        U, V, W, rank = peel_decomposition(U, V, W, n, m, p, deadline, start_time)

    # Verify correctness
    if time.time() - start_time < deadline - 5.0:
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        # Efficient verification using tensor contraction
        # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = U[:, r].reshape(n, m)
            v_r = V[:, r].reshape(m, p)
            w_r = W[:, r].reshape(p, n)
            # Term: (A * u_r) * (v_r * B) * w_r
            # Actually: sum_j A[i,j]*u_r[i,j] * sum_k v_r[j,k]*B[j,k] ... 
            # Let's do it directly
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        s += A[i, j] * B[j, k] * u_r[i, j] * v_r[j, k] * w_r[k, i]
                    C_computed[i, k] += s

        error = np.max(np.abs(C_computed - C_expected))
        if error > 1e-9:
            # Fallback to schoolbook
            rank_school = n * m * p
            U_sb = np.zeros((nm, rank_school))
            V_sb = np.zeros((mp, rank_school))
            W_sb = np.zeros((pn, rank_school))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, col] = 1
                        V_sb[j * p + k, col] = 1
                        W_sb[k * n + i, col] = 1
                        col += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def peel_decomposition(U, V, W, n, m, p, deadline, start_time):
    """
    Peeling algorithm: iteratively try to remove rank-1 terms by checking
    if the residual tensor has lower rank.
    
    Strategy: For each pair of terms, check if their sum can be replaced
    by a single term (i.e., if the 2-term combination has rank 1).
    """
    nm, mp, pn = n*m, m*p, p*n
    rank = U.shape[1]
    
    # Try to merge pairs of terms
    merged = False
    U_list = [U[:, r] for r in range(rank)]
    V_list = [V[:, r] for r in range(rank)]
    W_list = [W[:, r] for r in range(rank)]
    
    # Try all pairs to see if we can merge
    i = 0
    while i < len(U_list) and time.time() - start_time < deadline - 10.0:
        j = i + 1
        while j < len(U_list) and time.time() - start_time < deadline - 10.0:
            # Check if U_list[i] + U_list[j], V_list[i] + V_list[j], W_list[i] + W_list[j]
            # has rank 1
            u_sum = U_list[i] + U_list[j]
            v_sum = V_list[i] + V_list[j]
            w_sum = W_list[i] + W_list[j]
            
            # Check rank 1: u_sum ⊗ v_sum ⊗ w_sum should match the sum of the two terms
            # The sum of two rank-1 tensors has rank 1 iff they are proportional
            # Check if u_sum is proportional to u_list[i] and u_list[j]
            # This happens if the 2x2 matrix [u_list[i], u_list[j]] has rank 1
            # AND same for v and w
            
            # Simpler check: try to find scalars a,b such that
            # u_sum = a*u_list[i] + b*u_list[j] with a+b=1 (trivially true)
            # The real check is whether the outer product structure holds
            
            # Better approach: check if the tensor (u_sum ⊗ v_sum ⊗ w_sum)
            # equals u_list[i]⊗v_list[i]⊗w_list[i] + u_list[j]⊗v_list[j]⊗w_list[j]
            # This is equivalent to checking if the 2-term combination is rank 1
            
            # Quick check: if u_sum, v_sum, w_sum are all zero, skip
            if np.allclose(u_sum, 0) and np.allclose(v_sum, 0) and np.allclose(w_sum, 0):
                # These two terms cancel out
                del U_list[i], V_list[i], W_list[i]
                del U_list[j], V_list[j], W_list[j]
                merged = True
                break
            
            # Check if the sum is rank 1 by verifying the outer product
            # Compute the tensor and check its rank
            tensor_sum = np.einsum('a,b,c->abc', u_sum, v_sum, w_sum)
            tensor_orig = np.einsum('a,b,c->abc', U_list[i], V_list[i], W_list[i]) + \
                         np.einsum('a,b,c->abc', U_list[j], V_list[j], W_list[j])
            
            if np.allclose(tensor_sum, tensor_orig, atol=1e-10):
                # The sum is rank 1, so we can replace two terms with one
                U_list[i] = u_sum
                V_list[i] = v_sum
                W_list[i] = w_sum
                del U_list[j], V_list[j], W_list[j]
                merged = True
                break
            
            j += 1
        
        if merged:
            merged = False
            continue
        i += 1
    
    if len(U_list) == 0:
        return np.zeros((nm, 1)), np.zeros((mp, 1)), np.zeros((pn, 1)), 1
    
    U_new = np.column_stack(U_list)
    V_new = np.column_stack(V_list)
    W_new = np.column_stack(W_list)
    
    return U_new, V_new, W_new, len(U_list)
# EVOLVE-BLOCK-END
