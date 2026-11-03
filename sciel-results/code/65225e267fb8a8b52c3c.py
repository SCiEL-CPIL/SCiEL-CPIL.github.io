"""
Matrix multiplication 3x4 by 4x7 with rank-minimal tensor decomposition.
Uses block decomposition with Strassen for 2x2x2 blocks and schoolbook for smaller blocks.
Then applies peeling to remove linearly dependent rank-1 terms.
Target rank: 78 (possibly reducible).
"""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a low-rank tensor decomposition for 3x4 * 4x7 matrix multiplication.
    """
    start_time = time.time()
    deadline = 140.0

    n, m, p = DIMENSIONS  # 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [0.0] * (n * m)
        v = [0.0] * (m * p)
        w = [0.0] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += val
        for idx, val in zip(v_indices, v_vals):
            v[idx] += val
        for idx, val in zip(w_indices, w_vals):
            w[idx] += val

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def strassen_222_local(i0, j0, k0):
        """
        Strassen decomposition for 2x2 * 2x2 block starting at (i0, j0, k0).
        i0, j0, k0 are the starting indices.
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

    def m127_decomp(i0, j0, k0):
        """
        Decomposition for 1x2 * 2x7 block.
        Rank 14.
        """
        for j in range(2):
            for k in range(7):
                jj = j0 + j
                kk = k0 + k
                add_rank1_term([i0*m+jj], [1.0],
                               [jj*p+kk], [1.0],
                               [kk*n+i0], [1.0])

    def m227_decomp(i0, j0, k0):
        """
        Decomposition for 2x2 * 2x7 block.
        Rank 25.
        """
        strassen_222_local(i0, j0, k0)
        strassen_222_local(i0, j0, k0 + 2)
        strassen_222_local(i0, j0, k0 + 4)
        k6 = k0 + 6
        for i in range(2):
            for j in range(2):
                ii = i0 + i
                jj = j0 + j
                add_rank1_term([ii*m+jj], [1.0],
                               [jj*p+k6], [1.0],
                               [k6*n+ii], [1.0])

    def m347_decomp():
        """
        Full 3x4 * 4x7 decomposition.
        Split n=3 into 1+2.
        n=1 part: 1x4 * 4x7. Split m=4 into 2+2. Two 1x2x7 blocks.
        n=2 part: 2x4 * 4x7. Split m=4 into 2+2. Two 2x2x7 blocks.
        """
        # n=1 part (i=0)
        # m split: j0=0 (size 2), j0=2 (size 2)
        m127_decomp(0, 0, 0)
        m127_decomp(0, 2, 0)
        
        # n=2 part (i=1,2)
        # m split: j0=0 (size 2), j0=2 (size 2)
        m227_decomp(1, 0, 0)
        m227_decomp(1, 2, 0)

    m347_decomp()

    U = np.array(U_vecs).T
    V = np.array(V_vecs).T
    W = np.array(W_vecs).T

    # Peeling: try to remove linearly dependent rank-1 terms
    def try_peel(U_mat, V_mat, W_mat):
        """
        Attempt to reduce rank by identifying and removing linearly dependent
        rank-1 terms. A term r is redundant if it can be expressed as a linear
        combination of other terms, i.e., if the tensor contribution of term r
        lies in the span of the other terms' contributions.
        """
        U_curr = U_mat.copy()
        V_curr = V_mat.copy()
        W_curr = W_mat.copy()
        rank = U_curr.shape[1]
        removed = set()
        
        # We'll iteratively try to remove terms
        # For each term, check if removing it still preserves the decomposition
        # This is expensive, so we use a heuristic: check if the term's
        # "signature" (pattern of non-zero entries) is unique
        
        # Build a matrix where each column is a rank-1 term's tensor product
        # flattened. If columns are linearly dependent, we can remove one.
        
        # Actually, a simpler approach: check if any two terms have the same
        # support pattern and can be combined.
        
        # Let's try a different approach: use SVD on the flattened tensor
        # to find the numerical rank, then project.
        
        # For each term r, compute the tensor T_r = u_r ⊗ v_r ⊗ w_r
        # Stack all T_r as columns of a big matrix M (n*m*p x rank)
        # If rank(M) < rank, then some terms are redundant.
        
        # But this is very expensive for large dimensions.
        # Instead, let's use a simpler heuristic:
        # Check if removing term r still gives exact decomposition.
        
        n_terms = rank
        for r in range(n_terms):
            if time.time() > deadline - 10:
                break
            if r in removed:
                continue
            
            # Try removing term r
            # Check if the resulting decomposition is still valid
            # This requires checking all (i,j,k) triples, which is 3*4*7=84 checks
            # For each check, we sum over (rank-1) terms.
            # Total: 84 * (rank-1) multiplications per term.
            # For rank=78, that's 84*77 = 6468 ops per term, 78 terms total.
            # That's manageable.
            
            is_redundant = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        total = 0.0
                        for r2 in range(n_terms):
                            if r2 == r or r2 in removed:
                                continue
                            total += U_curr[i*m+j, r2] * V_curr[j*p+k, r2] * W_curr[k*n+i, r2]
                        if abs(total - 1.0) > 1e-9:
                            is_redundant = False
                            break
                    if not is_redundant:
                        break
                if not is_redundant:
                    break
            
            if is_redundant:
                removed.add(r)
        
        if not removed:
            return U_mat, V_mat, W_mat
        
        # Rebuild matrices without removed terms
        keep = [r for r in range(n_terms) if r not in removed]
        return U_curr[:, keep], V_curr[:, keep], W_curr[:, keep]

    # Apply peeling
    U, V, W = try_peel(U, V, W)

    rank = U.shape[1]
    
    # Final verification
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(total - 1.0) > 1e-9:
                    # If peeling broke something, fall back to original
                    U_orig = np.array(U_vecs).T
                    V_orig = np.array(V_vecs).T
                    W_orig = np.array(W_vecs).T
                    return {"U": U_orig.tolist(), "V": V_orig.tolist(), "W": W_orig.tolist()}
                    break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
