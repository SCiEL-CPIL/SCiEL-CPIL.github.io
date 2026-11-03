import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100.0

    def decompose_strassen(n, m, p):
        """
        Returns a list of (U_dict, V_dict, W_dict) for Strassen decomposition.
        """
        cols = []

        def add_strassen_2x2x2(i0, j0, k0):
            # 7 Strassen terms
            # M1 = (A11+A22)(B11+B22)
            cols.append((
                {i0*m+j0: 1, (i0+1)*m+(j0+1): 1},
                {j0*p+k0: 1, (j0+1)*p+(k0+1): 1},
                {k0*n+i0: 1, (k0+1)*n+(i0+1): 1}
            ))
            # M2 = (A21+A22)B11
            cols.append((
                {(i0+1)*m+j0: 1, (i0+1)*m+(j0+1): 1},
                {j0*p+k0: 1},
                {k0*n+(i0+1): 1, (k0+1)*n+(i0+1): -1}
            ))
            # M3 = A11(B12-B22)
            cols.append((
                {i0*m+j0: 1},
                {j0*p+(k0+1): 1, (j0+1)*p+(k0+1): -1},
                {(k0+1)*n+i0: 1, (k0+1)*n+(i0+1): 1}
            ))
            # M4 = A22(B21-B11)
            cols.append((
                {(i0+1)*m+(j0+1): 1},
                {(j0+1)*p+k0: 1, j0*p+k0: -1},
                {k0*n+i0: 1, k0*n+(i0+1): 1}
            ))
            # M5 = (A11+A12)B22
            cols.append((
                {i0*m+j0: 1, i0*m+(j0+1): 1},
                {(j0+1)*p+(k0+1): 1},
                {k0*n+i0: -1, (k0+1)*n+i0: 1}
            ))
            # M6 = (A21-A11)(B11+B12)
            cols.append((
                {(i0+1)*m+j0: 1, i0*m+j0: -1},
                {j0*p+k0: 1, j0*p+(k0+1): 1},
                {(k0+1)*n+(i0+1): 1}
            ))
            # M7 = (A12-A22)(B21+B22)
            cols.append((
                {i0*m+(j0+1): 1, (i0+1)*m+(j0+1): -1},
                {(j0+1)*p+k0: 1, (j0+1)*p+(k0+1): 1},
                {k0*n+i0: 1}
            ))

        def recurse(n_curr, m_curr, p_curr, i0, j0, k0):
            if n_curr == 1 and m_curr == 1 and p_curr == 1:
                u_idx = i0 * m + j0
                v_idx = j0 * p + k0
                w_idx = k0 * n + i0
                cols.append(({u_idx: 1.0}, {v_idx: 1.0}, {w_idx: 1.0}))
                return

            if n_curr >= 2 and m_curr >= 2 and p_curr >= 2:
                add_strassen_2x2x2(i0, j0, k0)
                if n_curr > 2:
                    recurse(n_curr - 2, m_curr, p_curr, i0 + 2, j0, k0)
                if m_curr > 2:
                    recurse(2, m_curr - 2, p_curr, i0, j0 + 2, k0)
                if p_curr > 2:
                    recurse(2, 2, p_curr - 2, i0, j0, k0 + 2)
                return

            if n_curr > 1:
                recurse(1, m_curr, p_curr, i0, j0, k0)
                recurse(n_curr - 1, m_curr, p_curr, i0 + 1, j0, k0)
                return
            elif m_curr > 1:
                recurse(n_curr, 1, p_curr, i0, j0, k0)
                recurse(n_curr, m_curr - 1, p_curr, i0, j0 + 1, k0)
                return
            elif p_curr > 1:
                recurse(n_curr, m_curr, 1, i0, j0, k0)
                recurse(n_curr, m_curr, p_curr - 1, i0, j0, k0 + 1)
                return

        recurse(n, m, p, 0, 0, 0)
        return cols

    # 1. Generate initial Strassen decomposition
    cols = decompose_strassen(n, m, p)
    rank = len(cols)
    
    # 2. Subspace Peeling: Check for linear dependencies among the rank-1 tensors
    # We represent each term as a vector in the full tensor space (size n*m*p)
    # If the rank of the matrix of these vectors is less than the number of vectors,
    # we can potentially reduce the rank.
    
    def vectorize_term(term):
        u, v, w = term
        vec = np.zeros(n * m * p, dtype=float)
        # This vectorization is for dependency check only.
        # The tensor T has indices (i, j, k).
        # Term is u[i,j] * v[j,k] * w[i,k]
        # We expand the sum over i,j,k
        for i, j in u:
            for j2, k in v:
                if j == j2:
                    for i2, k2 in w:
                        if i == i2 and k == k2:
                            idx = (i * m + j) * p + k
                            vec[idx] += u[(i,j)] * v[(j,k)] * w[(i,k)]
        return vec

    # Build matrix of vectors
    if rank > 0:
        vecs = [vectorize_term(c) for c in cols]
        M = np.vstack(vecs)
        
        # Calculate rank of M
        # M has shape (n*m*p, rank)
        # We want to see if columns are linearly independent.
        # Use SVD or QR with column pivoting
        try:
            # Rank of M (number of linearly independent columns)
            # Since n*m*p (72) is small, this is fast.
            U_svd, s, Vh = np.linalg.svd(M, full_matrices=False)
            tol = 1e-10 * s[0] if s[0] > 0 else 1e-10
            current_rank = np.sum(s > tol)
            
            if current_rank < rank:
                # There are dependencies. We can try to find a subset of columns
                # that spans the same space and solve for coefficients.
                # For simplicity and safety, if we detect dependency,
                # we can attempt to drop columns and re-solve.
                # However, finding the *minimal* subset is NP-hard.
                # We will just drop the columns corresponding to the smallest singular values
                # and verify if the resulting decomposition is still exact (within tolerance).
                
                # Indices of columns to keep (largest singular values)
                keep_idx = np.argsort(s)[::-1][:current_rank]
                # Note: Vh rows correspond to columns of M.
                # We need to map back to original terms.
                
                # This part is complex to implement correctly for *exact* integer recovery
                # in general. If we drop a term, the coefficients of remaining terms
                # must change to compensate.
                # Given the strict "integer/half-integer" constraint,
                # this "peeling" is high-risk.
                
                # SAFE FALLBACK:
                # If we detect that the theoretical rank is lower, we might still
                # stick to the 66 terms if we can't easily refit integer coefficients.
                # However, for this specific problem, if dependencies exist,
                # a simple greedy drop-and-solve might work.
                
                # To strictly adhere to the "improve if possible" and "safe fallback"
                # we will only output the reduced rank if we can verify it exactly.
                # For now, we return the original 66 if we can't guarantee exactness
                # of the reduced set.
                
                pass # No automatic reduction implemented to ensure validity
        except Exception:
            pass

    # 3. Construct final matrices
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    for r, (u_dict, v_dict, w_dict) in enumerate(cols):
        for idx, val in u_dict.items():
            U[idx, r] = val
        for idx, val in v_dict.items():
            V[idx, r] = val
        for idx, val in w_dict.items():
            W[idx, r] = val
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
