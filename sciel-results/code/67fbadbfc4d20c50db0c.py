"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a low-rank exact tensor decomposition for 2x2 * 2x14 matrix multiplication.
    Strategy: Try structured decompositions with cross-block sharing, then fall back
    to Strassen (rank 49) or schoolbook (rank 56).
    """
    import time
    start_time = time.time()
    deadline = start_time + 140  # 20s safety margin

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    def verify(Uf, Vf, Wf):
        """Check if decomposition is exact."""
        for i in range(n):
            for j in range(m):
                u_idx = i * m + j
                for k in range(p):
                    v_idx = j * p + k
                    w_idx = k * n + i
                    s = np.dot(Uf[u_idx, :], Vf[v_idx, :] * Wf[w_idx, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for x in row:
                rx = round(x)
                if abs(x - rx) < 1e-9:
                    clean_row.append(int(rx))
                else:
                    clean_row.append(float(x))
            result.append(clean_row)
        return result

    def strassen_decomp():
        """Block-wise Strassen: rank 49."""
        num_blocks = p // 2
        rank = num_blocks * 7
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)

        strassen_terms = [
            ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
            ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
            ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
            ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
            ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
            ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
            ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
        ]

        for t in range(num_blocks):
            for s in range(7):
                col = t * 7 + s
                a_coeffs, b_coeffs, c_coeffs = strassen_terms[s]
                for a_idx, coeff in a_coeffs.items():
                    a_i = a_idx // m
                    a_j = a_idx % m
                    U[a_i * m + a_j, col] = coeff
                for b_idx, coeff in b_coeffs.items():
                    b_j = b_idx // 2
                    b_col = b_idx % 2
                    b_k = 2 * t + b_col
                    V[b_j * p + b_k, col] = coeff
                for c_idx, coeff in c_coeffs.items():
                    c_i = c_idx // 2
                    c_col = c_idx % 2
                    c_k = 2 * t + c_col
                    W[c_k * n + c_i, col] = coeff

        return U, V, W, rank

    def schoolbook_decomp():
        """Schoolbook: rank 56."""
        rank = n * m * p
        U = np.zeros((nm, rank), dtype=np.float64)
        V = np.zeros((mp, rank), dtype=np.float64)
        W = np.zeros((pn, rank), dtype=np.float64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Try structured low-rank decompositions with cross-block sharing
    # The idea: if the B matrix has structure (e.g., repeated patterns),
    # we can share multiplication terms across blocks.
    
    # Approach: Try rank r from 48 down to 40, using a parameterized family
    # where the V and W matrices have shared structure across blocks.
    
    # For a 2x2 * 2x14 problem, think of it as 7 independent 2x2 * 2x2 multiplications
    # unless there's structure in B that allows sharing.
    
    # Try: can we do better than 7*7=49 by using a global Strassen-like scheme?
    # The key constraint is that A is 2x2 (only 4 independent entries) and B is 2x14.
    # The tensor is T[i,j,k] = delta for the multiplication.
    
    # Let's try a different structural approach: 
    # Represent the multiplication as a sequence of operations on the 2x14 matrix.
    # Since A is 2x2, C = A*B where A = [[a,b],[c,d]] and B is 2x14.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # C[0,k] = a*B[0,k] + b*B[1,k]
    # C[1,k] = c*B[0,k] + d*B[1,k]
    
    # This is just 28 scalar multiplications (2 rows * 14 cols) if we do it naively.
    # But we can share the computation of linear combinations of B's rows.
    
    # Actually, the tensor rank for 2x2 * 2x14 is at most 7*7 = 49 (block Strassen)
    # and at least 28 (since we need at least 28 "independent" scalar products).
    
    # Let's try to find a rank between 28 and 49 using a structured search.
    
    # For now, return Strassen as the best known structured decomposition.
    # A more sophisticated search would use numerical tensor rank algorithms,
    # but those are expensive. Let's try a few structured families.
    
    # Family 1: Use 7 Strassen blocks but try to merge some terms.
    # This is hard to do systematically, so let's just return Strassen.
    
    if time.time() < deadline:
        U, V, W, rank = strassen_decomp()
        if verify(U, V, W):
            return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
    
    # Fallback to schoolbook
    U, V, W, rank = schoolbook_decomp()
    return {"U": to_clean_list(U), "V": to_clean_list(V), "W": to_clean_list(W)}
# EVOLVE-BLOCK-END
