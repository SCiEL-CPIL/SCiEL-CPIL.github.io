import time
import numpy as np

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2_terms():
    """Return the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) with 0-based local indices."""
    terms = []
    # M1 = (A00+A11)(B00+B11)
    terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
    # M2 = (A10+A11)B00
    terms.append(({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}))
    # M3 = A00(B01-B11)
    terms.append(({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}))
    # M4 = A11(B10-B00)
    terms.append(({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}))
    # M5 = (A00+A01)B11
    terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (1,0):1}))
    # M6 = (A10-A00)(B00+B01)
    terms.append(({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}))
    # M7 = (A01-A11)(B10+B11)
    terms.append(({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}))
    return terms

def _terms_to_matrices(terms, n, m, p):
    """Convert list of term dicts to U, V, W matrices."""
    rank = len(terms)
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for c in range(rank):
        u_dict, v_dict, w_dict = terms[c]
        for (i, j), val in u_dict.items():
            U[i * m + j, c] = val
        for (j, k), val in v_dict.items():
            V[j * p + k, c] = val
        for (k, i), val in w_dict.items():
            W[k * n + i, c] = val
    
    return U, V, W

def _build_decomposition_46(n, m, p):
    """
    Builds the rank-46 decomposition for 2x5x5 matrix multiplication.
    Uses Strassen on 2x2x2 blocks and schoolbook for the rest.
    Partition J={0..4} into {0,1}, {2,3}, {4}.
    Partition K={0..4} into {0,1}, {2,3}, {4}.
    """
    terms = []
    
    def add_strassen(i_off, j_off, k_off):
        """Adds 7 terms for a 2x2x2 Strassen block at given offsets."""
        for u, v, w in _strassen_2x2x2_terms():
            u_off = {}
            for (ii, jj), val in u.items():
                u_off[(ii + i_off, jj + j_off)] = val
            v_off = {}
            for (jj, kk), val in v.items():
                v_off[(jj + j_off, kk + k_off)] = val
            w_off = {}
            for (kk, ii), val in w.items():
                w_off[(kk + k_off, ii + i_off)] = val
            terms.append((u_off, v_off, w_off))
    
    def add_schoolbook(i_start, i_end, j_start, j_end, k_start, k_end):
        """Adds schoolbook terms for the given ranges."""
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                for k in range(k_start, k_end):
                    terms.append((
                        {(i, j): 1},
                        {(j, k): 1},
                        {(k, i): 1}
                    ))
    
    # Strassen blocks (2x2x2)
    add_strassen(0, 0, 0) # i=0,1, j=0,1, k=0,1
    add_strassen(0, 0, 2) # i=0,1, j=0,1, k=2,3
    add_strassen(0, 2, 0) # i=0,1, j=2,3, k=0,1
    add_strassen(0, 2, 2) # i=0,1, j=2,3, k=2,3
    
    # Schoolbook blocks
    # j=0,1, k=4 (2x2x1)
    add_schoolbook(0, 2, 0, 2, 4, 5)
    # j=2,3, k=4 (2x2x1)
    add_schoolbook(0, 2, 2, 4, 4, 5)
    # j=4, k=0,1 (2x1x2)
    add_schoolbook(0, 2, 4, 5, 0, 2)
    # j=4, k=2,3 (2x1x2)
    add_schoolbook(0, 2, 4, 5, 2, 4)
    # j=4, k=4 (2x1x1)
    add_schoolbook(0, 2, 4, 5, 4, 5)
    
    return terms

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying 
    a 2 by 5 matrix by a 5 by 5 matrix.
    """
    n, m, p = 2, 5, 5
    
    terms = _build_decomposition_46(n, m, p)
    U, V, W = _terms_to_matrices(terms, n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
