"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _rank1_decomposition(n, m, p):
    """Schoolbook rank n*m*p decomposition for n x m * m * p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    """
    terms = []
    # P1
    terms.append((
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1},
        {(0,0): 1, (1,1): 1}
    ))
    # P2
    terms.append((
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,0): 1, (1,1): -1}
    ))
    # P3
    terms.append((
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1}
    ))
    # P4
    terms.append((
        {(1,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (1,0): 1}
    ))
    # P5
    terms.append((
        {(0,0): 1, (0,1): 1},
        {(1,1): 1},
        {(0,0): -1, (0,1): 1}
    ))
    # P6
    terms.append((
        {(1,0): 1, (0,0): -1},
        {(0,0): 1, (0,1): 1},
        {(1,1): 1}
    ))
    # P7
    terms.append((
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1}
    ))
    return terms

def _construct_2x3x9():
    """
    Construct rank-50 decomposition for 2x3 * 3x9.
    """
    n, m, p = 2, 3, 9
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    strassen_terms = _strassen_2x2x2_terms()
    
    # Part 1: A[:, 0:2] * B[0:2, :]
    # Split B[0:2, :] into 4 pairs of columns and 1 single column
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    
    for (k1, k2) in k_pairs:
        for (u_dict, v_dict, w_dict) in strassen_terms:
            U_term = {}
            for (li, lj), coeff in u_dict.items():
                U_term[(li, lj)] = coeff
            
            V_term = {}
            for (lj, lk), coeff in v_dict.items():
                V_term[(lj, k1 + lk)] = coeff
            
            W_term = {}
            for (li, lk), coeff in w_dict.items():
                W_term[(li, k1 + lk)] = coeff
            
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
            
    for k in k_single:
        for i in range(2):
            for j in range(2):
                U_terms.append({(i, j): 1})
                V_terms.append({(j, k): 1})
                W_terms.append({(i, k): 1})
                
    # Part 2: A[:, 2] * B[2, :]
    for i in range(2):
        for k in range(9):
            U_terms.append({(i, 2): 1})
            V_terms.append({(2, k): 1})
            W_terms.append({(i, k): 1})
            
    rank = len(U_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for (i, j), coeff in U_terms[col].items():
            U[i * m + j][col] = coeff
        for (j, k), coeff in V_terms[col].items():
            V[j * p + k][col] = coeff
        for (i, k), coeff in W_terms[col].items():
            W[k * n + i][col] = coeff
            
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly represents matrix multiplication."""
    rank = len(U[0]) if U else 0
    
    T_computed = np.zeros((n*m, m*p, p*n))
    for c in range(rank):
        u_vec = np.array([U[i][c] for i in range(n*m)])
        v_vec = np.array([V[i][c] for i in range(m*p)])
        w_vec = np.array([W[i][c] for i in range(p*n)])
        T_computed += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
        
    T_expected = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_expected[i*m+j, j*p+k, k*n+i] = 1
                
    diff = np.max(np.abs(T_computed - T_expected))
    return diff < 1e-10

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    """
    start = time.time()
    deadline = start + 135
    
    try:
        U, V, W = _construct_2x3x9()
        if _verify_decomposition(U, V, W, 2, 3, 9):
            return {"U": U, "V": V, "W": W}
        else:
            return {"U": _rank1_decomposition(2, 3, 9)[0], 
                    "V": _rank1_decomposition(2, 3, 9)[1], 
                    "W": _rank1_decomposition(2, 3, 9)[2]}
    except Exception:
        U, V, W = _rank1_decomposition(2, 3, 9)
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
