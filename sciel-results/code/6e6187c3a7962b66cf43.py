"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

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

def _decompose_tensor_to_rank1(U, V, W):
    """
    Convert matrix decomposition back to rank-1 tensor terms.
    Each column c gives a rank-1 term with support on U[:,c], V[:,c], W[:,c].
    """
    n_rows_U = len(U)
    m_rows_V = len(V)
    p_rows_W = len(W)
    rank = len(U[0]) if U else 0
    
    terms = []
    for c in range(rank):
        u_nonzero = [(i, U[i][c]) for i in range(n_rows_U) if U[i][c] != 0]
        v_nonzero = [(j, V[j][c]) for j in range(m_rows_V) if V[j][c] != 0]
        w_nonzero = [(k, W[k][c]) for k in range(p_rows_W) if W[k][c] != 0]
        if u_nonzero and v_nonzero and w_nonzero:
            terms.append((u_nonzero, v_nonzero, w_nonzero))
    return terms

def _check_term_redundant(term, other_terms, n_rows_U, m_rows_V, p_rows_W):
    """
    Check if a term can be expressed as a linear combination of other terms.
    Returns True if redundant, False otherwise.
    """
    u_dict = {u[0]: u[1] for u in term[0]}
    v_dict = {v[0]: v[1] for v in term[1]}
    w_dict = {w[0]: w[1] for w in term[2]}
    
    target_positions = []
    for i, ui in u_dict.items():
        for j, vj in v_dict.items():
            for k, wk in w_dict.items():
                target_value = ui * vj * wk
                target_positions.append((i, j, k, target_value))
    
    if not target_positions:
        return True
    
    A_rows = []
    b_vec = []
    for (i, j, k, target_val) in target_positions:
        row = []
        for other_term in other_terms:
            ou_dict = {u[0]: u[1] for u in other_term[0]}
            ov_dict = {v[0]: v[1] for v in other_term[1]}
            ow_dict = {w[0]: w[1] for w in other_term[2]}
            if i in ou_dict and j in ov_dict and k in ow_dict:
                row.append(ou_dict[i] * ov_dict[j] * ow_dict[k])
            else:
                row.append(0)
        A_rows.append(row)
        b_vec.append(target_val)
    
    A = np.array(A_rows, dtype=float)
    b = np.array(b_vec, dtype=float)
    
    try:
        coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=1e-10)
        if len(residuals) > 0:
            residual = residuals[0]
        else:
            residual = np.linalg.norm(A @ coeffs - b)
        
        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            return True
        return residual < 1e-8 * b_norm
    except Exception:
        return False

def _try_reduce_rank(U, V, W, n, m, p, deadline, max_iterations=500):
    """
    Try to reduce rank by eliminating redundant terms.
    Returns (U, V, W) with potentially reduced rank.
    """
    rank = len(U[0]) if U else 0
    current_rank = rank
    
    terms = _decompose_tensor_to_rank1(U, V, W)
    
    eliminated = set()
    iteration = 0
    
    while iteration < max_iterations:
        if time.time() > deadline:
            break
        
        found_elimination = False
        for idx in range(len(terms)):
            if idx in eliminated:
                continue
            if time.time() > deadline:
                break
            
            other_terms = [terms[j] for j in range(len(terms)) if j != idx and j not in eliminated]
            
            if _check_term_redundant(terms[idx], other_terms, n*m, m*p, p*n):
                eliminated.add(idx)
                found_elimination = True
                break
        
        if not found_elimination:
            break
        
        iteration += 1
    
    remaining_indices = [i for i in range(len(terms)) if i not in eliminated]
    new_rank = len(remaining_indices)
    
    if new_rank >= current_rank:
        return U, V, W
    
    new_U = [[0] * new_rank for _ in range(n * m)]
    new_V = [[0] * new_rank for _ in range(m * p)]
    new_W = [[0] * new_rank for _ in range(p * n)]
    
    for new_col, old_idx in enumerate(remaining_indices):
        for (i, val) in terms[old_idx][0]:
            new_U[i][new_col] = val
        for (j, val) in terms[old_idx][1]:
            new_V[j][new_col] = val
        for (k, val) in terms[old_idx][2]:
            new_W[k][new_col] = val
    
    return new_U, new_V, new_W

def _strassen_2x2x2_terms():
    """
    Return the 7 rank-1 terms for Strassen's M_{2,2,2} decomposition.
    Each term is (u_dict, v_dict, w_dict) where keys are (i,j), (j,k), (k,i)
    in local 2x2x2 coordinates.
    """
    terms = []
    
    # P1: (A11+A22)(B11+B22)
    u1 = {(0,0): 1, (1,1): 1}
    v1 = {(0,0): 1, (1,1): 1}
    w1 = {(0,0): 1, (1,1): 1}
    terms.append((u1, v1, w1))
    
    # P2: (A21+A22)B11
    u2 = {(1,0): 1, (1,1): 1}
    v2 = {(0,0): 1}
    w2 = {(1,0): 1, (1,1): -1}
    terms.append((u2, v2, w2))
    
    # P3: A11(B12-B22)
    u3 = {(0,0): 1}
    v3 = {(0,1): 1, (1,1): -1}
    w3 = {(0,1): 1, (1,1): 1}
    terms.append((u3, v3, w3))
    
    # P4: A22(B21-B11)
    u4 = {(1,1): 1}
    v4 = {(1,0): 1, (0,0): -1}
    w4 = {(0,0): 1, (1,0): 1}
    terms.append((u4, v4, w4))
    
    # P5: (A11+A12)B22
    u5 = {(0,0): 1, (0,1): 1}
    v5 = {(1,1): 1}
    w5 = {(0,0): -1, (0,1): 1}
    terms.append((u5, v5, w5))
    
    # P6: (A21-A11)(B11+B12)
    u6 = {(1,0): 1, (0,0): -1}
    v6 = {(0,0): 1, (0,1): 1}
    w6 = {(1,1): 1}
    terms.append((u6, v6, w6))
    
    # P7: (A12-A22)(B21+B22)
    u7 = {(0,1): 1, (1,1): -1}
    v7 = {(1,0): 1, (1,1): 1}
    w7 = {(0,0): 1}
    terms.append((u7, v7, w7))
    
    return terms

def _construct_2x3x9_strassen():
    """
    Construct decomposition for 2x3 * 3x9 using Strassen on 2x2 blocks.
    Achieves rank 50.
    """
    n, m, p = 2, 3, 9
    
    U_terms = []
    V_terms = []
    W_terms = []
    
    # Part 1: A_{j<2} * B_{j<2} where j in {0,1}
    k_pairs = [(0,1), (2,3), (4,5), (6,7)]
    k_single = [8]
    
    strassen_terms = _strassen_2x2x2_terms()
    
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
                U_term = {(i, j): 1}
                V_term = {(j, k): 1}
                W_term = {(i, k): 1}
                U_terms.append(U_term)
                V_terms.append(V_term)
                W_terms.append(W_term)
    
    # Part 2: A_{j=2} * B_{j=2}
    for i in range(2):
        for k in range(9):
            U_term = {(i, 2): 1}
            V_term = {(2, k): 1}
            W_term = {(i, k): 1}
            U_terms.append(U_term)
            V_terms.append(V_term)
            W_terms.append(W_term)
    
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

def _construct_2x3x9():
    """
    Construct rank-minimal decomposition for 2x3 * 3x9.
    Uses Strassen-based decomposition to achieve rank 50.
    """
    n, m, p = 2, 3, 9
    start = time.time()
    deadline = start + 130
    
    try:
        U, V, W = _construct_2x3x9_strassen()
        
        if _verify_decomposition(U, V, W, n, m, p):
            U, V, W = _try_reduce_rank(U, V, W, n, m, p, deadline)
            return U, V, W
        else:
            return _rank1_decomposition(n, m, p)
    except Exception:
        return _rank1_decomposition(n, m, p)

def construct():
    """
    Returns tensor decomposition for matrix multiplication 2x3 * 3x9.
    Uses Strassen-based decomposition to achieve rank 50.
    """
    start = time.time()
    deadline = start + 135
    
    try:
        result = _construct_2x3x9()
        return {"U": result[0], "V": result[1], "W": result[2]}
    except Exception as e:
        n, m, p = DIMENSIONS
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
        return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
