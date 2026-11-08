"""
Rank-minimal exact tensor decomposition for 4x4 * 4x7 matrix multiplication.

Strategy:
  - Start from a correct schoolbook decomposition (rank 112).
  - Systematically try to remove terms by checking if the remaining terms
    can still reproduce the multiplication tensor.
  - Use a smarter search: group terms by output column, try removing one
    term at a time from each group.
"""
import time
import numpy as np

_START = time.time()
_DEADLINE = 140.0  # Safety margin before 150s limit


def _build_schoolbook():
    """
    Build the schoolbook decomposition for 4x4 * 4x7.
    Rank = 4 * 4 * 7 = 112.
    """
    n, m, p = 4, 4, 7
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W


def _verify(U, V, W, tol=1e-8):
    """
    Verify that the decomposition exactly reproduces the multiplication tensor.
    U: (16, r), V: (28, r), W: (28, r)
    """
    n, m, p = 4, 4, 7
    r = U.shape[1]
    
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    
    expected = np.zeros((n, m, p))
    for j in range(min(m, p)):
        expected[:, j, j] = 1.0
    
    return np.allclose(T, expected, atol=tol)


def _try_remove_term(U, V, W, col_idx):
    """
    Try removing term col_idx. Return (success, new_U, new_V, new_W).
    """
    r = U.shape[1]
    if r <= 1:
        return False, U, V, W
    
    mask = [i for i in range(r) if i != col_idx]
    U_sub = U[:, mask]
    V_sub = V[:, mask]
    W_sub = W[:, mask]
    
    if _verify(U_sub, V_sub, W_sub):
        return True, U_sub, V_sub, W_sub
    else:
        return False, U, V, W


def _search_lower_rank(U, V, W, time_budget=120.0):
    """
    Iteratively try to remove terms to reduce rank.
    """
    deadline = _START + time_budget
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Repeatedly try to remove terms
    changed = True
    while changed and time.time() < deadline:
        changed = False
        r = best_U.shape[1]
        
        # Try removing each term
        for c in range(r):
            if time.time() > deadline:
                break
            
            success, U_new, V_new, W_new = _try_remove_term(
                best_U, best_V, best_W, c
            )
            
            if success:
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = best_U.shape[1]
                changed = True
                break
    
    return best_rank, best_U, best_V, best_W


def _strassen_4x4():
    """
    Build a Strassen-based decomposition for 4x4 * 4x4.
    Returns U, V, W arrays with shape (16, 49), (28, 49), (28, 49).
    """
    n, m, p = 4, 4, 7
    rank = 49  # 7 Strassen terms * 7 sub-terms
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # 2x2 Strassen terms
    s2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,1,0,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,0,1,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,0,0]),
        ([1,0,1,0], [0,0,0,1], [0,1,0,1]),
        ([0,1,0,-1], [0,0,1,1], [-1,0,1,0]),
        ([-1,0,1,0], [1,1,0,0], [1,1,0,0]),
    ]
    
    # Block maps for 4x4
    a_blocks = {
        'A00': [0, 1, 4, 5],
        'A01': [2, 3, 6, 7],
        'A10': [8, 9, 12, 13],
        'A11': [10, 11, 14, 15],
    }
    b_blocks = {
        'B00': [0, 1, 7, 8],
        'B01': [2, 3, 9, 10],
        'B10': [14, 15, 21, 22],
        'B11': [16, 17, 23, 24],
    }
    c_blocks = {
        'C00': [0, 1, 4, 5],
        'C01': [2, 3, 6, 7],
        'C10': [8, 9, 12, 13],
        'C11': [10, 11, 14, 15],
    }
    
    # 7 Strassen block products
    products = [
        ([('A00', 1), ('A11', 1)], [('B00', 1), ('B11', 1)], [('C00', 1), ('C11', 1)]),
        ([('A10', 1), ('A11', 1)], [('B00', 1)], [('C10', 1)]),
        ([('A00', 1)], [('B01', 1), ('B11', -1)], [('C01', 1)]),
        ([('A11', 1)], [('B10', 1), ('B00', -1)], [('C00', 1)]),
        ([('A00', 1), ('A10', 1)], [('B11', 1)], [('C10', 1), ('C11', 1)]),
        ([('A01', 1), ('A11', -1)], [('B10', 1), ('B11', 1)], [('C01', 1)]),
        ([('A10', 1), ('A00', -1)], [('B00', 1), ('B01', 1)], [('C00', 1), ('C10', 1)]),
    ]
    
    col = 0
    for a_prod, b_prod, c_prod in products:
        for s_idx in range(7):
            u_s, v_s, w_s = s2x2[s_idx]
            
            u_full = [0.0] * 16
            for block_name, coeff in a_prod:
                for local_idx, global_idx in enumerate(a_blocks[block_name]):
                    u_full[global_idx] += coeff * u_s[local_idx]
            
            v_full = [0.0] * 28
            for block_name, coeff in b_prod:
                for local_idx, global_idx in enumerate(b_blocks[block_name]):
                    v_full[global_idx] += coeff * v_s[local_idx]
            
            w_full = [0.0] * 28
            for block_name, coeff in c_prod:
                for local_idx, global_idx in enumerate(c_blocks[block_name]):
                    w_full[global_idx] += coeff * w_s[local_idx]
            
            for i in range(16):
                if u_full[i] != 0:
                    U[i, col] = u_full[i]
            for j in range(28):
                if v_full[j] != 0:
                    V[j, col] = v_full[j]
            for k in range(28):
                if w_full[k] != 0:
                    W[k, col] = w_full[k]
            
            col += 1
    
    return U, V, W


def _build_hybrid():
    """
    Build a hybrid decomposition: Strassen for 4x4*4x4 + schoolbook for 4x4*4x3.
    """
    n, m, p = 4, 4, 7
    U_str, V_str, W_str = _strassen_4x4()
    
    # Schoolbook for remaining 3 columns
    U_school = np.zeros((n * m, 48))
    V_school = np.zeros((m * p, 48))
    W_school = np.zeros((p * n, 48))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    U = np.hstack([U_str, U_school])
    V = np.hstack([V_str, V_school])
    W = np.hstack([W_str, W_school])
    
    return U, V, W


def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 4x4 * 4x7.
    Returns {'U': list, 'V': list, 'W': list} with shapes (16, rank), (28, rank), (28, rank).
    """
    # Try multiple starting points
    candidates = []
    
    # 1. Schoolbook (rank 112)
    U1, V1, W1 = _build_schoolbook()
    if _verify(U1, V1, W1):
        candidates.append((112, U1, V1, W1))
    
    # 2. Hybrid Strassen + schoolbook (rank 97)
    U2, V2, W2 = _build_hybrid()
    if _verify(U2, V2, W2):
        candidates.append((97, U2, V2, W2))
    
    if not candidates:
        # Fallback: pure schoolbook
        U, V, W = _build_schoolbook()
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Pick the best candidate
    candidates.sort(key=lambda x: x[0])
    best_rank, best_U, best_V, best_W = candidates[0]
    
    # Try to reduce rank further
    elapsed = time.time() - _START
    time_budget = max(10.0, _DEADLINE - elapsed)
    
    final_rank, final_U, final_V, final_W = _search_lower_rank(
        best_U, best_V, best_W, time_budget
    )
    
    return {"U": final_U.tolist(), "V": final_V.tolist(), "W": final_W.tolist()}


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
