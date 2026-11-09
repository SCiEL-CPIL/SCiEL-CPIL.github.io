import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

def _construct_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    return U, V, W

def _strassen_2x2_terms():
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]

def _strassen_4x4_terms():
    s2 = _strassen_2x2_terms()
    macro_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    terms = []
    for A_m, B_m, C_m in macro_terms:
        for u2, v2, w2 in s2:
            u_coeffs = {}
            v_coeffs = {}
            w_coeffs = {}
            for (li, lj), u_val in u2.items():
                for (si, sj), a_val in A_m.items():
                    gi = si * 2 + li
                    gj = sj * 2 + lj
                    u_coeffs[(gi, gj)] = u_coeffs.get((gi, gj), 0) + u_val * a_val
            for (lj, lk), v_val in v2.items():
                for (sj, sk), b_val in B_m.items():
                    gj = sj * 2 + lj
                    gk = sk * 2 + lk
                    v_coeffs[(gj, gk)] = v_coeffs.get((gj, gk), 0) + v_val * b_val
            for (li, lk), w_val in w2.items():
                for (si, sk), c_val in C_m.items():
                    gi = si * 2 + li
                    gk = sk * 2 + lk
                    w_coeffs[(gi, gk)] = w_coeffs.get((gi, gk), 0) + w_val * c_val
            u_coeffs = {k: v for k, v in u_coeffs.items() if v != 0}
            v_coeffs = {k: v for k, v in v_coeffs.items() if v != 0}
            w_coeffs = {k: v for k, v in w_coeffs.items() if v != 0}
            terms.append((u_coeffs, v_coeffs, w_coeffs))
    return terms

def _construct_4x1_decomposition():
    n, m, p = 5, 5, 5
    blocks = [[0, 1, 2, 3], [4]]
    s4_terms = _strassen_4x4_terms()
    columns = []
    
    def add_column(u_entries, v_entries, w_entries):
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    def add_strassen_block(i_list, j_list, k_list, terms):
        for u_coeffs, v_coeffs, w_coeffs in terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), val in u_coeffs.items():
                gi = i_list[li]
                gj = j_list[lj]
                u_entries.append((gi * m + gj, val))
            for (lj, lk), val in v_coeffs.items():
                gj = j_list[lj]
                gk = k_list[lk]
                v_entries.append((gj * p + gk, val))
            for (li, lk), val in w_coeffs.items():
                gi = i_list[li]
                gk = k_list[lk]
                w_entries.append((gk * n + gi, val))
            add_column(u_entries, v_entries, w_entries)
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                i_list = blocks[bi]
                j_list = blocks[bj]
                k_list = blocks[bk]
                if len(i_list) == 4 and len(j_list) == 4 and len(k_list) == 4:
                    add_strassen_block(i_list, j_list, k_list, s4_terms)
                else:
                    add_schoolbook_block(i_list, j_list, k_list)
    
    total_rank = len(columns)
    U = np.zeros((25, total_rank))
    V = np.zeros((25, total_rank))
    W = np.zeros((25, total_rank))
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx, c] = val
        for idx, val in v_d.items():
            V[idx, c] = val
        for idx, val in w_d.items():
            W[idx, c] = val
    return U, V, W

def _reduce_rank_fast(U, V, W, n=5, m=5, p=5):
    """Reduce rank by removing linearly dependent columns using SVD-based approach."""
    if _deadline_exceeded():
        return U, V, W
    
    num_cols = U.shape[1]
    if num_cols <= 1:
        return U, V, W
    
    # Build the combined tensor matrix for rank detection
    # For each column c, we have vectors U[:,c], V[:,c], W[:,c]
    # Two columns are equivalent if there exists alpha such that
    # U[:,d] = alpha*U[:,c], V[:,d] = alpha*V[:,c], W[:,d] = alpha*W[:,c]
    
    # Use a hashing approach: for each column, compute a canonical signature
    # based on the normalized direction of the triple (U[:,c], V[:,c], W[:,c])
    
    keep = list(range(num_cols))
    used = set()
    
    # First pass: find exact duplicates (same entries)
    for c in range(num_cols):
        if c in used:
            continue
        for d in range(c+1, num_cols):
            if d in used:
                continue
            if np.allclose(U[:, c], U[:, d]) and np.allclose(V[:, c], V[:, d]) and np.allclose(W[:, c], W[:, d]):
                used.add(d)
    
    # Second pass: find proportional columns
    for c in range(num_cols):
        if c in used:
            continue
        for d in range(c+1, num_cols):
            if d in used:
                continue
            # Find ratio from U
            nonzero_u = np.nonzero(U[:, c])[0]
            if len(nonzero_u) == 0:
                continue
            i0 = nonzero_u[0]
            if U[i0, c] == 0:
                continue
            ratio = U[i0, d] / U[i0, c]
            
            # Check proportionality
            if not np.allclose(U[:, d], ratio * U[:, c], atol=1e-10):
                continue
            if not np.allclose(V[:, d], ratio * V[:, c], atol=1e-10):
                continue
            if not np.allclose(W[:, d], ratio * W[:, c], atol=1e-10):
                continue
            used.add(d)
    
    if used:
        new_cols = [c for c in range(num_cols) if c not in used]
        U_new = U[:, new_cols]
        V_new = V[:, new_cols]
        W_new = W[:, new_cols]
        return U_new, V_new, W_new
    
    return U, V, W

def _verify_decomposition_fast(U, V, W, n=5, m=5, p=5):
    if _deadline_exceeded():
        return False
    
    # Check all valid (i,j,k) triples must give 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-8:
                    return False
    
    # Check invalid triples must give 0
    # Use vectorized check for all invalid triples
    rank = U.shape[1]
    
    # Build all possible (a,b,c) and check
    # Total 25*25*25 = 15625, 125 valid, 15500 invalid
    # Sample a large number
    rng = np.random.RandomState(42)
    samples = 10000
    for _ in range(samples):
        a = rng.randint(0, 25)
        b = rng.randint(0, 25)
        c = rng.randint(0, 25)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        k2 = c // n
        i2 = c % n
        if j == j2 and k == k2 and i == i2:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > 1e-8:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 150.0
    n, m, p = DIMENSIONS
    
    # Fallback: schoolbook
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    
    try:
        if time.time() - start_time < deadline - 80:
            U_block, V_block, W_block = _construct_4x1_decomposition()
            
            if time.time() - start_time < deadline - 40:
                U_red, V_red, W_red = _reduce_rank_fast(U_block, V_block, W_block, n, m, p)
                
                if time.time() - start_time < deadline - 25:
                    valid = _verify_decomposition_fast(U_red, V_red, W_red, n, m, p)
                    if valid:
                        return {
                            "U": U_red.tolist(),
                            "V": V_red.tolist(),
                            "W": W_red.tolist()
                        }
            
            if time.time() - start_time < deadline - 20:
                valid = _verify_decomposition_fast(U_block, V_block, W_block, n, m, p)
                if valid:
                    return {
                        "U": U_block.tolist(),
                        "V": V_block.tolist(),
                        "W": W_block.tolist()
                    }
    except Exception:
        pass
    
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
