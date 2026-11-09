import numpy as np
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)


def _construct_schoolbook(n, m, p):
    """
    Build the standard (schoolbook) tensor decomposition for multiplying
    an n x m matrix by an m x p matrix.
    """
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
    """
    Return the 7 rank-1 terms of Strassen's 2x2x2 decomposition.
    Each term is (u_coeffs, v_coeffs, w_coeffs) where each coeffs is a dict
    mapping (local_row, local_col) -> coefficient.
    """
    return [
        # M1 = (A11+A22)*(B11+B22)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A21+A22)*B11
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A11*(B12-B22)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A22*(B21-B11)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A11+A12)*B22
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (A21-A11)*(B11+B12)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A12-A22)*(B21+B22)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]


def _strassen_4x4_terms():
    """
    Return the 49 rank-1 terms of the recursive Strassen 4x4x4 decomposition.
    Each term is (u_coeffs, v_coeffs, w_coeffs) where each coeffs is a dict
    mapping (local_row, local_col) -> coefficient, with rows/cols in 0..3.
    """
    s2 = _strassen_2x2_terms()
    
    # The 7 macro terms for 4x4 Strassen, same structure as 2x2 but at block level
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
            
            # Remove zero entries
            u_coeffs = {k: v for k, v in u_coeffs.items() if v != 0}
            v_coeffs = {k: v for k, v in v_coeffs.items() if v != 0}
            w_coeffs = {k: v for k, v in w_coeffs.items() if v != 0}
            
            terms.append((u_coeffs, v_coeffs, w_coeffs))
    
    return terms


def _construct_4x1_decomposition():
    """
    Build the 4+1 block decomposition for 5x5x5 matrix multiplication.
    
    Partition: block 0 = indices [0,1,2,3] (size 4), block 1 = index [4] (size 1).
    
    For the (0,0,0) block (4x4x4), use recursive Strassen (rank 49).
    For all other blocks, use schoolbook.
    
    Total rank = 49 + (125 - 64) = 49 + 61 = 110.
    """
    n, m, p = 5, 5, 5
    
    blocks = [[0, 1, 2, 3], [4]]
    
    # Precompute Strassen terms
    s4_terms = _strassen_4x4_terms()
    s2_terms = _strassen_2x2_terms()
    
    # Collect all columns as sparse dicts
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
        """
        Add a Strassen decomposition for a block where i_list, j_list, k_list
        are lists of global indices of the same length as the local size of terms.
        """
        local_size = len(i_list)
        for u_coeffs, v_coeffs, w_coeffs in terms:
            u_entries = []
            v_entries = []
            w_entries = {}
            
            for (li, lj), val in u_coeffs.items():
                gi = i_list[li]
                gj = j_list[lj]
                u_entries.append((gi * m + gj, val))
            
            v_entries = []
            for (lj, lk), val in v_coeffs.items():
                gj = j_list[lj]
                gk = k_list[lk]
                v_entries.append((gj * p + gk, val))
            
            w_entries = []
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
                    # Use Strassen 4x4x4
                    add_strassen_block(i_list, j_list, k_list, s4_terms)
                elif len(i_list) == 2 and len(j_list) == 2 and len(k_list) == 2:
                    # Use Strassen 2x2x2
                    add_strassen_block(i_list, j_list, k_list, s2_terms)
                else:
                    # Schoolbook
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


def _verify_decomposition(U, V, W, n=5, m=5, p=5):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    Returns True if valid, False otherwise.
    """
    # The tensor T for n x m x p matrix mult has:
    # T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # and T[a, b, c] = 0 otherwise.
    
    # Check all entries efficiently
    # For each (a, b, c) where the tensor should be 1:
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a, :] * V[b, :], W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check some entries where the tensor should be 0
    # We check all entries that are not of the form (i*m+j, j*p+k, k*n+i)
    # There are 25*25*25 = 15625 total entries, and n*m*p = 125 should be 1.
    # So 15500 should be 0. Let's check a sample.
    
    # Actually, let's check all of them for correctness
    for a in range(25):
        for b in range(25):
            for c in range(25):
                # Determine if this is a valid tensor entry
                i = a // m
                j = a % m
                j2 = b // p
                k = b % p
                k2 = c // n
                i2 = c % n
                
                if j == j2 and k == k2 and i == i2:
                    # Should be 1
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
                else:
                    # Should be 0
                    val = np.dot(U[a, :] * V[b, :], W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _verify_decomposition_fast(U, V, W, n=5, m=5, p=5):
    """
    Faster verification using vectorized operations.
    """
    # Build the expected tensor
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    # Otherwise 0
    
    # The decomposition gives T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # We can compute this as: for each r, T += outer(U[:,r], V[:,r]) * W[:,r]
    # But that's O(rank * 25^3) which is too slow.
    
    # Instead, let's just check the 125 entries that should be 1,
    # and a random sample of entries that should be 0.
    
    # Check all 125 entries that should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check a random sample of entries that should be 0
    rng = np.random.RandomState(12345)
    samples = 5000
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
        
        val = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(val) > 1e-9:
            return False
    
    return True


def construct():
    start_time = time.time()
    deadline = 120.0  # seconds
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a fallback
    U_school, V_school, W_school = _construct_schoolbook(n, m, p)
    
    # Try the 4+1 block decomposition with recursive Strassen
    try:
        if time.time() - start_time < deadline - 60:
            U_block, V_block, W_block = _construct_4x1_decomposition()
            rank = U_block.shape[1]
            
            # Verify the decomposition
            if time.time() - start_time < deadline - 30:
                valid = _verify_decomposition_fast(U_block, V_block, W_block, n, m, p)
                if valid:
                    return {
                        "U": U_block.tolist(),
                        "V": V_block.tolist(),
                        "W": W_block.tolist()
                    }
    except Exception:
        pass
    
    # Fallback to schoolbook
    return {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }

# EVOLVE-BLOCK-END
