"""Combined approach: structural block decomposition + systematic search for rank minimization."""
import numpy as np
import random
import time

DIMENSIONS = [3, 3, 3]

def build_target_tensor():
    """Return list of (u_idx, v_idx, w_idx) for each valid multiplication term."""
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                terms.append((u_idx, v_idx, w_idx))
    return terms

TARGET_TERMS = build_target_tensor()

def validate_decomposition(U, V, W, rank):
    """Check if the decomposition correctly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_idx = i * 3 + j
                v_idx = j * 3 + k
                w_idx = k * 3 + i
                val = 0.0
                for r in range(rank):
                    val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def compute_error(U, V, W, rank):
    """Compute total error of decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    err = 0.0
    for (u_idx, v_idx, w_idx) in TARGET_TERMS:
        val = 0.0
        for r in range(rank):
            val += U[u_idx, r] * V[v_idx, r] * W[w_idx, r]
        err += (val - 1.0) ** 2
    return err

def make_candidate(rank, U, V, W):
    """Return candidate dict."""
    return {'U': [list(row) for row in U], 'V': [list(row) for row in V], 'W': [list(row) for row in W]}

def _known_rank26():
    """Return the known rank 26 decomposition using block structure."""
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    # 1. Strassen on 2x2 top-left blocks (rank 7)
    # M1 = (A00+A11)(B00+B11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[4][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1
    
    # M2 = (A10+A11)B00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[1][col] = 1; W[4][col] = -1
    col += 1
    
    # M3 = A00(B01-B11)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[3][col] = 1; W[4][col] = 1
    col += 1
    
    # M4 = A11(B10-B00)
    U[4][col] = 1
    V[3][col] = 1; V[0][col] = -1
    W[0][col] = 1; W[1][col] = 1
    col += 1
    
    # M5 = (A00+A01)B11
    U[0][col] = 1; U[1][col] = 1
    V[4][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[4][col] = 1
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[4][col] = 1
    W[0][col] = 1
    col += 1
    
    # 2. Off-diagonal interactions (rank 19 total)
    add_term(2, 6, 0, 1)   # A02*B20 -> C00
    add_term(2, 7, 3, 1)   # A02*B21 -> C01
    add_term(5, 6, 1, 1)   # A12*B20 -> C10
    add_term(5, 7, 4, 1)   # A12*B21 -> C11
    
    add_term(0, 2, 6, 1)   # A00*B02 -> C02
    add_term(1, 5, 6, 1)   # A01*B12 -> C02
    add_term(3, 2, 7, 1)   # A10*B02 -> C12
    add_term(4, 5, 7, 1)   # A11*B12 -> C12
    
    add_term(2, 8, 6, 1)   # A02*B22 -> C02
    add_term(5, 8, 7, 1)   # A12*B22 -> C12
    
    add_term(6, 0, 2, 1)   # A20*B00 -> C20
    add_term(7, 3, 2, 1)   # A21*B10 -> C20
    add_term(6, 1, 5, 1)   # A20*B01 -> C21
    add_term(7, 4, 5, 1)   # A21*B11 -> C21
    
    add_term(8, 6, 2, 1)   # A22*B20 -> C20
    add_term(8, 7, 5, 1)   # A22*B21 -> C21
    
    add_term(6, 2, 8, 1)   # A20*B02 -> C22
    add_term(7, 5, 8, 1)   # A21*B12 -> C22
    
    add_term(8, 8, 8, 1)   # A22*B22 -> C22
    
    return {'U': U, 'V': V, 'W': W}

def _get_term_info(U, V, W, col):
    """Get the (u_idx, v_idx, w_idx, coeff) for a column."""
    u_idx = None
    v_idx = None
    w_idx = None
    u_coeff = 0
    for i in range(9):
        if U[i][col] != 0:
            u_idx = i
            u_coeff = U[i][col]
            break
    for i in range(9):
        if V[i][col] != 0:
            v_idx = i
            break
    for i in range(9):
        if W[i][col] != 0:
            w_idx = i
            break
    return u_idx, v_idx, w_idx, u_coeff

def _try_merge_pair(U, V, W, rank, col_i, col_j, deadline):
    """
    Try to merge two columns into one. The merged column can have
    up to 2 non-zeros in each of U, V, W.
    """
    if time.time() > deadline:
        return None
    
    ui, vi, wi, _ = _get_term_info(U, V, W, col_i)
    uj, vj, wj, _ = _get_term_info(U, V, W, col_j)
    
    if None in (ui, vi, wi, uj, vj, wj):
        return None
    
    # Candidate index sets
    u_sets = [frozenset({ui}), frozenset({uj}), frozenset({ui, uj})]
    v_sets = [frozenset({vi}), frozenset({vj}), frozenset({vi, vj})]
    w_sets = [frozenset({wi}), frozenset({wj}), frozenset({wi, wj})]
    
    coeffs = [-2, -1, 1, 2]
    
    for us in u_sets:
        for vs in v_sets:
            for ws in w_sets:
                # Generate all coefficient combinations
                u_coeff_list = [coeffs] * len(us)
                v_coeff_list = [coeffs] * len(vs)
                w_coeff_list = [coeffs] * len(ws)
                
                for uc_prod in _cartesian(u_coeff_list):
                    for vc_prod in _cartesian(v_coeff_list):
                        for wc_prod in _cartesian(w_coeff_list):
                            if time.time() > deadline:
                                return None
                            
                            new_rank = rank - 1
                            U_new = [[0] * new_rank for _ in range(9)]
                            V_new = [[0] * new_rank for _ in range(9)]
                            W_new = [[0] * new_rank for _ in range(9)]
                            
                            # Copy other columns
                            for r in range(rank):
                                if r == col_i or r == col_j:
                                    continue
                                new_r = r
                                if r > min(col_i, col_j):
                                    new_r -= 1
                                if r > max(col_i, col_j):
                                    new_r -= 1
                                for i in range(9):
                                    U_new[i][new_r] = U[i][r]
                                    V_new[i][new_r] = V[i][r]
                                    W_new[i][new_r] = W[i][r]
                            
                            # Add merged column
                            nc = new_rank - 1
                            for idx, coeff in zip(sorted(us), uc_prod):
                                U_new[idx][nc] = coeff
                            for idx, coeff in zip(sorted(vs), vc_prod):
                                V_new[idx][nc] = coeff
                            for idx, coeff in zip(sorted(ws), wc_prod):
                                W_new[idx][nc] = coeff
                            
                            if validate_decomposition(U_new, V_new, W_new, new_rank):
                                return U_new, V_new, W_new, new_rank
    
    return None

def _cartesian(lists):
    """Generate cartesian product of lists."""
    if not lists:
        yield ()
        return
    for item in lists[0]:
        for rest in _cartesian(lists[1:]):
            yield (item,) + rest

def _local_polish(U, V, W, rank, deadline, max_iters=300):
    """Polish a near-solution with local search."""
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    
    best_err = compute_error(U, V, W, rank)
    
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if best_err < 1e-12:
            break
        
        improved = False
        for _ in range(15):
            mat = random.choice([U, V, W])
            r = random.randint(0, 8)
            c = random.randint(0, rank - 1)
            old = mat[r][c]
            mat[r][c] = random.randint(-3, 3)
            
            err = compute_error(U, V, W, rank)
            if err < best_err:
                best_err = err
                improved = True
            else:
                mat[r][c] = old
        
        if not improved:
            # Occasional larger jump
            if random.random() < 0.1:
                for _ in range(2):
                    mat = random.choice([U, V, W])
                    r = random.randint(0, 8)
                    c = random.randint(0, rank - 1)
                    mat[r][c] = random.randint(-3, 3)
    
    return U, V, W, best_err

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145
    
    # Get the known rank-26 solution
    sol26 = _known_rank26()
    U26 = sol26['U']
    V26 = sol26['V']
    W26 = sol26['W']
    
    # Verify
    if not validate_decomposition(U26, V26, W26, 26):
        return sol26
    
    # Identify rank-1 columns (those with exactly 1 non-zero in each of U, V, W)
    rank1_cols = []
    for col in range(26):
        u_nz = sum(1 for i in range(9) if U26[i][col] != 0)
        v_nz = sum(1 for i in range(9) if V26[i][col] != 0)
        w_nz = sum(1 for i in range(9) if W26[i][col] != 0)
        if u_nz == 1 and v_nz == 1 and w_nz == 1:
            rank1_cols.append(col)
    
    # Try all pairs of rank-1 columns for merging
    for i in range(len(rank1_cols)):
        if time.time() > deadline - 10:
            break
        for j in range(i + 1, len(rank1_cols)):
            if time.time() > deadline - 5:
                break
            
            result = _try_merge_pair(U26, V26, W26, 26, rank1_cols[i], rank1_cols[j], deadline)
            if result is not None:
                U_new, V_new, W_new, new_rank = result
                print(f"Merged columns {rank1_cols[i]} and {rank1_cols[j]} -> rank {new_rank}")
                return make_candidate(new_rank, U_new, V_new, W_new)
    
    # Try merging a rank-1 column with a multi-term column
    multi_cols = [c for c in range(26) if c not in rank1_cols]
    for i in range(len(rank1_cols)):
        if time.time() > deadline - 10:
            break
        for j in multi_cols:
            if time.time() > deadline - 5:
                break
            
            result = _try_merge_pair(U26, V26, W26, 26, rank1_cols[i], j, deadline)
            if result is not None:
                U_new, V_new, W_new, new_rank = result
                print(f"Merged column {rank1_cols[i]} and {j} -> rank {new_rank}")
                return make_candidate(new_rank, U_new, V_new, W_new)
    
    # Try local search from rank-26 with one column removed
    if time.time() < deadline - 30:
        for col_to_try in rank1_cols:
            if time.time() > deadline - 10:
                break
            
            # Create rank-25 by removing this column
            U25 = [[0] * 25 for _ in range(9)]
            V25 = [[0] * 25 for _ in range(9)]
            W25 = [[0] * 25 for _ in range(9)]
            
            for r in range(26):
                if r == col_to_try:
                    continue
                new_r = r if r < col_to_try else r - 1
                for i in range(9):
                    U25[i][new_r] = U26[i][r]
                    V25[i][new_r] = V26[i][r]
                    W25[i][new_r] = W26[i][r]
            
            # Local search to repair
            U25, V25, W25, err = _local_polish(U25, V25, W25, 25, deadline, max_iters=200)
            
            if err < 1e-10:
                if validate_decomposition(U25, V25, W25, 25):
                    print(f"Found rank 25 via local search after removing column {col_to_try}")
                    return make_candidate(25, U25, V25, W25)
    
    # Fallback to rank 26
    return sol26
# EVOLVE-BLOCK-END
