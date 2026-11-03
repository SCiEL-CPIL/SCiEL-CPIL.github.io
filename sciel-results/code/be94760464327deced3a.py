"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    block_size = 2
    n_blocks = n // block_size  # 3

    # Rank-189 block Strassen: 3x3 blocks of 2x2, each 2x2 uses rank-7 Strassen
    rank = n_blocks ** 3 * 7

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    # Index helpers
    def u_idx(ib, kb, r, c):
        return (ib * block_size + r) * m + (kb * block_size + c)

    def v_idx(kb, jb, r, c):
        return (kb * block_size + r) * p + (jb * block_size + c)

    def w_idx(ib, jb, r, c):
        return (jb * block_size + c) * n + (ib * block_size + r)

    # Strassen 2x2 multiplication terms:
    # M1 = (A00+A11)(B00+B11) -> C00+C11
    # M2 = (A10+A11)(B00)     -> C10+C11
    # M3 = (A00)(B01-B11)     -> C01+C11
    # M4 = (A11)(B10-B00)     -> C00+C10
    # M5 = (A00+A01)(B11)     -> C00-C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00

    # Each term: list of (u_entries, v_entries, w_entries)
    # u_entries: [(r, c, coeff), ...] for A
    # v_entries: [(r, c, coeff), ...] for B
    # w_entries: [(r, c, coeff), ...] for C
    strassen_terms = [
        # M1
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    for ib in range(n_blocks):
        for jb in range(n_blocks):
            for kb in range(n_blocks):
                for u_entries, v_entries, w_entries in strassen_terms:
                    for (r, c, coeff) in u_entries:
                        U[u_idx(ib, kb, r, c), col] += coeff
                    for (r, c, coeff) in v_entries:
                        V[v_idx(kb, jb, r, c), col] += coeff
                    for (r, c, coeff) in w_entries:
                        W[w_idx(ib, jb, r, c), col] += coeff
                    col += 1

    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Attempt rank reduction by identifying linearly dependent columns
    # The tensor is T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
    # We need to find the minimal set of columns that still span the tensor
    
    # Build the full tensor to verify
    def build_tensor(U_mat, V_mat, W_mat):
        rank_t = U_mat.shape[1]
        T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for r in range(rank_t):
            T += np.einsum('i,j,k->ijk', U_mat[:,r], V_mat[:,r], W_mat[:,r])
        return T

    # Verify initial decomposition
    T_target = build_tensor(U, V, W)
    
    # Check if decomposition is valid: T[i,j,k] should equal sum over (a,b) where 
    # i=(a, j_idx), j=(j_idx, b), k=(b, a) in appropriate block structure
    # For matrix mult: C[i,j] = sum_k A[i,k]*B[k,j]
    # In our indexing: U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # W[k*n+i, r] corresponds to C[i,j] where k is the B row, i is A row, j is B col
    # So T[i*m + k, k*p + j, j*n + i] should = 1 for the matrix mult tensor
    
    # Actually, let's just try greedy column elimination
    # Keep a set of active columns, try removing each one and check validity
    
    active_cols = list(range(rank))
    
    # Greedy removal: try to remove columns one by one
    removed = []
    for trial in range(rank):
        if time.time() > deadline - 20:
            break
        if not active_cols:
            break
        
        # Try removing each active column
        best_remove = None
        for idx in active_cols:
            test_active = [c for c in active_cols if c != idx]
            if not test_active:
                break
            
            # Build tensor with only test_active columns
            T_test = build_tensor(U[:, test_active], V[:, test_active], W[:, test_active])
            
            # Check if T_test == T_target
            if np.array_equal(T_test, T_target):
                best_remove = idx
                break
        
        if best_remove is not None:
            active_cols.remove(best_remove)
            removed.append(best_remove)
        else:
            break
    
    if removed:
        U_final = U[:, active_cols]
        V_final = V[:, active_cols]
        W_final = W[:, active_cols]
    else:
        U_final = U
        V_final = V
        W_final = W

    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
# EVOLVE-BLOCK-END
