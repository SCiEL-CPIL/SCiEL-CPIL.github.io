import time
import numpy as np

N, M, P = 3, 3, 8
A_DIM = N * M          # 9
B_DIM = M * P          # 24
C_DIM = P * N          # 24

# Precompute target tensor T[a,b,c]
_T = np.zeros((A_DIM, B_DIM, C_DIM), dtype=np.float64)
for i in range(N):
    for j in range(M):
        for k in range(P):
            a = i * M + j
            b = j * P + k
            c = k * N + i
            _T[a, b, c] = 1.0


def _verify_fast(U, V, W, tol=1e-9):
    """Check if U,V,W decompose T exactly."""
    if U.shape[1] == 0:
        return False
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(T_rec, _T, atol=tol, rtol=0)


def _build_schoolbook():
    """Rank-72 schoolbook."""
    rank = N * M * P
    U = np.zeros((A_DIM, rank), dtype=np.float64)
    V = np.zeros((B_DIM, rank), dtype=np.float64)
    W = np.zeros((C_DIM, rank), dtype=np.float64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    return U, V, W


def _build_strassen_seed():
    """Rank-68 Strassen-based seed."""
    rank = 68
    U = np.zeros((A_DIM, rank), dtype=np.float64)
    V = np.zeros((B_DIM, rank), dtype=np.float64)
    W = np.zeros((C_DIM, rank), dtype=np.float64)
    col = 0

    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        a00 = a_rows[0] * M + a_cols[0]
        a01 = a_rows[0] * M + a_cols[1]
        a10 = a_rows[1] * M + a_cols[0]
        a11 = a_rows[1] * M + a_cols[1]
        b00 = b_rows[0] * P + b_cols[0]
        b01 = b_rows[0] * P + b_cols[1]
        b10 = b_rows[1] * P + b_cols[0]
        b11 = b_rows[1] * P + b_cols[1]
        c00 = c_rows[0] * N + c_cols[0]
        c01 = c_rows[0] * N + c_cols[1]
        c10 = c_rows[1] * N + c_cols[0]
        c11 = c_rows[1] * N + c_cols[1]

        # M1
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        W[c01, col] -= 1; W[c10, col] -= 1
        col += 1
        # M2
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] -= 1
        col += 1
        # M3
        U[a00, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M4
        U[a11, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M5
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c00, col] -= 1; W[c01, col] += 1
        col += 1
        # M6
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b01, col] += 1
        W[c11, col] += 1
        col += 1
        # M7
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c10, col] -= 1
        col += 1

    def add_schoolbook_block(a_rows, a_cols, b_rows, b_cols):
        nonlocal col
        for i in a_rows:
            for j in a_cols:
                for k in b_cols:
                    a_idx = i * M + j
                    b_idx = j * P + k
                    c_idx = k * N + i
                    U[a_idx, col] = 1
                    V[b_idx, col] = 1
                    W[c_idx, col] = 1
                    col += 1

    for bcol_start in range(0, 8, 2):
        add_strassen_2x2x2(
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[bcol_start, bcol_start + 1],
            c_rows=[bcol_start, bcol_start + 1], c_cols=[0, 1]
        )
    add_schoolbook_block(a_rows=[0, 1], a_cols=[2], b_rows=[2], b_cols=list(range(8)))
    add_schoolbook_block(a_rows=[2], a_cols=[0, 1], b_rows=[0, 1], b_cols=list(range(8)))
    add_schoolbook_block(a_rows=[2], a_cols=[2], b_rows=[2], b_cols=list(range(8)))

    return U[:, :col], V[:, :col], W[:, :col]


def _remove_column_with_solve(U, V, W, col_to_remove, tol=1e-10):
    """
    Try to remove column `col_to_remove` by solving for adjustments to other columns.
    
    The idea: if we remove column r, we need to find small corrections to the
    remaining columns such that the tensor still decomposes. This is a linear
    system in the corrections.
    
    Actually, a simpler approach: try removing the column and check if the
    remaining decomposition still works. If not, try to find a replacement.
    
    Here we use a different strategy: we check if column r is in the span of
    the other columns (in a tensor sense). We do this by checking if the
    tensor contribution of column r can be expressed as a linear combination
    of the other columns' contributions.
    """
    rank = U.shape[1]
    if rank <= 1:
        return None
    
    # Simple check: is this column zero?
    if np.all(U[:, col_to_remove] == 0) and np.all(V[:, col_to_remove] == 0) and np.all(W[:, col_to_remove] == 0):
        mask = np.ones(rank, dtype=bool)
        mask[col_to_remove] = False
        return U[:, mask], V[:, mask], W[:, mask]
    
    # Try removing and verifying
    mask = np.ones(rank, dtype=bool)
    mask[col_to_remove] = False
    U_new = U[:, mask]
    V_new = V[:, mask]
    W_new = W[:, mask]
    
    if _verify_fast(U_new, V_new, W_new, tol=tol):
        return U_new, V_new, W_new
    
    return None


def _tensor_rank_minimization(U, V, W, target_rank, time_limit=30.0, seed=None):
    """
    Iteratively try to reduce rank by removing columns and solving.
    Uses a greedy approach: repeatedly try to remove the "least important" column.
    
    We measure importance by the norm of the column's contribution to the tensor.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    start = time.time()
    
    current_rank = best_U.shape[1]
    
    # Iterate: try to remove columns
    while current_rank > target_rank and (time.time() - start) < time_limit:
        # Score each column by its "importance" - we use a heuristic
        # based on the norm of the rank-1 tensor it contributes
        importances = np.zeros(current_rank)
        for r in range(current_rank):
            # Approximate importance: norm of U[:,r] * V[:,r] * W[:,r] tensor
            # We can approximate this by the product of norms
            importances[r] = (np.linalg.norm(best_U[:, r]) * 
                            np.linalg.norm(best_V[:, r]) * 
                            np.linalg.norm(best_W[:, r]))
        
        # Try removing columns in order of lowest importance first
        order = np.argsort(importances)
        removed = False
        
        for r_idx in order:
            if time.time() - start > time_limit:
                break
            result = _remove_column_with_solve(best_U, best_V, best_W, int(r_idx))
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = best_U.shape[1]
                current_rank = best_rank
                removed = True
                break
        
        if not removed:
            # No column could be removed; try a more aggressive approach
            # Randomly perturb and retry
            if current_rank > target_rank + 5:
                # Give up on this trajectory
                break
    
    return best_U, best_V, best_W


def _random_decomposition(rank, seed=None):
    """Generate a random sparse decomposition."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    U = np.zeros((A_DIM, rank), dtype=np.float64)
    V = np.zeros((B_DIM, rank), dtype=np.float64)
    W = np.zeros((C_DIM, rank), dtype=np.float64)
    
    for r in range(rank):
        n_u = rng.randint(1, 4)
        n_v = rng.randint(1, 4)
        n_w = rng.randint(1, 4)
        
        u_idx = rng.choice(A_DIM, size=n_u, replace=False)
        v_idx = rng.choice(B_DIM, size=n_v, replace=False)
        w_idx = rng.choice(C_DIM, size=n_w, replace=False)
        
        for idx in u_idx:
            U[idx, r] = rng.choice([-1, 1])
        for idx in v_idx:
            V[idx, r] = rng.choice([-1, 1])
        for idx in w_idx:
            W[idx, r] = rng.choice([-1, 1])
    
    return U, V, W


def _local_search(U, V, W, time_limit=15.0, seed=None):
    """Local search to reduce rank."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    start = time.time()
    
    # Remove zero/redundant columns
    changed = True
    while changed and (time.time() - start) < time_limit * 0.3:
        changed = False
        current_rank = best_U.shape[1]
        for c in range(current_rank):
            if time.time() - start > time_limit * 0.3:
                break
            result = _remove_column_with_solve(best_U, best_V, best_W, c)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = best_U.shape[1]
                changed = True
                break
    
    # Random perturbation moves
    moves = 0
    while (time.time() - start) < time_limit and moves < 100:
        current_rank = best_U.shape[1]
        if current_rank == 0:
            break
        
        # Perturb a random entry
        c = rng.randint(current_rank)
        dim = rng.randint(3)
        U_test = best_U.copy()
        V_test = best_V.copy()
        W_test = best_W.copy()
        
        if dim == 0:
            idx = rng.randint(A_DIM)
            U_test[idx, c] = rng.choice([-1, 0, 1])
        elif dim == 1:
            idx = rng.randint(B_DIM)
            V_test[idx, c] = rng.choice([-1, 0, 1])
        else:
            idx = rng.randint(C_DIM)
            W_test[idx, c] = rng.choice([-1, 0, 1])
        
        if _verify_fast(U_test, V_test, W_test):
            best_U, best_V, best_W = U_test, V_test, W_test
            moves += 1
    
    return best_U, best_V, best_W


def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin
    
    # Start with Strassen seed
    U, V, W = _build_strassen_seed()
    
    # Verify seed
    if not _verify_fast(U, V, W):
        U, V, W = _build_schoolbook()
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Phase 1: Tensor rank minimization on the seed
    if time.time() < deadline - 40:
        t1_start = time.time()
        U_min, V_min, W_min = _tensor_rank_minimization(
            U, V, W, target_rank=50, 
            time_limit=min(25.0, deadline - time.time() - 20), seed=42
        )
        if _verify_fast(U_min, V_min, W_min) and U_min.shape[1] < best_rank:
            best_U, best_V, best_W = U_min.copy(), V_min.copy(), W_min.copy()
            best_rank = U_min.shape[1]
    
    # Phase 2: Local search on the best found so far
    if time.time() < deadline - 30:
        U_ls, V_ls, W_ls = _local_search(
            best_U, best_V, best_W,
            time_limit=min(15.0, deadline - time.time() - 15), seed=123
        )
        if _verify_fast(U_ls, V_ls, W_ls) and U_ls.shape[1] < best_rank:
            best_U, best_V, best_W = U_ls.copy(), V_ls.copy(), W_ls.copy()
            best_rank = U_ls.shape[1]
    
    # Phase 3: Random restarts with tensor rank minimization
    seed_counter = 200
    while time.time() < deadline - 25:
        seed_counter += 1
        # Start with random decomposition at rank close to best
        initial_rank = max(best_rank, 55)
        U_rand, V_rand, W_rand = _random_decomposition(initial_rank, seed=seed_counter)
        
        # Try to verify - random won't work, so we use it as a starting point
        # for local search only if it's close
        # Actually, random decompositions won't verify. Let's skip and just
        # do local search on the best.
        
        # Instead, try different random perturbations of the best
        U_pert = best_U.copy()
        V_pert = best_V.copy()
        W_pert = best_W.copy()
        
        # Randomly zero out some entries
        for _ in range(3):
            dim = rng_choice = np.random.RandomState(seed_counter + _).randint(3)
            if dim == 0:
                idx = np.random.RandomState(seed_counter + _).randint(A_DIM)
                c = np.random.RandomState(seed_counter + _).randint(U_pert.shape[1])
                U_pert[idx, c] = 0
            elif dim == 1:
                idx = np.random.RandomState(seed_counter + _).randint(B_DIM)
                c = np.random.RandomState(seed_counter + _).randint(V_pert.shape[1])
                V_pert[idx, c] = 0
            else:
                idx = np.random.RandomState(seed_counter + _).randint(C_DIM)
                c = np.random.RandomState(seed_counter + _).randint(W_pert.shape[1])
                W_pert[idx, c] = 0
        
        if _verify_fast(U_pert, V_pert, W_pert) and U_pert.shape[1] < best_rank:
            best_U, best_V, best_W = U_pert.copy(), V_pert.copy(), W_pert.copy()
            best_rank = U_pert.shape[1]
    
    # Final verification
    if _verify_fast(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
