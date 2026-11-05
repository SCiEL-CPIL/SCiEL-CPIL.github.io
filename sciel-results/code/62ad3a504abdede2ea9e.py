import time
import numpy as np

DIMENSIONS = [3, 4, 5]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 115  # Safety margin

def _time_left():
    return DEADLINE - time.time()

def _verify(U, V, W, n=N, m=M, p=P):
    """Check exact decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    rank = U.shape[1]
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9, rtol=0)

def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition."""
    U = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, 0, 0, 0, 1],
    ], dtype=np.float64)
    V = np.array([
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, -1, 1, 0, 0, 1],
        [1, 0, 1, 0, 0, 1, 0],
        [1, 1, 0, -1, 1, 0, 0],
    ], dtype=np.float64)
    W = np.array([
        [1, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, -1, 0, 0],
    ], dtype=np.float64)
    return U, V, W

def _construct_structured(n, m, p):
    """
    Construct a structured decomposition using Strassen blocks and lower-rank blocks.
    Returns U, V, W as numpy arrays of shape (n*m, rank), (m*p, rank), (p*n, rank).
    """
    U_cols = []
    V_cols = []
    W_cols = []

    def add_strassen_block(i_off, j_off, k_off):
        U_s, V_s, W_s = _strassen_2x2x2()
        for c in range(7):
            U_col = np.zeros(n * m)
            V_col = np.zeros(m * p)
            W_col = np.zeros(p * n)
            for ii in range(2):
                for jj in range(2):
                    U_col[(i_off + ii) * m + (j_off + jj)] = U_s[ii * 2 + jj, c]
            for jj in range(2):
                for kk in range(2):
                    V_col[(j_off + jj) * p + (k_off + kk)] = V_s[jj * 2 + kk, c]
            for kk in range(2):
                for ii in range(2):
                    W_col[(k_off + kk) * n + (i_off + ii)] = W_s[kk * 2 + ii, c]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    def add_rank1_block(i, j, k):
        U_col = np.zeros(n * m)
        V_col = np.zeros(m * p)
        W_col = np.zeros(p * n)
        U_col[i * m + j] = 1.0
        V_col[j * p + k] = 1.0
        W_col[k * n + i] = 1.0
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)

    def add_rank2_block_1x2x2(i, j0, j1, k0, k1):
        """Rank-2 block for 1x2x2 tensor at (i, {j0,j1}, {k0,k1})."""
        for c in range(2):
            U_col = np.zeros(n * m)
            V_col = np.zeros(m * p)
            W_col = np.zeros(p * n)
            if c == 0:
                U_col[i * m + j0] = 1.0
                V_col[j0 * p + k0] = 1.0
                V_col[j0 * p + k1] = 1.0
                W_col[k0 * n + i] = 1.0
                W_col[k1 * n + i] = 1.0
            else:
                U_col[i * m + j1] = 1.0
                V_col[j1 * p + k0] = 1.0
                V_col[j1 * p + k1] = 1.0
                W_col[k0 * n + i] = 1.0
                W_col[k1 * n + i] = 1.0
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)

    # For n=3, m=4, p=5:
    # i in {0,1}: use Strassen on 2x2x2 blocks
    # i=2: use rank-2 blocks for 1x2x2

    # i=0:
    add_strassen_block(0, 0, 0)  # j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 2)  # j in {0,1}, k in {2,3}
    add_strassen_block(0, 2, 0)  # j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 2)  # j in {2,3}, k in {2,3}
    # k=4: j in {0,1} and j in {2,3}
    add_rank1_block(0, 0, 4)
    add_rank1_block(0, 1, 4)
    add_rank1_block(0, 2, 4)
    add_rank1_block(0, 3, 4)

    # i=1:
    add_strassen_block(1, 0, 0)
    add_strassen_block(1, 0, 2)
    add_strassen_block(1, 2, 0)
    add_strassen_block(1, 2, 2)
    add_rank1_block(1, 0, 4)
    add_rank1_block(1, 1, 4)
    add_rank1_block(1, 2, 4)
    add_rank1_block(1, 3, 4)

    # i=2:
    # j in {0,1}, k in {0,1}: rank 2
    add_rank2_block_1x2x2(2, 0, 1, 0, 1)
    # j in {0,1}, k in {2,3}: rank 2
    add_rank2_block_1x2x2(2, 0, 1, 2, 3)
    # j in {2,3}, k in {0,1}: rank 2
    add_rank2_block_1x2x2(2, 2, 3, 0, 1)
    # j in {2,3}, k in {2,3}: rank 2
    add_rank2_block_1x2x2(2, 2, 3, 2, 3)
    # k=4: j in {0,1} and j in {2,3}
    add_rank1_block(2, 0, 4)
    add_rank1_block(2, 1, 4)
    add_rank1_block(2, 2, 4)
    add_rank1_block(2, 3, 4)

    rank = len(U_cols)
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    return U, V, W

def _search_rank_from_init(U_init, V_init, W_init, rank, seed, time_budget):
    """
    Search for integer solution starting from initial guess.
    Returns U, V, W (halves of integer values) if found.
    """
    np.random.seed(seed)
    n, m, p = N, M, P
    
    # Convert to integer (multiply by 2)
    U_p = (U_init * 2).round().astype(np.int64).reshape(-1, rank)
    V_p = (V_init * 2).round().astype(np.int64).reshape(-1, rank)
    W_p = (W_init * 2).round().astype(np.int64).reshape(-1, rank)
    
    U3 = U_p.reshape(n, m, rank)
    V3 = V_p.reshape(m, p, rank)
    W3 = W_p.reshape(p, n, rank)
    
    target = 8
    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    E = np.sum((S - target) ** 2)
    
    if E == 0:
        return U_p / 2.0, V_p / 2.0, W_p / 2.0

    # Precompute partials
    P_U = np.einsum('jkr,kir->ijrk', V3, W3)
    P_V = np.einsum('ijr,kir->jkri', U3, W3)
    P_W = np.einsum('ijr,jkr->kirj', U3, V3)
    
    def get_delta_E_U(i, j, r, delta):
        S_ijk = S[i, j, :]
        delta_S = delta * P_U[i, j, r, :]
        new_S = S_ijk + delta_S
        old_err = (S_ijk - target) ** 2
        new_err = (new_S - target) ** 2
        return np.sum(new_err - old_err)

    def get_delta_E_V(j, k, r, delta):
        S_ijk = S[:, j, k]
        delta_S = delta * P_V[j, k, r, :]
        new_S = S_ijk + delta_S
        old_err = (S_ijk - target) ** 2
        new_err = (new_S - target) ** 2
        return np.sum(new_err - old_err)

    def get_delta_E_W(k, i, r, delta):
        S_ijk = S[i, :, k]
        delta_S = delta * P_W[k, i, r, :]
        new_S = S_ijk + delta_S
        old_err = (S_ijk - target) ** 2
        new_err = (new_S - target) ** 2
        return np.sum(new_err - old_err)

    def apply_U(i, j, r, delta):
        U3[i, j, r] += delta
        U_p[i * m + j, r] += delta
        S[i, j, :] += delta * P_U[i, j, r, :]
        P_V[:, :, r, i] += delta * W3[:, i, r]
        P_W[:, i, r, j] += delta * V3[j, :, r]

    def apply_V(j, k, r, delta):
        V3[j, k, r] += delta
        V_p[j * p + k, r] += delta
        S[:, j, k] += delta * P_V[j, k, r, :]
        P_U[:, j, r, k] += delta * W3[k, :, r]
        P_W[k, :, r, j] += delta * U3[:, j, r]

    def apply_W(k, i, r, delta):
        W3[k, i, r] += delta
        W_p[k * n + i, r] += delta
        S[i, :, k] += delta * P_W[k, i, r, :]
        P_U[i, :, r, k] += delta * V3[:, k, r]
        P_V[:, k, r, i] += delta * U3[i, :, r]

    deltas = [-2, -1, 1, 2]
    stuck_count = 0
    max_stuck = 500
    
    while _time_left() > 1.0:
        if E == 0:
            return U_p / 2.0, V_p / 2.0, W_p / 2.0
            
        choice = np.random.randint(3)
        best_delta = 0
        best_delta_E = 0
        best_apply = None
        
        if choice == 0:
            i = np.random.randint(n)
            j = np.random.randint(m)
            r = np.random.randint(rank)
            for d in deltas:
                dE = get_delta_E_U(i, j, r, d)
                if dE < best_delta_E:
                    best_delta_E = dE
                    best_delta = d
            if best_delta != 0:
                best_apply = lambda: apply_U(i, j, r, best_delta)
        elif choice == 1:
            j = np.random.randint(m)
            k = np.random.randint(p)
            r = np.random.randint(rank)
            for d in deltas:
                dE = get_delta_E_V(j, k, r, d)
                if dE < best_delta_E:
                    best_delta_E = dE
                    best_delta = d
            if best_delta != 0:
                best_apply = lambda: apply_V(j, k, r, best_delta)
        else:
            k = np.random.randint(p)
            i = np.random.randint(n)
            r = np.random.randint(rank)
            for d in deltas:
                dE = get_delta_E_W(k, i, r, d)
                if dE < best_delta_E:
                    best_delta_E = dE
                    best_delta = d
            if best_delta != 0:
                best_apply = lambda: apply_W(k, i, r, best_delta)
                
        if best_apply:
            best_apply()
            E += best_delta_E
            stuck_count = 0
        else:
            stuck_count += 1
            if stuck_count > max_stuck:
                for _ in range(5):
                    choice = np.random.randint(3)
                    d = np.random.choice(deltas)
                    if choice == 0:
                        i = np.random.randint(n)
                        j = np.random.randint(m)
                        r = np.random.randint(rank)
                        apply_U(i, j, r, d)
                    elif choice == 1:
                        j = np.random.randint(m)
                        k = np.random.randint(p)
                        r = np.random.randint(rank)
                        apply_V(j, k, r, d)
                    else:
                        k = np.random.randint(p)
                        i = np.random.randint(n)
                        r = np.random.randint(rank)
                        apply_W(k, i, r, d)
                    S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
                    E = np.sum((S - target) ** 2)
                    P_U = np.einsum('jkr,kir->ijrk', V3, W3)
                    P_V = np.einsum('ijr,kir->jkri', U3, W3)
                    P_W = np.einsum('ijr,jkr->kirj', U3, V3)
                    stuck_count = 0

    if E == 0:
        return U_p / 2.0, V_p / 2.0, W_p / 2.0
    return None

def construct():
    """Search for rank-minimal exact half-integer tensor decomposition."""
    n, m, p = N, M, P
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_fb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_fb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_fb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_schoolbook

    # Try structured construction first
    U_struct, V_struct, W_struct = _construct_structured(n, m, p)
    struct_rank = U_struct.shape[1]
    if _verify(U_struct, V_struct, W_struct):
        best_result = {"U": U_struct.tolist(), "V": V_struct.tolist(), "W": W_struct.tolist()}
        best_rank = struct_rank

    # Search for lower ranks using local search from structured init
    for rank in range(20, best_rank + 1):
        if _time_left() < 10:
            break
        
        # Create initial guess by truncating structured solution
        if struct_rank >= rank:
            U_init = U_struct[:, :rank]
            V_init = V_struct[:, :rank]
            W_init = W_struct[:, :rank]
        else:
            # Pad with zeros
            U_init = np.zeros((n * m, rank))
            V_init = np.zeros((m * p, rank))
            W_init = np.zeros((p * n, rank))
            min_r = min(struct_rank, rank)
            U_init[:, :min_r] = U_struct[:, :min_r]
            V_init[:, :min_r] = V_struct[:, :min_r]
            W_init[:, :min_r] = W_struct[:, :min_r]
        
        found = False
        for seed in range(3):
            if _time_left() < 5:
                break
            res = _search_rank_from_init(U_init, V_init, W_init, rank, seed, _time_left() - 2)
            if res is not None:
                U, V, W = res
                if _verify(U, V, W):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return best_result
