import time
import numpy as np

N, M, P = 3, 3, 8
SCHOOLBOOK = N * M * P  # 72
A_DIM = N * M          # 9
B_DIM = M * P          # 24
C_DIM = P * N          # 24

# Precompute the target tensor T[a,b,c] = 1 iff (a,b,c) is a valid multiplication term
# a = i*M + j  (i in [0,N), j in [0,M))
# b = j*P + k  (j in [0,M), k in [0,P))
# c = k*N + i  (k in [0,P), i in [0,N))
_T = np.zeros((A_DIM, B_DIM, C_DIM), dtype=np.float64)
for i in range(N):
    for j in range(M):
        for k in range(P):
            a = i * M + j
            b = j * P + k
            c = k * N + i
            _T[a, b, c] = 1.0

# For fast verification: for each column c, the contribution to T is U[:,c] * V[:,c] * W[:,c]
# sum over rank columns should equal T.
# We'll verify by: for each rank col r, compute outer = U[:,r][:,None] * V[:,r][None,:]
# and add to a matrix; then check against T reshaped.
# Actually T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r].
# Reshape: T is (A,B,C). For a fixed c, T[:,:,c] = sum_r (U[:,r]*V[:,r]) * W[c,r].
# So T[:,:,c] = (U * W[c,:] * V.T) ... let's just do:
# T_reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
# But that's O(A*B*C*rank) which for rank~68 is 9*24*24*68 = 350k, fine.


def _verify_fast(U, V, W, tol=1e-9):
    """Vectorized verification: check U,V,W decompose T exactly."""
    rank = U.shape[1]
    if rank == 0:
        return False
    # Reconstruct tensor
    T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
    return np.allclose(T_rec, _T, atol=tol, rtol=0)


def _build_schoolbook():
    """Return (U,V,W) for rank-72 schoolbook decomposition."""
    rank = SCHOOLBOOK
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
    """
    Build a seed decomposition using Strassen-style 2x2 blocks where possible,
    plus schoolbook for the rest. This gives a lower-rank starting point.
    
    Strategy: partition the 3x3 A matrix into 2x2 + 1x2 + 2x1 + 1x1 blocks.
    For each 2x2 block of A multiplied by corresponding 2x? block of B,
    use Strassen (7 mults instead of 4).
    
    We'll use a systematic approach:
    - A is 3x3, B is 3x8.
    - Split A into: top-left 2x2 (rows 0-1, cols 0-1), top-right 2x1 (rows 0-1, col 2),
      bottom-left 1x2 (row 2, cols 0-1), bottom-right 1x1 (row 2, col 2).
    - For each A-block, multiply by all 8 columns of B.
    - For 2x2 A-block * 2x8 B-block: use Strassen per 2x2x2 sub-block (4 sub-blocks of B).
      Actually B is 3x8, so for A top-left 2x2, B corresponding is rows 0-1, all 8 cols.
      That's 2x8. We can split B into 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7.
      Each 2x2*2x2 uses 7 Strassen mults. Total: 4*7 = 28 for this A-block.
    - For 2x1 A-block * 1x8 B-block: that's 2x8, which is 4 blocks of 2x2? No, B is 1x8.
      2x1 * 1x8 = 2x8. We can do schoolbook: 2*8 = 16 mults. Or split into 2x2*2x2? 
      B is only 1 row, so can't make 2x2 from B. So schoolbook: 16.
    - Similarly for 1x2 A-block * 2x8 B-block: schoolbook 1*2*8 = 16.
    - 1x1 * 1x8: 8 mults.
    
    Total: 28 + 16 + 16 + 8 = 68. Same as current seed.
    
    But we can try different block placements to see if we get better interaction.
    """
    rank = 68
    U = np.zeros((A_DIM, rank), dtype=np.float64)
    V = np.zeros((B_DIM, rank), dtype=np.float64)
    W = np.zeros((C_DIM, rank), dtype=np.float64)
    col = 0

    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        """Add a 2x2x2 Strassen block.
        a_rows: 2 row indices in A (i indices)
        a_cols: 2 col indices in A (j indices)
        b_rows: 2 row indices in B (j indices) - must match a_cols
        b_cols: 2 col indices in B (k indices)
        c_rows: 2 row indices in C (k indices) - must match b_cols
        c_cols: 2 col indices in C (i indices) - must match a_rows
        """
        nonlocal col
        # A indices: (i, j) -> i*M + j
        a00 = a_rows[0] * M + a_cols[0]
        a01 = a_rows[0] * M + a_cols[1]
        a10 = a_rows[1] * M + a_cols[0]
        a11 = a_rows[1] * M + a_cols[1]
        
        # B indices: (j, k) -> j*P + k
        b00 = b_rows[0] * P + b_cols[0]
        b01 = b_rows[0] * P + b_cols[1]
        b10 = b_rows[1] * P + b_cols[0]
        b11 = b_rows[1] * P + b_cols[1]
        
        # C indices: (k, i) -> k*N + i
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
        """Add schoolbook for A-block (a_rows x a_cols) * B-block (b_rows x b_cols).
        b_rows must match a_cols dimension (j indices).
        Output C-block: c_rows = b_cols (k indices), c_cols = a_rows (i indices).
        """
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

    # Block 1: A[0:2, 0:2] * B[0:2, 0:8]
    # Split B into 4 blocks of 2x2: cols 0-1, 2-3, 4-5, 6-7
    for bcol_start in range(0, 8, 2):
        add_strassen_2x2x2(
            a_rows=[0, 1], a_cols=[0, 1],
            b_rows=[0, 1], b_cols=[bcol_start, bcol_start + 1],
            c_rows=[bcol_start, bcol_start + 1], c_cols=[0, 1]
        )
    # That's 4 * 7 = 28

    # Block 2: A[0:2, 2] * B[2, 0:8]
    # 2x1 * 1x8 = 2x8, schoolbook
    add_schoolbook_block(a_rows=[0, 1], a_cols=[2], b_rows=[2], b_cols=list(range(8)))
    # 2*8 = 16

    # Block 3: A[2, 0:2] * B[0:2, 0:8]
    # 1x2 * 2x8 = 1x8, schoolbook
    add_schoolbook_block(a_rows=[2], a_cols=[0, 1], b_rows=[0, 1], b_cols=list(range(8)))
    # 1*2*8 = 16

    # Block 4: A[2, 2] * B[2, 0:8]
    # 1x1 * 1x8 = 1x8, schoolbook
    add_schoolbook_block(a_rows=[2], a_cols=[2], b_rows=[2], b_cols=list(range(8)))
    # 8

    return U[:, :col], V[:, :col], W[:, :col]


def _random_sparse_decomposition(rank, seed=None):
    """
    Generate a random sparse decomposition of given rank.
    Each column has a small number of non-zero entries in U, V, W.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    U = np.zeros((A_DIM, rank), dtype=np.float64)
    V = np.zeros((B_DIM, rank), dtype=np.float64)
    W = np.zeros((C_DIM, rank), dtype=np.float64)
    
    for r in range(rank):
        # Each column: pick 1-3 non-zero entries in each of U, V, W
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


def _local_search(U, V, W, time_limit=10.0, seed=None):
    """
    Local search: try to reduce rank by merging columns or removing redundant ones.
    Also try random perturbations of existing columns.
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    start = time.time()
    
    # First, try to remove any column that is entirely zero or redundant
    changed = True
    while changed and (time.time() - start) < time_limit * 0.5:
        changed = False
        current_rank = best_U.shape[1]
        for c in range(current_rank):
            if time.time() - start > time_limit * 0.5:
                break
            # Check if column c is all zeros
            if np.all(best_U[:, c] == 0) and np.all(best_V[:, c] == 0) and np.all(best_W[:, c] == 0):
                mask = np.ones(current_rank, dtype=bool)
                mask[c] = False
                best_U = best_U[:, mask]
                best_V = best_V[:, mask]
                best_W = best_W[:, mask]
                best_rank = best_U.shape[1]
                changed = True
                break
        
        # Try to remove a column and see if the decomposition still holds
        current_rank = best_U.shape[1]
        for c in range(current_rank):
            if time.time() - start > time_limit * 0.5:
                break
            mask = np.ones(current_rank, dtype=bool)
            mask[c] = False
            U_test = best_U[:, mask]
            V_test = best_V[:, mask]
            W_test = best_W[:, mask]
            if _verify_fast(U_test, V_test, W_test):
                best_U, best_V, best_W = U_test, V_test, W_test
                best_rank = best_U.shape[1]
                changed = True
                break
    
    # Now try random local moves: perturb a column, or add a new column and remove an old one
    moves_made = 0
    while (time.time() - start) < time_limit and moves_made < 200:
        if time.time() - start > time_limit:
            break
        current_rank = best_U.shape[1]
        
        # Move type 1: perturb a random column
        if current_rank > 0 and rng.random() < 0.5:
            c = rng.randint(current_rank)
            U_test = best_U.copy()
            V_test = best_V.copy()
            W_test = best_W.copy()
            
            # Perturb one entry
            dim = rng.choice([0, 1, 2])
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
                moves_made += 1
        else:
            # Move type 2: try to add a column and then remove one
            if current_rank >= 1:
                U_test = np.zeros((A_DIM, current_rank + 1), dtype=np.float64)
                V_test = np.zeros((B_DIM, current_rank + 1), dtype=np.float64)
                W_test = np.zeros((C_DIM, current_rank + 1), dtype=np.float64)
                U_test[:, :current_rank] = best_U
                V_test[:, :current_rank] = best_V
                W_test[:, :current_rank] = best_W
                
                # Add a random sparse column
                n_u = rng.randint(1, 3)
                n_v = rng.randint(1, 3)
                n_w = rng.randint(1, 3)
                u_idx = rng.choice(A_DIM, size=n_u, replace=False)
                v_idx = rng.choice(B_DIM, size=n_v, replace=False)
                w_idx = rng.choice(C_DIM, size=n_w, replace=False)
                for idx in u_idx:
                    U_test[idx, current_rank] = rng.choice([-1, 1])
                for idx in v_idx:
                    V_test[idx, current_rank] = rng.choice([-1, 1])
                for idx in w_idx:
                    W_test[idx, current_rank] = rng.choice([-1, 1])
                
                # Now try to remove one of the original columns
                c_remove = rng.randint(current_rank)
                mask = np.ones(current_rank + 1, dtype=bool)
                mask[c_remove] = False
                U_test2 = U_test[:, mask]
                V_test2 = V_test[:, mask]
                W_test2 = W_test[:, mask]
                
                if _verify_fast(U_test2, V_test2, W_test2):
                    best_U, best_V, best_W = U_test2, V_test2, W_test2
                    best_rank = best_U.shape[1]
                    moves_made += 1
    
    return best_U, best_V, best_W


def construct():
    start = time.time()
    deadline = start + 130.0  # Safety margin before 150s kill
    
    # Start with Strassen seed
    U, V, W = _build_strassen_seed()
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = U.shape[1]
    
    # Verify seed
    if not _verify_fast(U, V, W):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = U.shape[1]
    
    # Run local search on the seed
    if time.time() < deadline - 30:
        U, V, W = _local_search(U, V, W, time_limit=min(30.0, deadline - time.time() - 10), seed=42)
        if _verify_fast(U, V, W) and U.shape[1] < best_rank:
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = U.shape[1]
    
    # Try random restarts with local search
    seed_counter = 100
    while time.time() < deadline - 20:
        seed_counter += 1
        # Start with a random decomposition slightly below current best rank
        initial_rank = max(best_rank - 2, 40)
        U_rand, V_rand, W_rand = _random_sparse_decomposition(initial_rank, seed=seed_counter)
        
        # Run local search
        remaining = deadline - time.time() - 15
        if remaining < 5:
            break
        U_rand, V_rand, W_rand = _local_search(
            U_rand, V_rand, W_rand, 
            time_limit=min(10.0, remaining), 
            seed=seed_counter
        )
        
        if _verify_fast(U_rand, V_rand, W_rand) and U_rand.shape[1] < best_rank:
            best_U, best_V, best_W = U_rand.copy(), V_rand.copy(), W_rand.copy()
            best_rank = U_rand.shape[1]
    
    # Final verification
    if _verify_fast(best_U, best_V, best_W):
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fallback: schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
