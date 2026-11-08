"""Block-partitioned Strassen-based decomposition for 4x5x6 matrix multiplication.
Uses rank-7 2x2x2 Strassen decomposition on 2x2x2 sub-blocks, with careful
index mapping for the irregular 4x5x6 shape. Then applies exact linear-algebra
based column reduction to find minimal rank."""
import time
import random
from fractions import Fraction

try:
    import numpy as np
except Exception:
    np = None

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication.
    
    Each term is (u, v, w) where:
    - u is a 2-vector for A (indexed by (i,j) with i,j in {0,1})
    - v is a 2-vector for B (indexed by (j,k) with j,k in {0,1})  
    - w is a scalar for C (indexed by (k,i) with k,i in {0,1})
    
    C[i,k] = sum_t w_t * (sum_j u_t[i,j] * A[i,j]) * (sum_j v_t[j,k] * B[j,k])
    
    For this to work: sum_t w_t * u_t[i,j] * v_t[j',k] = delta_{j,j'} for all i,k,j,j'
    """
    # Strassen's 7 multiplications for 2x2:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)b00
    # m3 = a00(b10-b11)
    # m4 = a11(b01-b00)
    # m5 = (a00+a01)b11
    # m6 = (a10-a00)(b00+b10)
    # m7 = (a01-a11)(b01+b11)
    #
    # C[0,0] = m1 - m4 + m5 - m7
    # C[0,1] = m3 + m4
    # C[1,0] = m2 + m7
    # C[1,1] = m1 - m2 + m3 + m6
    #
    # In our tensor convention:
    # U[i*m+j, t] = u_t[i,j] for i,j in {0,1} (local indices)
    # V[j*p+k, t] = v_t[j,k] for j,k in {0,1} (local indices)
    # W[k*n+i, t] = w_t for k,i in {0,1} (local indices)
    #
    # Term 0: m1 = (a00+a11)(b00+b11)
    #   u: A[0,0]=1, A[1,1]=1 -> u = [1,0; 0,1] (in (i,j) order: (0,0)=1,(0,1)=0,(1,0)=0,(1,1)=1)
    #   v: B[0,0]=1, B[1,1]=1 -> v = [1,0; 0,1]
    #   w: contributes to C[0,0] (+1), C[1,1] (+1)
    # Term 1: m2 = (a10+a11)b00
    #   u: A[1,0]=1, A[1,1]=1
    #   v: B[0,0]=1
    #   w: C[1,0] (+1), C[1,1] (-1)
    # Term 2: m3 = a00(b10-b11)
    #   u: A[0,0]=1
    #   v: B[1,0]=1, B[1,1]=-1
    #   w: C[0,1] (+1), C[1,1] (+1)
    # Term 3: m4 = a11(b01-b00)
    #   u: A[1,1]=1
    #   v: B[0,1]=1, B[0,0]=-1
    #   w: C[0,0] (-1), C[0,1] (+1)
    # Term 4: m5 = (a00+a01)b11
    #   u: A[0,0]=1, A[0,1]=1
    #   v: B[1,1]=1
    #   w: C[0,0] (+1), C[1,1] (-1)
    # Term 5: m6 = (a10-a00)(b00+b10)
    #   u: A[1,0]=1, A[0,0]=-1
    #   v: B[0,0]=1, B[1,0]=1
    #   w: C[1,1] (+1)
    # Term 6: m7 = (a01-a11)(b01+b11)
    #   u: A[0,1]=1, A[1,1]=-1
    #   v: B[0,1]=1, B[1,1]=1
    #   w: C[0,0] (-1), C[1,0] (+1)
    
    # Represent as (u_flat, v_flat, w_dict)
    # u_flat: 4 entries for (i,j) in order (0,0),(0,1),(1,0),(1,1)
    # v_flat: 4 entries for (j,k) in order (0,0),(0,1),(1,0),(1,1)
    # w_dict: {(k,i): coeff}
    
    terms = [
        # m1
        (
            [1, 0, 0, 1],   # u: A[0,0]=1, A[1,1]=1
            [1, 0, 0, 1],   # v: B[0,0]=1, B[1,1]=1
            {(0,0): 1, (1,1): 1}  # w: C[0,0]+=1, C[1,1]+=1
        ),
        # m2
        (
            [0, 0, 1, 1],   # u: A[1,0]=1, A[1,1]=1
            [1, 0, 0, 0],   # v: B[0,0]=1
            {(0,1): 1, (1,1): -1}  # w: C[1,0]+=1, C[1,1]-=1
        ),
        # m3
        (
            [1, 0, 0, 0],   # u: A[0,0]=1
            [0, 0, 1, -1],  # v: B[1,0]=1, B[1,1]=-1
            {(1,0): 1, (1,1): 1}  # w: C[0,1]+=1, C[1,1]+=1
        ),
        # m4
        (
            [0, 0, 0, 1],   # u: A[1,1]=1
            [-1, 1, 0, 0],  # v: B[0,0]=-1, B[0,1]=1
            {(0,0): -1, (1,0): 1}  # w: C[0,0]-=1, C[0,1]+=1
        ),
        # m5
        (
            [1, 1, 0, 0],   # u: A[0,0]=1, A[0,1]=1
            [0, 0, 0, 1],   # v: B[1,1]=1
            {(0,0): 1, (1,1): -1}  # w: C[0,0]+=1, C[1,1]-=1
        ),
        # m6
        (
            [-1, 0, 1, 0],  # u: A[0,0]=-1, A[1,0]=1
            [1, 0, 1, 0],   # v: B[0,0]=1, B[1,0]=1
            {(1,1): 1}      # w: C[1,1]+=1
        ),
        # m7
        (
            [0, 1, 0, -1],  # u: A[0,1]=1, A[1,1]=-1
            [0, 1, 0, 1],   # v: B[0,1]=1, B[1,1]=1
            {(0,0): -1, (1,0): 1}  # w: C[0,0]-=1, C[1,0]+=1
        ),
    ]
    return terms


def _verify_strassen_local():
    """Verify the local 2x2x2 Strassen decomposition is correct."""
    terms = _strassen_2x2x2()
    n_local = 2
    for i in range(2):
        for k in range(2):
            for j in range(2):
                for j2 in range(2):
                    s = 0
                    for (u, v, w) in terms:
                        w_val = w.get((k, i), 0)
                        u_val = u[i * 2 + j]
                        v_val = v[j2 * 2 + k]
                        s += w_val * u_val * v_val
                    target = 1 if j == j2 else 0
                    if s != target:
                        return False
    return True


def _build_block_decomposition(n, m, p):
    """Build a structured decomposition using block-partitioned Strassen.
    
    Partition:
    - i blocks: [0,1], [2,3]
    - j blocks: [0,1], [2,3], [4]
    - k blocks: [0,1], [2,3], [4,5]
    
    For each combination, use appropriate decomposition.
    """
    i_blocks = [(0, 1), (2, 3)]
    j_blocks = [(0, 1), (2, 3), (4,)]
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    U = [[0] * 0 for _ in range(n * m)]  # Will be resized
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    col = 0
    terms_2x2x2 = _strassen_2x2x2()
    
    for i_block in i_blocks:
        for j_block in j_blocks:
            for k_block in k_blocks:
                if len(j_block) == 2 and len(k_block) == 2:
                    # 2x2x2 block: use Strassen rank 7
                    i0 = i_block[0]
                    j0 = j_block[0]
                    k0 = k_block[0]
                    for t in range(7):
                        u, v, w = terms_2x2x2[t]
                        # u is indexed by (i_local, j_local) in {0,1}x{0,1}
                        # Global: i = i0 + i_local, j = j0 + j_local
                        for il in range(2):
                            for jl in range(2):
                                if u[il * 2 + jl] != 0:
                                    U[i0 * m + (j0 + jl)][col] += u[il * 2 + jl]
                        # v is indexed by (j_local, k_local) in {0,1}x{0,1}
                        # Global: j = j0 + j_local, k = k0 + k_local
                        for jl in range(2):
                            for kl in range(2):
                                if v[jl * 2 + kl] != 0:
                                    V[(j0 + jl) * p + (k0 + kl)][col] += v[jl * 2 + kl]
                        # w is indexed by (k_local, i_local)
                        # Global: k = k0 + k_local, i = i0 + i_local
                        for (kl, il), wval in w.items():
                            if wval != 0:
                                W[(k0 + kl) * n + (i0 + il)][col] += wval
                        col += 1
                elif len(j_block) == 1 and len(k_block) == 2:
                    # 2x1x2 block: schoolbook rank 4
                    i0 = i_block[0]
                    j0 = j_block[0]
                    k0 = k_block[0]
                    for il in range(2):
                        for kl in range(2):
                            U[(i0 + il) * m + j0][col] = 1
                            V[j0 * p + (k0 + kl)][col] = 1
                            W[(k0 + kl) * n + (i0 + il)][col] = 1
                            col += 1
                else:
                    # Should not happen with our partition, but handle generically
                    # Schoolbook for this block
                    for i in i_block:
                        for j in j_block:
                            for k in k_block:
                                U[i * m + j][col] = 1
                                V[j * p + k][col] = 1
                                W[k * n + i][col] = 1
                                col += 1
    
    R = col
    # Trim rows to proper sizes
    U = [row[:R] for row in U]
    V = [row[:R] for row in V]
    W = [row[:R] for row in W]
    
    return U, V, W


def _verify_exact(U, V, W, n, m, p):
    """Exact integer verification."""
    R = len(U[0]) if R > 0 else 0
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for j2 in range(m):
                    s = 0
                    for c in range(R):
                        wv = W[k * n + i][c]
                        uu = U[i * m + j][c]
                        vv = V[j2 * p + k][c]
                        if wv == 0 or uu == 0 or vv == 0:
                            continue
                        s += wv * uu * vv
                    target = 1 if j == j2 else 0
                    if s != target:
                        return False
    return True


def _verify_numpy_fast(U, V, W, n, m, p):
    """Fast numpy verification."""
    if np is None:
        return False
    R = len(U[0])
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m), dtype=np.float64)
            for c in range(R):
                wc = Wa[k * n + i, c]
                if wc == 0:
                    continue
                uc = Ua[i * m: i * m + m, c]
                vc = Va[k * p: k * p + m, c]
                M += np.outer(uc, vc) * wc
            if not np.allclose(M, np.eye(m, dtype=np.float64), atol=1e-9):
                return False
    return True


def _build_constraint_matrix(U, V, W, n, m, p):
    """Build the constraint matrix for column reduction.
    
    For each (i,k,j,j2) constraint: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = delta_{j,j2}
    
    We want to find which columns can be removed. If we fix all columns except column c,
    the contribution of column c must be zero for all constraints (or we can adjust).
    
    Actually, for removal: if we remove column c, we need the remaining columns to still
    satisfy all constraints. This is a system of equations.
    
    Better approach: treat column removal as checking if column c is in the span of
    other columns in a certain sense.
    """
    pass


def _greedy_reduce_fast(U, V, W, n, m, p, deadline, rng):
    """Fast greedy column reduction using incremental verification."""
    R = len(U[0])
    active = list(range(R))
    
    # Precompute column contributions for faster checking
    # For each column c, store its (U_c, V_c, W_c) vectors
    
    def try_remove(c, active_cols):
        """Check if removing column c (from active_cols) still gives valid decomposition."""
        R_new = len(active_cols) - 1
        # Build reduced arrays
        U_new = []
        for r in range(n * m):
            U_new.append([U[r][cc] for cc in active_cols if cc != c])
        V_new = []
        for r in range(m * p):
            V_new.append([V[r][cc] for cc in active_cols if cc != c])
        W_new = []
        for r in range(p * n):
            W_new.append([W[r][cc] for cc in active_cols if cc != c])
        
        return _verify_numpy_fast(U_new, V_new, W_new, n, m, p)
    
    removed = 0
    max_rounds = 50
    for rnd in range(max_rounds):
        if time.time() > deadline - 20.0:
            break
        
        found = False
        # Try removing each active column
        for idx in range(len(active) - 1, -1, -1):
            if time.time() > deadline - 20.0:
                break
            c = active[idx]
            if try_remove(c, active):
                active.pop(idx)
                removed += 1
                found = True
                break
        
        if not found:
            break
    
    # Rebuild with only active columns
    if removed > 0:
        U_new = []
        for r in range(n * m):
            U_new.append([U[r][cc] for cc in active])
        V_new = []
        for r in range(m * p):
            V_new.append([V[r][cc] for cc in active])
        W_new = []
        for r in range(p * n):
            W_new.append([W[r][cc] for cc in active])
        return U_new, V_new, W_new, len(active)
    
    return U, V, W, R


def _linear_algebra_reduce(U, V, W, n, m, p, deadline):
    """Use linear algebra to find redundant columns.
    
    The constraints are: for all (i,k,j,j2): sum_c W[k*n+i,c] * U[i*m+j,c] * V[j2*p+k,c] = delta_{j,j2}
    
    This is a multilinear system. For column removal, we can check if a column's
    contribution is linearly dependent on others.
    
    Simpler: for each column c, check if setting it to zero and solving for adjustments
    in other columns (with integer constraints) is possible.
    """
    R = len(U[0])
    
    # For now, just do the greedy approach but with better ordering
    # Columns with more zeros are more likely to be removable
    def sparsity(c):
        z = 0
        for r in range(n * m):
            if U[r][c] == 0:
                z += 1
        for r in range(m * p):
            if V[r][c] == 0:
                z += 1
        for r in range(p * n):
            if W[r][c] == 0:
                z += 1
        return z
    
    order = sorted(range(R), key=lambda c: -sparsity(c))
    
    active = list(range(R))
    removed = 0
    
    for c in order:
        if time.time() > deadline - 15.0:
            break
        if c not in active:
            continue
        
        # Try removing c
        active_test = [cc for cc in active if cc != c]
        if not active_test:
            continue
        
        U_test = []
        for r in range(n * m):
            U_test.append([U[r][cc] for cc in active_test])
        V_test = []
        for r in range(m * p):
            V_test.append([V[r][cc] for cc in active_test])
        W_test = []
        for r in range(p * n):
            W_test.append([W[r][cc] for cc in active_test])
        
        if _verify_numpy_fast(U_test, V_test, W_test, n, m, p):
            active = active_test
            removed += 1
    
    if removed > 0:
        U_new = []
        for r in range(n * m):
            U_new.append([U[r][cc] for cc in active])
        V_new = []
        for r in range(m * p):
            V_new.append([V[r][cc] for cc in active])
        W_new = []
        for r in range(p * n):
            W_new.append([W[r][cc] for cc in active])
        return U_new, V_new, W_new, len(active)
    
    return U, V, W, R


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    n, m, p = DIMENSIONS
    rng = random.Random(42)
    
    # Verify Strassen local decomposition
    if not _verify_strassen_local():
        # Fallback to schoolbook
        U = [[0] * (n * m * p) for _ in range(n * m)]
        V = [[0] * (n * m * p) for _ in range(m * p)]
        W = [[0] * (n * m * p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Build block-partitioned decomposition
    U, V, W = _build_block_decomposition(n, m, p)
    R = len(U[0])
    
    # Verify
    if not _verify_exact(U, V, W, n, m, p):
        # Fallback to schoolbook
        U = [[0] * (n * m * p) for _ in range(n * m)]
        V = [[0] * (n * m * p) for _ in range(m * p)]
        W = [[0] * (n * m * p) for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Greedy reduction
    U, V, W, R = _greedy_reduce_fast(U, V, W, n, m, p, deadline, rng)
    
    # Final verification
    if _verify_exact(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback
    U = [[0] * (n * m * p) for _ in range(n * m)]
    V = [[0] * (n * m * p) for _ in range(m * p)]
    W = [[0] * (n * m * p) for _ in range(p * n)]
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
