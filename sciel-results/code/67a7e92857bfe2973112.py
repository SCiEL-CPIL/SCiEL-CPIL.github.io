import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid exact decomposition."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    U_t = U_arr.reshape(n, m, rank)
    V_t = V_arr.reshape(m, p, rank)
    W_t = W_arr.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True

def _build_strassen_rank39(n, m, p):
    """Build the rank-39 Strassen-based decomposition."""
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_term(u_indices, v_indices, w_indices, u_coeffs=None, v_coeffs=None, w_coeffs=None):
        nonlocal col
        if u_coeffs is None:
            u_coeffs = [1] * len(u_indices)
        if v_coeffs is None:
            v_coeffs = [1] * len(v_indices)
        if w_coeffs is None:
            w_coeffs = [1] * len(w_indices)
        for idx, c in zip(u_indices, u_coeffs):
            U[idx][col] += c
        for idx, c in zip(v_indices, v_coeffs):
            V[idx][col] += c
        for idx, c in zip(w_indices, w_coeffs):
            W[idx][col] += c
        col += 1

    def add_strassen_block(k_start):
        a00 = 0 * 3 + 0
        a01 = 0 * 3 + 1
        a10 = 1 * 3 + 0
        a11 = 1 * 3 + 1

        b00 = 0 * 7 + k_start
        b01 = 0 * 7 + k_start + 1
        b10 = 1 * 7 + k_start
        b11 = 1 * 7 + k_start + 1

        c00 = k_start * 2 + 0
        c01 = (k_start + 1) * 2 + 0
        c10 = k_start * 2 + 1
        c11 = (k_start + 1) * 2 + 1

        add_term([a00, a11], [b00, b11], [c00, c11])
        add_term([a10, a11], [b00], [c10, c11], w_coeffs=[1, -1])
        add_term([a00], [b01, b11], [c01, c11], v_coeffs=[1, -1])
        add_term([a11], [b10, b00], [c00, c10], v_coeffs=[1, -1])
        add_term([a00, a01], [b11], [c00, c01], w_coeffs=[-1, 1])
        add_term([a10, a00], [b00, b01], [c11], u_coeffs=[1, -1])
        add_term([a01, a11], [b10, b11], [c00], u_coeffs=[1, -1])

    add_strassen_block(0)
    add_strassen_block(2)
    add_strassen_block(4)

    b06 = 0 * 7 + 6
    b16 = 1 * 7 + 6
    c06 = 6 * 2 + 0
    c16 = 6 * 2 + 1

    add_term([0], [b06], [c06])
    add_term([1], [b16], [c06])
    add_term([3], [b06], [c16])
    add_term([4], [b16], [c16])

    for k in range(7):
        b2k = 2 * 7 + k
        c0k = k * 2 + 0
        c1k = k * 2 + 1
        add_term([2], [b2k], [c0k])
        add_term([5], [b2k], [c1k])

    return {"U": U, "V": V, "W": W}

def _peel_columns(U, V, W, n, m, p, max_rounds=200, tol=1e-10):
    """Aggressively peel removable columns. Returns (U, V, W) with fewer columns."""
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    for _ in range(max_rounds):
        if current_rank == 0:
            break
        removed = False
        for col in range(current_rank):
            U_test = [row[:col] + row[col+1:] for row in U]
            V_test = [row[:col] + row[col+1:] for row in V]
            W_test = [row[:col] + row[col+1:] for row in W]
            if _verify(U_test, V_test, W_test, n, m, p, tol):
                U, V, W = U_test, V_test, W_test
                current_rank -= 1
                removed = True
                break
        if not removed:
            break
    return U, V, W, current_rank

def _eliminate_dependent(U, V, W, n, m, p, tol=1e-10):
    """
    Try to eliminate a column r that is a linear combination of other columns,
    consistently across U, V, W.
    """
    U = [row[:] for row in U]
    V = [row[:] for row in V]
    W = [row[:] for row in W]
    current_rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)

    changed = True
    while changed and current_rank > 1:
        changed = False
        for r in range(current_rank):
            others = [s for s in range(current_rank) if s != r]
            U_others = U_arr[:, others]
            target_U = U_arr[:, r]
            try:
                alpha, residuals, rank_u, sv = np.linalg.lstsq(U_others, target_U, rcond=None)
            except np.linalg.LinAlgError:
                continue
            U_pred = U_others @ alpha
            if np.linalg.norm(U_pred - target_U) > tol * (1 + np.linalg.norm(target_U)):
                continue
            V_others = V_arr[:, others]
            target_V = V_arr[:, r]
            V_pred = V_others @ alpha
            if np.linalg.norm(V_pred - target_V) > tol * (1 + np.linalg.norm(target_V)):
                continue
            W_others = W_arr[:, others]
            target_W = W_arr[:, r]
            W_pred = W_others @ alpha
            if np.linalg.norm(W_pred - target_W) > tol * (1 + np.linalg.norm(target_W)):
                continue
            U_arr[:, r] = target_U - U_pred
            V_arr[:, r] = target_V - V_pred
            W_arr[:, r] = target_W - W_pred
            U_arr = np.round(U_arr, 8)
            V_arr = np.round(V_arr, 8)
            W_arr = np.round(W_arr, 8)
            U = U_arr.tolist()
            V = V_arr.tolist()
            W = W_arr.tolist()
            current_rank -= 1
            U = [row[:r] + row[r+1:] for row in U]
            V = [row[:r] + row[r+1:] for row in V]
            W = [row[:r] + row[r+1:] for row in W]
            U_arr = np.array(U, dtype=np.float64)
            V_arr = np.array(V, dtype=np.float64)
            W_arr = np.array(W, dtype=np.float64)
            changed = True
            break
    return U, V, W, current_rank

def _try_rank_reduction(rank_target, U_init, V_init, W_init, n, m, p, deadline):
    """Attempt to reduce rank below rank_target via peeling + dependent elimination."""
    U, V, W, cur = _peel_columns(U_init, V_init, W_init, n, m, p)
    U, V, W, cur = _eliminate_dependent(U, V, W, n, m, p)
    if cur < rank_target and _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    return None

def _recursive_slab_decomposition(n, m, p, deadline):
    """
    Try recursive slab decomposition: peel off one column of B at a time.
    2x3x7 = 2x3x1 (rank 2) + 2x3x6
    2x3x6 = 2x3x1 (rank 2) + 2x3x5
    ...
    Base case: 2x3x1 has rank 2 (just A*B for each output row)
    """
    if time.time() > deadline:
        return None
    
    # Base cases
    if p == 1:
        # 2x3x1: rank 2
        rank = 2
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for i in range(n):
            for j in range(m):
                U[i*m+j][i] = 1
                V[j*p+0][i] = 1
                W[0*n+i][i] = 1
        return {"U": U, "V": V, "W": W}
    
    if p == 0:
        return None
    
    # Try to use Strassen for 2x2x2 blocks when p >= 2
    if p >= 2 and m >= 2 and n >= 2:
        # Use Strassen for the first 2 columns, then recurse on the rest
        if p == 2:
            # 2x3x2: use Strassen on 2x2x2 part (rows 0,1 of B) + schoolbook on row 2
            # Strassen rank 7 for 2x2x2 + rank 2 for 2x1x2 = 9 total
            rank = 9
            U = [[0]*rank for _ in range(n*m)]
            V = [[0]*rank for _ in range(m*p)]
            W = [[0]*rank for _ in range(p*n)]
            
            col = 0
            # Strassen for A[:,0:2] * B[0:2,0:2] -> C[0:2,0:2]
            a00 = 0*3+0; a01 = 0*3+1
            a10 = 1*3+0; a11 = 1*3+1
            b00 = 0*2+0; b01 = 0*2+1
            b10 = 1*2+0; b11 = 1*2+1
            c00 = 0*2+0; c01 = 0*2+1
            c10 = 1*2+0; c11 = 1*2+1
            
            # M1 = (A00+A11)(B00+B11)
            U[a00][col]=1; U[a11][col]=1
            V[b00][col]=1; V[b11][col]=1
            W[c00][col]=1; W[c11][col]=1
            col+=1
            # M2 = (A10+A11)B00
            U[a10][col]=1; U[a11][col]=1
            V[b00][col]=1
            W[c10][col]=1; W[c11][col]=-1
            col+=1
            # M3 = A00(B01+B11)
            U[a00][col]=1
            V[b01][col]=1; V[b11][col]=1
            W[c01][col]=1; W[c11][col]=1
            col+=1
            # M4 = A11(B10+B00)
            U[a11][col]=1
            V[b10][col]=1; V[b00][col]=1
            W[c00][col]=1; W[c10][col]=1
            col+=1
            # M5 = (A00+A01)B11
            U[a00][col]=1; U[a01][col]=1
            V[b11][col]=1
            W[c00][col]=-1; W[c01][col]=1
            col+=1
            # M6 = (A10+A00)(B00+B01)
            U[a10][col]=1; U[a00][col]=-1
            V[b00][col]=1; V[b01][col]=1
            W[c11][col]=1
            col+=1
            # M7 = (A01+A11)(B10+B11)
            U[a01][col]=1; U[a11][col]=-1
            V[b10][col]=1; V[b11][col]=1
            W[c00][col]=1
            col+=1
            
            # Schoolbook for A[:,2] * B[2,0:2] -> C[0:2,2]
            for j in [2]:  # row 2 of A
                for k in range(2):
                    for i in range(n):
                        U[i*3+j][col] = 1
                        V[j*2+k][col] = 1
                        W[k*2+i][col] = 1
                    col+=1
            
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # General recursive approach: peel one column of B
    # 2x3xp = 2x3x1 (using column 0 of B) + 2x3x(p-1) (using columns 1..p-1 of B)
    # But we need to handle the indexing carefully
    
    # Try direct construction for small p
    if p <= 3:
        # Schoolbook might be competitive
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _build_direct_sum(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """Direct sum of two decompositions: (n,m,p1) and (n,m,p2) -> (n,m,p1+p2)"""
    rank1 = len(U1[0])
    rank2 = len(U2[0])
    rank = rank1 + rank2
    p = p1 + p2
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Copy first decomposition
    for i in range(n*m):
        for r in range(rank1):
            U[i][r] = U1[i][r]
    for j in range(m):
        for k in range(p1):
            for r in range(rank1):
                V[j*p+k][r] = V1[j*p1+k][r]
    for k in range(p1):
        for i in range(n):
            for r in range(rank1):
                W[k*n+i][r] = W1[k*n+i][r]
    
    # Copy second decomposition (offset in V and W by p1)
    for i in range(n*m):
        for r in range(rank2):
            U[i][rank1+r] = U2[i][r]
    for j in range(m):
        for k in range(p2):
            for r in range(rank2):
                V[j*p + p1 + k][rank1+r] = V2[j*p2+k][r]
    for k in range(p2):
        for i in range(n):
            for r in range(rank2):
                W[(p1+k)*n+i][rank1+r] = W2[k*n+i][r]
    
    return {"U": U, "V": V, "W": W}

def _recursive_decompose(n, m, p, deadline):
    """
    Recursively decompose n x m x p multiplication.
    Strategy: if p >= 2 and n >= 2 and m >= 2, use Strassen on 2x2x2 subblock.
    Otherwise, use schoolbook for the remaining part.
    """
    if time.time() > deadline:
        return None
    
    if p == 0:
        return None
    
    if p == 1:
        # n x m x 1: rank n*m... but we can do better
        # Actually for 2x3x1, rank is 2 (not 6)
        # C[i,0] = sum_j A[i,j] B[j,0]
        # We need rank = min(n*m, n, m) ... actually rank of this tensor
        # For 2x3x1: rank 2
        rank = min(n*m, max(n,m))
        # Actually let's just use schoolbook for simplicity
        rank = n * m
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j*p+0][col] = 1
                W[0*n+i][col] = 1
                col += 1
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        return None
    
    # If we can use Strassen on a 2x2x2 block
    if n >= 2 and m >= 2 and p >= 2:
        # Use Strassen on A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # Then handle the rest with schoolbook
        # This gives rank 7 + (n*m*p - 8) for the rest... not great
        
        # Better: use Strassen for all 2x2x2 blocks we can form
        # For 2x3x7: use Strassen on 3 blocks of 2x2x2 (covering B cols 0-1, 2-3, 4-5)
        # Plus schoolbook for B col 6 and row 2 of A
        pass
    
    # Fallback: schoolbook
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    result = _build_strassen_rank39(n, m, p)

    if _verify(result["U"], result["V"], result["W"], n, m, p):
        # Try to reduce rank aggressively
        for target_rank in [38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28]:
            if time.time() > deadline - 10:
                break
            reduced = _try_rank_reduction(target_rank, result["U"], result["V"], result["W"], n, m, p, deadline)
            if reduced is not None:
                result = reduced
                break

        return result

    # Fallback: schoolbook
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

if __name__ == "__main__":
    res = construct()
