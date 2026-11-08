import numpy as np
import time
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 140

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, num_trials=4):
    """Verify exact integer decomposition using small integer test matrices."""
    rank = len(U[0])
    np.random.seed(42)
    for _ in range(num_trials):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_true = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            x = np.dot(A.flatten(), [U[i*m+j][r] for i in range(n) for j in range(m)])
            y = np.dot(B.flatten(), [V[j*p+k][r] for j in range(m) for k in range(p)])
            for i in range(n):
                for k in range(p):
                    C_decomp[i,k] += x * y * W[k*n+i][r]
        if not np.array_equal(C_true, C_decomp):
            return False
    return True

def _build_85():
    """Build the known rank-85 solution: 49 (Strassen 4x4x4) + 16 (4x4x1) + 20 (4x1x5)."""
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # 2x2x2 Strassen micro factors
    u_micro = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, -1], [0, 1, 0, -1]
    ]
    v_micro = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1],
        [0, 0, 1, -1], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    w_micro = [
        [1, 0, 0, 1], [0, 1, 0, 0], [0, 1, 0, -1],
        [1, 0, 0, 1], [-1, 0, 0, 1], [0, 0, 1, 0], [-1, 0, 1, 0]
    ]

    col = 0
    for k in range(7):
        for l in range(7):
            for b in range(4):
                if u_micro[k][b] == 0: continue
                br, bc = b // 2, b % 2
                for o in range(4):
                    if u_micro[l][o] == 0: continue
                    er, ec = o // 2, o % 2
                    gi, gj = br*2+er, bc*2+ec
                    U[gi*5+gj][col] += u_micro[k][b] * u_micro[l][o]
            for b in range(4):
                if v_micro[k][b] == 0: continue
                br, bc = b // 2, b % 2
                for o in range(4):
                    if v_micro[l][o] == 0: continue
                    er, ec = o // 2, o % 2
                    gj, gk = br*2+er, bc*2+ec
                    V[gj*5+gk][col] += v_micro[k][b] * v_micro[l][o]
            for b in range(4):
                if w_micro[k][b] == 0: continue
                br, bc = b // 2, b % 2
                for o in range(4):
                    if w_micro[l][o] == 0: continue
                    er, ec = o // 2, o % 2
                    gi, gk = br*2+er, bc*2+ec
                    W[gk*4+gi][col] += w_micro[k][b] * w_micro[l][o]
            col += 1

    # 4x4x1 part (rank 16): A(4x4) @ B(4x1) -> C(4x1)
    for i in range(4):
        for j in range(4):
            U[i*5+j][col] = 1
            V[j*5+4][col] = 1
            W[4*4+i][col] = 1
            col += 1

    # 4x1x5 part (rank 20): A(4x1) @ B(1x5) -> C(4x5)
    for i in range(4):
        for k in range(5):
            U[i*5+4][col] = 1
            V[4*5+k][col] = 1
            W[k*4+i][col] = 1
            col += 1

    return U, V, W

def _try_reduce(U, V, W, n, m, p, target_rank, deadline):
    """
    Attempt to reduce the rank of a valid decomposition to target_rank.
    Strategy: The 36 'naive' terms (indices 49..84) are simple rank-1 terms.
    We check if any subset of them can be merged by finding linear dependencies
    in the combined (U,V,W) tensor. Specifically, we look for a rank-revealing
    structure: if two terms have proportional U-factors or V-factors, they might
    be combinable. We do a greedy search: for each pair of 'naive' terms, check
    if their tensor contribution is linearly dependent with an existing term.
    """
    rank = len(U[0])
    if rank <= target_rank:
        return U, V, W

    # Convert to numpy for efficient checks
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    # The naive terms are the last 36 (indices 49 to 84)
    naive_start = 49
    # Build a list of tensor contributions for naive terms
    # Each term r contributes: U[:,r] * V[:,r] * W[:,r] (outer product structure)
    # We want to find if any naive term is a linear combination of others.
    # For exact integer arithmetic, we use the fact that the naive terms are
    # rank-1 with unit entries. We check for duplicates or sign-flips.

    # Actually, a more structural approach: the 4x4x1 and 4x1x5 parts share
    # the index j=4 or i=4. Let's check if we can merge terms that involve
    # the same (i,j) or (j,k) pair.
    # For 4x4x1: term (i,j) has U[i*5+j]=1, V[j*5+4]=1, W[16+i]=1
    # For 4x1x5: term (i,k) has U[i*5+4]=1, V[4*5+k]=1, W[k*4+i]=1
    # These are all distinct rank-1 tensors, so no merging is possible within
    # this structure. The rank 85 is likely tight for this block partition.

    # Try a different structural approach: use a different block partition.
    # Instead of splitting A as [4x4 | 4x1], try [3x5 | 1x5] or other splits.
    # But that would require different Strassen-like factors for 3x3 or 3x5.

    # For now, return the original if we can't reduce.
    return U, V, W

def _build_alternative_partition(n, m, p):
    """
    Try an alternative block partition: split A as [2x5 | 2x5] and use
    2x5x5 sub-problems. Each 2x5x5 can be done with rank 2*5*5=50 naive,
    but maybe with Strassen-like tricks for the 2x2 blocks.
    Actually, 2x5x5: split B as [2x5 | 3x5]? No, B is 5x5.
    Let's try: A(4x5) = [A1(2x5); A2(2x5)], B(5x5) stays.
    C(4x5) = [A1@B; A2@B]. Each A1@B is 2x5 * 5x5 = 2x5.
    2x5x5 naive rank = 2*5*5 = 50. Two of them = 100. Worse.
    """
    pass

def construct():
    n, m, p = 4, 5, 5
    deadline = _START_TIME + TIME_LIMIT - 5

    # Build the known rank-85 solution
    U, V, W = _build_85()

    # Verify it
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: Kronecker rank-100
    rank_k = n * m * p
    Uk = [[0]*rank_k for _ in range(n*m)]
    Vk = [[0]*rank_k for _ in range(m*p)]
    Wk = [[0]*rank_k for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                Uk[i*m+j][r] = 1
                Vk[j*p+k][r] = 1
                Wk[k*n+i][r] = 1
    return {"U": Uk, "V": Vk, "W": Wk}
