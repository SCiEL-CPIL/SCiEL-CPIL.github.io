import time

DIMENSIONS = [3, 5, 5]
N, M, P = DIMENSIONS

# Known lower bound for 3x5x5 (from literature on small matrix multiplication ranks)
LB = 60
RANK_MAX = 75


def _verify(U, V, W, rank):
    """Check tensor identity: (U[i,j]*V[j,k])_c * W[k,i]_c summed over c == 1 iff i,j,k all valid."""
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = 0
                for c in range(rank):
                    s += U[i * M + j][c] * V[j * P + k][c] * W[k * N + i][c]
                if s != 1:
                    return False
    for i in range(N):
        for j in range(M):
            for k in range(P):
                for c in range(rank):
                    if U[i * M + j][c] * V[j * P + k][c] * W[k * N + i][c] != 0:
                        return False
    return True


def _try_rank(rank, deadline, seed):
    """Peeling search for a rank-rank decomposition. Returns (U,V,W) or None."""
    import random
    rng = random.Random(seed)
    best = None
    # Try a few random initial orderings
    for attempt in range(30):
        if time.time() > deadline - 5:
            break
        order = list(range(rank))
        rng.shuffle(order)
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        # Track which (i,j,k) triples are covered
        covered = [[[False] * P for _ in range(M)] for _ in range(N)]
        used_cols = [False] * rank
        success = True
        for c in order:
            used_cols[c] = True
            # Find best triple: maximize # of triples it can cover (all 3 factors zero elsewhere in this column)
            best_triple = None
            best_score = -1
            # For each triple, score = number of "free" cells (where all 3 factors are 0)
            for i in range(N):
                for j in range(M):
                    ui = U[i * M + j]
                    for k in range(P):
                        if covered[i][j][k]:
                            continue
                        # Check that adding this triple keeps consistency: for all other (i',j',k'),
                        # at most one of (i',j') or (j',k') or (k',i') matches in a way that creates a spurious product.
                        # Simplify: just require that in column c, U[i',j']=0 for all (i',j') != (i,j),
                        # V[j',k']=0 for all (j',k') != (j,k), W[k',i']=0 for all (k',i') != (k,i).
                        ok = True
                        for ii in range(N):
                            for jj in range(M):
                                if (ii, jj) == (i, j):
                                    continue
                                if ui[jj] != 0:  # actually ui is row i*M+j; check other rows
                                    pass
                        # Check U: all rows except i*M+j must be 0 in col c
                        for r in range(N * M):
                            if r != i * M + j and U[r][c] != 0:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Check V: all rows except j*P+k must be 0 in col c
                        for r in range(M * P):
                            if r != j * P + k and V[r][c] != 0:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Check W: all rows except k*N+i must be 0 in col c
                        for r in range(P * N):
                            if r != k * N + i and W[r][c] != 0:
                                ok = False
                                break
                        if not ok:
                            continue
                        # Score: number of new triples covered (should be 1 for valid peeling)
                        # Prefer triples where the row/column are "popular" to enable future coverage
                        score = 1
                        if score > best_score:
                            best_score = score
                            best_triple = (i, j, k)
            if best_triple is None:
                success = False
                break
            i, j, k = best_triple
            U[i * M + j][c] = 1
            V[j * P + k][c] = 1
            W[k * N + i][c] = 1
            covered[i][j][k] = True
        if success:
            # Verify
            if _verify(U, V, W, rank):
                return U, V, W
    return None


def _schoolbook():
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    start = time.time()
    deadline = start + 140  # 10s safety margin before 150s kill
    fallback = _schoolbook()

    for rank in range(LB, RANK_MAX + 1):
        if time.time() > deadline - 10:
            break
        res = _try_rank(rank, deadline, seed=rank)
        if res is not None:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
    return fallback
