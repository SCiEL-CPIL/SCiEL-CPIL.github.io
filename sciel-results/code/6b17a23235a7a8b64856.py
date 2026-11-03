"""Matrix multiplication 3x3 * 3x4 - rank-minimal exact tensor decomposition.

Strategy:
- The matmul tensor T[i,j,k]=1 for i in [0,3), j in [0,3), k in [0,4)
  can be decomposed as sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r].
- We use a structured search over integer/half-integer entries with
  efficient gradient-based polishing and coordinate descent on the
  half-integer lattice.
- Key insight: for small dimensions, the minimal rank is likely around
  15-20. We search from low ranks upward with multiple restarts.
- We exploit the fact that the tensor is rank-36 (schoolbook) but may
  have lower tensor rank due to the all-ones structure.
"""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]
N, M, P = DIMENSIONS
START_TIME = time.time()
DEADLINE = START_TIME + 130.0  # 130s deadline, 20s safety margin


def _time_left():
    return DEADLINE - time.time()


def _schoolbook():
    """Rank-36 schoolbook construction."""
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


def _evaluate(U, V, W):
    """Compute max |sum_r U[i*M+j,r]*V[j*P+k,r]*W[k*N+i,r] - 1| over all (i,j,k)."""
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    max_err = 0.0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                s = float(np.dot(Ua[i * M + j], Va[j * P + k] * Wa[k * N + i]))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err


def _is_exact(U, V, W, tol=1e-9):
    return _evaluate(U, V, W) < tol


def _snap_half(arr):
    """Snap to nearest half-integer."""
    return np.round(arr * 2.0) / 2.0


def _gradient_step(Ua, Va, Wa, lr=0.05):
    """One gradient descent step on the squared loss."""
    rank = Ua.shape[1]
    gU = np.zeros_like(Ua)
    gV = np.zeros_like(Va)
    gW = np.zeros_like(Wa)
    loss = 0.0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                ui = i * M + j
                vj = j * P + k
                wk = k * N + i
                s = float(np.dot(Ua[ui], Va[vj] * Wa[wk]))
                d = s - 1.0
                loss += d * d
                gU[ui] += 2.0 * d * (Va[vj] * Wa[wk])
                gV[vj] += 2.0 * d * (Ua[ui] * Wa[wk])
                gW[wk] += 2.0 * d * (Ua[ui] * Va[vj])
    Ua -= lr * gU
    Va -= lr * gV
    Wa -= lr * gW
    return Ua, Va, Wa, loss


def _adam_polish(Ua, Va, Wa, deadline, max_iters=500, lr=0.03):
    """Adam optimizer polish with snap to half-integers."""
    rank = Ua.shape[1]
    b1, b2, eps = 0.9, 0.999, 1e-8
    mU = np.zeros_like(Ua)
    vU = np.zeros_like(Ua)
    mV = np.zeros_like(Va)
    vV = np.zeros_like(Va)
    mW = np.zeros_like(Wa)
    vW = np.zeros_like(Wa)
    t = 0

    cur_loss = _evaluate(Ua.tolist(), Va.tolist(), Wa.tolist()) ** 2
    best_loss = cur_loss
    best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()

    for it in range(max_iters):
        if time.time() > deadline:
            break
        t += 1
        # Compute gradient
        gU = np.zeros_like(Ua)
        gV = np.zeros_like(Va)
        gW = np.zeros_like(Wa)
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    ui = i * M + j
                    vj = j * P + k
                    wk = k * N + i
                    s = float(np.dot(Ua[ui], Va[vj] * Wa[wk]))
                    d = s - 1.0
                    gU[ui] += 2.0 * d * (Va[vj] * Wa[wk])
                    gV[vj] += 2.0 * d * (Ua[ui] * Wa[wk])
                    gW[wk] += 2.0 * d * (Ua[ui] * Va[vj])

        mU = b1 * mU + (1 - b1) * gU
        vU = b2 * vU + (1 - b2) * gU ** 2
        mU_hat = mU / (1 - b1 ** t)
        vU_hat = vU / (1 - b2 ** t)

        mV = b1 * mV + (1 - b1) * gV
        vV = b2 * vV + (1 - b2) * gV ** 2
        mV_hat = mV / (1 - b1 ** t)
        vV_hat = vV / (1 - b2 ** t)

        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * gW ** 2
        mW_hat = mW / (1 - b1 ** t)
        vW_hat = vW / (1 - b2 ** t)

        Ua -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        Va -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        Wa -= lr * mW_hat / (np.sqrt(vW_hat) + eps)

        cur_loss = _evaluate(Ua.tolist(), Va.tolist(), Wa.tolist()) ** 2
        if cur_loss < best_loss:
            best_loss = cur_loss
            best_U, best_V, best_W = Ua.copy(), Va.copy(), Wa.copy()
        if cur_loss < 1e-16:
            break

    # Snap to half-integers
    Ua = _snap_half(best_U)
    Va = _snap_half(best_V)
    Wa = _snap_half(best_W)

    if _is_exact(Ua.tolist(), Va.tolist(), Wa.tolist()):
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None


def _coordinate_descent(Ua, Va, Wa, deadline, max_iters=5000):
    """Local search: try single-coordinate half-integer steps."""
    rank = Ua.shape[1]

    def _residual():
        max_err = 0.0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    s = float(np.dot(Ua[i * M + j], Va[j * P + k] * Wa[k * N + i]))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err

    cur_res = _residual()
    if cur_res < 1e-9:
        return Ua.tolist(), Va.tolist(), Wa.tolist()

    for it in range(max_iters):
        if time.time() > deadline:
            break

        factor_choice = random.randint(0, 2)
        if factor_choice == 0:
            mat, nrows = Ua, N * M
        elif factor_choice == 1:
            mat, nrows = Va, M * P
        else:
            mat, nrows = Wa, P * N

        row = random.randint(0, nrows - 1)
        col = random.randint(0, rank - 1)
        step = random.choice([-0.5, 0.5, -1.0, 1.0, -1.5, 1.5])

        old_val = mat[row, col]
        mat[row, col] = old_val + step

        new_res = _residual()
        if new_res < cur_res:
            cur_res = new_res
            if cur_res < 1e-9:
                return Ua.tolist(), Va.tolist(), Wa.tolist()
        else:
            mat[row, col] = old_val

    return None


def _structured_init(rank, seed=0):
    """Create a structured initialization exploiting matmul structure."""
    rng = np.random.RandomState(seed)
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))

    for r in range(rank):
        i = r % N
        j = (r // N) % M
        k = (r // (N * M)) % P

        if rng.random() < 0.4:
            i = rng.randint(0, N)
            j = rng.randint(0, M)
            k = rng.randint(0, P)

        U[i * M + j, r] = 1.0
        V[j * P + k, r] = 1.0
        W[k * N + i, r] = 1.0

    U += rng.normal(0, 0.15, U.shape)
    V += rng.normal(0, 0.15, V.shape)
    W += rng.normal(0, 0.15, W.shape)

    return U, V, W


def _try_rank(rank, deadline, num_restarts=30):
    """Attempt to find a rank-r decomposition."""
    for restart in range(num_restarts):
        if time.time() > deadline - 3.0:
            break

        U, V, W = _structured_init(rank, seed=restart)

        # Polish with Adam
        result = _adam_polish(U, V, W, deadline - 1.0)
        if result is not None:
            return result

        # Try coordinate descent on the snapped result
        Ua = _snap_half(U)
        Va = _snap_half(V)
        Wa = _snap_half(W)
        result = _coordinate_descent(Ua, Va, Wa, deadline - 1.0)
        if result is not None:
            return result

    return None


def _try_hadamard_construction(deadline):
    """Try Hadamard-based construction with L=4,8."""
    for L in [4, 8]:
        if time.time() > deadline:
            break

        # Hadamard matrix H of order L
        H = np.ones((L, L))
        for a in range(L):
            for b in range(L):
                x = a & b
                pc = 0
                while x:
                    pc += 1
                    x &= x - 1
                H[a, b] = 1.0 if pc % 2 == 0 else -1.0

        # Try linear mappings a(i,j) = alpha*i + beta*j, etc.
        for alpha in range(L):
            if time.time() > deadline:
                break
            for beta in range(L):
                if time.time() > deadline:
                    break
                for delta in range(L):
                    if time.time() > deadline:
                        break

                    rank = L
                    U = np.zeros((N * M, rank))
                    V = np.zeros((M * P, rank))
                    W = np.zeros((P * N, rank))

                    for i in range(N):
                        for j in range(M):
                            a = (alpha * i + beta * j) % L
                            U[i * M + j] = H[a]
                    for j in range(M):
                        for k in range(P):
                            b = (-beta * j + delta * k) % L
                            V[j * P + k] = H[b]
                    for k in range(P):
                        for i in range(N):
                            c = (-delta * k - alpha * i) % L
                            W[k * N + i] = H[c]

                    # Check if unscaled works
                    if _is_exact(U.tolist(), V.tolist(), W.tolist()):
                        return U.tolist(), V.tolist(), W.tolist()

                    # Try scaling one factor by 1/L
                    for factor in ['U', 'V', 'W']:
                        if factor == 'U':
                            Us = _snap_half(U / L)
                            if _is_exact(Us.tolist(), V.tolist(), W.tolist()):
                                return Us.tolist(), V.tolist(), W.tolist()
                        elif factor == 'V':
                            Vs = _snap_half(V / L)
                            if _is_exact(U.tolist(), Vs.tolist(), W.tolist()):
                                return U.tolist(), Vs.tolist(), W.tolist()
                        else:
                            Ws = _snap_half(W / L)
                            if _is_exact(U.tolist(), V.tolist(), Ws.tolist()):
                                return U.tolist(), V.tolist(), Ws.tolist()

    return None


def _try_block_construction(deadline):
    """
    Try to construct a low-rank decomposition by splitting into blocks.
    For 3x3x4, we can try to use the structure of overlapping 2x2 blocks.
    """
    if time.time() > deadline:
        return None

    # Try rank 15-20 with block-structured initialization
    for rank in [15, 16, 18, 20]:
        if time.time() > deadline - 5:
            break

        for restart in range(20):
            if time.time() > deadline - 5:
                break

            rng = np.random.RandomState(restart + rank * 100)
            U = np.zeros((N * M, rank))
            V = np.zeros((M * P, rank))
            W = np.zeros((P * N, rank))

            # Block-structured: divide ranks into blocks
            block_size = max(1, rank // 4)
            for block in range(min(4, rank // block_size)):
                base = block * block_size
                sz = min(block_size, rank - base)
                for r in range(base, base + sz):
                    # Each block handles a subset of (i,j,k)
                    i = (block + r) % N
                    j = (block + 1 + r) % M
                    k = (block + 2 + r) % P
                    U[i * M + j, r] = 1.0
                    V[j * P + k, r] = 1.0
                    W[k * N + i, r] = 1.0

            U += rng.normal(0, 0.1, U.shape)
            V += rng.normal(0, 0.1, V.shape)
            W += rng.normal(0, 0.1, W.shape)

            result = _adam_polish(U, V, W, deadline - 2.0)
            if result is not None:
                return result

            Ua = _snap_half(U)
            Va = _snap_half(V)
            Wa = _snap_half(W)
            result = _coordinate_descent(Ua, Va, Wa, deadline - 2.0)
            if result is not None:
                return result

    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x4 matmul.

    Returns {'U', 'V', 'W'} with factor matrices of shapes:
    U: (9, rank)
    V: (12, rank)
    W: (12, rank)
    """
    # Try Hadamard construction first (fast, structured)
    result = _try_hadamard_construction(DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Try block construction
    result = _try_block_construction(DEADLINE)
    if result is not None:
        U, V, W = result
        return {"U": U, "V": V, "W": W}

    # Search from low ranks upward
    for rank in range(12, 36):
        if time.time() > DEADLINE - 5:
            break
        num_restarts = 40 if rank <= 18 else 25
        result = _try_rank(rank, DEADLINE - 2.0, num_restarts=num_restarts)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook rank-36
    return _schoolbook()
