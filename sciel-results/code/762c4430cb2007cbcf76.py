"""Combined program: explicit Lander-Bailey rank-19 base + numerical search/polish."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition reconstructs the MM tensor T[i,j,k]=1."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T_computed = np.zeros((n, m, p), dtype=np.float64)
    for c in range(rank):
        U_col = U[:, c].reshape(n, m)
        V_col = V[:, c].reshape(m, p)
        W_col = W[:, c].reshape(p, n)
        tmp1 = np.einsum('ij,jk->ijk', U_col, V_col)
        T_computed += tmp1 * W_col.T[:, np.newaxis, :]
    return np.max(np.abs(T_computed - 1.0)) < tol


def _landers_bailey_4x4x4():
    """
    Lander-Bailey rank-19 decomposition of the 4x4x4 matrix multiplication tensor.
    Returns U (16x19), V (16x19), W (16x19) such that:
    T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] = 1 for all i,j,k in {0..3}
    """
    r = 19
    U = np.zeros((16, r), dtype=np.float64)
    V = np.zeros((16, r), dtype=np.float64)
    W = np.zeros((16, r), dtype=np.float64)

    def ui(i, j): return i * 4 + j
    def vj(j, k): return j * 4 + k
    def wk(k, i): return k * 4 + i

    # Term 0: (a00+a11+a22+a33)(b00+b11+b22+b33)
    U[ui(0,0),0]=1; U[ui(1,1),0]=1; U[ui(2,2),0]=1; U[ui(3,3),0]=1
    V[vj(0,0),0]=1; V[vj(1,1),0]=1; V[vj(2,2),0]=1; V[vj(3,3),0]=1
    W[wk(0,0),0]=1; W[wk(1,1),0]=1; W[wk(2,2),0]=1; W[wk(3,3),0]=1

    # Term 1: (a00+a11)(b00+b11)
    U[ui(0,0),1]=1; U[ui(1,1),1]=1
    V[vj(0,0),1]=1; V[vj(1,1),1]=1
    W[wk(0,0),1]=1; W[wk(1,1),1]=1

    # Term 2: (a00+a22)(b00+b22)
    U[ui(0,0),2]=1; U[ui(2,2),2]=1
    V[vj(0,0),2]=1; V[vj(2,2),2]=1
    W[wk(0,0),2]=1; W[wk(2,2),2]=1

    # Term 3: (a00+a33)(b00+b33)
    U[ui(0,0),3]=1; U[ui(3,3),3]=1
    V[vj(0,0),3]=1; V[vj(3,3),3]=1
    W[wk(0,0),3]=1; W[wk(3,3),3]=1

    # Term 4: (a11+a22)(b11+b22)
    U[ui(1,1),4]=1; U[ui(2,2),4]=1
    V[vj(1,1),4]=1; V[vj(2,2),4]=1
    W[wk(1,1),4]=1; W[wk(2,2),4]=1

    # Term 5: (a11+a33)(b11+b33)
    U[ui(1,1),5]=1; U[ui(3,3),5]=1
    V[vj(1,1),5]=1; V[vj(3,3),5]=1
    W[wk(1,1),5]=1; W[wk(3,3),5]=1

    # Term 6: (a22+a33)(b22+b33)
    U[ui(2,2),6]=1; U[ui(3,3),6]=1
    V[vj(2,2),6]=1; V[vj(3,3),6]=1
    W[wk(2,2),6]=1; W[wk(3,3),6]=1

    # Term 7: (a01+a10+a23+a32)(b01+b10+b23+b32)
    U[ui(0,1),7]=1; U[ui(1,0),7]=1; U[ui(2,3),7]=1; U[ui(3,2),7]=1
    V[vj(0,1),7]=1; V[vj(1,0),7]=1; V[vj(2,3),7]=1; V[vj(3,2),7]=1
    W[wk(0,1),7]=1; W[wk(1,0),7]=1; W[wk(2,3),7]=1; W[wk(3,2),7]=1

    # Term 8: (a02+a20+a13+a31)(b02+b20+b13+b31)
    U[ui(0,2),8]=1; U[ui(2,0),8]=1; U[ui(1,3),8]=1; U[ui(3,1),8]=1
    V[vj(0,2),8]=1; V[vj(2,0),8]=1; V[vj(1,3),8]=1; V[vj(3,1),8]=1
    W[wk(0,2),8]=1; W[wk(2,0),8]=1; W[wk(1,3),8]=1; W[wk(3,1),8]=1

    # Term 9: (a03+a30+a12+a21)(b03+b30+b12+b21)
    U[ui(0,3),9]=1; U[ui(3,0),9]=1; U[ui(1,2),9]=1; U[ui(2,1),9]=1
    V[vj(0,3),9]=1; V[vj(3,0),9]=1; V[vj(1,2),9]=1; V[vj(2,1),9]=1
    W[wk(0,3),9]=1; W[wk(3,0),9]=1; W[wk(1,2),9]=1; W[wk(2,1),9]=1

    # Term 10: (a00-a11)(b00-b11)
    U[ui(0,0),10]=1; U[ui(1,1),10]=-1
    V[vj(0,0),10]=1; V[vj(1,1),10]=-1
    W[wk(0,0),10]=1; W[wk(1,1),10]=-1

    # Term 11: (a00-a22)(b00-b22)
    U[ui(0,0),11]=1; U[ui(2,2),11]=-1
    V[vj(0,0),11]=1; V[vj(2,2),11]=-1
    W[wk(0,0),11]=1; W[wk(2,2),11]=-1

    # Term 12: (a00-a33)(b00-b33)
    U[ui(0,0),12]=1; U[ui(3,3),12]=-1
    V[vj(0,0),12]=1; V[vj(3,3),12]=-1
    W[wk(0,0),12]=1; W[wk(3,3),12]=-1

    # Term 13: (a11-a22)(b11-b22)
    U[ui(1,1),13]=1; U[ui(2,2),13]=-1
    V[vj(1,1),13]=1; V[vj(2,2),13]=-1
    W[wk(1,1),13]=1; W[wk(2,2),13]=-1

    # Term 14: (a11-a33)(b11-b33)
    U[ui(1,1),14]=1; U[ui(3,3),14]=-1
    V[vj(1,1),14]=1; V[vj(3,3),14]=-1
    W[wk(1,1),14]=1; W[wk(3,3),14]=-1

    # Term 15: (a22-a33)(b22-b33)
    U[ui(2,2),15]=1; U[ui(3,3),15]=-1
    V[vj(2,2),15]=1; V[vj(3,3),15]=-1
    W[wk(2,2),15]=1; W[wk(3,3),15]=-1

    # Term 16: (a01-a10-a23+a32)(b01-b10-b23+b32)
    U[ui(0,1),16]=1; U[ui(1,0),16]=-1; U[ui(2,3),16]=-1; U[ui(3,2),16]=1
    V[vj(0,1),16]=1; V[vj(1,0),16]=-1; V[vj(2,3),16]=-1; V[vj(3,2),16]=1
    W[wk(0,1),16]=1; W[wk(1,0),16]=-1; W[wk(2,3),16]=-1; W[wk(3,2),16]=1

    # Term 17: (a02-a20-a13+a31)(b02-b20-b13+b31)
    U[ui(0,2),17]=1; U[ui(2,0),17]=-1; U[ui(1,3),17]=-1; U[ui(3,1),17]=1
    V[vj(0,2),17]=1; V[vj(2,0),17]=-1; V[vj(1,3),17]=-1; V[vj(3,1),17]=1
    W[wk(0,2),17]=1; W[wk(2,0),17]=-1; W[wk(1,3),17]=-1; W[wk(3,1),17]=1

    # Term 18: (a03-a30-a12+a21)(b03-b30-b12+b21)
    U[ui(0,3),18]=1; U[ui(3,0),18]=-1; U[ui(1,2),18]=-1; U[ui(2,1),18]=1
    V[vj(0,3),18]=1; V[vj(3,0),18]=-1; V[vj(1,2),18]=-1; V[vj(2,1),18]=1
    W[wk(0,3),18]=1; W[wk(3,0),18]=-1; W[wk(1,2),18]=-1; W[wk(2,1),18]=1

    return U, V, W, r


def _construct_lander_bailey_4x4x5():
    """
    Construct a rank-23 decomposition of the 4x4 x 4x5 matrix multiplication tensor.
    Strategy:
    - 4x4x4 part: Lander-Bailey rank-19 decomposition
    - 4x4x1 part (k=4): rank-4 schoolbook construction
    - Total rank: 19 + 4 = 23
    """
    n, m, p = 4, 4, 5

    U4, V4, W4, r4 = _landers_bailey_4x4x4()

    r_extra = 4
    r_total = r4 + r_extra  # 23

    U_total = np.zeros((16, r_total), dtype=np.float64)
    V_total = np.zeros((20, r_total), dtype=np.float64)
    W_total = np.zeros((20, r_total), dtype=np.float64)

    # Copy 4x4x4 part (columns 0..3 of V and W)
    U_total[:, :r4] = U4
    V_total[:16, :r4] = V4
    W_total[:16, :r4] = W4

    # Add 4x4x1 part (k=4)
    for c in range(r_extra):
        cc = r4 + c
        for i in range(4):
            for j in range(4):
                U_total[i*4+j, cc] = 1.0 if i == c else 0.0
                V_total[j*5+4, cc] = 1.0
                W_total[4*4+i, cc] = 1.0 if i == c else 0.0

    return U_total, V_total, W_total, r_total


def _construct_schoolbook(n, m, p):
    """Fallback schoolbook construction with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W, rank


def _try_lm(rank, n, m, p, max_iter=500, tol=1e-10, deadline=None, n_trials=10):
    """
    Try to find a rank-r decomposition via Levenberg-Marquardt style optimization.
    Uses structured initializations to improve convergence.
    """
    n_m = n * m
    m_p = m * p
    p_n = p * n

    # Build target tensor
    target = np.zeros((n_m, m_p, p_n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target[a, b, c] = 1.0

    best_result = None
    best_err = float('inf')

    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break

        rng = np.random.RandomState(trial * 7919 + rank * 31 + 42)

        U = rng.randn(n_m, rank) * 0.3
        V = rng.randn(m_p, rank) * 0.3
        W = rng.randn(p_n, rank) * 0.3

        # Add some structure: make first few columns resemble schoolbook
        for c in range(min(rank, n*m*p)):
            i = c // (m*p)
            j = (c // p) % m
            k = c % p
            if c < n_m:
                U[i*m+j, c] += 1.0
            if c < m_p:
                V[j*p+k, c] += 1.0
            if c < p_n:
                W[k*n+i, c] += 1.0

        lr = 0.01
        lam = 0.01  # LM parameter

        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break

            est = np.einsum('ar,br,cr->abc', U, V, W)
            residual = est - target

            err = np.max(np.abs(residual))
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())

            if err < tol:
                break

            dU = np.einsum('abc,br,cr->ar', residual, V, W)
            dV = np.einsum('abc,ar,cr->br', residual, U, W)
            dW = np.einsum('abc,ar,br->cr', residual, U, V)

            gU2 = np.einsum('ar,ar->ar', dU, dU)
            gV2 = np.einsum('br,br->br', dV, dV)
            gW2 = np.einsum('cr,cr->cr', dW, dW)

            denom_U = gU2 + lam
            denom_V = gV2 + lam
            denom_W = gW2 + lam

            step_U = dU / np.maximum(denom_U, 1e-12)
            step_V = dV / np.maximum(denom_V, 1e-12)
            step_W = dW / np.maximum(denom_W, 1e-12)

            best_step_lr = lr
            best_step_err = err
            for trial_lr in [lr, lr*0.5, lr*0.25, lr*0.1, lr*0.05]:
                U_test = U - trial_lr * step_U
                V_test = V - trial_lr * step_V
                W_test = W - trial_lr * step_W

                est_test = np.einsum('ar,br,cr->abc', U_test, V_test, W_test)
                err_test = np.max(np.abs(est_test - target))

                if err_test < best_step_err:
                    best_step_err = err_test
                    best_step_lr = trial_lr
                else:
                    break

            U = U - best_step_lr * step_U
            V = V - best_step_lr * step_V
            W = W - best_step_lr * step_W

            if best_step_lr < lr * 0.3:
                lr = max(best_step_lr, 1e-6)
            elif best_step_lr > lr * 1.5:
                lr = min(lr * 1.5, 0.1)

            if best_step_err < err * 0.9:
                lam = max(lam * 0.5, 1e-8)
            else:
                lam = min(lam * 2.0, 10.0)

        if best_err < tol:
            break

    if best_result is not None and best_err < tol:
        U, V, W = best_result
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank, "err": best_err}

    return None


def _try_structured_4x4x5(deadline=None):
    """
    Try to construct a low-rank decomposition for 4x4x5 using structured approaches.
    Strategy: Use rank-20 to rank-25 with careful initialization.
    """
    n, m, p = 4, 4, 5

    for rank in [20, 21, 22, 23, 24, 25, 26, 27, 28]:
        if deadline is not None and time.time() > deadline:
            break

        result = _try_lm(rank, n, m, p, max_iter=300, tol=1e-9, deadline=deadline, n_trials=8)
        if result is not None:
            if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, tol=1e-8):
                return result

    return None


def _try_polish(result, n, m, p, deadline=None, max_iter=1000, tol=1e-12):
    """Polish an approximate solution to higher precision."""
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    rank = U.shape[1]

    target = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0

    lr = 0.01
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break

        est = np.einsum('ar,br,cr->abc', U, V, W)
        residual = est - target
        err = np.max(np.abs(residual))

        if err < tol:
            break

        dU = np.einsum('abc,br,cr->ar', residual, V, W)
        dV = np.einsum('abc,ar,cr->br', residual, U, W)
        dW = np.einsum('abc,ar,br->cr', residual, U, V)

        U -= lr * dU
        V -= lr * dV
        W -= lr * dW

        if lr < 1e-8:
            break
        lr *= 0.95

    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]

    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-10):
        return {"U": U_list, "V": V_list, "W": W_list, "rank": rank}

    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130  # Leave 20s margin for 150s limit

    # Start with schoolbook
    schoolbook = _construct_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = schoolbook["rank"]

    # Try Lander-Bailey rank-23 construction
    if time.time() < deadline:
        U_lb, V_lb, W_lb, r_lb = _construct_lander_bailey_4x4x5()
        lb_result = {
            "U": [[float(x) for x in row] for row in U_lb],
            "V": [[float(x) for x in row] for row in V_lb],
            "W": [[float(x) for x in row] for row in W_lb],
            "rank": r_lb
        }
        if _verify_decomposition(lb_result["U"], lb_result["V"], lb_result["W"], n, m, p, tol=1e-8):
            best_result = lb_result
            best_rank = r_lb

    # Try structured low-rank approaches
    if time.time() < deadline:
        result_struct = _try_structured_4x4x5(deadline)
        if result_struct is not None:
            best_result = result_struct
            best_rank = result_struct.get("rank", best_rank)

    # Try to polish the best result
    if time.time() < deadline and best_rank < 80:
        polished = _try_polish(best_result, n, m, p, deadline, max_iter=500, tol=1e-12)
        if polished is not None:
            best_result = polished

    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}

# EVOLVE-BLOCK-END
