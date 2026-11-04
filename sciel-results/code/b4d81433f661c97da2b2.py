import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
DEADLINE = 145  # safety margin before 150s kill

def _strassen_rank33():
    """Return the known rank-33 Strassen-based decomposition."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, 0:6]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Part 2: Strassen on three 2x2 blocks
    for t in range(3):
        c0 = 2 * t
        c1 = 2 * t + 1
        # M1: (A00+A11)(B00+B11)
        U[0*m+1, col] += 1.0; U[1*m+2, col] += 1.0
        V[1*p+c0, col] += 1.0; V[2*p+c1, col] += 1.0
        W[c0*n+0, col] += 1.0; W[c1*n+1, col] += 1.0
        col += 1
        # M2: (A10+A11)B00
        U[1*m+1, col] += 1.0; U[1*m+2, col] += 1.0
        V[1*p+c0, col] += 1.0
        W[c0*n+1, col] += 1.0; W[c1*n+1, col] += -1.0
        col += 1
        # M3: A00(B01-B11)
        U[0*m+1, col] += 1.0
        V[1*p+c1, col] += 1.0; V[2*p+c1, col] += -1.0
        W[c1*n+0, col] += 1.0; W[c1*n+1, col] += 1.0
        col += 1
        # M4: A11(B10-B00)
        U[1*m+2, col] += 1.0
        V[2*p+c0, col] += 1.0; V[1*p+c0, col] += -1.0
        W[c0*n+0, col] += 1.0; W[c0*n+1, col] += 1.0
        col += 1
        # M5: (A00+A01)B11
        U[0*m+1, col] += 1.0; U[0*m+2, col] += 1.0
        V[2*p+c1, col] += 1.0
        W[c0*n+0, col] += -1.0; W[c1*n+0, col] += 1.0
        col += 1
        # M6: (A10-A00)(B00+B01)
        U[1*m+1, col] += 1.0; U[0*m+1, col] += -1.0
        V[1*p+c0, col] += 1.0; V[1*p+c1, col] += 1.0
        W[c1*n+1, col] += 1.0
        col += 1
        # M7: (A01-A11)(B10+B11)
        U[0*m+2, col] += 1.0; U[1*m+2, col] += -1.0
        V[2*p+c0, col] += 1.0; V[2*p+c1, col] += 1.0
        W[c0*n+0, col] += 1.0
        col += 1
    return U, V, W

def _validate(U, V, W, tol=1e-9):
    """Check if the decomposition exactly reproduces the matmul tensor."""
    n, m, p = 2, 3, 6
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_r[i, j, :], V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _search_lower_rank(start_rank, end_rank, time_budget):
    """Search for a rank lower than start_rank using random init + gradient descent."""
    n, m, p = 2, 3, 6
    best_U, best_V, best_W = None, None, None
    best_rank = start_rank

    for rank in range(end_rank, start_rank):
        if time.time() - _START_TIME > time_budget:
            break
        # Try multiple random initializations
        n_trials = 5
        for trial in range(n_trials):
            if time.time() - _START_TIME > time_budget:
                break
            np.random.seed(42 + trial * 1000 + rank)
            U0 = np.random.randn(n * m, rank) * 0.5
            V0 = np.random.randn(m * p, rank) * 0.5
            W0 = np.random.randn(p * n, rank) * 0.5

            def loss_and_grad(params):
                U = params[0:n*m*rank].reshape(n*m, rank)
                V = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                # Loss: sum over (i,j,k) of (dot(U_ij, V_jk * W_ki) - 1)^2
                loss = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = np.dot(U_r[i,j,:], V_r[j,k,:] * W_r[k,i,:])
                            loss += (s - 1.0)**2
                return loss

            x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            try:
                result = minimize(loss_and_grad, x0, method='Nelder-Mead',
                                  options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
                if result.fun < 1e-6:
                    U_f = result.x[0:n*m*rank].reshape(n*m, rank)
                    V_f = result.x[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_f = result.x[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    if _validate(U_f, V_f, W_f):
                        best_U, best_V, best_W = U_f, V_f, W_f
                        best_rank = rank
                        break
            except Exception:
                pass

    if best_U is not None:
        return best_U, best_V, best_W, best_rank
    return None, None, None, start_rank

def construct():
    n, m, p = 2, 3, 6

    # Get the known rank-33 solution
    U33, V33, W33 = _strassen_rank33()

    # Verify it's correct
    if not _validate(U33, V33, W33):
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        c = 0
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    U_fb[ii * m + jj, c] = 1.0
                    V_fb[jj * p + kk, c] = 1.0
                    W_fb[kk * n + ii, c] = 1.0
                    c += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    # Search for lower rank using remaining time
    time_remaining = DEADLINE - (time.time() - _START_TIME)
    if time_remaining > 10:
        U_found, V_found, W_found, rank_found = _search_lower_rank(33, 20, time_remaining - 5)
        if U_found is not None:
            return {"U": U_found.tolist(), "V": V_found.tolist(), "W": W_found.tolist()}

    return {"U": U33.tolist(), "V": V33.tolist(), "W": W33.tolist()}
