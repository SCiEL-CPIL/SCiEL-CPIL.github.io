"""Combined program: exploit tensor structure of 3x3x6 via 3x3x3 core + direct sum."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=6, tol=1e-9):
    """Verify exact tensor decomposition."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    nm, mp, pn = n * m, m * p, p * n
    r = U.shape[1]
    if U.shape[0] != nm or V.shape[0] != mp or W.shape[0] != pn:
        return False
    if V.shape[1] != r or W.shape[1] != r:
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.dot(U[a], V[b] * W[c])
                if abs(val - 1.0) > tol:
                    return False
    for a in range(nm):
        ia, ja = divmod(a, m)
        for b in range(mp):
            jb, kb = divmod(b, p)
            for c in range(pn):
                kc, ic = divmod(c, n)
                if not (ja == jb and kb == kc and ic == ia):
                    val = np.dot(U[a], V[b] * W[c])
                    if abs(val) > tol:
                        return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank


def _verify_3x3x3(U1, V1, W1, tol=1e-9):
    """Verify 3x3x3 decomposition: U1,V1,W1 all (9, r)."""
    for a in range(9):
        ia, ja = divmod(a, 3)
        for b in range(9):
            jb, kb = divmod(b, 3)
            for c in range(9):
                kc, ic = divmod(c, 3)
                target = 1.0 if (ja == jb and kb == kc and ic == ia) else 0.0
                val = np.dot(U1[a], V1[b] * W1[c])
                if abs(val - target) > tol:
                    return False
    return True


def _find_3x3x3_decomp(target_rank, max_iter=3000, seed=42, deadline=None):
    """Try to find a rank-target_rank decomposition for 3x3x3 via gradient descent."""
    rng = np.random.RandomState(seed)
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    constraints = []
    for a in range(9):
        ia, ja = divmod(a, 3)
        for b in range(9):
            jb, kb = divmod(b, 3)
            for c in range(9):
                kc, ic = divmod(c, 3)
                target = 1.0 if (ja == jb and kb == kc and ic == ia) else 0.0
                constraints.append((a, b, c, target))
    
    for attempt in range(3):
        U = rng.randn(9, target_rank) * 0.3
        V = rng.randn(9, target_rank) * 0.3
        W = rng.randn(9, target_rank) * 0.3
        
        lr = 0.005
        for it in range(max_iter):
            if deadline and time.time() > deadline - 20:
                break
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            err = 0.0
            for a, b, c, target in constraints:
                val = np.dot(U[a], V[b] * W[c])
                e = val - target
                err += e * e
                dU[a] += 2 * e * V[b] * W[c]
                dV[b] += 2 * e * U[a] * W[c]
                dW[c] += 2 * e * U[a] * V[b]
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            if err < 1e-14:
                break
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        if best_err < 1e-10:
            break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    if time.time() > deadline - 10:
        return best_result
    
    # Try to find low-rank 3x3x3 decomposition, then extend to 3x3x6 via direct sum
    for r in range(19, 28):
        if time.time() > deadline - 30:
            break
        result = _find_3x3x3_decomp(r, max_iter=2000, seed=r, deadline=deadline)
        if result is None:
            continue
        U1, V1, W1 = result
        if not _verify_3x3x3(U1, V1, W1):
            continue
        
        # Build 3x3x6 via direct sum
        rank_36 = 2 * r
        U36 = np.zeros((9, rank_36))
        V36 = np.zeros((18, rank_36))
        W36 = np.zeros((18, rank_36))
        
        for i in range(3):
            for j in range(3):
                U36[i*3+j, :r] = U1[i*3+j, :]
                U36[i*3+j, r:] = U1[i*3+j, :]
        
        for j in range(3):
            for k in range(3):
                V36[j*6+k, :r] = V1[j*3+k, :]
                V36[j*6+3+k, r:] = V1[j*3+k, :]
        
        for k in range(3):
            for i in range(3):
                W36[k*3+i, :r] = W1[k*3+i, :]
                W36[(3+k)*3+i, r:] = W1[k*3+i, :]
        
        if _verify(U36, V36, W36, 3, 3, 6):
            best_rank = rank_36
            best_result = {"U": U36.tolist(), "V": V36.tolist(), "W": W36.tolist()}
            break
    
    return best_result
# EVOLVE-BLOCK-END
