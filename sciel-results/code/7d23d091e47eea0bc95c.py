import time
import random
import numpy as np

DIMENSIONS = [3, 3, 5]

def _validate(U, V, W, n, m, p):
    """Check if the decomposition is exact."""
    rank = len(V[0]) if V else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor for matrix multiplication C[i,k] = sum_j A[i,j]*B[j,k]
    # We need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all valid i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _target(n, m, p):
    """Build the target tensor T[i,j,k] = 1 for all valid i,j,k."""
    T = np.ones((n, m, p), dtype=float)
    return T

def _tensor_from(U, V, W, n, m, p, r):
    """Compute the tensor from the decomposition."""
    Ua = U.reshape(n, m, r)
    Va = V.reshape(m, p, r)
    Wa = W.reshape(p, n, r)
    return np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)

def _error_fast(U, V, W, n, m, p, r):
    """Faster error computation using vectorized computation."""
    R = _tensor_from(U, V, W, n, m, p, r)
    return float(np.sum((R - 1.0) ** 2))

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 45

    # Coefficients to try: integers and half-integers
    # Expand range slightly
    coeffs = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    rng = random.Random(777)

    def rand_init(r, seed):
        r_rng = random.Random(seed)
        U = np.array([[r_rng.choice(coeffs) for _ in range(r)] for _ in range(n*m)], dtype=float)
        V = np.array([[r_rng.choice(coeffs) for _ in range(r)] for _ in range(m*p)], dtype=float)
        W = np.array([[r_rng.choice(coeffs) for _ in range(r)] for _ in range(p*n)], dtype=float)
        return U, V, W

    def local_search(r, iters, seed):
        r_rng = random.Random(seed)
        U, V, W = rand_init(r, seed)
        cur_err = _error_fast(U, V, W, n, m, p, r)
        best_err = cur_err
        best_state = (U.copy(), V.copy(), W.copy())
        
        for it in range(iters):
            if time.time() > deadline - 2:
                break
            which = r_rng.randint(0, 2)
            if which == 0:
                row = r_rng.randint(0, n*m-1); col = r_rng.randint(0, r-1)
                old = U[row, col]; U[row, col] = r_rng.choice(coeffs)
                new_err = _error_fast(U, V, W, n, m, p, r)
                if new_err < cur_err:
                    cur_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_state = (U.copy(), V.copy(), W.copy())
                        if best_err < 1e-8:
                            break
                else:
                    U[row, col] = old
            elif which == 1:
                row = r_rng.randint(0, m*p-1); col = r_rng.randint(0, r-1)
                old = V[row, col]; V[row, col] = r_rng.choice(coeffs)
                new_err = _error_fast(U, V, W, n, m, p, r)
                if new_err < cur_err:
                    cur_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_state = (U.copy(), V.copy(), W.copy())
                        if best_err < 1e-8:
                            break
                else:
                    V[row, col] = old
            else:
                row = r_rng.randint(0, p*n-1); col = r_rng.randint(0, r-1)
                old = W[row, col]; W[row, col] = r_rng.choice(coeffs)
                new_err = _error_fast(U, V, W, n, m, p, r)
                if new_err < cur_err:
                    cur_err = new_err
                    if new_err < best_err:
                        best_err = new_err
                        best_state = (U.copy(), V.copy(), W.copy())
                        if best_err < 1e-8:
                            break
                else:
                    W[row, col] = old
        return best_err, best_state

    def polish(U, V, W, r, seed):
        """Final polish pass with more iterations and stricter acceptance."""
        p_rng = random.Random(seed)
        cur_err = _error_fast(U, V, W, n, m, p, r)
        best_err = cur_err
        best_state = (U.copy(), V.copy(), W.copy())
        
        for it in range(5000):
            if time.time() > deadline - 3:
                break
            which = p_rng.randint(0, 2)
            if which == 0:
                row = p_rng.randint(0, n*m-1); col = p_rng.randint(0, r-1)
                old = U[row, col]
                for c in coeffs:
                    if c == old:
                        continue
                    U[row, col] = c
                    new_err = _error_fast(U, V, W, n, m, p, r)
                    if new_err < cur_err:
                        cur_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_state = (U.copy(), V.copy(), W.copy())
                            if best_err < 1e-8:
                                return best_state
                        break
                    elif new_err == cur_err and p_rng.random() < 0.1:
                        break
                else:
                    U[row, col] = old
            elif which == 1:
                row = p_rng.randint(0, m*p-1); col = p_rng.randint(0, r-1)
                old = V[row, col]
                for c in coeffs:
                    if c == old:
                        continue
                    V[row, col] = c
                    new_err = _error_fast(U, V, W, n, m, p, r)
                    if new_err < cur_err:
                        cur_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_state = (U.copy(), V.copy(), W.copy())
                            if best_err < 1e-8:
                                return best_state
                        break
                    elif new_err == cur_err and p_rng.random() < 0.1:
                        break
                else:
                    V[row, col] = old
            else:
                row = p_rng.randint(0, p*n-1); col = p_rng.randint(0, r-1)
                old = W[row, col]
                for c in coeffs:
                    if c == old:
                        continue
                    W[row, col] = c
                    new_err = _error_fast(U, V, W, n, m, p, r)
                    if new_err < cur_err:
                        cur_err = new_err
                        if new_err < best_err:
                            best_err = new_err
                            best_state = (U.copy(), V.copy(), W.copy())
                            if best_err < 1e-8:
                                return best_state
                        break
                    elif new_err == cur_err and p_rng.random() < 0.1:
                        break
                else:
                    W[row, col] = old
        return best_state

    start_time = time.time()
    deadline = start_time + 145  # Leave 5s safety margin

    # Try ranks below schoolbook, from low to high.
    # The lower bound is max(n*m, m*p, p*n) = max(9, 15, 15) = 15.
    for r in range(15, best_rank):
        if time.time() > deadline - 8:
            break
        # a few restarts per rank
        for restart in range(10):
            if time.time() > deadline - 5:
                break
            # Scale iterations based on remaining time
            remaining = deadline - time.time()
            iters = max(2000, int(20000 * remaining / 30))
            err, state = local_search(r, iters, seed=1000 + r*10 + restart)
            if err < 1e-8:
                U2, V2, W2 = state
                # verify with original error function
                if _validate(U2.tolist(), V2.tolist(), W2.tolist(), n, m, p):
                    best_rank = r
                    best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                    return best
            elif err < 10 and time.time() < deadline - 10:
                # If we're close, do a polish pass
                U2, V2, W2 = state
                polished = polish(U2, V2, W2, r, seed=5000 + r*10 + restart)
                U3, V3, W3 = polished
                if _validate(U3.tolist(), V3.tolist(), W3.tolist(), n, m, p):
                    best_rank = r
                    best = {"U": U3.tolist(), "V": V3.tolist(), "W": W3.tolist()}
                    return best
    return best
