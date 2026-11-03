"""Structured Strassen-like decomposition for 3x3x6 with ALS refinement."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 3, 3, 6

    # Start with schoolbook (rank 54)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p

    def verify_decomposition(U, V, W, rank, n, m, p, tol=1e-9):
        """Verify exact decomposition using numpy for speed."""
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            u3 = u.reshape(n, m)
            v3 = v.reshape(m, p)
            w3 = w.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += u3[i, j] * (v3[j, :] * w3[:, i])
        return np.all(np.abs(T - 1.0) <= tol)

    def compute_residual(U, V, W, rank):
        """Compute the residual tensor S - 1."""
        S = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            u = U[:, c]
            v = V[:, c]
            w = W[:, c]
            u3 = u.reshape(n, m)
            v3 = v.reshape(m, p)
            w3 = w.reshape(p, n)
            for i in range(n):
                for j in range(m):
                    S[i, j, :] += u3[i, j] * (v3[j, :] * w3[:, i])
        return S - 1.0

    def als_refine(U, V, W, rank, max_iters=200, deadline=None):
        """Alternating least squares refinement for the decomposition."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        best_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5.0:
                break
            
            # Solve for U
            for i in range(n):
                for j in range(m):
                    # For fixed (i,j), we need: sum_k U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:] = 1
                    # This is a linear system in U[i*m+j,:]
                    # M[k, r] = V[j*p+k, r] * W[k*n+i, r]
                    # We want M^T * U[i*m+j,:] = ones(p)
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(M.T, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    # For fixed (j,k), we need: sum_i U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:] = 1
                    # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(M.T, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    # For fixed (k,i), we need: sum_j U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:] = 1
                    # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(M.T, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Calculate error
            err = np.sum(compute_residual(U, V, W, rank) ** 2)
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_err < 1e-10:
                break
        
        return best_U, best_V, best_W, best_err

    def find_3x3x3_decomposition(max_rank=23, min_rank=15, max_time=30):
        """Find a rank-r decomposition for 3x3x3 matrix multiplication."""
        n3, m3, p3 = 3, 3, 3
        best = None
        best_r = max_rank + 1

        def verify_3x3x3(U, V, W, r):
            U = np.array(U, dtype=np.float64)
            V = np.array(V, dtype=np.float64)
            W = np.array(W, dtype=np.float64)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        s = np.sum(U[i*3+j] * V[j*3+k] * W[k*3+i])
                        if abs(s - 1.0) > 1e-9:
                            return False
            return True

        for r in range(max_rank, min_rank - 1, -1):
            if time.time() > start_time + max_time:
                break
            for attempt in range(50):
                if time.time() > start_time + max_time:
                    break
                rng = np.random.RandomState(attempt * 7919 + r * 31)
                vals = [-1, 0, 1]
                U = rng.choice(vals, size=(9, r))
                V = rng.choice(vals, size=(9, r))
                W = rng.choice(vals, size=(9, r))
                if verify_3x3x3(U, V, W, r):
                    best = (U, V, W, r)
                    best_r = r
                    break
            if best is not None:
                break
        return best

    # Try to find 3x3x3 decomposition
    dec3 = find_3x3x3_decomposition(max_rank=23, min_rank=15, max_time=20)

    if dec3 is not None:
        U3, V3, W3, r3 = dec3
        # Build 3x3x6 decomposition from 3x3x3
        r6 = 2 * r3
        U6 = np.zeros((9, r6))
        V6 = np.zeros((18, r6))
        W6 = np.zeros((18, r6))

        # Block 1: c in 0..r3-1, k in 0,1,2
        U6[:, :r3] = U3
        for j in range(3):
            for k in range(3):
                V6[j*6+k, :r3] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W6[k*3+i, :r3] = W3[k*3+i]

        # Block 2: c in r3..2*r3-1, k in 3,4,5
        U6[:, r3:] = U3
        for j in range(3):
            for k in range(3):
                V6[j*6+3+k, r3:] = V3[j*3+k]
        for k in range(3):
            for i in range(3):
                W6[(3+k)*3+i, r3:] = W3[k*3+i]

        if verify_decomposition(U6.tolist(), V6.tolist(), W6.tolist(), r6, n, m, p):
            if r6 < best_rank:
                best_rank = r6
                best_result = {
                    "U": U6.tolist(),
                    "V": V6.tolist(),
                    "W": W6.tolist()
                }

    # ============================================================
    # Strategy: Direct random search for 3x3x6 with ALS refinement
    # ============================================================
    for rank in range(min(50, best_rank - 1), 30, -1):
        if time.time() > deadline - 20:
            break

        max_attempts = 10 if rank > 40 else 15
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break

            rng = np.random.RandomState(attempt * 104729 + rank * 13)
            density = 0.4
            vals = [-1, 0, 1]
            U = rng.choice(vals, size=(9, rank))
            V = rng.choice(vals, size=(18, rank))
            W = rng.choice(vals, size=(18, rank))

            # Apply sparsity
            U = U * (rng.random((9, rank)) < density)
            V = V * (rng.random((18, rank)) < density)
            W = W * (rng.random((18, rank)) < density)

            # Try ALS refinement
            U_ref, V_ref, W_ref, err = als_refine(U, V, W, rank, max_iters=50, deadline=deadline)
            
            if err < 1e-6:
                # Round to half-integers and verify
                U_round = np.round(U_ref * 2) / 2
                V_round = np.round(V_ref * 2) / 2
                W_round = np.round(W_ref * 2) / 2
                
                if verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist(), rank, n, m, p):
                    best_rank = rank
                    best_result = {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                    break

    # ============================================================
    # Strategy: Local search - try to reduce rank from current best
    # ============================================================
    if best_rank < 54 and time.time() < deadline - 15:
        U_best = np.array(best_result["U"], dtype=np.float64)
        V_best = np.array(best_result["V"], dtype=np.float64)
        W_best = np.array(best_result["W"], dtype=np.float64)
        r_best = best_rank

        # Try removing columns (reducing rank)
        for target_rank in range(r_best - 1, max(30, r_best - 5), -1):
            if time.time() > deadline - 10:
                break
            for attempt in range(10):
                if time.time() > deadline - 5:
                    break
                rng = np.random.RandomState(attempt * 9973 + target_rank)
                cols = rng.choice(r_best, size=target_rank, replace=False)
                cols.sort()
                U_sub = U_best[:, cols]
                V_sub = V_best[:, cols]
                W_sub = W_best[:, cols]
                if verify_decomposition(U_sub.tolist(), V_sub.tolist(), W_sub.tolist(), target_rank, n, m, p):
                    best_rank = target_rank
                    best_result = {
                        "U": U_sub.tolist(),
                        "V": V_sub.tolist(),
                        "W": W_sub.tolist()
                    }
                    break

    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
# EVOLVE-BLOCK-END
