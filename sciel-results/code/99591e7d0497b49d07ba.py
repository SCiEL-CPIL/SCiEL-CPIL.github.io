"""Matrix multiplication 2x3x9: rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3x9 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try numerical search for lower rank
    lower_bound = max(n * m, m * p, p * n)
    
    for rank in range(lower_bound, best_rank):
        if time.time() > deadline - 30:
            break
        result = _try_rank_als(n, m, p, rank, deadline)
        if result is not None and _verify_exact(n, m, p, result):
            best_rank = rank
            best_result = result
    
    if time.time() > deadline - 10:
        return best_result
    
    # Rationalize
    if best_rank < n * m * p:
        cleaned = _rationalize(best_result)
        if cleaned is not None and _verify_exact(n, m, p, cleaned):
            best_result = cleaned
    
    return best_result

def _schoolbook(n, m, p):
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

def _verify_exact(n, m, p, result):
    U = np.array(result["U"], dtype=np.float64)
    V = np.array(result["V"], dtype=np.float64)
    W = np.array(result["W"], dtype=np.float64)
    rank = U.shape[1]
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _try_rank_als(n, m, p, rank, deadline):
    n_m, m_p, p_n = n * m, m * p, p * n
    for seed in range(15):
        if time.time() > deadline - 20:
            break
        np.random.seed(seed * 1000 + rank)
        U = np.random.randn(n_m, rank) * 0.3
        V = np.random.randn(m_p, rank) * 0.3
        W = np.random.randn(p_n, rank) * 0.3
        for it in range(80):
            if time.time() > deadline - 15:
                break
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    try:
                        U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                    except:
                        pass
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    try:
                        V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    except:
                        pass
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    try:
                        W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                    except:
                        pass
            max_res = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                        max_res = max(max_res, abs(s - 1.0))
            if max_res < 1e-8:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return None

def _rationalize(result):
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    U2, V2, W2 = np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2
    if np.allclose(U2, U, atol=1e-6) and np.allclose(V2, V, atol=1e-6) and np.allclose(W2, W, atol=1e-6):
        return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
    return None

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0]) if result else 'None'}")
