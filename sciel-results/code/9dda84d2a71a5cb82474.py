import time
import numpy as np

DIMENSIONS = [3, 4, 8]
START_TIME = time.time()
TIME_LIMIT = 115.0  # Safety margin before 150s kill

def _build_true_tensor(n, m, p):
    """Build the true matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, num_tests=50):
    """Verify that the decomposition correctly computes C = A @ B using random integer tests."""
    import random
    rng = random.Random(12345)
    for _ in range(num_tests):
        A = [[rng.randint(-10, 10) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-10, 10) for _ in range(p)] for _ in range(m)]
        
        # True product
        true_C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Decomposition product
        rank = len(U[0]) if U else 0
        
        # u[i][t] = sum_j U[i*m+j][t] * A[i,j]
        u = [[0] * rank for _ in range(n)]
        for i in range(n):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += U[i * m + j][t] * A[i][j]
                u[i][t] = s
        
        # v[k][t] = sum_j V[j*p+k][t] * B[j,k]
        v = [[0] * rank for _ in range(p)]
        for k in range(p):
            for t in range(rank):
                s = 0
                for j in range(m):
                    s += V[j * p + k][t] * B[j, k]
                v[k][t] = s
        
        # C[k*n+i] = sum_t W[k*n+i][t] * u[i][t] * v[k][t]
        for i in range(n):
            for k in range(p):
                s = 0
                for t in range(rank):
                    s += W[k * n + i][t] * u[i][t] * v[k][t]
                if s != true_C[i][k]:
                    return False
    return True

def _verify_tensor(U, V, W, T_true, atol=1e-8):
    """Verify tensor decomposition via einsum."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    T_recon = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    return np.allclose(T_recon, T_true, atol=atol)

def _snap_to_half_inters(U, V, W):
    """Snap values to nearest half-integer."""
    U_r = np.round(np.array(U) * 2) / 2.0
    V_r = np.round(np.array(V) * 2) / 2.0
    W_r = np.round(np.array(W) * 2) / 2.0
    return U_r.tolist(), V_r.tolist(), W_r.tolist()

def _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10):
    """Run gradient descent to minimize ||T_recon - T_true||_F^2."""
    n, m, p = T_true.shape
    rank = U.shape[1]
    
    best_err = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T_true - T_recon
        err = np.max(np.abs(R))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < tol:
            break
        
        # Compute gradients
        grad_U = np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = np.einsum('abc,ar,br->cr', R, U, V)
        
        U += eta * grad_U
        V += eta * grad_V
        W += eta * grad_W
    
    return best_U, best_V, best_W, best_err

def _try_kronecker_structured(rank, T_true, nm, mp, pn, time_budget):
    """Try a Kronecker-structured initialization and optimize."""
    n, m, p = 3, 4, 8
    
    # Structure: j = 2*j1 + j2, k = 4*k1 + k2
    # For the 3x2x2 part (i, j1, k1): use schoolbook rank 12
    # For the 2x2x4 part (j2, k2, ...): this is tricky
    
    # Alternative: use random initialization with structure hints
    rng = np.random.RandomState(rank * 137)
    
    U = rng.randn(nm, rank) * 0.3
    V = rng.randn(mp, rank) * 0.3
    W = rng.randn(pn, rank) * 0.3
    
    eta = 0.005
    max_iters = 3000
    
    for phase in range(4):
        U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
        if err < 1e-8:
            # Try snapping
            U_s, V_s, W_s = _snap_to_half_inters(U, V, W)
            if _verify_tensor(U_s, V_s, W_s, T_true, atol=1e-6):
                return U_s, V_s, W_s
            if _verify_tensor(U.tolist(), V.tolist(), W.tolist(), T_true, atol=1e-6):
                return U.tolist(), V.tolist(), W.tolist()
        eta *= 0.5
        max_iters = max(200, max_iters // 2)
    
    return None

def _try_sparse_schoolbook(rank, T_true, nm, mp, pn, time_budget):
    """Start from schoolbook and try to find lower-rank by selecting subset and optimizing."""
    n, m, p = 3, 4, 8
    schoolbook_rank = n * m * p  # 96
    
    if rank >= schoolbook_rank:
        return None
    
    # Build schoolbook factors
    U_sb = np.zeros((nm, schoolbook_rank))
    V_sb = np.zeros((mp, schoolbook_rank))
    W_sb = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1
    
    for restart in range(8):
        if time.time() - START_TIME > time_budget:
            break
        
        rng = np.random.RandomState(restart * 999 + rank * 31)
        
        # Select random subset
        selected = rng.choice(schoolbook_rank, size=rank, replace=False)
        U = U_sb[:, selected].copy()
        V = V_sb[:, selected].copy()
        W = W_sb[:, selected].copy()
        
        # Add perturbation
        U += rng.randn(nm, rank) * 0.05
        V += rng.randn(mp, rank) * 0.05
        W += rng.randn(pn, rank) * 0.05
        
        eta = 0.008
        max_iters = 2500
        
        for phase in range(3):
            U, V, W, err = _gradient_descent(U, V, W, T_true, eta, max_iters, tol=1e-10)
            if err < 1e-8:
                U_s, V_s, W_s = _snap_to_half_inters(U, V, W)
                if _verify_tensor(U_s, V_s, W_s, T_true, atol=1e-6):
                    return U_s, V_s, W_s
                if _verify_tensor(U.tolist(), V.tolist(), W.tolist(), T_true, atol=1e-6):
                    return U.tolist(), V.tolist(), W.tolist()
            eta *= 0.5
            max_iters = max(150, max_iters // 2)
    
    return None

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition."""
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
    return U, V, W

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    deadline = START_TIME + TIME_LIMIT
    
    # Schoolbook fallback
    def schoolbook():
        U, V, W = _build_schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}
    
    T_true = _build_true_tensor(n, m, p)
    
    # Try to find lower-rank decompositions
    # Known lower bound: rank >= max(n*m, m*p, p*n) / something
    # For 3x4x8, try ranks from 48 down to 32
    
    best_result = schoolbook()
    best_rank = n * m * p  # 96
    
    # Strategy 1: Sparse schoolbook (select subset and optimize)
    for r in [64, 56, 48, 40, 36, 32]:
        if time.time() > deadline - 25:
            break
        if r >= best_rank:
            break
        
        result = _try_sparse_schoolbook(r, T_true, nm, mp, pn, deadline - 20)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p, num_tests=30):
                return {"U": U, "V": V, "W": W}
    
    # Strategy 2: Random initialization with gradient descent
    for r in [48, 40, 36, 32, 28]:
        if time.time() > deadline - 15:
            break
        if r >= best_rank:
            break
        
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            result = _try_kronecker_structured(r, T_true, nm, mp, pn, deadline - 10)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p, num_tests=30):
                    found = True
                    break
        
        if found:
            return {"U": U, "V": V, "W": W}
    
    return best_result
