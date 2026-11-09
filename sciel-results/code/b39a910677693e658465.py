"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def _verify(U, V, W, tol=1e-9):
    n, m, p = DIMENSIONS
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                # U index: i*m+j
                # V index: j*p+k
                # W index: k*n+i
                actual = np.dot(Ua[i*m+j], Va[j*p+k] * Wa[k*n+i])
                if abs(actual - expected) > tol:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _snap_to_half_int(A):
    """Snap values to nearest half-integer (0, 0.5, 1, 1.5, ...)"""
    return np.round(A * 2) / 2.0

def _compute_error(U, V, W, n, m, p):
    """Compute total squared error"""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                err += (val - 1.0)**2
    return err

def _local_search(U, V, W, max_iters=30, tol=1e-6, deadline=None):
    """
    Perform a local search on half-integer values to minimize error.
    Tries flipping signs or shifting by 0.5.
    """
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    current_err = _compute_error(U, V, W, n, m, p)
    if current_err < tol:
        return U, V, W, current_err

    U_c = U.copy()
    V_c = V.copy()
    W_c = W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try perturbing U
        for r in range(rank):
            for c in range(n*m):
                # Try adding 0.5
                U_c[c, r] += 0.5
                err = _compute_error(U_c, V_c, W_c, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    U_c[c, r] -= 0.5
                
                # Try subtracting 0.5
                U_c[c, r] -= 0.5
                err = _compute_error(U_c, V_c, W_c, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    U_c[c, r] += 0.5
        
        # Try perturbing V
        for r in range(rank):
            for c in range(m*p):
                V_c[c, r] += 0.5
                err = _compute_error(U_c, V_c, W_c, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    V_c[c, r] -= 0.5
                
                V_c[c, r] -= 0.5
                err = _compute_error(U_c, V_c, W_c, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    V_c[c, r] += 0.5

        # Try perturbing W
        for r in range(rank):
            for c in range(p*n):
                W_c[c, r] += 0.5
                err = _compute_error(U_c, V_c, W_c, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    W_c[c, r] -= 0.5
                
                W_c[c, r] -= 0.5
                err = _compute_error(U_c, V_c, W_c, n, m, p)
                if err < current_err - 1e-9:
                    current_err = err
                    improved = True
                else:
                    W_c[c, r] += 0.5

        if not improved:
            break
            
    return U_c, V_c, W_c, current_err

def _als_decompose_fast(rank, deadline, seed=42, num_restarts=15):
    """
    Fast vectorized ALS with snapping and local search.
    """
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n*m, m*p, p*n
    
    best_result = None
    best_err = float('inf')
    
    for restart in range(num_restarts):
        if time.time() > deadline - 2:
            break
            
        rng = np.random.RandomState(seed + restart)
        # Initialize with small random values
        scale = 0.3
        U = rng.randn(dim_U, rank) * scale
        V = rng.randn(dim_V, rank) * scale
        W = rng.randn(dim_W, rank) * scale
        
        for it in range(150):
            if time.time() > deadline - 2:
                break
                
            # Update U
            for i in range(n):
                for j in range(m):
                    M = np.zeros((rank, p))
                    for k in range(p):
                        M[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, n))
                    for i in range(n):
                        M[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
                        
            # Update W
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, m))
                    for j in range(m):
                        M[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check error
            err = _compute_error(U, V, W, n, m, p)
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                
            if best_err < 1e-12:
                break
                
        # Check if best solution is already close to half-integers
        if best_err < 1e-6:
            # Try to verify directly
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Snap to half-integers
        U_snap = _snap_to_half_int(best_U)
        V_snap = _snap_to_half_int(best_V)
        W_snap = _snap_to_half_int(best_W)
        
        # Local search
        U_ls, V_ls, W_ls, err_ls = _local_search(U_snap, V_snap, W_snap, max_iters=15, deadline=deadline-2)
        
        if err_ls < 1e-6:
            U_list = [[float(x) for x in row] for row in U_ls]
            V_list = [[float(x) for x in row] for row in V_ls]
            W_list = [[float(x) for x in row] for row in W_ls]
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def _try_structured_init(rank, deadline, seed=0):
    """
    Try structured initializations based on the problem structure.
    """
    n, m, p = DIMENSIONS
    dim_U, dim_V, dim_W = n*m, m*p, p*n
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Initialize based on schoolbook pattern
    # Each term corresponds to a (i,j,k) triple
    # Try to group terms
    
    for attempt in range(5):
        if time.time() > deadline - 2:
            break
            
        if attempt == 0:
            # Random but with structure: make some entries larger
            U = rng.randn(dim_U, rank)
            V = rng.randn(dim_V, rank)
            W = rng.randn(dim_W, rank)
            # Normalize columns
            for r in range(rank):
                nU = np.linalg.norm(U[:, r])
                nV = np.linalg.norm(V[:, r])
                nW = np.linalg.norm(W[:, r])
                if nU > 1e-10 and nV > 1e-10 and nW > 1e-10:
                    U[:, r] /= nU
                    V[:, r] /= nV
                    W[:, r] /= (nU * nV * nW)
        elif attempt == 1:
            # Use a rank-1 like structure
            U = np.zeros((dim_U, rank))
            V = np.zeros((dim_V, rank))
            W = np.zeros((dim_W, rank))
            for r in range(rank):
                # Random unit vectors
                u = rng.randn(dim_U)
                v = rng.randn(dim_V)
                w = rng.randn(dim_W)
                U[:, r] = u / np.linalg.norm(u)
                V[:, r] = v / np.linalg.norm(v)
                W[:, r] = w / np.linalg.norm(w)
        else:
            # Sparse initialization
            U = np.zeros((dim_U, rank))
            V = np.zeros((dim_V, rank))
            W = np.zeros((dim_W, rank))
            for r in range(rank):
                # Pick random non-zero entries
                nz_U = rng.choice(dim_U, size=max(1, dim_U//3), replace=False)
                nz_V = rng.choice(dim_V, size=max(1, dim_V//3), replace=False)
                nz_W = rng.choice(dim_W, size=max(1, dim_W//3), replace=False)
                U[nz_U, r] = rng.choice([-1, 1], size=len(nz_U))
                V[nz_V, r] = rng.choice([-1, 1], size=len(nz_V))
                W[nz_W, r] = rng.choice([-1, 1], size=len(nz_W))
        
        # Run ALS
        for it in range(100):
            if time.time() > deadline - 2:
                break
                
            for i in range(n):
                for j in range(m):
                    M = np.zeros((rank, p))
                    for k in range(p):
                        M[:, k] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[i*m+j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((rank, n))
                    for i in range(n):
                        M[:, i] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[j*p+k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
                        
            for k in range(p):
                for i in range(n):
                    M = np.zeros((rank, m))
                    for j in range(m):
                        M[:, j] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[k*n+i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            err = _compute_error(U, V, W, n, m, p)
            if err < 1e-12:
                break
        
        # Check
        U_snap = _snap_to_half_int(U)
        V_snap = _snap_to_half_int(V)
        W_snap = _snap_to_half_int(W)
        
        U_ls, V_ls, W_ls, err_ls = _local_search(U_snap, V_snap, W_snap, max_iters=10, deadline=deadline-2)
        
        if err_ls < 1e-6:
            U_list = [[float(x) for x in row] for row in U_ls]
            V_list = [[float(x) for x in row] for row in V_ls]
            W_list = [[float(x) for x in row] for row in W_ls]
            
            if _verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
                
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Fallback to schoolbook
    schoolbook_result = _schoolbook()
    best_rank = len(schoolbook_result['U'][0])
    best_result = schoolbook_result
    
    # Try to find lower rank decompositions
    # The theoretical lower bound for 2x3x5 is at least 2*3*5/2 = 15 (by Strassen-like bounds)
    # But let's try from 20 downwards... actually we should try from low to high
    # Known: rank(2,3,5) is at most 30 (schoolbook)
    # Let's try 20, 21, 22, ... up to 29
    
    for target_rank in range(20, best_rank):
        if time.time() > deadline - 5:
            break
            
        # Try structured initialization first
        result = _try_structured_init(target_rank, deadline, seed=target_rank)
        if result is None:
            # Try random ALS
            result = _als_decompose_fast(target_rank, deadline, num_restarts=8)
        
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found the lowest possible in this range
            
    return best_result
# EVOLVE-BLOCK-END
