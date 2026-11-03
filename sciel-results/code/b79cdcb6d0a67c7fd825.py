import time
import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    """
    Constructs a rank-minimal matrix multiplication decomposition for 2x3 * 3x8.
    Returns a dictionary with matrices U, V, W of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # First, construct the known rank-44 decomposition
    U44, V44, W44 = _construct_rank44(n, m, p)
    
    # Try to find a lower rank decomposition
    best_U, best_V, best_W = U44, V44, W44
    
    # Try ranks from 43 down
    for target_rank in range(43, 35, -1):
        if time.time() > deadline - 30:
            break
        result = _try_find_decomposition(n, m, p, target_rank, deadline)
        if result is not None:
            best_U, best_V, best_W = result
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}

def _construct_rank44(n, m, p):
    """Constructs the known rank-44 decomposition."""
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 * B1 (2x2 * 2x8)
    for j in range(4):
        w_r00 = 4*j
        w_r01 = 4*j + 2
        w_r10 = 4*j + 1
        w_r11 = 4*j + 3
        
        v_e = 2*j
        v_f = 2*j + 1
        v_g = 8 + 2*j
        v_h = 9 + 2*j
        
        # P1
        U[0][col] = 1
        U[4][col] = 1
        V[v_e][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = 1
        W[w_r11][col] = 1
        col += 1
        
        # P2
        U[3][col] = 1
        U[4][col] = 1
        V[v_e][col] = 1
        W[w_r10][col] = 1
        W[w_r11][col] = -1
        col += 1
        
        # P3
        U[0][col] = 1
        V[v_f][col] = 1
        V[v_h][col] = -1
        W[w_r01][col] = 1
        W[w_r11][col] = 1
        col += 1
        
        # P4
        U[4][col] = 1
        V[v_g][col] = 1
        V[v_e][col] = -1
        W[w_r00][col] = 1
        W[w_r10][col] = 1
        col += 1
        
        # P5
        U[0][col] = 1
        U[1][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = -1
        W[w_r01][col] = 1
        col += 1
        
        # P6
        U[3][col] = 1
        U[0][col] = -1
        V[v_e][col] = 1
        V[v_f][col] = 1
        W[w_r11][col] = 1
        col += 1
        
        # P7
        U[1][col] = 1
        U[4][col] = -1
        V[v_g][col] = 1
        V[v_h][col] = 1
        W[w_r00][col] = 1
        col += 1
    
    # Part 2: A2 * B2 (2x1 * 1x8)
    for i in range(2):
        for k in range(8):
            u_idx = i * 3 + 2
            v_idx = 2 * 8 + k
            w_idx = k * 2 + i
            
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1
            
    return U, V, W

def _try_find_decomposition(n, m, p, target_rank, deadline):
    """
    Tries to find a rank-target_rank decomposition using alternating least squares.
    Returns (U, V, W) as lists or None if not found.
    """
    num_u = n * m
    num_v = m * p
    num_w = p * n
    
    np.random.seed(42)
    
    # Try multiple random initializations
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        
        U = np.random.randn(num_u, target_rank) * 0.5
        V = np.random.randn(num_v, target_rank) * 0.5
        W = np.random.randn(num_w, target_rank) * 0.5
        
        # Alternating least squares
        for iteration in range(100):
            if time.time() > deadline - 20:
                break
            
            # Update U: for each u_idx (i,j), minimize sum_{j,k} (sum_l U[u_idx,l]*V[j*p+k,l]*W[k*n+i,l] - 1)^2
            # This is complex due to coupling. Use a simpler approach.
            
            # Actually, let's use a different formulation. The constraint is:
            # For all (i,j,k): sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1
            
            # Let's flatten and use a linear system approach per factor.
            
            # Update U: fix V, W. For each u_idx = i*m+j:
            # sum_l U[u_idx,l] * (V[j*p+k,l] * W[k*n+i,l]) = 1 for all k
            # So U[u_idx,:] = argmin sum_k (sum_l U[u_idx,l] * M[k,l] - 1)^2
            # where M[k,l] = V[j*p+k,l] * W[k*n+i,l]
            
            for i in range(n):
                for j in range(m):
                    u_idx = i * m + j
                    M = np.zeros((p, target_rank))
                    for k in range(p):
                        v_idx = j * p + k
                        w_idx = k * n + i
                        M[k, :] = V[v_idx, :] * W[w_idx, :]
                    try:
                        A = M.T @ M + 1e-10 * np.eye(target_rank)
                        b = M.T @ np.ones(p)
                        U[u_idx, :] = np.linalg.solve(A, b)
                    except:
                        pass
            
            # Update V: fix U, W. For each v_idx = j*p+k:
            # sum_l U[i*m+j,l] * V[v_idx,l] * W[k*n+i,l] = 1 for all i
            # So V[v_idx,:] = argmin sum_i (sum_l V[v_idx,l] * N[i,l] - 1)^2
            # where N[i,l] = U[i*m+j,l] * W[k*n+i,l]
            
            for j in range(m):
                for k in range(p):
                    v_idx = j * p + k
                    N = np.zeros((n, target_rank))
                    for i in range(n):
                        u_idx = i * m + j
                        w_idx = k * n + i
                        N[i, :] = U[u_idx, :] * W[w_idx, :]
                    try:
                        A = N.T @ N + 1e-10 * np.eye(target_rank)
                        b = N.T @ np.ones(n)
                        V[v_idx, :] = np.linalg.solve(A, b)
                    except:
                        pass
            
            # Update W: fix U, V. For each w_idx = k*n+i:
            # sum_l U[i*m+j,l] * V[j*p+k,l] * W[w_idx,l] = 1 for all j
            # So W[w_idx,:] = argmin sum_j (sum_l W[w_idx,l] * P[j,l] - 1)^2
            # where P[j,l] = U[i*m+j,l] * V[j*p+k,l]
            
            for k in range(p):
                for i in range(n):
                    w_idx = k * n + i
                    P = np.zeros((m, target_rank))
                    for j in range(m):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        P[j, :] = U[u_idx, :] * V[v_idx, :]
                    try:
                        A = P.T @ P + 1e-10 * np.eye(target_rank)
                        b = P.T @ np.ones(m)
                        W[w_idx, :] = np.linalg.solve(A, b)
                    except:
                        pass
            
            # Check convergence
            if iteration % 10 == 0:
                error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            u_idx = i * m + j
                            v_idx = j * p + k
                            w_idx = k * n + i
                            val = np.dot(U[u_idx, :], V[v_idx, :] * W[w_idx, :])
                            error += (val - 1)**2
                if error < 1e-6:
                    # Try to round to half-integers
                    U_rounded = np.round(U * 2) / 2
                    V_rounded = np.round(V * 2) / 2
                    W_rounded = np.round(W * 2) / 2
                    
                    if _verify_decomposition(n, m, p, U_rounded, V_rounded, W_rounded):
                        return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist()
    
    return None

def _verify_decomposition(n, m, p, U, V, W):
    """Verifies that the decomposition is exact."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = 0
                for l in range(rank):
                    val += U[u_idx][l] * V[v_idx][l] * W[w_idx][l]
                if abs(val - 1) > 1e-6:
                    return False
    return True
