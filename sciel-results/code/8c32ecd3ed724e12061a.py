"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if len(U) > 0 else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # T[(i,j), (jp,k), (kp,ip)] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    # Check: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 0.0
                            for c in range(rank):
                                lhs += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(lhs - rhs) > 1e-6:
                                return False
    return True

def _build_linear_systems(n, m, p):
    """
    Pre-build the linear systems for alternating least squares.
    For each factor, we need to solve a linear least squares problem.
    
    The tensor equation is:
    sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    
    For fixed U and V, solving for W:
    For each (kp, ip), we want to find W[kp*n+ip, :] such that
    sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    
    This is a linear system in W[kp*n+ip, :].
    Let's define for each (kp, ip):
    A[(i,j,jp,k), c] = U[i*m+j, c] * V[jp*p+k, c]
    b[(i,j,jp,k)] = delta(i,ip)*delta(j,jp)*delta(k,kp)
    
    Then A @ W[kp*n+ip, :] = b
    
    But note: the equation must hold for ALL (i,j,jp,k) given fixed (kp,ip).
    So we have n*m*m*p equations and r unknowns for each (kp,ip).
    
    Similarly for U and V.
    """
    # We'll build the matrices on the fly since they depend on the other two factors.
    # Instead, let's pre-compute the index mappings.
    
    # For W update: for each (kp, ip), build A matrix of shape (n*m*m*p, r)
    # A[idx, c] = U[i*m+j, c] * V[jp*p+k, c] where idx = ((i*m+j)*m+jp)*p+k
    # b[idx] = 1 if (i==ip and j==jp and k==kp) else 0
    
    # For U update: for each (i, j), build A matrix of shape (m*p*p*n, r)
    # A[idx, c] = V[jp*p+k, c] * W[kp*n+ip, c] where idx = ((jp*p+k)*p+kp)*n+ip
    # b[idx] = 1 if (i==ip and j==jp and k==kp) else 0
    
    # For V update: for each (jp, k), build A matrix of shape (n*m*p*n, r)
    # A[idx, c] = U[i*m+j, c] * W[kp*n+ip, c] where idx = ((i*m+j)*p+kp)*n+ip
    # b[idx] = 1 if (i==ip and j==jp and k==kp) else 0
    
    # Pre-compute the index mappings and delta vectors
    # For W:
    w_idx_map = []  # List of (i, j, jp, k) for each linear index
    w_delta = np.zeros(n*m*m*p)
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    idx = ((i*m+j)*m+jp)*p+k
                    w_idx_map.append((i, j, jp, k))
                    if j == jp:  # k and kp are handled per (kp, ip)
                        pass  # We'll handle delta in the b vector construction
    
    # Actually, it's easier to just build the systems on the fly with numpy.
    # Let's create helper functions.
    
    def build_w_system(U, V, ip, kp):
        """Build linear system for W[kp*n+ip, :] given U, V, and fixed (ip, kp)."""
        n_eq = n * m * m * p
        A = np.zeros((n_eq, len(U[0])))
        b = np.zeros(n_eq)
        idx = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        # A[idx, c] = U[i*m+j, c] * V[jp*p+k, c]
                        A[idx, :] = U[i*m+j, :] * V[jp*p+k, :]
                        # b[idx] = 1 if (i==ip and j==jp and k==kp) else 0
                        if i == ip and j == jp and k == kp:
                            b[idx] = 1.0
                        idx += 1
        return A, b
    
    def build_u_system(V, W, i, j):
        """Build linear system for U[i*m+j, :] given V, W, and fixed (i, j)."""
        n_eq = m * p * p * n
        A = np.zeros((n_eq, len(U[0]) if 'U' in dir() else len(V[0])))
        # Wait, we need to know the rank. Let's pass it or infer from V.
        r = V.shape[1]
        A = np.zeros((n_eq, r))
        b = np.zeros(n_eq)
        idx = 0
        for jp in range(m):
            for k in range(p):
                for kp in range(p):
                    for ip in range(n):
                        # A[idx, c] = V[jp*p+k, c] * W[kp*n+ip, c]
                        A[idx, :] = V[jp*p+k, :] * W[kp*n+ip, :]
                        # b[idx] = 1 if (i==ip and j==jp and k==kp) else 0
                        if j == jp and k == kp and i == ip:
                            b[idx] = 1.0
                        idx += 1
        return A, b
    
    def build_v_system(U, W, jp, k):
        """Build linear system for V[jp*p+k, :] given U, W, and fixed (jp, k)."""
        n_eq = n * m * p * n
        r = U.shape[1]
        A = np.zeros((n_eq, r))
        b = np.zeros(n_eq)
        idx = 0
        for i in range(n):
            for j in range(m):
                for kp in range(p):
                    for ip in range(n):
                        # A[idx, c] = U[i*m+j, c] * W[kp*n+ip, c]
                        A[idx, :] = U[i*m+j, :] * W[kp*n+ip, :]
                        # b[idx] = 1 if (i==ip and j==jp and k==kp) else 0
                        if j == jp and k == kp and i == ip:
                            b[idx] = 1.0
                        idx += 1
        return A, b
    
    return build_w_system, build_u_system, build_v_system

def _als_decompose(r, n, m, p, max_iters=500, n_restarts=20, deadline=None, seed_base=42):
    """
    Alternating Least Squares for tensor decomposition.
    """
    build_w_system, build_u_system, build_v_system = _build_linear_systems(n, m, p)
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 10:
            break
        
        np.random.seed(seed_base + restart * 7919 + r * 131)
        
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 10:
                break
            
            # Update W
            for ip in range(n):
                for kp in range(p):
                    A, b = build_w_system(U, V, ip, kp)
                    try:
                        W[kp*n+ip, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W[kp*n+ip, :] = np.zeros(r)
            
            # Update U
            for i in range(n):
                for j in range(m):
                    A, b = build_u_system(V, W, i, j)
                    try:
                        U[i*m+j, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U[i*m+j, :] = np.zeros(r)
            
            # Update V
            for jp in range(m):
                for k in range(p):
                    A, b = build_v_system(U, W, jp, k)
                    try:
                        V[jp*p+k, :], residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V[jp*p+k, :] = np.zeros(r)
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    lhs = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                                    loss += (lhs - rhs) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
        
        if best_loss < 1e-6:
            # Try to round to rational numbers
            for denom in [1, 2, 10, 100]:
                U_rounded = np.round(best_U * denom) / denom
                V_rounded = np.round(best_V * denom) / denom
                W_rounded = np.round(best_W * denom) / denom
                
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
    
    return None

def _try_rank_rational(r, n, m, p, max_iters=100, n_restarts=10, deadline=None):
    """
    Try to find a rank-r decomposition using alternating least squares with rational rounding.
    Uses the structure of the multiplication tensor more effectively.
    """
    if deadline is not None and time.time() > deadline - 5:
        return None
    
    for restart in range(n_restarts):
        if deadline is not None and time.time() > deadline - 5:
            break
        
        np.random.seed(restart * 7919 + r * 131 + 42)
        
        # Initialize with random values
        U = np.random.randn(n*m, r) * 0.1
        V = np.random.randn(m*p, r) * 0.1
        W = np.random.randn(p*n, r) * 0.1
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Alternating least squares
            # Fix U and V, solve for W
            # The tensor equation: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
            # For fixed (kp, ip), we need to find W[kp*n+ip, :] 
            # Actually, let's use a different approach: treat it as a tensor completion problem
            
            # Compute the "effective" target for each factor
            # For W: given U and V, the contribution to T from factors U and V is known
            # T_target[(i,j),(jp,k),(kp,ip)] = delta(i,ip)*delta(j,jp)*delta(k,kp)
            # T_approx[(i,j),(jp,k),(kp,ip)] = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
            
            # To update W: for each (kp, ip), we want to minimize
            # sum_{i,j,jp,k} (delta(i,ip)*delta(j,jp)*delta(k,kp) - sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c])^2
            # This is a linear least squares problem in W[kp*n+ip, :]
            
            # Build the system: for each (kp, ip), we have n*m*m*p equations
            # Let's vectorize: for fixed (kp, ip), define
            # A_{(i,j,jp,k), c} = U[i*m+j, c] * V[jp*p+k, c]
            # b_{(i,j,jp,k)} = delta(i,ip)*delta(j,jp)*delta(k,kp)
            # Then we solve A * W[kp*n+ip, :] = b in least squares sense
            
            # But this is n*m*m*p = 8*16 = 128 equations per (kp,ip) pair, and there are p*n = 12 such pairs
            # Total: 12 * 128 = 1536 equations, 12*r unknowns
            
            # Actually, let's just do gradient descent on the full loss for simplicity and speed
            
            # Compute loss and gradients
            loss = 0.0
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for jp in range(m):
                        for k in range(p):
                            for kp in range(p):
                                for ip in range(n):
                                    rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                    # Compute s = sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c]
                                    s = np.sum(U[i*m+j, :] * V[jp*p+k, :] * W[kp*n+ip, :])
                                    err = s - rhs
                                    loss += err * err
                                    # Gradients
                                    dU[i*m+j, :] += 2 * err * V[jp*p+k, :] * W[kp*n+ip, :]
                                    dV[jp*p+k, :] += 2 * err * U[i*m+j, :] * W[kp*n+ip, :]
                                    dW[kp*n+ip, :] += 2 * err * U[i*m+j, :] * V[jp*p+k, :]
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Adaptive learning rate
            lr = 0.01 / (1 + it * 0.01)
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_loss < 1e-6:
            # Try to round to rational numbers
            U_rounded = np.round(best_U * 1000) / 1000
            V_rounded = np.round(best_V * 1000) / 1000
            W_rounded = np.round(best_W * 1000) / 1000
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
            
            # If rounding didn't work, try with more precision
            U_rounded = np.round(best_U * 10000) / 10000
            V_rounded = np.round(best_V * 10000) / 10000
            W_rounded = np.round(best_W * 10000) / 10000
            
            U_list = U_rounded.tolist()
            V_list = V_rounded.tolist()
            W_list = W_rounded.tolist()
            
            if _verify(U_list, V_list, W_list, n, m, p):
                return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s safety margin before 120s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try to find lower rank decompositions
    # For 2x4x6, the known lower bound is around 2*4*6 - something
    # Try ranks from best_rank-1 down to a reasonable minimum
    min_rank_to_try = max(30, best_rank - 15)
    
    for r in range(best_rank - 1, min_rank_to_try - 1, -1):
        if time.time() > deadline:
            break
        
        result = _als_decompose(r, n, m, p, max_iters=100, n_restarts=5, deadline=deadline)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            print(f"Found rank {r} decomposition")
            break
    
    return best_result
# EVOLVE-BLOCK-END
