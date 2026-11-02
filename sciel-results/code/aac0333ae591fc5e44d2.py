import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """
    Verify if U, V, W form an exact decomposition for n x m by m x p matrix multiplication.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    """
    r = U.shape[1]
    # Precompute tensor indices for speed
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = 0.0
                            # Vectorized dot product for the r-th column
                            # U[a*m+b, :] dot V[c*p+d, :] dot W[k*n+i, :]
                            # This is a tensor contraction.
                            # s = sum_l U[a*m+b, l] * V[c*p+d, l] * W[k*n+i, l]
                            u_vec = U[a * m + b, :]
                            v_vec = V[c * p + d, :]
                            w_vec = W[k * n + i, :]
                            s = np.dot(np.dot(u_vec, v_vec), w_vec) # Wait, this is (u*v) then dot w?
                            # No, sum_l u_l v_l w_l.
                            # np.dot(u_vec, v_vec) is scalar. That's wrong.
                            # It's sum_l (u_l * v_l * w_l).
                            s = np.sum(u_vec * v_vec * w_vec)
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def get_residual(U, V, W, n, m, p):
    """
    Calculate the squared error of the decomposition.
    """
    r = U.shape[1]
    error = 0.0
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            u_vec = U[a * m + b, :]
                            v_vec = V[c * p + d, :]
                            w_vec = W[k * n + i, :]
                            s = np.sum(u_vec * v_vec * w_vec)
                            error += (s - target) ** 2
    return error

def solve_233(rank, time_limit=10):
    """
    Try to find a rank decomposition for 2x3x3 multiplication.
    Returns (U, V, W) or None.
    """
    n, m, p = 2, 3, 3
    U_dim = n * m # 6
    V_dim = m * p # 9
    W_dim = p * n # 6
    
    best_U = None
    best_V = None
    best_W = None
    best_err = float('inf')
    
    start_time = time.time()
    
    for restart in range(50): # Multiple restarts
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = np.random.uniform(-1, 1, (U_dim, rank))
        V = np.random.uniform(-1, 1, (V_dim, rank))
        W = np.random.uniform(-1, 1, (W_dim, rank))
        
        # Flatten
        x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x_flat):
            U_f = x_flat[:U_dim*rank].reshape(U_dim, rank)
            V_f = x_flat[U_dim*rank:U_dim*rank+V_dim*rank].reshape(V_dim, rank)
            W_f = x_flat[U_dim*rank+V_dim*rank:].reshape(W_dim, rank)
            return get_residual(U_f, V_f, W_f, n, m, p)
        
        # Use Nelder-Mead for local optimization (robust to non-smoothness)
        res = minimize(objective, x, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        
        if res.fun < best_err:
            best_err = res.fun
            x_best = res.x
            best_U = x_best[:U_dim*rank].reshape(U_dim, rank)
            best_V = x_best[U_dim*rank:U_dim*rank+V_dim*rank].reshape(V_dim, rank)
            best_W = x_best[U_dim*rank+V_dim*rank:].reshape(W_dim, rank)
            
            # Check if we found an exact solution (or very close)
            if best_err < 1e-6:
                # Try to snap to half-integers
                U_snap = np.round(best_U * 2) / 2
                V_snap = np.round(best_V * 2) / 2
                W_snap = np.round(best_W * 2) / 2
                
                if check_decomposition(U_snap, V_snap, W_snap, n, m, p):
                    return U_snap, V_snap, W_snap
                
                # If snapping fails, try integer snapping
                U_snap_int = np.round(best_U)
                V_snap_int = np.round(best_V)
                W_snap_int = np.round(best_W)
                if check_decomposition(U_snap_int, V_snap_int, W_snap_int, n, m, p):
                    return U_snap_int, V_snap_int, W_snap_int

    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Fallback: Known Rank 34 solution (from current program logic, cleaned up)
    def get_fallback_rank34():
        U_comb = np.zeros((6, 34))
        V_comb = np.zeros((18, 34))
        W_comb = np.zeros((12, 34))
        
        U233_seed = np.zeros((6, 17))
        V233_seed = np.zeros((9, 17))
        W233_seed = np.zeros((6, 17))
        
        # Part 1: M_222 (Strassen) - Rank 7
        U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
        V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
        W233_seed[0, 0] = 1; W233_seed[3, 0] = 1
        
        U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
        V233_seed[0, 1] = 1
        W233_seed[1, 1] = 1; W233_seed[3, 1] = -1
        
        U233_seed[0, 2] = 1
        V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
        W233_seed[2, 2] = 1; W233_seed[3, 2] = 1
        
        U233_seed[4, 3] = 1
        V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
        W233_seed[0, 3] = 1; W233_seed[1, 3] = 1
        
        U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
        V233_seed[4, 4] = 1
        W233_seed[0, 4] = -1; W233_seed[2, 4] = 1
        
        U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
        V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
        W233_seed[3, 5] = 1
        
        U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
        V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
        W233_seed[0, 6] = 1
        
        # Part 2: M_221 (Rank 4)
        U233_seed[0, 7] = 1
        V233_seed[2, 7] = 1
        W233_seed[4, 7] = 1
        
        U233_seed[1, 8] = 1
        V233_seed[5, 8] = 1
        W233_seed[4, 8] = 1
        
        U233_seed[3, 9] = 1
        V233_seed[2, 9] = 1
        W233_seed[5, 9] = 1
        
        U233_seed[4, 10] = 1
        V233_seed[5, 10] = 1
        W233_seed[5, 10] = 1
        
        # Part 3: M_213 (Rank 6)
        U233_seed[2, 11] = 1
        V233_seed[6, 11] = 1
        W233_seed[0, 11] = 1
        
        U233_seed[2, 12] = 1
        V233_seed[7, 12] = 1
        W233_seed[2, 12] = 1
        
        U233_seed[2, 13] = 1
        V233_seed[8, 13] = 1
        W233_seed[4, 13] = 1
        
        U233_seed[5, 14] = 1
        V233_seed[6, 14] = 1
        W233_seed[1, 14] = 1
        
        U233_seed[5, 15] = 1
        V233_seed[7, 15] = 1
        W233_seed[3, 15] = 1
        
        U233_seed[5, 16] = 1
        V233_seed[8, 16] = 1
        W233_seed[5, 16] = 1
        
        U_comb[:, :17] = U233_seed
        U_comb[:, 17:] = U233_seed
        
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
                V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
                
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
                W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
                
        return U_comb, V_comb, W_comb

    # Try to find a better decomposition for 2x3x3
    # If we find rank 16, total rank is 32.
    # If we find rank 15, total rank is 30.
    
    # We try ranks 16 and 17. 18 is schoolbook.
    # We allocate time to search.
    
    found_233 = None
    best_rank_233 = 18 # Schoolbook
    
    # Try Rank 16
    if time.time() < deadline - 20:
        res16 = solve_233(16, time_limit=15)
        if res16 is not None:
            found_233 = res16
            best_rank_233 = 16
    
    # Try Rank 17 (only if 16 failed or if we have time)
    if found_233 is None and time.time() < deadline - 10:
        res17 = solve_233(17, time_limit=10)
        if res17 is not None:
            found_233 = res17
            best_rank_233 = 17
            
    if found_233 is not None:
        U233, V233, W233 = found_233
        R = best_rank_233
        
        # Construct 2x3x6 by duplicating the 2x3x3 solution
        # U: (6, 2R)
        # V: (18, 2R)
        # W: (12, 2R)
        
        U_comb = np.zeros((6, 2 * R))
        V_comb = np.zeros((18, 2 * R))
        W_comb = np.zeros((12, 2 * R))
        
        # Copy to first half
        U_comb[:, :R] = U233
        U_comb[:, R:] = U233
        
        # V mapping: 
        # V233 is 9xR. Indices j*3+k.
        # V_comb is 18x2R. Indices j*6+k.
        # Block 1 (k=0..2) uses V233. Block 2 (k=3..5) uses V233.
        for j in range(3):
            for k in range(3):
                V_comb[j * 6 + k, :R] = V233[j * 3 + k]
                V_comb[j * 6 + 3 + k, R:] = V233[j * 3 + k]
                
        # W mapping:
        # W233 is 6xR. Indices k*2+i.
        # W_comb is 12x2R. Indices k*2+i.
        # Block 1 (k=0..2) uses W233. Block 2 (k=3..5) uses W233.
        for k in range(3):
            for i in range(2):
                W_comb[k * 2 + i, :R] = W233[k * 2 + i]
                W_comb[(k + 3) * 2 + i, R:] = W233[k * 2 + i]
                
        # Final verification
        if check_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
        else:
            # If verification fails, fallback
            pass
            
    # Fallback to Rank 34
    U_comb, V_comb, W_comb = get_fallback_rank34()
    if check_decomposition(U_comb, V_comb, W_comb, n, m, p):
        return {
            "U": U_comb.tolist(),
            "V": V_comb.tolist(),
            "W": W_comb.tolist()
        }
        
    # Last resort: Schoolbook Rank 36
    U_comb = np.zeros((6, 36))
    V_comb = np.zeros((18, 36))
    W_comb = np.zeros((12, 36))
    
    idx = 0
    for a in range(2):
        for b in range(3):
            for k in range(6):
                U_comb[a * 3 + b, idx] = 1
                V_comb[b * 6 + k, idx] = 1
                W_comb[k * 2 + a, idx] = 1
                idx += 1
                
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
