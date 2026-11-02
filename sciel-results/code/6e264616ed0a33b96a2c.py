import time
import numpy as np
import scipy.optimize

DIMENSIONS = [2, 3, 3]
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin

def construct():
    n, m, p = DIMENSIONS
    rank_sb = n * m * p # 18
    
    # Schoolbook decomposition
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
                
    def to_half_int_list(matrix):
        # Round to nearest 0.5
        scaled = np.rint(matrix * 2.0)
        result = scaled / 2.0
        return [[float(v) for v in row] for row in result]

    def validate(U, V, W):
        # Check if U, V, W (numpy arrays) decompose the tensor exactly
        # T[u,v,w] = sum_c U[u,c] V[v,c] W[w,c]
        # u in 0..n*m-1, v in 0..m*p-1, w in 0..p*n-1
        # Target is 1 if (u,v,w) corresponds to (i,j,k) with u=i*m+j, v=j*p+k, w=k*n+i
        # and 0 otherwise.
        
        # We can check all 324 entries
        for u in range(n*m):
            i = u // m
            j = u % m
            for v in range(m*p):
                v_j = v // p
                k = v % p
                if j != v_j:
                    target = 0.0
                else:
                    # Check w consistency
                    # w must be k*n + i
                    w_target = k*n + i
                    # If w != w_target, target is 0
                    # But we are iterating w in the loop? No, we need to check all w.
                    pass
            
        # More efficient: Iterate all u, v, w
        # This is 6*9*6 = 324 iterations. Fast enough.
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    val = np.sum(U[u, :] * V[v, :] * W[w, :])
                    
                    # Determine target
                    i_u = u // m
                    j_u = u % m
                    j_v = v // p
                    k_v = v % p
                    k_w = w // n
                    i_w = w % n
                    
                    if i_u == i_w and j_u == j_v and k_v == k_w:
                        target = 1.0
                    else:
                        target = 0.0
                        
                    if abs(val - target) > 1e-8:
                        return False
        return True

    # Try to find lower rank
    # We will try rank 17
    # Use least_squares
    
    def residual(x, rank):
        # x is flattened U, V, W
        # U size: n*m*rank
        # V size: m*p*rank
        # W size: p*n*rank
        u_size = n * m * rank
        v_size = m * p * rank
        w_size = p * n * rank
        
        U_flat = x[:u_size]
        V_flat = x[u_size:u_size+v_size]
        W_flat = x[u_size+v_size:u_size+v_size+w_size]
        
        U = U_flat.reshape((n*m, rank))
        V = V_flat.reshape((m*p, rank))
        W = W_flat.reshape((p*n, rank))
        
        res = []
        for u in range(n*m):
            for v in range(m*p):
                for w in range(p*n):
                    val = np.dot(U[u, :], V[v, :] * W[w, :])
                    
                    i_u = u // m
                    j_u = u % m
                    j_v = v // p
                    k_v = v % p
                    k_w = w // n
                    i_w = w % n
                    
                    if i_u == i_w and j_u == j_v and k_v == k_w:
                        target = 1.0
                    else:
                        target = 0.0
                        
                    res.append(val - target)
        return res

    def try_rank(rank):
        if time.time() > START_TIME + TIME_LIMIT - 10:
            return None
            
        # Initial guess: random or schoolbook truncated?
        # Schoolbook truncated might be a bad start for rank < 18 because the removed column creates a large error.
        # Random start is better.
        
        attempts = 20
        for _ in range(attempts):
            if time.time() > START_TIME + TIME_LIMIT - 5:
                break
                
            u_size = n * m * rank
            v_size = m * p * rank
            w_size = p * n * rank
            total_size = u_size + v_size + w_size
            
            # Random initialization
            x0 = np.random.randn(total_size) * 0.5
            
            try:
                sol = scipy.optimize.least_squares(residual, x0, args=(rank,), method='lm', max_nfev=1000)
            except Exception:
                continue
                
            if sol.success or sol.cost < 1e-12:
                # Check if it's a valid solution
                U_sol = sol.x[:u_size].reshape((n*m, rank))
                V_sol = sol.x[u_size:u_size+v_size].reshape((m*p, rank))
                W_sol = sol.x[u_size+v_size:u_size+v_size+w_size].reshape((p*n, rank))
                
                # Snap to half-integers
                U_snapped = np.rint(U_sol * 2) / 2
                V_snapped = np.rint(V_sol * 2) / 2
                W_snapped = np.rint(W_sol * 2) / 2
                
                if validate(U_snapped, V_snapped, W_snapped):
                    return (U_snapped, V_snapped, W_snapped)
                    
        return None

    best_result = None
    for r in [17, 16]:
        res = try_rank(r)
        if res:
            best_result = res
            break
            
    if best_result:
        U, V, W = best_result
        return {"U": to_half_int_list(U), "V": to_half_int_list(V), "W": to_half_int_list(W)}
    else:
        # Return schoolbook
        return {"U": to_half_int_list(U_sb), "V": to_half_int_list(V_sb), "W": to_half_int_list(W_sb)}
