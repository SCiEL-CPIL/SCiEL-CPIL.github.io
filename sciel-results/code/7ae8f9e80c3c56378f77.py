"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    schoolbook_rank = n * m * p  # 80
    
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build schoolbook decomposition as fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = schoolbook_rank
    
    def compute_error(U, V, W):
        """Compute the error between the decomposition and target tensor."""
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    error += (val - 1.0) ** 2
        return error
    
    def try_rank(target_rank, num_attempts=10):
        """Try to find a decomposition of the given rank."""
        for attempt in range(num_attempts):
            if time.time() > deadline - 10:
                return None
            
            # Random initialization
            U_try = np.random.randn(n * m, target_rank) * 0.5
            V_try = np.random.randn(m * p, target_rank) * 0.5
            W_try = np.random.randn(p * n, target_rank) * 0.5
            
            # Coordinate descent optimization
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for r in range(target_rank):
                    # For each (i,j), find optimal U[i*m+j, r]
                    for i in range(n):
                        for j in range(m):
                            # sum_k W[k*n+i, r] * V[j*p+k, r] * U[i*m+j, r] should contribute to T[i,j,k]
                            # Optimal U[i*m+j, r] = sum_k (1 - sum_{r'!=r} U[i*m+j,r']*V[j*p+k,r']*W[k*n+i,r']) * V[j*p+k,r] * W[k*n+i,r] / sum_k (V[j*p+k,r]*W[k*n+i,r])**2
                            num = 0.0
                            den = 0.0
                            for k in range(p):
                                v_val = V_try[j * p + k, r]
                                w_val = W_try[k * n + i, r]
                                # Compute current contribution from other ranks
                                other_contrib = 0.0
                                for r2 in range(target_rank):
                                    if r2 != r:
                                        other_contrib += U_try[i * m + j, r2] * V_try[j * p + k, r2] * W_try[k * n + i, r2]
                                target = 1.0 - other_contrib
                                num += target * v_val * w_val
                                den += (v_val * w_val) ** 2
                            if den > 1e-10:
                                U_try[i * m + j, r] = num / den
                
                # Update V
                for r in range(target_rank):
                    for j in range(m):
                        for k in range(p):
                            num = 0.0
                            den = 0.0
                            for i in range(n):
                                u_val = U_try[i * m + j, r]
                                w_val = W_try[k * n + i, r]
                                other_contrib = 0.0
                                for r2 in range(target_rank):
                                    if r2 != r:
                                        other_contrib += U_try[i * m + j, r2] * V_try[j * p + k, r2] * W_try[k * n + i, r2]
                                target = 1.0 - other_contrib
                                num += target * u_val * w_val
                                den += (u_val * w_val) ** 2
                            if den > 1e-10:
                                V_try[j * p + k, r] = num / den
                
                # Update W
                for r in range(target_rank):
                    for k in range(p):
                        for i in range(n):
                            num = 0.0
                            den = 0.0
                            for j in range(m):
                                u_val = U_try[i * m + j, r]
                                v_val = V_try[j * p + k, r]
                                other_contrib = 0.0
                                for r2 in range(target_rank):
                                    if r2 != r:
                                        other_contrib += U_try[i * m + j, r2] * V_try[j * p + k, r2] * W_try[k * n + i, r2]
                                target = 1.0 - other_contrib
                                num += target * u_val * v_val
                                den += (u_val * v_val) ** 2
                            if den > 1e-10:
                                W_try[k * n + i, r] = num / den
                
                # Check convergence
                if iteration % 20 == 0:
                    err = compute_error(U_try, V_try, W_try)
                    if err < 1e-6:
                        return (U_try, V_try, W_try)
        
        return None
    
    # Try decreasing ranks
    for target_rank in [50, 49, 48, 47, 46, 45, 40, 35, 30]:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(target_rank, num_attempts=5)
        if result is not None:
            U_try, V_try, W_try = result
            err = compute_error(U_try, V_try, W_try)
            if err < 1e-6:
                best_U = U_try
                best_V = V_try
                best_W = W_try
                best_rank = target_rank
                break
    
    # Convert to lists with integer/half-integer entries if possible
    def to_list_with_ints(arr):
        result = []
        for i in range(arr.shape[0]):
            row = []
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Check if it's close to an integer or half-integer
                if abs(val - round(val)) < 1e-6:
                    row.append(int(round(val)))
                elif abs(val - (round(val * 2) / 2)) < 1e-6:
                    row.append(round(val * 2) / 2)
                else:
                    row.append(float(val))
            result.append(row)
        return result
    
    U_list = to_list_with_ints(best_U)
    V_list = to_list_with_ints(best_V)
    W_list = to_list_with_ints(best_W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
