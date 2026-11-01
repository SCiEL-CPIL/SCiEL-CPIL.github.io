import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # 10s safety margin
    
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    for target_rank in range(29, 15, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random initializations
        max_iters = 50
        for attempt in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randint(-2, 3, (n*m, target_rank)) * 0.5
            V = np.random.randint(-2, 3, (m*p, target_rank)) * 0.5
            W = np.random.randint(-2, 3, (p*n, target_rank)) * 0.5
            
            # Check if this satisfies the constraints
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                        if abs(val - 1) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                best_rank = target_rank
                best_U, best_V, best_W = U, V, W
                break
        
        if best_rank < rank_sb:
            break
    
    # Convert to lists with integer/half-integer entries
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    return {"U": to_list(best_U), "V": to_list(best_V), "W": to_list(best_W)}
