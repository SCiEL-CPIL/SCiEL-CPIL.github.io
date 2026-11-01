import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe
    
    # Schoolbook solution as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n*m, rank_schoolbook))
    V_sb = np.zeros((m*p, rank_schoolbook))
    W_sb = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    def to_list(mat):
        return [[float(x) for x in row] for row in mat]
    
    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition"""
        n, m, p = DIMENSIONS
        # T[i,j,k] should equal sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # For matrix multiplication, T[i,j,k] = 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(len(U[0])):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to find lower rank solutions
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try ranks from 23 down to 10
    for rank in range(rank_schoolbook - 1, 9, -1):
        if time.time() > deadline - 5:
            break
        
        # Try random half-integer matrices
        for attempt in range(1000):
            if time.time() > deadline - 5:
                break
            
            # Generate random half-integer matrices
            U = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(n*m, rank))
            V = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(m*p, rank))
            W = np.random.choice([-1, -0.5, 0, 0.5, 1], size=(p*n, rank))
            
            if verify(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                break
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }
