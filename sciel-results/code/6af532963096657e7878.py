import numpy as np
import time

def construct():
    """
    Find rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = 2, 3, 6
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin

    # Schoolbook solution: Rank = n*m*p = 36
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    # Try to find lower rank decompositions
    # Lower bound is max(nm, mp, pn) = 18
    # Try ranks from 18 to 35
    for rank in range(18, 36):
        if time.time() > deadline - 5:
            break
            
        # Try a few random initializations
        found = False
        for attempt in range(100):
            if time.time() > deadline - 2:
                break
                
            np.random.seed(attempt)
            # Initialize with random integers -1, 0, 1
            U = np.random.randint(-1, 2, size=(n * m, rank))
            V = np.random.randint(-1, 2, size=(m * p, rank))
            W = np.random.randint(-1, 2, size=(p * n, rank))
            
            # Check validity
            if _check_validity(n, m, p, U, V, W):
                best_U, best_V, best_W = U, V, W
                best_rank = rank
                found = True
                break
                
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _check_validity(n, m, p, U, V, W):
    """
    Check if the decomposition is valid.
    Condition: sum_c W[k*n+i, c] * U[i'*m+j, c] * V[j'*p+k', c] == delta(i,i')*delta(j,j')*delta(k,k')
    """
    rank = U.shape[1]
    
    # Precompute for efficiency?
    # The condition must hold for all i, i', j, j', k, k'
    # There are n^2 * m^2 * p^2 = 4 * 9 * 36 = 1296 checks
    # Each check sums over rank.
    
    for i in range(n):
        for i_prime in range(n):
            delta_ii = 1 if i == i_prime else 0
            for j in range(m):
                for j_prime in range(m):
                    delta_jj = 1 if j == j_prime else 0
                    for k in range(p):
                        for k_prime in range(p):
                            delta_kk = 1 if k == k_prime else 0
                            
                            target = delta_ii * delta_jj * delta_kk
                            
                            total = 0.0
                            for c in range(rank):
                                total += W[k * n + i, c] * U[i_prime * m + j, c] * V[j_prime * p + k_prime, c]
                                
                            if abs(total - target) > 1e-9:
                                return False
    return True
