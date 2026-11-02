import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 2, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 8, 32, 16
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W, rank
    
    # Strassen coefficients for 2x2 * 2x2
    strassen_terms = [
        {'x': {(0,0):1, (1,1):1}, 'y': {(0,0):1, (1,1):1}, 'z': {(0,0):1, (1,1):1}},
        {'x': {(1,0):1, (1,1):1}, 'y': {(0,0):1}, 'z': {(1,0):1, (1,1):-1}},
        {'x': {(0,0):1}, 'y': {(0,1):1, (1,1):-1}, 'z': {(0,1):1, (1,1):1}},
        {'x': {(1,1):1}, 'y': {(1,0):1, (0,0):-1}, 'z': {(0,0):1, (1,0):1}},
        {'x': {(0,0):1, (0,1):1}, 'y': {(1,1):1}, 'z': {(0,0):-1, (0,1):1}},
        {'x': {(1,0):1, (0,0):-1}, 'y': {(0,0):1, (0,1):1}, 'z': {(1,1):1}},
        {'x': {(0,1):1, (1,1):-1}, 'y': {(1,0):1, (1,1):1}, 'z': {(0,0):1}},
    ]
    
    def strassen_56():
        """8 independent 2x2 x 2x2 Strassen blocks, rank 56."""
        rank = 56
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        col = 0
        for a in range(2):  # A block column: 2a, 2a+1
            for b in range(4):  # B column block: 2b, 2b+1
                for t in range(7):
                    term = strassen_terms[t]
                    for (i_prime, j_prime), coeff in term['x'].items():
                        u_idx = i_prime * m + (2*a + j_prime)
                        U[u_idx, col] += coeff
                    for (j_prime, k_prime), coeff in term['y'].items():
                        v_idx = (2*a + j_prime) * p + (2*b + k_prime)
                        V[v_idx, col] += coeff
                    for (i_prime, k_prime), coeff in term['z'].items():
                        w_idx = (2*b + k_prime) * n + i_prime
                        W[w_idx, col] += coeff
                    col += 1
        
        return U, V, W, rank
    
    def verify(U, V, W, rank, atol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :rank] * V[j*p+k, :rank] * W[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=atol):
                        return False
        return True
    
    # Start with Strassen (rank 56)
    U_best, V_best, W_best, rank_best = strassen_56()
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    # Try to find linear dependencies among the 56 terms by checking
    # if any subset of terms can be removed while maintaining correctness
    # This is a combinatorial search, so we'll try a greedy approach
    
    def try_remove_term(col_idx, U, V, W, rank):
        """Check if removing term at col_idx still gives correct result."""
        if time.time() > deadline - 20:
            return None
        
        U_new = U.copy()
        V_new = V.copy()
        W_new = W.copy()
        
        # Zero out the column
        U_new[:, col_idx] = 0
        V_new[:, col_idx] = 0
        W_new[:, col_idx] = 0
        
        # Verify
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_new[i*m+j, :rank] * V_new[j*p+k, :rank] * W_new[k*n+i, :rank])
                    if not np.isclose(val, 1.0, atol=1e-9):
                        return None
        return U_new, V_new, W_new, rank
    
    # Try removing each term one by one (greedy)
    removed = []
    for col in range(rank_best):
        if time.time() > deadline - 20:
            break
        result = try_remove_term(col, U_best, V_best, W_best, rank_best)
        if result is not None:
            U_best, V_best, W_best, rank_best = result
            removed.append(col)
    
    # Final verification
    if not verify(U_best, V_best, W_best, rank_best):
        U_best, V_best, W_best, rank_best = schoolbook()
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
