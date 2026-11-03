import numpy as np
import time
from fractions import Fraction
import itertools

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin

    def _verify(U, V, W):
        """
        Verify if U, V, W form a valid decomposition for M_{n,m,p}.
        U: (nm, rank), V: (mp, rank), W: (pn, rank)
        Condition: sum_c W[k*n+i, c] * U[i*m+j, c] * V[j'*p+k, c] == delta(j, j') for all i, k, j, j'
        """
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        s = 0
                        for c in range(rank):
                            s += W[k * n + i][c] * U[i * m + j][c] * V[jp * p + k][c]
                        target = 1 if j == jp else 0
                        if s != target:
                            return False
        return True

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Strategy: Try to find a rank 12 decomposition using Winograd/Fourier insights or structured search.
    # If not found quickly, fall back to schoolbook.
    
    # Known bound for 2x3x4 is often cited around 12-14. 
    # Let's try a structured search for rank 12 first.
    # If we find a valid rank 12, we return it.
    
    # To make the search tractable, we can try to fix some structure or use random restarts with local optimization.
    # However, given the time limit, a simple random search for rank 12 with integer/half-integer entries 
    # might not converge. 
    
    # Instead, let's try a few known small ranks or use a heuristic.
    # Actually, the prompt asks to combine strategies. 
    # Inspiration 2 used ALS. Inspiration 3 used a specific target rank 12.
    
    # Let's attempt a quick random search for rank 12 with small integer values.
    # If it fails within a short time, we return schoolbook.
    
    target_rank = 12
    max_restarts = 500
    max_local_steps = 20
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
            
        # Initialize with random small integers
        U = np.random.randint(-2, 3, size=(nm, target_rank))
        V = np.random.randint(-2, 3, size=(mp, target_rank))
        W = np.random.randint(-2, 3, size=(pn, target_rank))
        
        # Check if exact
        if _verify(U.tolist(), V.tolist(), W.tolist()):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
        # Local search: try to reduce error by flipping single elements
        # Error function: sum (sum_c W[k*n+i,c] U[i*m+j,c] V[jp*p+k,c] - delta(j,jp))^2
        # This is expensive to compute fully each time, but n,m,p are small.
        
        for _ in range(max_local_steps):
            if time.time() > deadline - 10:
                break
                
            # Pick a random matrix and a random element
            mat_name = np.random.choice(['U', 'V', 'W'])
            if mat_name == 'U':
                idx = np.random.randint(nm * target_rank)
                row, col = divmod(idx, target_rank)
                old_val = U[row, col]
                U[row, col] = np.random.randint(-2, 3)
            elif mat_name == 'V':
                idx = np.random.randint(mp * target_rank)
                row, col = divmod(idx, target_rank)
                old_val = V[row, col]
                V[row, col] = np.random.randint(-2, 3)
            else:
                idx = np.random.randint(pn * target_rank)
                row, col = divmod(idx, target_rank)
                old_val = W[row, col]
                W[row, col] = np.random.randint(-2, 3)
                
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # Revert if not exact (simple hill climbing on 0/1 exactness)
                if mat_name == 'U':
                    U[row, col] = old_val
                elif mat_name == 'V':
                    V[row, col] = old_val
                else:
                    W[row, col] = old_val

    # If no rank 12 found, try rank 13? Or just fallback.
    # Given the difficulty, fallback to schoolbook is safest for validity.
    return _schoolbook()

# EVOLVE-BLOCK-START
# EVOLVE-BLOCK-END
