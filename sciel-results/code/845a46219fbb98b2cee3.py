import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Default schoolbook solution
    def _construct_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_solution = _construct_schoolbook()
    
    # Helper: verify solution exactly
    def _verify(U_sol, V_sol, W_sol, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s_val = 0.0
                    for r in range(rank):
                        s_val += U_sol[i * m + j][r] * V_sol[j * p + k][r] * W_sol[k * n + i][r]
                    if abs(s_val - 1.0) > 1e-9:
                        return False
        return True
    
    # Helper: try to complete a partial solution (U, V fixed) to find W
    def _try_complete(U, V, rank):
        num_vars = p * n * rank
        num_eqs = n * m * p
        
        A = np.zeros((num_eqs, num_vars))
        b = np.ones(num_eqs)
        
        eq_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        coeff = U[i * m + j, r] * V[j * p + k, r]
                        if coeff != 0:
                            var_idx = (k * n + i) * rank + r
                            A[eq_idx, var_idx] = coeff
                    eq_idx += 1
        
        # Solve A W = b
        try:
            W_flat, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
        except:
            return None
        
        # Check if the solution is exact
        if not np.allclose(A @ W_flat, b, atol=1e-9):
            return None
        
        # Check if W entries are half-integers
        W_check = 2 * W_flat
        if not np.allclose(W_check, np.round(W_check), atol=1e-9):
            return None
        
        # Construct the solution matrices
        W_sol = [[0] * rank for _ in range(p * n)]
        for k in range(p):
            for i in range(n):
                for r in range(rank):
                    val = W_flat[(k * n + i) * rank + r]
                    val_int = int(np.round(val * 2)) / 2.0
                    W_sol[k * n + i][r] = val_int
        
        U_sol = U.tolist()
        V_sol = V.tolist()
        
        if _verify(U_sol, V_sol, W_sol, rank):
            return {"U": U_sol, "V": V_sol, "W": W_sol}
        
        return None
    
    # Search for better rank
    for rank in range(49, 39, -1):
        if time.time() > deadline - 5:
            break
        
        # Strategy 1: Random ±1, 0 entries
        for attempt in range(30):
            if time.time() > deadline - 2:
                break
            
            U = np.random.randint(-1, 2, (n * m, rank))
            V = np.random.randint(-1, 2, (m * p, rank))
            
            sol = _try_complete(U, V, rank)
            if sol is not None:
                return sol
        
        # Strategy 2: Structured U based on row/column patterns
        for attempt in range(20):
            if time.time() > deadline - 2:
                break
            
            # U[i,j,r]: pattern based on i, j, r
            U = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        # Random sign with some structure
                        if np.random.random() < 0.5:
                            U[i * m + j, r] = np.random.choice([-1, 1])
            
            V = np.random.randint(-1, 2, (m * p, rank))
            
            sol = _try_complete(U, V, rank)
            if sol is not None:
                return sol
        
        # Strategy 3: Kronecker-inspired structure
        # Split the 2 rows: for row 0, use some columns; for row 1, use others
        for attempt in range(10):
            if time.time() > deadline - 2:
                break
            
            # Try: U has block structure where row 0 and row 1 share some pattern
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            
            # For each j (column of A, row of B), assign r values
            # Simple: U[i,j,r] = 1 if (i + j) % rank == r, else random
            for i in range(n):
                for j in range(m):
                    for r in range(rank):
                        U[i * m + j, r] = np.random.choice([-1, 0, 1])
            
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        V[j * p + k, r] = np.random.choice([-1, 0, 1])
            
            sol = _try_complete(U, V, rank)
            if sol is not None:
                return sol
    
    return best_solution

def _construct_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
