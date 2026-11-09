"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _to_rational_array(mat, tol=1e-8):
    """Convert a numeric matrix to a matrix of Fractions (exact rationals)."""
    rows, cols = mat.shape
    result = [[Fraction(0) for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            val = mat[i, j]
            if abs(val) < tol:
                result[i][j] = Fraction(0)
                continue
            found = False
            for den in range(1, 201):
                num = round(val * den)
                if abs(val - num / den) < tol:
                    result[i][j] = Fraction(num, den)
                    found = True
                    break
            if not found:
                result[i][j] = Fraction(val).limit_denominator(1000000)
    return result

def _verify_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Verify decomposition exactly."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Check all valid (i,j,k) triplets sum to 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check invalid triplets sum to 0
    import random
    random.seed(42)
    for _ in range(200):
        i = random.randint(0, n-1)
        j = random.randint(0, m-1)
        k = random.randint(0, p-1)
        i2 = random.randint(0, n-1)
        j2 = random.randint(0, m-1)
        k2 = random.randint(0, p-1)
        if i == i2 and j == j2 and k == k2:
            continue
        s = 0.0
        for c in range(rank):
            s += U[i*m+j, c] * V[j2*p+k, c] * W[k*n+i2, c]
        if abs(s) > tol:
            return False
    return True

def construct():
    start_time = time.time()
    DEADLINE = start_time + 130.0  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 24
    
    # Build the schoolbook solution as fallback
    def make_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = schoolbook_rank
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    # T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if i==i', j==j', k==k else 0
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    import random
    random.seed(12345)
    np.random.seed(12345)
    
    lower_bound = max(n*m, m*p, p*n)  # 12
    
    # Proper ALS implementation
    def als_solve_U(T, V, W, rank, n, m, p):
        """Solve for U given V, W.
        T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        For each a: T[a, :, :] = sum_r U[a, r] * outer(V[:, r], W[:, r])
        Flatten: T_flat[a, (b,c)] = sum_r U[a, r] * B[(b,c), r]
        where B[(b,c), r] = V[b, r] * W[c, r]
        So T_flat = U @ B^T, meaning U = T_flat @ pinv(B^T) = T_flat @ pinv(B).T
        Actually: T_flat[a, bc] = sum_r U[a, r] * B[bc, r]
        => T_flat = U @ B.T
        => U = T_flat @ pinv(B.T)
        """
        # B has shape (m*p * p*n, rank), B[b * p*n + c, r] = V[b, r] * W[c, r]
        B = np.zeros((m * p * p * n, rank))
        for b in range(m * p):
            for c in range(p * n):
                for r in range(rank):
                    B[b * p * n + c, r] = V[b, r] * W[c, r]
        
        T_flat = T.reshape(n * m, m * p * p * n)
        
        # Solve U @ B.T = T_flat
        # U = T_flat @ pinv(B.T)
        try:
            U_sol = T_flat @ np.linalg.pinv(B.T)
        except:
            U_sol = np.random.randn(n * m, rank)
        return U_sol
    
    def als_solve_V(T, U, W, rank, n, m, p):
        """Solve for V given U, W.
        T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        For each (a, c): T[a, :, c] = sum_r (U[a, r] * W[c, r]) * V[:, r]
        Let D[a*p*n + c, r] = U[a, r] * W[c, r]
        T_reshaped[(a,c), b] = T[a, b, c]
        Then T_reshaped = D @ V.T where V.T has shape (rank, m*p), V.T[r, b] = V[b, r]
        So V.T = pinv(D) @ T_reshaped
        """
        # D has shape (n*m * p*n, rank)
        D = np.zeros((n * m * p * n, rank))
        for a in range(n * m):
            for c in range(p * n):
                for r in range(rank):
                    D[a * p * n + c, r] = U[a, r] * W[c, r]
        
        # T_reshaped has shape (n*m * p*n, m*p)
        # T_reshaped[(a,c), b] = T[a, b, c]
        T_reshaped = np.zeros((n * m * p * n, m * p))
        for a in range(n * m):
            for c in range(p * n):
                for b in range(m * p):
                    T_reshaped[a * p * n + c, b] = T[a, b, c]
        
        # V.T = pinv(D) @ T_reshaped
        try:
            V_T = np.linalg.pinv(D) @ T_reshaped
            V_sol = V_T.T
        except:
            V_sol = np.random.randn(m * p, rank)
        return V_sol
    
    def als_solve_W(T, U, V, rank, n, m, p):
        """Solve for W given U, V.
        T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        For each (a, b): T[a, b, :] = sum_r (U[a, r] * V[b, r]) * W[:, r]
        Let D[a*m*p + b, r] = U[a, r] * V[b, r]
        T_reshaped[(a,b), c] = T[a, b, c]
        Then T_reshaped = D @ W.T where W.T has shape (rank, p*n)
        So W.T = pinv(D) @ T_reshaped
        """
        # D has shape (n*m * m*p, rank)
        D = np.zeros((n * m * m * p, rank))
        for a in range(n * m):
            for b in range(m * p):
                for r in range(rank):
                    D[a * m * p + b, r] = U[a, r] * V[b, r]
        
        # T_reshaped has shape (n*m * m*p, p*n)
        T_reshaped = np.zeros((n * m * m * p, p * n))
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    T_reshaped[a * m * p + b, c] = T[a, b, c]
        
        # W.T = pinv(D) @ T_reshaped
        try:
            W_T = np.linalg.pinv(D) @ T_reshaped
            W_sol = W_T.T
        except:
            W_sol = np.random.randn(p * n, rank)
        return W_sol
    
    # Try ranks from lower bound up to schoolbook
    for target_rank in range(lower_bound, schoolbook_rank):
        if time.time() > DEADLINE - 25:
            break
        
        max_tries = 30
        found = False
        
        for trial in range(max_tries):
            if time.time() > DEADLINE - 15:
                break
            
            # Random initialization
            U_al = np.random.randn(n * m, target_rank)
            V_al = np.random.randn(m * p, target_rank)
            W_al = np.random.randn(p * n, target_rank)
            
            # Run ALS iterations
            for it in range(50):
                if time.time() > DEADLINE - 10:
                    break
                U_al = als_solve_U(T, V_al, W_al, target_rank, n, m, p)
                V_al = als_solve_V(T, U_al, W_al, target_rank, n, m, p)
                W_al = als_solve_W(T, U_al, V_al, target_rank, n, m, p)
            
            if time.time() > DEADLINE - 5:
                break
            
            # Verify
            valid = _verify_exact(U_al, V_al, W_al, n, m, p, target_rank, tol=1e-4)
            if valid:
                # Convert to rational
                U_rat = _to_rational_array(U_al)
                V_rat = _to_rational_array(V_al)
                W_rat = _to_rational_array(W_al)
                
                # Verify with rationals
                valid_rat = _verify_exact(U_rat, V_rat, W_rat, n, m, p, target_rank, tol=1e-6)
                if valid_rat:
                    best_result = {"U": U_rat, "V": V_rat, "W": W_rat}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
