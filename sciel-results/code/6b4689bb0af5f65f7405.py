import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _verify_333(U, V, W, rank):
    """Verify 3x3x3 decomposition using vectorized operations."""
    n, m, p = 3, 3, 3
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Build tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # U shape: (n*m, rank) -> reshape to (n, m, rank)
    # V shape: (m*p, rank) -> reshape to (m, p, rank)
    # W shape: (p*n, rank) -> reshape to (p, n, rank)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    # T[i,j,k] = sum_r U3[i,j,r] * V3[j,k,r] * W3[k,i,r]
    # Use einsum
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _verify_336(U, V, W, rank):
    """Verify 3x3x6 decomposition using vectorized operations."""
    n, m, p = 3, 3, 6
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    U3 = U_arr.reshape(n, m, rank)
    V3 = V_arr.reshape(m, p, rank)
    W3 = W_arr.reshape(p, n, rank)
    T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_3x3():
    """
    Known rank-23 decomposition for 3x3x3 matrix multiplication.
    Based on Bini et al. construction.
    """
    rank = 23
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Bini's 23-multiplication algorithm for 3x3
    # This is a known exact decomposition
    # Using the construction from Bini, Capovani, Lotti, Romani (1979)
    
    # The 23 multiplications for 3x3 matrix multiplication
    # Indices: U[i*3+j], V[j*3+k], W[k*3+i]
    
    # Multiplication 1: a00*b00
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # Multiplication 2: a01*b10
    U[1, 1] = 1; V[3, 1] = 1; W[1, 1] = 1
    # Multiplication 3: a02*b20
    U[2, 2] = 1; V[6, 2] = 1; W[2, 2] = 1
    # Multiplication 4: a10*b01
    U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    # Multiplication 5: a11*b11
    U[4, 4] = 1; V[4, 4] = 1; W[4, 4] = 1
    # Multiplication 6: a12*b21
    U[5, 5] = 1; V[7, 5] = 1; W[5, 5] = 1
    # Multiplication 7: a20*b02
    U[6, 6] = 1; V[2, 6] = 1; W[6, 6] = 1
    # Multiplication 8: a21*b12
    U[7, 7] = 1; V[5, 7] = 1; W[7, 7] = 1
    # Multiplication 9: a22*b22
    U[8, 8] = 1; V[8, 8] = 1; W[8, 8] = 1
    
    # Now we need 14 more multiplications to get to rank 23
    # These are the "cross" terms that allow cancellation
    
    # Multiplication 10: (a00+a01)*(b10+b11) - a00*b11 - a01*b10
    # But we need exact integer/half-integer coefficients
    
    # Let's use a different approach: try to find rank-23 via search
    return None  # We'll search instead

def _structured_search_333(rank, max_attempts=200, deadline=None):
    """
    Structured search for rank-r decomposition of 3x3x3.
    Uses patterns based on symmetry and known constructions.
    """
    n, m, p = 3, 3, 3
    
    # Try various structured patterns
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            break
        
        rng = np.random.RandomState(attempt * 7 + 13)
        
        # Pattern 1: Block diagonal structure
        # Divide the 9x9 matrices into blocks
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Use a mix of dense and sparse patterns
        density = 0.3 + 0.3 * (attempt % 3) / 2.0
        
        for col in range(rank):
            # Randomly choose which entries are non-zero
            u_mask = rng.random(9) < density
            v_mask = rng.random(9) < density
            w_mask = rng.random(9) < density
            
            u_vals = rng.choice([-1, 0, 1], size=9)
            v_vals = rng.choice([-1, 0, 1], size=9)
            w_vals = rng.choice([-1, 0, 1], size=9)
            
            U[:, col] = u_vals * u_mask
            V[:, col] = v_vals * v_mask
            W[:, col] = w_vals * w_mask
        
        if _verify_333(U.tolist(), V.tolist(), W.tolist(), rank):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _refinement_search_333(rank, max_iterations=500, deadline=None):
    """
    Local refinement search: start from random initial point and try to
    minimize the error using coordinate descent.
    """
    n, m, p = 3, 3, 3
    
    best_error = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(20):
        if deadline is not None and time.time() > deadline:
            break
        
        rng = np.random.RandomState(attempt * 31 + 7)
        density = 0.4
        
        U = rng.choice([-1, 0, 1], size=(9, rank))
        V = rng.choice([-1, 0, 1], size=(9, rank))
        W = rng.choice([-1, 0, 1], size=(9, rank))
        
        mask_u = rng.random((9, rank)) < density
        mask_v = rng.random((9, rank)) < density
        mask_w = rng.random((9, rank)) < density
        U = U * mask_u
        V = V * mask_v
        W = W * mask_w
        
        for iteration in range(max_iterations):
            if deadline is not None and time.time() > deadline:
                break
            
            # Compute current error
            U3 = U.reshape(3, 3, rank)
            V3 = V.reshape(3, 3, rank)
            W3 = W.reshape(3, 3, rank)
            T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
            error = np.sum((T - 1.0)**2)
            
            if error < 1e-9:
                return U.tolist(), V.tolist(), W.tolist()
            
            if error < best_error:
                best_error = error
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            # Try flipping random entries
            if rng.random() < 0.7:
                idx = rng.randint(0, 9)
                col = rng.randint(0, rank)
                U[idx, col] *= -1
            elif rng.random() < 0.7:
                idx = rng.randint(0, 9)
                col = rng.randint(0, rank)
                V[idx, col] *= -1
            else:
                idx = rng.randint(0, 9)
                col = rng.randint(0, rank)
                W[idx, col] *= -1
            
            # Occasionally set to 0 or 1
            if rng.random() < 0.1:
                idx = rng.randint(0, 9)
                col = rng.randint(0, rank)
                U[idx, col] = rng.choice([0, 1, -1])
            if rng.random() < 0.1:
                idx = rng.randint(0, 9)
                col = rng.randint(0, rank)
                V[idx, col] = rng.choice([0, 1, -1])
            if rng.random() < 0.1:
                idx = rng.randint(0, 9)
                col = rng.randint(0, rank)
                W[idx, col] = rng.choice([0, 1, -1])
    
    # Check best found
    if best_error < 1e-9:
        return best_U.tolist(), best_V.tolist(), best_W.tolist()
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 54
    
    # Initialize with schoolbook decomposition
    best_result = _schoolbook(n, m, p)
    best_rank = rank_schoolbook
    
    # Strategy: Find a low-rank decomposition for 3x3x3 and double it
    # Try ranks from 26 down to 23
    for r3 in [26, 25, 24, 23]:
        if time.time() > deadline - 30:
            break
        
        # Try structured search
        result_333 = _structured_search_333(r3, max_attempts=100, deadline=deadline - 20)
        
        if result_333 is None:
            # Try refinement search
            result_333 = _refinement_search_333(r3, max_iterations=200, deadline=deadline - 20)
        
        if result_333:
            U3, V3, W3 = result_333
            U3_arr = np.array(U3)
            V3_arr = np.array(V3)
            W3_arr = np.array(W3)
            
            # Construct 3x3x6 decomposition
            U6 = np.zeros((n*m, 2*r3))
            V6 = np.zeros((m*p, 2*r3))
            W6 = np.zeros((p*n, 2*r3))
            
            # First half: for k in 0..2
            U6[:, :r3] = U3_arr
            for j in range(m):
                for k in range(3):
                    V6[j*6+k, :r3] = V3_arr[j*3+k]
            for k in range(3):
                for i in range(n):
                    W6[k*3+i, :r3] = W3_arr[k*3+i]
            
            # Second half: for k in 3..5
            U6[:, r3:] = U3_arr
            for j in range(m):
                for k in range(3, 6):
                    V6[j*6+k, r3:] = V3_arr[j*3+(k-3)]
            for k in range(3, 6):
                for i in range(n):
                    W6[k*3+i, r3:] = W3_arr[(k-3)*3+i]
            
            # Verify the 3x3x6 decomposition
            if _verify_336(U6.tolist(), V6.tolist(), W6.tolist(), 2*r3):
                if 2*r3 < best_rank:
                    best_rank = 2*r3
                    best_result = {
                        "U": U6.tolist(),
                        "V": V6.tolist(),
                        "W": W6.tolist()
                    }
                break  # Found a valid one, no need to try lower r3 for this strategy
    
    return best_result

def _schoolbook(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
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
