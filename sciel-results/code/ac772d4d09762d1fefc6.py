"""Matrix multiplication 2x3x3 tensor decomposition - rank minimization."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (9, rank)  
    # W: (p*n, rank) = (6, rank)
    
    def tensor_residual(U, V, W):
        """Compute residual efficiently using matrix operations."""
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    total += (s - 1.0) ** 2
        return total
    
    def is_half_integer(arr, tol=1e-8):
        """Check if all entries are integer or half-integer."""
        flat = arr.flatten()
        v = flat * 2
        return np.all(np.abs(v - np.round(v)) < tol)
    
    def round_to_half(arr):
        return np.round(arr * 2) / 2
    
    def verify_solution(U, V, W, tol=1e-6):
        """Verify that the decomposition is exact and has half-integer entries."""
        if not (is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8)):
            return False
        res = tensor_residual(U, V, W)
        return res < tol
    
    # Schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, rank
    
    U_sb, V_sb, W_sb, rank_sb = schoolbook()
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Try to find a rank 16 decomposition by combining:
    # - 2x2x3 sub-problem (rank 10) for columns 0,1 of the 3x3 matrix
    # - 2x1x3 sub-problem (rank 6) for column 2 of the 3x3 matrix
    
    def find_223_rank10():
        """Search for rank-10 decomposition of 2x2x3 matrix multiplication."""
        nn, mm, pp = 2, 2, 3
        target_rank = 10
        
        # Known result: rank of 2x2x3 is 10
        # Let's try to find it with random search + rounding
        
        for attempt in range(200):
            if time.time() > deadline - 15:
                return None
            
            np.random.seed(12345 + attempt)
            
            # Initialize with half-integer values
            U = np.random.randint(-2, 3, size=(nn*mm, target_rank)) / 2.0
            V = np.random.randint(-2, 3, size=(mm*pp, target_rank)) / 2.0
            W = np.random.randint(-2, 3, size=(pp*nn, target_rank)) / 2.0
            
            res = 0.0
            for i in range(nn):
                for j in range(mm):
                    for k in range(pp):
                        s = np.dot(U[i*mm+j], V[j*pp+k] * W[k*nn+i])
                        res += (s - 1.0) ** 2
            
            if res < 1e-4:
                # Verify and return
                if is_half_integer(U, 1e-8) and is_half_integer(V, 1e-8) and is_half_integer(W, 1e-8):
                    if res < 1e-6:
                        return U, V, W
        
        return None
    
    def find_213_rank6():
        """Find rank-6 decomposition of 2x1x3 matrix multiplication (trivial)."""
        nn, mm, pp = 2, 1, 3
        target_rank = 6
        U = np.zeros((nn*mm, target_rank))
        V = np.zeros((mm*pp, target_rank))
        W = np.zeros((pp*nn, target_rank))
        
        col = 0
        for i in range(nn):
            for k in range(pp):
                U[i*mm + 0, col] = 1.0
                V[0*pp + k, col] = 1.0
                W[k*nn + i, col] = 1.0
                col += 1
        
        return U, V, W
    
    # Try to construct rank-16 solution
    def try_rank16():
        """Try to build a rank-16 decomposition."""
        nonlocal best, best_rank
        
        # Get the 2x1x3 part (column 2 of the 3x3 matrix)
        U_213, V_213, W_213 = find_213_rank6()
        
        # Try to find 2x2x3 part (columns 0,1 of the 3x3 matrix)
        result_223 = find_223_rank10()
        
        if result_223 is None:
            return False
        
        U_223, V_223, W_223 = result_223
        
        # Combine: the full problem is 2x3x3
        # We split the 3 columns of the 3x3 matrix into first 2 and last 1
        # U_full: (6, 16)
        # V_full: (9, 16)  
        # W_full: (6, 16)
        
        U_full = np.zeros((n*m, 16))
        V_full = np.zeros((m*p, 16))
        W_full = np.zeros((p*n, 16))
        
        # First 10 terms: 2x2x3 for columns 0,1
        # U_223 has shape (4, 10) - for 2x2 matrix
        # We need to map this to 2x3 matrix (first 2 columns)
        # Row index in U: i*m + j where i in [0,2), j in [0,3)
        # For j=0,1: use U_223[i*2 + j]
        # For j=2: use U_213[i*1 + 0]
        
        # V_223 has shape (4, 10) - for 2x3 matrix (mm=2, pp=3)
        # Row index in V: j*p + k where j in [0,3), k in [0,3)
        # For j=0,1: use V_223[j*3 + k]
        # For j=2: use V_213[0*3 + k]
        
        # W_223 has shape (6, 10) - for 3x2 matrix (pp=3, nn=2)
        # Row index in W: k*n + i where k in [0,3), i in [0,2)
        # Use W_223[k*2 + i]
        
        # W_213 has shape (6, 6) - for 3x2 matrix (pp=3, nn=2)
        # Row index in W: k*n + i where k in [0,3), i in [0,2)
        # Use W_213[k*2 + i]
        
        # Fill first 10 columns
        for i in range(2):
            for j in range(2):
                U_full[i*3 + j, 0:10] = U_223[i*2 + j, :]
        
        for j in range(2):
            for k in range(3):
                V_full[j*3 + k, 0:10] = V_223[j*3 + k, :]
        
        for k in range(3):
            for i in range(2):
                W_full[k*2 + i, 0:10] = W_223[k*2 + i, :]
        
        # Fill last 6 columns
        for i in range(2):
            U_full[i*3 + 2, 10:16] = U_213[i*1 + 0, :]
        
        for k in range(3):
            V_full[2*3 + k, 10:16] = V_213[0*3 + k, :]
        
        for k in range(3):
            for i in range(2):
                W_full[k*2 + i, 10:16] = W_213[k*2 + i, :]
        
        # Verify
        if verify_solution(U_full, V_full, W_full, 1e-6):
            best = {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
            best_rank = 16
            return True
        
        return False
    
    # Try rank 16
    if try_rank16():
        return best
    
    # Try other approaches: random search for lower ranks
    def try_rank(target_rank, max_attempts=50):
        """Try to find a decomposition for given target_rank."""
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return False
            
            np.random.seed(10000 + target_rank * 100 + attempt)
            
            # Random half-integer initialization
            U = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
            V = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
            W = np.random.randint(-2, 3, size=(p*n, target_rank)) / 2.0
            
            res = tensor_residual(U, V, W)
            
            if res < 1e-4:
                if verify_solution(U, V, W, 1e-6):
                    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                    best_rank = target_rank
                    return True
        
        return False
    
    # Try ranks from 14 to 15 (16 already tried)
    for target_rank in range(14, 16):
        if time.time() > deadline - 20:
            break
        try_rank(target_rank, max_attempts=30)
    
    return best
# EVOLVE-BLOCK-END
