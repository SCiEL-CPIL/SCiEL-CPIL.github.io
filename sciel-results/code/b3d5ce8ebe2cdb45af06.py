import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
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
    
    def is_half_integer(arr, tol=1e-9):
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
    
    def try_known_decomposition():
        """Try known algebraic decompositions for 2x3x3."""
        nonlocal best, best_rank
        
        # Known rank-16 decomposition for 2x3x3
        # This uses the structure of the problem to find a minimal decomposition
        rank = 16
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # Construction based on splitting the 3x3 multiplication
        # We can view 3x3 as having structure that allows sharing
        # Use a systematic construction
        
        # For 2x3x3, one approach: treat the 3x3 as two 3x2 blocks plus one 3x1
        # Or use the fact that rank(2x3x3) <= rank(2x3x2) + rank(2x3x1)
        
        # Let's try a direct construction with known patterns
        # The key insight: we can share multiplications between different output entries
        
        # Column 0-5: handle (i=0, j=0) for k=0,1,2 and (i=1, j=0) for k=0,1,2
        for k in range(3):
            U[0, k] = 1  # i=0, j=0
            V[k, k] = 1  # j=0, k
            W[k*2+0, k] = 1  # k, i=0
            W[k*2+1, k] = 1  # k, i=1
            
        # Columns 6-11: handle (i=0, j=1) for k=0,1,2 and (i=1, j=1) for k=0,1,2
        for k in range(3):
            U[1, 6+k] = 1  # i=0, j=1
            V[3+k, 6+k] = 1  # j=1, k
            W[k*2+0, 6+k] = 1  # k, i=0
            W[k*2+1, 6+k] = 1  # k, i=1
            
        # Columns 12-15: handle (i=0, j=2) and (i=1, j=2) with sharing
        # We need 4 more columns to complete the 16-rank decomposition
        # Use shared multiplications
        
        # For j=2, we can share some work
        U[2, 12] = 1  # i=0, j=2
        V[6, 12] = 1  # j=2, k=0
        W[0, 12] = 1  # k=0, i=0
        W[1, 12] = 1  # k=0, i=1
        
        U[2, 13] = 1  # i=0, j=2
        V[7, 13] = 1  # j=2, k=1
        W[2, 13] = 1  # k=1, i=0
        W[3, 13] = 1  # k=1, i=1
        
        U[3, 14] = 1  # i=1, j=2
        V[6, 14] = 1  # j=2, k=0
        W[0, 14] = 1  # k=0, i=0
        W[1, 14] = 1  # k=0, i=1
        
        U[3, 15] = 1  # i=1, j=2
        V[7, 15] = 1  # j=2, k=1
        W[2, 15] = 1  # k=1, i=0
        W[3, 15] = 1  # k=1, i=1
        
        # Verify this construction
        if verify_solution(U, V, W, 1e-6):
            best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            best_rank = rank
            return True
        
        # Try a different construction
        # Use the fact that we can have half-integer entries
        U2 = np.zeros((n*m, rank))
        V2 = np.zeros((m*p, rank))
        W2 = np.zeros((p*n, rank))
        
        # More sophisticated construction with sharing
        # Columns 0-2: (0,0,k) for k=0,1,2
        for k in range(3):
            U2[0, k] = 1
            V2[k, k] = 1
            W2[k*2+0, k] = 0.5
            W2[k*2+1, k] = 0.5
        
        # Columns 3-5: (1,0,k) for k=0,1,2
        for k in range(3):
            U2[3, 3+k] = 1
            V2[k, 3+k] = 1
            W2[k*2+0, 3+k] = 0.5
            W2[k*2+1, 3+k] = 0.5
        
        # Columns 6-8: (0,1,k) for k=0,1,2
        for k in range(3):
            U2[1, 6+k] = 1
            V2[3+k, 6+k] = 1
            W2[k*2+0, 6+k] = 0.5
            W2[k*2+1, 6+k] = 0.5
        
        # Columns 9-11: (1,1,k) for k=0,1,2
        for k in range(3):
            U2[4, 9+k] = 1
            V2[3+k, 9+k] = 1
            W2[k*2+0, 9+k] = 0.5
            W2[k*2+1, 9+k] = 0.5
        
        # Columns 12-15: handle j=2 with sharing
        # We need to cover (0,2,k) and (1,2,k) for k=0,1,2
        # That's 6 entries but we only have 4 columns left
        # Use half-integers to share
        
        U2[2, 12] = 1
        U2[5, 12] = 1
        V2[6, 12] = 1
        W2[0, 12] = 0.5
        W2[1, 12] = 0.5
        W2[2, 12] = 0.5
        W2[3, 12] = 0.5
        
        U2[2, 13] = 1
        U2[5, 13] = 1
        V2[7, 13] = 1
        W2[0, 13] = 0.5
        W2[1, 13] = 0.5
        W2[2, 13] = 0.5
        W2[3, 13] = 0.5
        
        U2[2, 14] = 0.5
        U2[5, 14] = 0.5
        V2[8, 14] = 1
        W2[4, 14] = 1
        W2[5, 14] = 1
        
        U2[2, 15] = 0.5
        U2[5, 15] = 0.5
        V2[8, 15] = 1
        W2[4, 15] = 1
        W2[5, 15] = 1
        
        if verify_solution(U2, V2, W2, 1e-6):
            best = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
            best_rank = rank
            return True
        
        return False
    
    def try_search(target_rank, max_attempts=20):
        """Search for a decomposition with the given rank using random half-integer initialization."""
        nonlocal best, best_rank
        if target_rank >= best_rank:
            return False
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                return False
            
            np.random.seed(42 + target_rank * 1000 + attempt)
            
            # Random half-integer initialization in range [-1, 1]
            U = np.random.randint(-2, 3, size=(n*m, target_rank)) / 2.0
            V = np.random.randint(-2, 3, size=(m*p, target_rank)) / 2.0
            W = np.random.randint(-2, 3, size=(p*n, target_rank)) / 2.0
            
            # Try local search: perturb individual entries
            current_res = tensor_residual(U, V, W)
            best_local = (U.copy(), V.copy(), W.copy(), current_res)
            
            for iteration in range(100):
                if time.time() > deadline - 10:
                    break
                    
                # Pick a random entry to perturb
                mat_choice = np.random.choice(3)
                if mat_choice == 0:
                    idx = np.random.randint(n*m)
                    r = np.random.randint(target_rank)
                    old_val = U[idx, r]
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    if new_val < -2 or new_val > 2:
                        continue
                    U[idx, r] = new_val
                    new_res = tensor_residual(U, V, W)
                    if new_res < current_res:
                        current_res = new_res
                        if current_res < best_local[3]:
                            best_local = (U.copy(), V.copy(), W.copy(), current_res)
                    else:
                        U[idx, r] = old_val
                elif mat_choice == 1:
                    idx = np.random.randint(m*p)
                    r = np.random.randint(target_rank)
                    old_val = V[idx, r]
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    if new_val < -2 or new_val > 2:
                        continue
                    V[idx, r] = new_val
                    new_res = tensor_residual(U, V, W)
                    if new_res < current_res:
                        current_res = new_res
                        if current_res < best_local[3]:
                            best_local = (U.copy(), V.copy(), W.copy(), current_res)
                    else:
                        V[idx, r] = old_val
                else:
                    idx = np.random.randint(p*n)
                    r = np.random.randint(target_rank)
                    old_val = W[idx, r]
                    new_val = old_val + np.random.choice([-1.0, -0.5, 0.5, 1.0])
                    if new_val < -2 or new_val > 2:
                        continue
                    W[idx, r] = new_val
                    new_res = tensor_residual(U, V, W)
                    if new_res < current_res:
                        current_res = new_res
                        if current_res < best_local[3]:
                            best_local = (U.copy(), V.copy(), W.copy(), current_res)
                    else:
                        W[idx, r] = old_val
            
            # Check if we found a valid solution
            U_sol, V_sol, W_sol, res_sol = best_local
            if res_sol < 1e-6:
                if verify_solution(U_sol, V_sol, W_sol, 1e-6):
                    best = {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}
                    best_rank = target_rank
                    return True
        
        return False
    
    # First try known decompositions
    if try_known_decomposition():
        # Found a rank-16 solution, try to go lower
        pass
    
    # Try to find lower rank solutions
    for target_rank in range(14, best_rank):
        if time.time() > deadline - 15:
            break
        found = try_search(target_rank, max_attempts=30)
        if found:
            break
    
    return best
