"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x4 matrix multiplication.
    Returns {'U': list, 'V': list, 'W': list} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 4
    deadline = time.time() + 145.0  # Safety margin before 150s kill

    def schoolbook():
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

    def verify(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    def verify_mod2(U, V, W):
        """Verify decomposition over F_2"""
        U_mod = np.array(U, dtype=np.uint8) % 2
        V_mod = np.array(V, dtype=np.uint8) % 2
        W_mod = np.array(W, dtype=np.uint8) % 2
        rank = U_mod.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0
                    for c in range(rank):
                        val = (val + U_mod[i*m+j, c] * V_mod[j*p+k, c] * W_mod[k*n+i, c]) % 2
                    if val != 1:
                        return False
        return True

    def rank_F2(M):
        """Compute rank of binary matrix over F_2"""
        M = M.copy()
        rows = M.shape[0]
        cols = M.shape[1]
        if rows == 0 or cols == 0:
            return 0
        rank = 0
        for col in range(cols):
            # Find pivot
            pivot = -1
            for row in range(rank, rows):
                if M[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap rows
            M[[rank, pivot]] = M[[pivot, rank]]
            # Eliminate
            for row in range(rows):
                if row != rank and M[row, col] == 1:
                    M[row] = (M[row] + M[rank]) % 2
            rank += 1
        return rank

    def solve_linear_F2(A, b):
        """
        Solve A x = b over F_2.
        Returns a solution x or None if inconsistent.
        A is (num_eq x num_vars), b is (num_eq,)
        """
        num_eq, num_vars = A.shape
        M = np.column_stack([A, b.astype(np.uint8)])
        rows, cols = M.shape
        rank = 0
        pivot_cols = []
        
        for col in range(num_vars):
            pivot = -1
            for row in range(rank, rows):
                if M[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            M[[rank, pivot]] = M[[pivot, rank]]
            pivot_cols.append(col)
            for row in range(rows):
                if row != rank and M[row, col] == 1:
                    M[row] = (M[row] + M[rank]) % 2
            rank += 1
        
        # Check consistency
        for row in range(rank, rows):
            if M[row, num_vars] == 1:
                return None  # Inconsistent
        
        # Find a particular solution (set free variables to 0)
        x = np.zeros(num_vars, dtype=np.uint8)
        for i, col in enumerate(pivot_cols):
            x[col] = M[i, num_vars]
        return x

    def try_gaussian_elimination_F2(target_rank):
        """
        Use linear algebra over F_2 to find a decomposition.
        Fix U and V, solve for W.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        random.seed(123)
        
        for trial in range(200):
            if time.time() > deadline - 15:
                return None
            
            # Generate random U and V
            U = np.zeros((nm, target_rank), dtype=np.uint8)
            V = np.zeros((mp, target_rank), dtype=np.uint8)
            
            for c in range(target_rank):
                # Make U and V sparse
                for _ in range(random.randint(1, 3)):
                    U[random.randint(0, nm-1), c] = 1
                for _ in range(random.randint(1, 3)):
                    V[random.randint(0, mp-1), c] = 1
            
            # Build linear system for W
            # For each (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # Group by W-row index r = k*n+i
            
            W = np.zeros((pn, target_rank), dtype=np.uint8)
            consistent = True
            
            for r in range(pn):
                i = r % n
                k = r // n
                
                # Build m x target_rank matrix A_r where A_r[j, c] = U[i*m+j,c] * V[j*p+k,c]
                A_r = np.zeros((m, target_rank), dtype=np.uint8)
                for j in range(m):
                    for c in range(target_rank):
                        A_r[j, c] = (U[i*m+j, c] * V[j*p+k, c]) % 2
                
                # We need A_r * W[r, :]^T = [1, 1, ..., 1]^T
                b = np.ones(m, dtype=np.uint8)
                
                # Solve the linear system
                W_r = solve_linear_F2(A_r, b)
                
                if W_r is None:
                    consistent = False
                    break
                
                W[r, :] = W_r
            
            if not consistent:
                continue
            
            # Verify the complete solution
            if verify_mod2(U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    def try_random_search_F2(target_rank, max_iter=20000):
        """
        Random search for a rank-target decomposition over F_2.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        random.seed(42)
        
        for iter in range(max_iter):
            if time.time() > deadline - 15:
                return None
            
            # Generate random sparse binary vectors
            U = np.zeros((nm, target_rank), dtype=np.uint8)
            V = np.zeros((mp, target_rank), dtype=np.uint8)
            W = np.zeros((pn, target_rank), dtype=np.uint8)
            
            for c in range(target_rank):
                # Choose random positions for U, V, W
                u_pos = random.randint(0, nm - 1)
                v_pos = random.randint(0, mp - 1)
                w_pos = random.randint(0, pn - 1)
                
                # With some probability, make it sparse (multiple non-zeros)
                if random.random() < 0.3:
                    for _ in range(random.randint(1, 2)):
                        u_pos2 = random.randint(0, nm - 1)
                        v_pos2 = random.randint(0, mp - 1)
                        w_pos2 = random.randint(0, pn - 1)
                        U[u_pos2, c] = 1
                        V[v_pos2, c] = 1
                        W[w_pos2, c] = 1
                
                U[u_pos, c] = 1
                V[v_pos, c] = 1
                W[w_pos, c] = 1
            
            if verify_mod2(U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    def try_structured_decomposition(target_rank):
        """
        Try a structured decomposition based on known algebraic techniques.
        For 3x3x4, try to exploit the tensor structure.
        """
        nm = n * m
        mp = m * p
        pn = p * n
        
        # Try a specific structured approach for rank 24
        # This uses the idea of sharing terms across different j values
        if target_rank >= 24:
            U = np.zeros((nm, target_rank), dtype=np.uint8)
            V = np.zeros((mp, target_rank), dtype=np.uint8)
            W = np.zeros((pn, target_rank), dtype=np.uint8)
            
            # Fill in a candidate structured decomposition
            # Based on the idea of using the algebraic structure of matrix multiplication
            
            # For now, try a simple structured pattern
            for c in range(target_rank):
                # Assign positions based on a structured pattern
                i = c % n
                j = (c // n) % m
                k = (c // (n * m)) % p
                
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
            
            if verify_mod2(U, V, W):
                return (U.tolist(), V.tolist(), W.tolist())
        
        return None

    # Try to find a decomposition with rank < 36
    # Start from rank 24 and go up
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Try structured approaches first
    for target_rank in [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]:
        if time.time() > deadline - 20:
            break
        
        # Try structured decomposition
        result = try_structured_decomposition(target_rank)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try Gaussian elimination approach
        result = try_gaussian_elimination_F2(target_rank)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try random search
        result = try_random_search_F2(target_rank, max_iter=5000)
        if result is not None:
            U_list, V_list, W_list = result
            if verify(U_list, V_list, W_list):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    return best_result

# EVOLVE-BLOCK-END
