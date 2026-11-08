"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction
import random
from itertools import product

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x4 matrix.
    Uses randomized linear algebra over finite fields to search for low-rank decompositions.
    """
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 3, 3, 4
    n_m = n * m  # 9
    m_p = m * p  # 12
    p_n = p * n  # 12
    
    # Schoolbook rank = n*m*p = 36
    schoolbook_rank = n * m * p
    
    def verify_decomposition_frac(U, V, W, rank):
        """Verify with exact rational arithmetic."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for r in range(rank):
                        s += Fraction(U[i*m+j][r]) * Fraction(V[j*p+k][r]) * Fraction(W[k*n+i][r])
                    if s != 1:
                        return False
        return True
    
    def verify_decomposition_float(U, V, W, rank, tol=1e-9):
        """Quick verification with floating point."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Start with schoolbook decomposition
    best_rank = schoolbook_rank
    best_U = [[0] * schoolbook_rank for _ in range(n_m)]
    best_V = [[0] * schoolbook_rank for _ in range(m_p)]
    best_W = [[0] * schoolbook_rank for _ in range(p_n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1
    
    # Strategy: Use randomized linear algebra over GF(3) to search for low-rank decompositions
    # The key idea: for a rank-r decomposition, we have r columns in U, V, W
    # Each column (u_r, v_r, w_r) contributes to the tensor
    # We can set up a linear system over GF(3) and solve it
    
    def try_rank_gf3(r, max_attempts=100):
        """Try to find a rank-r decomposition using linear algebra over GF(3)."""
        if time.time() > deadline:
            return None
        
        # Number of equations: n*m*p = 36
        # Number of unknowns per column: n_m + m_p + p_n = 9 + 12 + 12 = 33
        # For r columns, we have 33*r unknowns and 36 equations
        
        # We'll use a randomized approach:
        # 1. Randomly generate W matrix over GF(3)
        # 2. For each (i,k) pair, the equation is:
        #    sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # 3. Fix W, then the equations become quadratic in U and V
        # 4. Instead, we can fix one of the matrices and solve for the others
        
        # Alternative approach: use the fact that we can decompose the tensor
        # Let's try a different strategy:
        # For each rank r, try to find U, V, W such that the tensor equation holds
        
        # We'll use a greedy approach with random restarts
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Randomly initialize W over GF(3) with entries in {0, 1, 2}
            W_gf3 = np.zeros((p_n, r), dtype=int)
            for col_idx in range(r):
                for row_idx in range(p_n):
                    W_gf3[row_idx, col_idx] = random.randint(0, 2)
            
            # Now we need to find U and V such that:
            # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 (mod 3)
            
            # This is still nonlinear. Let's try a different approach.
            # We'll fix U as well and solve for V.
            
            # Actually, let's try a simpler approach:
            # Generate random U, V, W over GF(3) and check if they work
            # This is unlikely to work directly, but we can use it as a starting point
            
            # Better approach: use the structure of the problem
            # The tensor is T[i,j,k] = delta_{i,k} * delta_{j,?} ... no
            # Actually T[i,j,k] = 1 for all (i,j,k) in the range, but that's not right either
            
            # The matrix multiplication tensor has T[i,j,k] = 1 if the multiplication
            # produces output (i,k) from inputs (i,j) and (j,k)
            # For standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is T[i,j,k] = 1 for all valid (i,j,k)
            
            # Let me reconsider. The tensor for matrix multiplication is:
            # T_{i,j,k} = 1 if we're computing C[i,k] from A[i,j] and B[j,k]
            # But actually, the standard tensor is:
            # T = sum_{i,j,k} e_i^A ⊗ e_j^B ⊗ e_k^C where the multiplication
            # C[i,k] = sum_j A[i,j] B[j,k]
            
            # Wait, I need to be more careful. The tensor is:
            # T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k
            # where e_i is the basis vector for the i-th row of A,
            # e_j is the basis vector for the j-th column of A / row of B,
            # e_k is the basis vector for the k-th column of B / output C
            
            # Actually, for matrix multiplication, the tensor is:
            # T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_i^A ⊗ e_j^{A,B} ⊗ e_k^B
            # No, that's not right either.
            
            # Let me use the convention from the problem:
            # U has shape (n*m, rank), indexed by (i,j)
            # V has shape (m*p, rank), indexed by (j,k)
            # W has shape (p*n, rank), indexed by (k,i)
            
            # The tensor equation is:
            # sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for all (i,j,k)
            
            # This is a 3-tensor with entries all equal to 1.
            
            # For a rank-r decomposition, we need:
            # T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
            
            # Let's try a constructive approach. We can try to build the decomposition
            # column by column.
            
            # For now, let's try a simple random search with verification
            U_gf3 = np.zeros((n_m, r), dtype=int)
            V_gf3 = np.zeros((m_p, r), dtype=int)
            
            for col_idx in range(r):
                for row_idx in range(n_m):
                    U_gf3[row_idx, col_idx] = random.randint(0, 2)
                for row_idx in range(m_p):
                    V_gf3[row_idx, col_idx] = random.randint(0, 2)
            
            # Check if this works over GF(3)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r_idx in range(r):
                            s = (s + U_gf3[i*m+j, r_idx] * V_gf3[j*p+k, r_idx] * W_gf3[k*n+i, r_idx]) % 3
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Found a solution over GF(3), lift to integers
                U_int = U_gf3.tolist()
                V_int = V_gf3.tolist()
                W_int = W_gf3.tolist()
                
                # Verify with exact arithmetic
                if verify_decomposition_frac(U_int, V_int, W_int, r):
                    return (U_int, V_int, W_int)
        
        return None
    
    # Try to find better decompositions using GF(3) approach
    for r in range(35, 14, -1):
        if time.time() > deadline:
            break
        
        result = try_rank_gf3(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U = U
            best_V = V
            best_W = W
            break
    
    # If we didn't find anything better, try the original random search
    if best_rank == schoolbook_rank:
        random.seed(42)
        np.random.seed(42)
        
        def try_find_decomposition(r, max_attempts=50):
            """Try to find a decomposition of rank r."""
            if time.time() > deadline:
                return None
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    return None
                
                U = np.zeros((n_m, r), dtype=int)
                V = np.zeros((m_p, r), dtype=int)
                W = np.zeros((p_n, r), dtype=int)
                
                for _ in range(r):
                    u_rows = np.random.choice(n_m, size=min(r, n_m), replace=False)
                    v_rows = np.random.choice(m_p, size=min(r, m_p), replace=False)
                    w_rows = np.random.choice(p_n, size=min(r, p_n), replace=False)
                    
                    for idx, row in enumerate(u_rows):
                        if idx < r:
                            U[row, idx] = random.choice([-1, 1])
                    for idx, row in enumerate(v_rows):
                        if idx < r:
                            V[row, idx] = random.choice([-1, 1])
                    for idx, row in enumerate(w_rows):
                        if idx < r:
                            W[row, idx] = random.choice([-1, 1])
                
                if verify_decomposition_float(U.tolist(), V.tolist(), W.tolist(), r):
                    if verify_decomposition_frac(U.tolist(), V.tolist(), W.tolist(), r):
                        return (U.tolist(), V.tolist(), W.tolist())
            
            return None
        
        for r in range(35, 14, -1):
            if time.time() > deadline:
                break
            
            result = try_find_decomposition(r)
            if result is not None:
                U, V, W = result
                best_rank = r
                best_U = U
                best_V = V
                best_W = W
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
