"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 150s limit
    
    # Schoolbook decomposition (rank 44)
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
    
    best_result = schoolbook()
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-10):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower-rank decomposition using structured search
    # For 2x2xp, we can try ranks from 33 up to 43
    # Use a structured approach: assume V has a specific pattern
    
    # Strategy: For rank R, we try to solve for U, V, W
    # using the fact that the tensor is sparse and structured.
    
    # Key insight: For 2x2xp, the multiplication can be viewed as
    # computing C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, k in {0,...,p-1}
    # 
    # A structured decomposition might look like:
    # - For each "Strassen-like" term, we combine operations across columns
    # - Specifically, we can try to share terms between adjacent k values
    
    # Let's try a direct construction for rank 33 (3p):
    # For 2x2x2, rank 7 is minimal (Strassen).
    # For 2x2xp, a naive extension gives 7p = 77, which is worse than 44.
    # But there are better constructions.
    #
    # Actually, for 2x2xp, the rank is known to be between 3p and 4p for large p.
    # For p=11, rank 33 is a natural target.
    
    # Let's try a randomized search with structured initialization.
    # We'll try ranks 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43
    
    for target_rank in range(33, 44):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts for this rank
        found = False
        for restart in range(50):
            if time.time() > deadline - 5:
                break
            
            # Initialize with structured pattern
            # U: (n*m, R) = (4, R)
            # V: (m*p, R) = (22, R)
            # W: (p*n, R) = (22, R)
            
            rng = np.random.RandomState(target_rank * 100 + restart)
            
            # Initialize V with a pattern that respects the column structure
            # Each row of V corresponds to (j, k) where j in {0,1}, k in {0,...,10}
            # We want V[j*p+k, c] to be non-zero only for certain c's
            
            V = np.zeros((m * p, target_rank))
            U = np.zeros((n * m, target_rank))
            W = np.zeros((p * n, target_rank))
            
            # Assign each column of the factor matrices to handle specific (i,j,k) triples
            # with some sharing
            
            # Simple approach: random initialization with small values
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # Iterative refinement
            # We want sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
            # and = 0 for all other combinations (but there are no other combinations
            # since the tensor is exactly the multiplication tensor)
            
            # Actually, the tensor T[i,j,k] = 1 if (i,j,k) is a valid multiplication index,
            # and the decomposition must satisfy:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = T[i,j,k]
            # where T[i,j,k] = 1 for all i in {0,1}, j in {0,1}, k in {0,...,10}
            
            # Wait, that's not right. The multiplication tensor has entries:
            # T[i, j, k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
            # So the tensor is indexed by (i, j, k) where i is row of C (and A),
            # j is column of A (and row of B), k is column of C (and B).
            # T[i,j,k] = 1 for all valid (i,j,k), and the decomposition must reproduce this.
            
            # Actually, the tensor for matrix multiplication C = A * B where
            # A is n x m, B is m x p, C is n x p has entries:
            # T[i, j, k] = 1 if i < n, j < m, k < p
            # and the multiplication is C[i,k] = sum_j A[i,j] * B[j,k]
            # So the tensor is 1 for all (i,j,k) in [n]x[m]x[p].
            
            # The decomposition requires:
            # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
            
            # This is a system of n*m*p = 44 equations in U*V*W = 4*R + 22*R + 22*R = 48R unknowns.
            # For R=33, we have 1584 unknowns and 44 equations, so it's underdetermined.
            # We just need to find ANY solution.
            
            # Use a simple iterative method:
            # Fix two matrices, solve for the third.
            
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        idx = i * m + j
                        # We need sum_c U[idx, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                        # This is a system: sum_c U[idx, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
                        # Let's define M[k, c] = V[j*p+k, c] * W[k*n+i, c]
                        # Then sum_c U[idx, c] * M[k, c] = 1 for all k
                        # This is a linear system: M @ U[idx, :] = ones(p)
                        
                        M = np.zeros((p, target_rank))
                        for k in range(p):
                            for c in range(target_rank):
                                M[k, c] = V[j * p + k, c] * W[k * n + i, c]
                        
                        # Solve M @ x = ones(p) using least squares
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                            U[idx, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        idx = j * p + k
                        # We need sum_c U[i*m+j, c] * V[idx, c] * W[k*n+i, c] = 1 for all i
                        # Let's define M[i, c] = U[i*m+j, c] * W[k*n+i, c]
                        # Then sum_c V[idx, c] * M[i, c] = 1 for all i
                        # This is a linear system: M @ V[idx, :] = ones(n)
                        
                        M = np.zeros((n, target_rank))
                        for i in range(n):
                            for c in range(target_rank):
                                M[i, c] = U[i * m + j, c] * W[k * n + i, c]
                        
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                            V[idx, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        idx = k * n + i
                        # We need sum_c U[i*m+j, c] * V[j*p+k, c] * W[idx, c] = 1 for all j
                        # Let's define M[j, c] = U[i*m+j, c] * V[j*p+k, c]
                        # Then sum_c W[idx, c] * M[j, c] = 1 for all j
                        # This is a linear system: M @ W[idx, :] = ones(m)
                        
                        M = np.zeros((m, target_rank))
                        for j in range(m):
                            for c in range(target_rank):
                                M[j, c] = U[i * m + j, c] * V[j * p + k, c]
                        
                        try:
                            x, residuals, rank_m, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                            W[idx, :] = x
                        except:
                            pass
                
                # Check convergence
                if it % 20 == 0:
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0.0
                                for c in range(target_rank):
                                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                                err += (s - 1.0) ** 2
                    if err < 1e-12:
                        # Convert to lists
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        # Round to nearest half-integer or integer if close
                        def round_values(lst):
                            result = []
                            for row in lst:
                                new_row = []
                                for val in row:
                                    # Round to nearest 0.5
                                    rounded = round(val * 2) / 2
                                    if abs(val - rounded) < 1e-9:
                                        new_row.append(rounded)
                                    else:
                                        new_row.append(val)
                                result.append(new_row)
                            return result
                        
                        U_list = round_values(U_list)
                        V_list = round_values(V_list)
                        W_list = round_values(W_list)
                        
                        if verify(U_list, V_list, W_list):
                            best_result = {"U": U_list, "V": V_list, "W": W_list}
                            found = True
                            break
                        else:
                            # Try without rounding
                            if verify(U.tolist(), V.tolist(), W.tolist()):
                                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                                found = True
                                break
            
            if found:
                break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
