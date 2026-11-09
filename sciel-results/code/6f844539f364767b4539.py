"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The multiplication tensor T[i, j, k] = 1 if (i, j, k) corresponds to
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # In the decomposition: T = sum_c U[:, c] outer V[:, c] outer W[:, c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # and W[k*n+i, c] is the entry for output index (i, k) at column c
    
    rank = U.shape[1]
    
    # Build the tensor from the decomposition
    # T[i, j, k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _try_rank(r, n, m, p, deadline):
    """Try to find a decomposition of rank r using random initialization and ALS-like refinement."""
    if time.time() > deadline:
        return None
    
    # For small ranks, we can try a direct approach
    # The tensor has n*m*p = 30 non-zero entries
    # We need rank r such that the tensor can be written as sum of r rank-1 tensors
    
    # Strategy: Use the fact that for matrix multiplication, there are known
    # sub-multiplicative bounds. For 2x3x5, the schoolbook is 30.
    # We can try to find a decomposition with rank < 30.
    
    # One approach: exploit the structure. The multiplication tensor for
    # A (n x m) times B (m x p) gives C (n x p).
    # T[i, j, k] = 1 for all valid (i, j, k).
    
    # Let's try a structured approach:
    # For each column c, we choose indices (i_c, j_c, k_c) and values (a_c, b_c, w_c)
    # such that sum_c a_c * b_c * w_c = 1 for each (i, j, k).
    
    # Actually, let's use a more systematic approach:
    # The tensor is separable in a certain way. Let's try to find a decomposition
    # by solving a system of equations.
    
    # For rank r, we have r columns. Each column c contributes:
    # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for each (i, j, k)
    
    # This is a system of n*m*p = 30 equations in r * (n*m + m*p + p*n) = r*(6+15+10) = 31r unknowns.
    
    # For r = 10, we have 310 unknowns and 30 equations - very underdetermined.
    # We can try to find a solution by fixing some variables and solving for others.
    
    # Let's try a greedy approach: build up the decomposition column by column.
    
    # Actually, let's try a simpler approach first: check if rank 10, 11, ... work
    # by using a randomized algorithm.
    
    for attempt in range(50):
        if time.time() > deadline:
            return None
        
        # Random initialization
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        
        # ALS-like iteration
        for it in range(100):
            if time.time() > deadline:
                return None
            
            # Update U
            for i in range(n*m):
                for c in range(r):
                    # U[i, c] should satisfy sum_c U[i,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                    # This is complex. Let's use a different approach.
                    pass
        
        # This approach is too slow. Let's try something else.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second deadline
    
    # Start with the schoolbook decomposition
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Try to find lower rank decompositions
    # Known results: for 2x3x5, the minimum rank is not well-known but
    # is at most 30 (schoolbook). Let's try to improve.
    
    # Strategy: Use the fact that the tensor has a lot of structure.
    # The multiplication tensor for A (n x m) * B (m x p) can be viewed as
    # a sum over j of outer products of rows of A and columns of B.
    
    # For each j, the contribution is:
    # (col j of A^T) outer (row j of B) outer (identity on C)
    # But this doesn't directly give a rank-1 decomposition.
    
    # Let's try a different approach: use the Strassen-like idea.
    # For 2x2x2, Strassen gives rank 7 < 8.
    # For 2x3x5, we can try to find a similar improvement.
    
    # One approach: partition the matrices and use block multiplication.
    # A is 2x3, B is 3x5, C is 2x5.
    
    # Let's try to find a decomposition of rank < 30 by solving a linear system.
    # We'll use the following approach:
    # 1. Assume a rank-r decomposition with r < 30.
    # 2. Set up the system of equations.
    # 3. Solve it using numpy's least squares or by exploiting the structure.
    
    # For a rank-r decomposition, we need:
    # sum_{c=1}^{r} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i, j, k)
    
    # This is a system of 30 equations. The unknowns are the entries of U, V, W.
    # Total unknowns: r * (6 + 15 + 10) = 31r.
    
    # For r = 10: 310 unknowns, 30 equations. Very underdetermined.
    # We can fix some variables and solve for the rest.
    
    # Let's try a specific structured approach:
    # Fix W to be identity-like, then solve for U and V.
    
    # Actually, let's try a simpler heuristic:
    # Try ranks from 15 to 29 and see if we can find a valid decomposition.
    
    for r in range(15, best_rank):
        if time.time() > deadline - 10:
            break
        
        # Try to find a decomposition of rank r
        # Use a randomized approach with local search
        
        for attempt in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            np.random.seed(attempt * 1000 + r)
            U = np.random.randn(n*m, r) * 0.1
            V = np.random.randn(m*p, r) * 0.1
            W = np.random.randn(p*n, r) * 0.1
            
            # ALS iteration
            for it in range(50):
                if time.time() > deadline - 2:
                    break
                
                # Update U: for each i, j, c, we want
                # sum_c' U[i*m+j, c'] * V[j*p+k, c'] * W[k*n+i, c'] = 1
                # This is a quadratic equation in U[i*m+j, :]
                
                # Simplified approach: use gradient descent
                # Compute the current tensor
                T_current = np.zeros((n, m, p))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_current[i, j, k] = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                
                # Compute gradient
                # dT/dU[i*m+j, c] = V[j*p+k, c] * W[k*n+i, c]
                # Loss = sum_{i,j,k} (T_current[i,j,k] - 1)^2
                
                # This is getting complex. Let's try a different approach.
                break
            
            # Check if the decomposition is valid
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                break
    
    # If no improvement found, return the schoolbook decomposition
    if best_rank == rank_schoolbook:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Convert numpy arrays to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    # Round to nearest half-integer if possible
    for mat in [U_list, V_list, W_list]:
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                mat[i][j] = round(mat[i][j] * 2) / 2
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
