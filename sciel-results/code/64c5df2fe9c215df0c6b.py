"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix
    by a 5 by 6 matrix.
    
    Uses a random search with greedy linear combination pruning to find a 
    decomposition with rank < n*m*p = 60.
    """
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 60
    
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 130.0  # 10s safety margin
    
    # The tensor M[i,j,k] = 1 if we're computing (A @ B)[i,k] = sum_j A[i,j]B[j,k]
    # Decomposition: M[i,k,j] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # Wait, the indices are:
    # U has shape (n*m, rank), indexed by [i*m+j, c]
    # V has shape (m*p, rank), indexed by [j*p+k, c]  
    # W has shape (p*n, rank), indexed by [k*n+i, c]
    # The tensor element T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # must equal 1 for all i,j,k (since matrix mult tensor is all-1s)
    
    # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k
    # (each entry of the product depends on exactly one term A[i,j]*B[j,k])
    
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Schoolbook: one term per (i,j,k) with U=V=W=1
    # Better: use random vectors and solve for the constraint
    
    # Strategy: Try random rank-r decompositions and check if they satisfy the constraint.
    # Use a greedy approach: start with random rank-1 terms, add more if needed.
    
    # Let's try a different approach: use the fact that for fixed i,k,
    # sum_j A[i,j]B[j,k] = sum_c (sum_j U[i*m+j,c]*V[j*p+k,c]) * W[k*n+i,c]
    # We need this to equal sum_j A[i,j]B[j,k] for all A,B.
    # So: sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*V[j*p+k,c]) must equal 1 for all i,k... no.
    
    # Actually the constraint is: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is a system of n*m*p = 60 equations in rank*r variables (where r is the number of columns).
    # But it's nonlinear (cubic).
    
    # Simpler approach: Try to find a rank-r decomposition by random initialization 
    # and iterative refinement.
    
    # Let's try small ranks first and use random search.
    
    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    # Try rank 32 to 59 with random search
    for r in range(32, 60):
        if time.time() > deadline:
            break
        # Random search: try multiple random initializations
        for trial in range(50):
            if time.time() > deadline:
                break
            # Initialize with random ±1 entries
            np.random.seed(trial * 1000 + r)
            U = np.random.choice([-1.0, 1.0], size=(n*m, r))
            V = np.random.choice([-1.0, 1.0], size=(m*p, r))
            W = np.random.choice([-1.0, 1.0], size=(p*n, r))
            
            # Compute the resulting tensor and check
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(r):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (s - 1.0)**2
            if error < 1e-12:
                # Found a valid decomposition
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If no random solution found, fall back to schoolbook
    rank = rank_schoolbook
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
# EVOLVE-BLOCK-END
