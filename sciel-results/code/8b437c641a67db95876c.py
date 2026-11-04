"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W

    def full_verify(U, V, W, n, m, p, tol=1e-6):
        """Verify complete tensor decomposition."""
        rank = U.shape[1]
        # Check all entries of the tensor
        for i in range(n):
            for j in range(m):
                a = i * m + j
                for j2 in range(m):
                    for k in range(p):
                        b = j2 * p + k
                        for k2 in range(p):
                            for i2 in range(n):
                                c = k2 * n + i2
                                val = 0.0
                                for cc in range(rank):
                                    val += U[a, cc] * V[b, cc] * W[c, cc]
                                # Expected: 1 if i==i2 and j==j2 and k==k2, else 0
                                if i == i2 and j == j2 and k == k2:
                                    expected = 1.0
                                else:
                                    expected = 0.0
                                if abs(val - expected) > tol:
                                    return False
        return True

    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    
    # Try to find rank reductions using algebraic identities
    # For 4x5x5, one approach is to use block decomposition
    
    # Split the 4x5 matrix into two 2x5 blocks
    # C = [A1; A2] * B where A1, A2 are 2x5 and B is 5x5
    # C1 = A1 * B (2x5), C2 = A2 * B (2x5)
    
    # For each 2x5x5 multiplication, schoolbook rank is 50
    # Total schoolbook: 100
    
    # Try to find if 2x5x5 can be done in less than 50
    # This is an open problem in general, but for specific cases we might find reductions
    
    # Alternative: use the known fact that for n x m x p, there are sometimes
    # linear dependencies that can be exploited
    
    # For now, let's try a numerical approach with a time limit
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p
    
    # Try rank 99 first
    if time.time() < deadline - 30:
        np.random.seed(42)
        for attempt in range(10):
            if time.time() > deadline - 30:
                break
            
            # Initialize by perturbing schoolbook
            U = U_sb.copy()
            V = V_sb.copy()
            W = W_sb.copy()
            
            # Remove one column and try to compensate
            col_to_remove = np.random.randint(0, n*m*p)
            U[:, col_to_remove] = 0
            V[:, col_to_remove] = 0
            W[:, col_to_remove] = 0
            
            # Try to optimize the remaining columns
            # This is a hard non-convex problem, so use a simple heuristic
            # For now, just check if removing a column still works (it won't)
            # So this approach is flawed
            
            break
    
    # If no reduction found, return schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
# EVOLVE-BLOCK-END
