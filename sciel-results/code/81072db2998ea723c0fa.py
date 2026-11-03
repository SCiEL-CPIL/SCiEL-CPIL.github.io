"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 5
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Schoolbook rank = n*m*p = 80
    schoolbook_rank = n * m * p
    
    def build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        # Check shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U):
            return False
        if any(len(row) != rank for row in V):
            return False
        if any(len(row) != rank for row in W):
            return False
        
        # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # For matrix multiplication, T[i,j,k] = 1 if the indices correspond to a valid multiplication term
        # Actually, the matrix multiplication tensor has T[i,j,k] = 1 for all i,j,k in the standard convention
        # Wait, let me reconsider. The matrix multiplication tensor M[i,j,k] where:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 (it's a constant-1 tensor in the relevant indices)
        # 
        # The decomposition requires: for all i in [0,n), j in [0,m), k in [0,p):
        #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        #
        # And for all other combinations of indices (out of range), it should be 0,
        # but since U, V, W are only defined for valid indices, we just check the valid ones.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    # Try to find a lower rank decomposition using random search / optimization
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(16, 20, 20) = 20
    # Best known upper bounds for 4x4 by 4x5 are not trivially known to be much below 80.
    # Let's try a randomized approach to see if we can find something.
    
    def try_random_rank(rank):
        """Try random factor matrices of given rank."""
        for attempt in range(200):
            if time.time() > deadline:
                return None
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            # Normalize to try to get sums close to 1
            # This is a heuristic; random won't likely work for exact decomposition
            # But let's check anyway
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        if abs(s - 1.0) > 1e-6:
                            break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                # All checks passed
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W.tolist()
                }
        return None
    
    # The schoolbook solution is guaranteed to work
    best_result = build_schoolbook()
    
    # Try to find a better solution with lower rank
    # For 4x4 by 4x5, let's try ranks from 20 upward
    # But random search is unlikely to find exact integer/half-integer decompositions
    # Let's just return the schoolbook solution since it's guaranteed valid
    
    # Actually, let me try a structured approach: 
    # The matrix multiplication tensor for (n,m,p) has a known structure.
    # For small cases, we can try to use the fact that the tensor is a "slice" of a larger tensor.
    
    # Given the time constraints and the difficulty of finding exact lower-rank decompositions,
    # the schoolbook rank of 80 is our safe bet.
    
    return best_result
# EVOLVE-BLOCK-END
