"""
5x5 matrix multiplication - randomized algebraic rank-reduction heuristic.
Uses random initialization with integer-preserving projection to minimize rank.
"""
import random
import numpy as np
from itertools import product

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Build the 3D tensor T[i,j,k] = 1 for matrix multiplication
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,j,k] = 1 for all i,j,k in range(5)
    
    # We need to find rank-r decomposition where:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_{a,b} U[a*m+b, c] * A[a,b]) * (sum_{b',d} V[b'*p+d, c] * B[b',d])
    
    # This equals sum_c W[k*n+i,c] * (U_col_c · A_flat) * (V_col_c · B_flat)
    
    # The constraint is that for all A, B:
    # sum_c W[k*n+i,c] * (U[:,c] · A_flat) * (V[:,c] · B_flat) = sum_j A[i,j] * B[j,k]
    
    # Strategy: Try ranks from a lower bound upward, using random initialization
    # and checking validity.
    
    # Lower bound for 5x5 matrix mult: max(n*m, m*p, p*n) = 25
    # Upper bound: n*m*p = 125 (schoolbook)
    # Best known: around 103 for 5x5
    
    # Try a few random attempts at different ranks
    best_result = None
    best_rank = n * m * p  # schoolbook
    
    # First, always have schoolbook as fallback
    U_s = np.zeros((n*m, best_rank), dtype=float)
    V_s = np.zeros((m*p, best_rank), dtype=float)
    W_s = np.zeros((p*n, best_rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1
                V_s[j*p+k, col] = 1
                W_s[k*n+i, col] = 1
                col += 1
    
    # Try to find lower rank decompositions using random search
    # We'll try ranks from 120 down to maybe 100
    
    def validate_decomposition(U, V, W, rank):
        """Check if the decomposition is valid."""
        # For random test matrices A and B, check if C = A*B
        for _ in range(20):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A @ B
            
            A_flat = A.flatten()
            B_flat = B.flatten()
            
            # Compute using decomposition
            U_A = U @ A_flat  # shape (rank,)
            V_B = V @ B_flat  # shape (rank,)
            C_computed = np.zeros((n, p))
            
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] = np.sum(W[k*n+i, :] * U_A * V_B)
            
            if not np.allclose(C_computed, C_expected, atol=1e-8):
                return False
        return True
    
    def random_decomposition(rank, seed=None):
        """Generate a random decomposition attempt."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Initialize with small random integers
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        
        # Use structured initialization based on schoolbook but with merging
        # For each column, assign it to a random (i,j,k) triple
        for c in range(rank):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            
            # With some probability, add contributions to multiple positions
            # to allow for Strassen-like cancellations
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0
            
            # Sometimes add a second term for cancellation
            if random.random() < 0.3:
                i2 = random.randint(0, n-1)
                j2 = random.randint(0, m-1)
                k2 = random.randint(0, p-1)
                sign = random.choice([-1.0, 1.0])
                U[i2*m+j2, c] += sign
                V[j2*p+k2, c] += sign
                W[k2*n+i2, c] += sign
        
        return U, V, W
    
    # Try several ranks with multiple random attempts
    for target_rank in range(120, 99, -1):
        found = False
        for attempt in range(50):
            U, V, W = random_decomposition(target_rank, seed=attempt * 1000 + target_rank)
            if validate_decomposition(U, V, W, target_rank):
                found = True
                best_rank = target_rank
                best_result = (U, V, W)
                break
        if found:
            break
    
    if best_result is not None:
        U, V, W = best_result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fall back to schoolbook
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
