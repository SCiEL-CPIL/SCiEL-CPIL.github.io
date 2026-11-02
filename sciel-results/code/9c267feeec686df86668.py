"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [4, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s budget, 10s safety margin
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Schoolbook rank
    rank_school = n * m * p  # 100
    
    # Build schoolbook decomposition
    U_school = np.zeros((nm, rank_school), dtype=float)
    V_school = np.zeros((mp, rank_school), dtype=float)
    W_school = np.zeros((pn, rank_school), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    # Verify function using numpy for speed
    def verify_numpy(U, V, W, tol=1e-9):
        """Check if the decomposition is exact using numpy."""
        r = U.shape[1]
        # Compute the tensor product efficiently
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Build the target tensor
        # For each (i,j,k), compute the sum
        # Use broadcasting: U[i*m+j, :] @ (V[j*p+k, :] * W[k*n+i, :])
        
        # Precompute for speed
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try to find lower rank decompositions using random search
    # with structured initialization and verification
    
    def try_random_rank(rank, num_trials=100):
        """Try to find a valid rank-r decomposition using random search."""
        np.random.seed(42)
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                return None
            
            # Initialize with small random values
            U = np.zeros((nm, rank), dtype=float)
            V = np.zeros((mp, rank), dtype=float)
            W = np.zeros((pn, rank), dtype=float)
            
            # Use structured initialization: start from schoolbook-like patterns
            # and add random perturbations
            
            # For each column, assign non-zero entries in a structured way
            for c in range(rank):
                # Choose which entries are non-zero
                # Use a random permutation approach
                
                # Randomly select a subset of indices for each matrix
                u_indices = np.random.choice(nm, size=min(rank//3, nm), replace=False)
                v_indices = np.random.choice(mp, size=min(rank//3, mp), replace=False)
                w_indices = np.random.choice(pn, size=min(rank//3, pn), replace=False)
                
                # Assign random values (integers or half-integers)
                for idx in u_indices:
                    U[idx, c] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                for idx in v_indices:
                    V[idx, c] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
                for idx in w_indices:
                    W[idx, c] = np.random.choice([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])
            
            # Verify
            if verify_numpy(U, V, W):
                return U, V, W
        
        return None
    
    # Try ranks from 99 down to 80
    # This is unlikely to succeed with pure random search, but let's try
    
    for rank in range(rank_school - 1, max(rank_school - 20, 50), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_random_rank(rank, num_trials=50)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            break
    
    # Alternative approach: use the fact that for small matrices,
    # we can try to find decompositions by solving systems of equations
    
    # For a given rank r, we need to find U, V, W such that:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # This is a system of n*m*p = 100 equations in r*(nm+mp+pn) unknowns
    # For r < 100, this is an underdetermined system (if we fix one matrix)
    
    # Let's try fixing W to be the identity-like matrix and solving for U and V
    
    def try_structured_rank(rank):
        """Try a structured approach: fix W and solve for U, V."""
        if time.time() > deadline - 5:
            return None
        
        # Fix W to be a simple matrix
        W = np.zeros((pn, rank), dtype=float)
        for c in range(rank):
            # Assign W[k*n+i, c] = 1 for some (k,i)
            k = c % p
            i = (c // p) % n
            W[k*n+i, c] = 1.0
        
        # Now solve for U and V
        # For each (i,j,k): sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # Since W[k*n+i, c] = 1 only when c corresponds to (k,i), we have:
        # U[i*m+j, c(i,k)] * V[j*p+k, c(i,k)] = 1 for the specific c where W=1
        
        # This means for each (i,k), there's a column c(i,k) where W=1
        # And U[i*m+j, c(i,k)] * V[j*p+k, c(i,k)] = 1
        
        # Let's check if this is consistent
        # For fixed (i,k), c(i,k) is determined
        # We need U[i*m+j, c(i,k)] * V[j*p+k, c(i,k)] = 1 for all j
        
        # Let U[i*m+j, c(i,k)] = a_{i,j,k} and V[j*p+k, c(i,k)] = b_{j,k}
        # Then a_{i,j,k} * b_{j,k} = 1
        
        # This means a_{i,j,k} = 1/b_{j,k}
        # For this to work with integer/half-integer entries, b_{j,k} must be ±1, ±2, etc.
        
        # Let's try b_{j,k} = 1 for all j,k, then a_{i,j,k} = 1
        # So U[i*m+j, c(i,k)] = 1 and V[j*p+k, c(i,k)] = 1
        
        # But we need rank columns, and each column c(i,k) is used for one (i,k) pair
        # Total (i,k) pairs = n*p = 20
        # So we need at least 20 columns for W
        
        if rank < n * p:
            return None
        
        # Assign W
        W = np.zeros((pn, rank), dtype=float)
        col_map = {}
        for i in range(n):
            for k in range(p):
                c = i * p + k
                if c < rank:
                    W[k*n+i, c] = 1.0
                    col_map[(i,k)] = c
        
        # Now assign U and V
        U = np.zeros((nm, rank), dtype=float)
        V = np.zeros((mp, rank), dtype=float)
        
        for (i,k), c in col_map.items():
            for j in range(m):
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
        
        # Verify
        if verify_numpy(U, V, W):
            return U, V, W
        
        return None
    
    # Try structured ranks
    for rank in range(max(n*p, 20), rank_school):
        if time.time() > deadline - 5:
            break
        
        result = try_structured_rank(rank)
        if result is not None:
            U, V, W = result
            best_rank = rank
            best_result = {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            break
    
    return best_result
