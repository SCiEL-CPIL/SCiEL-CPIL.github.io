"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def construct():
    """Find rank-minimal exact tensor decomposition for 2x3x3 matrix multiplication."""
    deadline = time.time() + 140
    n, m, p = DIMENSIONS  # 2, 3, 3
    
    # Precompute the set of valid (a, b, c) indices where the tensor is 1
    # a = i*m+j, b = j*p+k, c = k*n+i
    valid_indices = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_indices.add((a, b, c))
    
    # All possible (a, b, c) indices
    all_a = list(range(n * m))
    all_b = list(range(m * p))
    all_c = list(range(p * n))
    
    def verify_fast(U, V, W, rank):
        """Fast verification using numpy. U, V, W are numpy arrays of shape (dim, rank)."""
        # Compute the full tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Use einsum for efficiency
        T = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Check valid entries equal 1
        for (a, b, c) in valid_indices:
            if T[a, b, c] != 1:
                return False
        
        # Check invalid entries equal 0
        for a in all_a:
            for b in all_b:
                for c in all_c:
                    if (a, b, c) not in valid_indices:
                        if T[a, b, c] != 0:
                            return False
        return True
    
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
    
    U, V, W, rank = schoolbook()
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions using structured search
    # Key insight: for matrix multiplication tensors, there are known algebraic
    # structures. Let's try systematic search with small integer coefficients.
    
    import random
    random.seed(42)
    
    # Try ranks from 17 down to 10
    for test_rank in range(17, 9, -1):
        if time.time() > deadline - 30:
            break
        
        # Strategy: Use structured patterns based on the tensor's symmetry
        # For 2x3x3, try various algebraic constructions
        
        found = False
        
        # Method 1: Random search with small integers
        for trial in range(20000):
            if time.time() > deadline - 15:
                break
            
            # Generate random matrices with small integer entries
            # Focus on sparse patterns that might exploit tensor structure
            max_val = 2
            
            # Create sparse random matrices - each row has few non-zero entries
            U_try = np.zeros((n * m, test_rank))
            V_try = np.zeros((m * p, test_rank))
            W_try = np.zeros((p * n, test_rank))
            
            for a in range(n * m):
                # Each row of U has 1-3 non-zero entries
                num_nz = random.randint(1, min(3, test_rank))
                cols = random.sample(range(test_rank), num_nz)
                for c in cols:
                    U_try[a, c] = random.randint(-max_val, max_val)
            
            for b in range(m * p):
                num_nz = random.randint(1, min(3, test_rank))
                cols = random.sample(range(test_rank), num_nz)
                for c in cols:
                    V_try[b, c] = random.randint(-max_val, max_val)
            
            for c in range(p * n):
                num_nz = random.randint(1, min(3, test_rank))
                cols = random.sample(range(test_rank), num_nz)
                for cc in cols:
                    W_try[c, cc] = random.randint(-max_val, max_val)
            
            if verify_fast(U_try, V_try, W_try, test_rank):
                best_rank = test_rank
                best_U, best_V, best_W = U_try.copy(), V_try.copy(), W_try.copy()
                found = True
                break
        
        if found:
            break
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
