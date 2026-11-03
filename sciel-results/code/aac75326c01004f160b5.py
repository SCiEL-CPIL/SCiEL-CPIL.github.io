import numpy as np
import time
import random
from itertools import product

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe
    
    def schoolbook_decomposition():
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def try_random_rank(r, num_attempts=50, deadline=None):
        """Try to find a rank-r decomposition using random U,V and solving for W."""
        for attempt in range(num_attempts):
            if deadline is not None and time.time() > deadline:
                return None
            
            # Create random sparse U and V with entries from {-1, 0, 1}
            U = np.zeros((n * m, r), dtype=np.float64)
            V = np.zeros((m * p, r), dtype=np.float64)
            
            # Each row gets a few nonzeros
            for row in range(n * m):
                nnz = random.randint(1, min(4, r))
                cols = random.sample(range(r), nnz)
                for c in cols:
                    U[row, c] = random.choice([-1.0, 1.0])
            
            for row in range(m * p):
                nnz = random.randint(1, min(4, r))
                cols = random.sample(range(r), nnz)
                for c in cols:
                    V[row, c] = random.choice([-1.0, 1.0])
            
            # Build linear system for W
            # For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Unknowns: W has p*n*r variables, indexed by (k*n+i, r)
            
            num_eqs = n * m * p
            num_unk = p * n * r
            
            A = np.zeros((num_eqs, num_unk), dtype=np.float64)
            b = np.ones(num_eqs, dtype=np.float64)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(r):
                            coeff = U[i * m + j, r] * V[j * p + k, r]
                            if coeff != 0:
                                A[eq_idx, (k * n + i) * r + r] = coeff
                        eq_idx += 1
            
            # Solve the linear system
            try:
                # Use least squares since the system may be underdetermined
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
                
                # Check if the solution is valid
                if residuals is not None and len(residuals) > 0:
                    if residuals[0] > 1e-6:
                        continue
                
                # Reconstruct W
                W = W_flat.reshape((p * n, r))
                
                # Verify
                if verify_decomposition(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            except np.linalg.LinAlgError:
                continue
            
            continue
        
        return None
    
    def try_structured_kronecker():
        """
        Try to construct a decomposition using Kronecker product structure.
        For 3x3x5, we can think of it as combining a 3x3x3 and 3x3x2 part,
        or use other structured approaches.
        
        Strategy: Use the fact that 5 = 3 + 2, and try to build from
        known decompositions of 3x3x3 and 3x3x2.
        """
        # For 3x3x2, the schoolbook rank is 18.
        # For 3x3x3, the schoolbook rank is 27.
        # But we can do better.
        
        # Let's try a direct structured approach:
        # Partition the 5 columns into groups and use shared terms.
        
        # Approach: Use a block structure where some terms are shared
        # across different k values.
        
        # Try rank 44 first: can we eliminate one term?
        # This is hard to do systematically, so let's try the random approach
        # for ranks 44, 43, 42, etc.
        
        pass
    
    # First, try structured approaches
    try_structured_kronecker()
    
    # Try decreasing ranks from 44 down to some minimum
    # Start with ranks close to schoolbook for efficiency
    best_result = None
    
    # Try ranks from 44 down to 30
    for r in range(rank_schoolbook - 1, 29, -1):
        if time.time() > deadline:
            break
        
        result = try_random_rank(r, num_attempts=30, deadline=deadline)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fall back to schoolbook
    return schoolbook_decomposition()
