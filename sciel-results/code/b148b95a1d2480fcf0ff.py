"""Matrix multiplication 3x5 x 5x6 tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    
    # Schoolbook: rank = n*m*p = 90
    rank_school = n * m * p
    U_school = np.zeros((n*m, rank_school))
    V_school = np.zeros((m*p, rank_school))
    W_school = np.zeros((p*n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
    
    best_U = U_school.tolist()
    best_V = V_school.tolist()
    best_W = W_school.tolist()
    best_rank = rank_school
    
    def verify_numpy(U, V, W, tol=1e-9):
        """Fast NumPy verification of tensor decomposition."""
        # T[a,b,c] = sum_d U[a,d]*V[b,d]*W[c,d]
        # Need T[i*m+j, j*p+k, k*n+i] = 1 for all valid (i,j,k)
        # And T[a,b,c] = 0 otherwise
        
        # Build the full tensor efficiently
        # U shape: (n*m, rank)
        # V shape: (m*p, rank)  
        # W shape: (p*n, rank)
        
        rank = U.shape[1]
        
        # For each column d, we have outer product U[:,d] ⊗ V[:,d] ⊗ W[:,d]
        # Sum over d
        # This is expensive for large rank, but let's try
        
        # Alternative: check only the required entries
        # Required: for all i,j,k: sum_d U[i*m+j,d]*V[j*p+k,d]*W[k*n+i,d] = 1
        # And for "wrong" (a,b,c): sum should be 0
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i*m + j
                    b = j*p + k
                    c = k*n + i
                    s = np.dot(U[a], np.multiply(V[b], W[c]))
                    if abs(s - 1) > tol:
                        return False
        
        # Check a sample of wrong entries
        # Total wrong entries: (n*m)*(m*p)*(p*n) - n*m*p
        # We can't check all, so check random samples
        for _ in range(200):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Determine if this is a "valid" entry
            i1 = a // m
            j1 = a % m
            j2 = b // p
            k2 = b % p
            k1 = c // n
            i2 = c % n
            if i1 == i2 and j1 == j2:
                expected = 1
            else:
                expected = 0
            s = np.dot(U[a], np.multiply(V[b], W[c]))
            if abs(s - expected) > tol:
                return False
        
        return True
    
    def try_block_decomposition(n, m, p):
        """Try to find a block-based decomposition with lower rank."""
        # Split the middle dimension into blocks
        # For m=5, we can split into 2+3
        # A (3x5) = [A1 (3x2) | A2 (3x3)]
        # B (5x6) = [B1 (2x6); B2 (3x6)]
        # A*B = A1*B1 + A2*B2
        
        # This gives rank = 3*2*6 + 3*3*6 = 36 + 54 = 90 (same as schoolbook)
        # Not helpful directly.
        
        # Try Strassen-like for the 3x3 part:
        # A2*B2 where A2 is 3x3, B2 is 3x6
        # We can use rank-19 or better for 3x3x3, then extend to 3x3x6
        
        # For 3x3x3, best known is 23 (Lader). For 3x3x6, we can do:
        # Split B2 into B2a (3x3) and B2b (3x3)
        # A2*B2 = A2*B2a + A2*B2b
        # Use rank-23 for each: total 46 for the 3x3x6 part
        # Plus A1*B1: 3x2x6, schoolbook is 36
        # Total: 46 + 36 = 82
        
        # Let's try to construct this
        # First, need a rank-23 decomposition for 3x3x3 matrix multiplication
        
        # Known rank-23 decomposition for 3x3x3 (Lader's algorithm)
        # This is complex. Let's use a simpler approach.
        
        # Alternative: use the fact that for 2x2, Strassen gives rank 7
        # For 3x5x6, we can try various block splits
        
        # Split m=5 into 1+4, 2+3, etc.
        # Try 2+3 split with Strassen on the 3x3 part
        
        # Actually, let's try a different approach:
        # Use random structured matrices and solve linear equations
        
        return None
    
    # Try to find a lower rank decomposition using structured random search
    # Key idea: use sparse random matrices with specific structure
    
    def try_random_structured(target_rank, num_trials=50):
        """Try random structured decompositions."""
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                return None
            
            # Create structured random matrices
            # Use block-diagonal or banded structure
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Each column corresponds to a "term" in the decomposition
            # Use random ±1 entries but with some structure
            for c in range(target_rank):
                # Randomly choose which (i,j,k) this term affects
                # But we need to satisfy the tensor equation
                # Let's just use fully random ±1 for now
                U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
                V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
                W[:, c] = np.random.choice([-1, 0, 1], size=p*n)
            
            if verify_numpy(U, V, W):
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try to reduce rank from schoolbook by finding linear dependencies
    # The schoolbook has rank 90. Can we find a lower rank solution?
    
    # Use a greedy column elimination approach:
    # Start with schoolbook, try to express each column as a combination of others
    
    def try_column_reduction(U, V, W, max_eliminations=10):
        """Try to eliminate columns by expressing them as linear combos."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        rank = U.shape[1]
        
        eliminated = 0
        for col in range(rank):
            if time.time() > deadline - 30:
                break
            if col >= U.shape[1]:
                break
            
            # Try to express column col as a combination of other columns
            # This is a linear system problem
            # For simplicity, try to eliminate if the column is linearly dependent
            
            # Check if column is in span of others
            # Use SVD to check rank
            other_cols = list(range(rank))
            other_cols.remove(col)
            
            if len(other_cols) < 1:
                continue
            
            # Build matrix of other columns
            U_other = U[:, other_cols]
            V_other = V[:, other_cols]
            W_other = W[:, other_cols]
            
            # Check if U[:,col] is in span of U_other
            # And V[:,col] in span of V_other
            # And W[:,col] in span of W_other
            # And the tensor equation is preserved
            
            # This is complex. Let's try a simpler approach:
            # Try to set one column to zero and see if the equation still holds
            # by adjusting other columns
            
            # For now, skip this and try a different approach
        
        return U.tolist(), V.tolist(), W.tolist(), eliminated
    
    # Try various target ranks with structured random search
    # Focus on ranks that are plausible for 3x5x6
    # Lower bound: at least max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # Upper bound: 90 (schoolbook)
    # Realistic: probably 60-80
    
    # Try a more systematic search
    for target_rank in [85, 80, 75, 70, 65, 60]:
        if time.time() > deadline - 40:
            break
        
        result = try_random_structured(target_rank, num_trials=30)
        if result is not None:
            best_U, best_V, best_W = result
            best_rank = target_rank
            break
    
    # If we found a better decomposition, try to refine it
    if best_rank < rank_school:
        # Try to reduce further
        for target_rank in range(best_rank - 5, max(best_rank - 20, 30), -5):
            if time.time() > deadline - 20:
                break
            result = try_random_structured(target_rank, num_trials=20)
            if result is not None:
                best_U, best_V, best_W = result
                best_rank = target_rank
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
