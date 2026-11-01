"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    deadline = time.time() + 140  # 140s deadline for 150s wall clock
    
    # First, try to find a rank-17 decomposition by searching for a single
    # modification to the rank-18 Strassen+schoolbook structure.
    # If that fails, fall back to the known rank-18 solution.
    
    def build_rank18():
        rank = 18
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_strassen_block(col0, col1, base):
            r = base
            U[0][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 1
            U[2][r] = 1; U[3][r] = 1
            V[0*p+col0][r] = 1
            W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
            
            r = base + 2
            U[0][r] = 1
            V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
            W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
            
            r = base + 3
            U[3][r] = 1
            V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
            W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
            
            r = base + 4
            U[0][r] = 1; U[1][r] = 1
            V[1*p+col1][r] = 1
            W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
            
            r = base + 5
            U[2][r] = 1; U[0][r] = -1
            V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
            W[col1*n+1][r] = 1
            
            r = base + 6
            U[1][r] = 1; U[3][r] = -1
            V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
            W[col0*n+0][r] = 1
        
        add_strassen_block(0, 1, 0)
        add_strassen_block(2, 3, 7)
        
        r = 14
        U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
        r = 15
        U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
        r = 16
        U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
        r = 17
        U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
        
        return U, V, W
    
    def check_decomposition(U, V, W, rank):
        """Verify that the decomposition is exact."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reconstruct: for each (i,j,k), sum_r U[i,r]*V[j,r]*W[k,r] should equal
        # the matrix multiplication tensor T[i,j,k] = A[i,:]*B[:,k] coefficient
        # T[i,j,k] = 1 if (i,j) is a valid entry of A and j is a valid entry of B
        # Specifically, for 2x2 * 2x5: A is 2x2, B is 2x5, C is 2x5
        # C[a,c] = sum_b A[a,b] * B[b,c]
        # Tensor T[a,b,c] = 1 if this is the (a,b) of A times (b,c) of B
        # So T[a,b,c] = 1 for all valid a,b,c where b is the shared index
        
        # U has shape (n*m, rank) = (4, rank): index = a*m + b = a*2 + b
        # V has shape (m*p, rank) = (10, rank): index = b*p + c = b*5 + c  
        # W has shape (p*n, rank) = (10, rank): index = c*n + a = c*2 + a
        
        # T[a,b,c] should equal sum_r U[a*2+b, r] * V[b*5+c, r] * W[c*2+a, r]
        # T[a,b,c] = 1 for all a in {0,1}, b in {0,1}, c in {0,1,2,3,4}
        
        for a in range(n):
            for b in range(m):
                for c in range(p):
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U_arr[a*m+b, r] * V_arr[b*p+c, r] * W_arr[c*n+a, r]
                    if abs(actual - expected) > 1e-9:
                        return False
        return True
    
    # Start with rank 18
    U18, V18, W18 = build_rank18()
    
    # Try to reduce to rank 17 by finding a linear dependency or
    # by modifying one term to absorb another.
    # Strategy: try to express one of the rank-18 terms as a linear combination
    # of the others, or find a perturbation that reduces rank.
    
    # For now, try a direct search: modify the last 4 schoolbook terms to use
    # 3 terms instead of 4 by sharing with Strassen terms.
    
    # Attempt: Can we do column 4 in 3 terms by sharing a U-factor with Strassen?
    # The schoolbook terms for col 4 are:
    #   term 14: U=[1,0,0,0], V=[1,0,0,0,0,0,0,0,0,0] (B04), W=[0,0,0,0,1,0,0,0,0,0] (C[0,4])
    #   term 15: U=[0,1,0,0], V=[0,0,0,0,0,1,0,0,0,0] (B14), W=[0,0,0,0,1,0,0,0,0,0] (C[0,4])
    #   term 16: U=[0,0,1,0], V=[1,0,0,0,0,0,0,0,0,0] (B04), W=[0,0,0,0,0,1,0,0,0,0] (C[1,4])
    #   term 17: U=[0,0,0,1], V=[0,0,0,0,0,1,0,0,0,0] (B14), W=[0,0,0,0,0,1,0,0,0,0] (C[1,4])
    
    # These are 4 independent rank-1 tensors. To reduce to 3, we'd need to find
    # a 3-term decomposition of the 2x2 x 2x1 tensor, which has rank 4. So impossible.
    
    # Alternative: modify Strassen terms to incorporate column 4 contributions.
    # This is complex; let's try a numerical search for rank 17.
    
    # Given time constraints, return the known rank-18 solution if search fails.
    if time.time() > deadline - 5:
        return {"U": U18, "V": V18, "W": W18}
    
    # Try a simple ALS-based search for rank 17
    best_result = None
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
        
        # Initialize randomly
        rank = 17
        np.random.seed(attempt * 12345 + 67890)
        U_rand = np.random.randn(n*m, rank)
        V_rand = np.random.randn(m*p, rank)
        W_rand = np.random.randn(p*n, rank)
        
        # ALS iterations
        for it in range(200):
            if time.time() > deadline - 10:
                break
            
            # Update U
            for i in range(n*m):
                # U[i,r] = sum_{j,k} T[i,j,k] * V[j,r] * W[k,r] / sum_{j,k} V[j,r]^2 * W[k,r]^2
                num = np.zeros(rank)
                den = np.zeros(rank)
                for j in range(m*p):
                    for k in range(p*n):
                        # Check if (i,j,k) is a valid tensor entry
                        # i = a*m+b, j = b'*p+c, k = c'*n+a'
                        # Valid if b == b' and c == c' and a == a'
                        a = i // m
                        b = i % m
                        b2 = j // p
                        c = j % p
                        c2 = k // n
                        a2 = k % n
                        if b == b2 and c == c2 and a == a2:
                            for r in range(rank):
                                num[r] += V_rand[j, r] * W_rand[k, r]
                                den[r] += V_rand[j, r]**2 * W_rand[k, r]**2
                mask = den > 1e-10
                U_rand[i, mask] = num[mask] / den[mask]
            
            # Update V
            for j in range(m*p):
                num = np.zeros(rank)
                den = np.zeros(rank)
                for i in range(n*m):
                    for k in range(p*n):
                        a = i // m
                        b = i % m
                        b2 = j // p
                        c = j % p
                        c2 = k // n
                        a2 = k % n
                        if b == b2 and c == c2 and a == a2:
                            for r in range(rank):
                                num[r] += U_rand[i, r] * W_rand[k, r]
                                den[r] += U_rand[i, r]**2 * W_rand[k, r]**2
                mask = den > 1e-10
                V_rand[j, mask] = num[mask] / den[mask]
            
            # Update W
            for k in range(p*n):
                num = np.zeros(rank)
                den = np.zeros(rank)
                for i in range(n*m):
                    for j in range(m*p):
                        a = i // m
                        b = i % m
                        b2 = j // p
                        c = j % p
                        c2 = k // n
                        a2 = k % n
                        if b == b2 and c == c2 and a == a2:
                            for r in range(rank):
                                num[r] += U_rand[i, r] * V_rand[j, r]
                                den[r] += U_rand[i, r]**2 * V_rand[j, r]**2
                mask = den > 1e-10
                W_rand[k, mask] = num[mask] / den[mask]
        
        # Check if we got an exact solution
        if check_decomposition(U_rand.tolist(), V_rand.tolist(), W_rand.tolist(), rank):
            best_result = (U_rand, V_rand, W_rand)
            break
    
    if best_result is not None:
        U_r, V_r, W_r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
    
    # Fall back to rank 18
    return {"U": U18, "V": V18, "W": W18}
# EVOLVE-BLOCK-END
