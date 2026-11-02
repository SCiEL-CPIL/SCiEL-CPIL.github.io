"""Rewritten program for Matrix multiplication 2,2,13"""

import time
import itertools

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    
    # Strategy: Try to find rank-10 decomposition for (2,2,3)
    # Then use 4 groups of 3 columns + 1 schoolbook column = 4*10 + 4 = 44
    
    # First, let's try to construct a (2,2,3) rank-10 decomposition
    # The (2,2,3) tensor has:
    # A: 2x2 (4 entries), B: 2x3 (6 entries), C: 3x2 (6 entries)
    # We need to find 10 rank-1 terms that sum to the multiplication tensor
    
    # Let me try a systematic approach using known results
    # For (2,2,3), we can try to extend Strassen-like ideas
    
    # Actually, let me first implement the known 46-rank solution as fallback
    # Then try to improve
    
    def build_46():
        rank = 46
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        def add_term(u, v, w):
            nonlocal col
            for i, val in u.items():
                U[i][col] += val
            for i, val in v.items():
                V[i][col] += val
            for i, val in w.items():
                W[i][col] += val
            col += 1
        
        for b in range(6):
            k0 = 2*b
            k1 = 2*b+1
            add_term({0:1, 3:1}, {k0:1, 13+k1:1}, {2*k0:1, 2*k1+1:1})
            add_term({2:1, 3:1}, {k0:1}, {2*k0+1:1, 2*k1+1:-1})
            add_term({0:1}, {k1:1, 13+k1:-1}, {2*k1:1, 2*k1+1:1})
            add_term({3:1}, {13+k0:1, k0:-1}, {2*k0:1, 2*k0+1:1})
            add_term({0:1, 1:1}, {13+k1:1}, {2*k0:-1, 2*k1:1})
            add_term({2:1, 0:-1}, {k0:1, k1:1}, {2*k1+1:1})
            add_term({1:1, 3:-1}, {13+k0:1, 13+k1:1}, {2*k0:1})
        
        k = 12
        add_term({0:1}, {k:1}, {2*k:1})
        add_term({1:1}, {13+k:1}, {2*k:1})
        add_term({2:1}, {k:1}, {2*k+1:1})
        add_term({3:1}, {13+k:1}, {2*k+1:1})
        
        return {"U": U, "V": V, "W": W}
    
    # Try to find a (2,2,3) rank-10 decomposition
    # The multiplication tensor for (2,2,3):
    # T[(i,j), (j,k), (k,i)] = 1 for all valid i,j,k
    # i in {0,1}, j in {0,1}, k in {0,1,2}
    
    # Let's try a specific construction
    # We want 10 terms. Let me try to use a mix of Strassen-like and additional terms.
    
    # For (2,2,3), one approach:
    # Use the 7 Strassen terms for columns 0,1 and 4 schoolbook terms for column 2 = 11
    # Can we save 1 term?
    
    # Let me try to find a rank-10 decomposition by solving a system of equations
    # or by using a known construction.
    
    # Actually, let me try a different grouping: 13 = 3 + 3 + 3 + 3 + 1
    # If (2,2,3) has rank 10, then 4*10 + 4 = 44
    
    # Let me try to verify if rank-10 for (2,2,3) exists by attempting to construct it
    
    # I'll try a specific ansatz for (2,2,3) rank 10
    # Based on literature, (2,2,3) might have rank 10
    
    # Let me try to construct it:
    # M1: (a00+a11)(b00+b10) -> c00 += 1, c10 += 1  [modified Strassen for 3 cols]
    # This is getting complex. Let me just implement the 46 solution and verify it works.
    
    result = build_46()
    
    # Verify the solution is correct
    import numpy as np
    U = np.array(result['U'], dtype=float)
    V = np.array(result['V'], dtype=float)
    W = np.array(result['W'], dtype=float)
    
    # Test with random matrices
    np.random.seed(42)
    A_test = np.random.rand(n, m)
    B_test = np.random.rand(m, p)
    C_expected = A_test @ B_test
    
    # Compute using the decomposition
    # C[i,k] = sum_r U[i*? + ?][r] * V[?][r] * W[k*? + ?][r]
    # U is (n*m, rank) -> U[i*m + j, r] for A_{ij}
    # V is (m*p, rank) -> V[j*p + k, r] for B_{jk}
    # W is (p*n, rank) -> W[k*n + i, r] for C_{ki}
    
    C_computed = np.zeros((p, n))
    for i in range(n):
        for k in range(p):
            total = 0
            for r in range(len(U[0])):
                total += sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for j in range(m))
            C_computed[k, i] = total
    
    # C_expected is (n, p), C_computed is (p, n) - they should be transposes
    error = np.max(np.abs(C_computed.T - C_expected))
    
    if error > 1e-10:
        # Fallback to a simple schoolbook implementation
        rank = n * m * p  # 52
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1
        
        result = {"U": U, "V": V, "W": W}
    
    if time.time() > deadline:
        return result
    
    return result
# EVOLVE-BLOCK-END
