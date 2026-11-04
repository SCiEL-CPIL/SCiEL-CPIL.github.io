"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Default: trivial decomposition
    def trivial():
        rank = n * m * p
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
    
    best_result = trivial()
    
    # Try to find better decompositions using structured patterns
    # For 3x3 * 3x7, the tensor has dimensions (3*3) x (3*7) x (7*3) = 9 x 21 x 21
    # Rank lower bound: max(9, 21) = 21, upper bound: 9*21 = 189
    # Known: Strassen-like algorithms can reduce rank for certain sizes
    
    # Try rank values from lower bound upward
    # Use randomized search with structured constraints
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose matrix multiplication"""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check tensor decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # For matrix multiplication: T[i,j,k] = 1 if this is a valid multiplication term
        # Actually, the matrix multiplication tensor is:
        # T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'}
        # Wait, let me reconsider. The standard formulation:
        # (A*B)_{i,k} = sum_j A_{i,j} * B_{j,k}
        # So the tensor T has indices (i,j) for A, (j,k) for B, and output (i,k)
        # T[(i,j), (j',k), (i',k')] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
        
        # Using the W[k*n+i, r] convention:
        # W index: k*n + i where k is column of B, i is row of A
        # U index: i*m + j where i is row of A, j is column of A
        # V index: j*p + k where j is row of B, k is column of B
        
        # The decomposition should satisfy:
        # For all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for all other combinations: sum_r ... = 0
        
        # Let's verify by checking specific cases
        # Case 1: i=i', j=j', k=k' -> should be 1
        # Case 2: any mismatch -> should be 0
        
        # Efficient verification: build the tensor from U,V,W and check
        # T_out[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        errors = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # This should be 1 (the "diagonal" term)
                    val = 0.0
                    for r in range(rank):
                        val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        errors += 1
                        if errors > 10:
                            return False
        
        # Check some off-diagonal terms (sampled for speed)
        import random
        random.seed(42)
        checks = 0
        for _ in range(100):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            k = random.randint(0, p-1)
            # Pick a different combination
            i2 = (i + random.randint(1, n)) % n
            if i2 == i:
                i2 = (i + 1) % n
            j2 = (j + random.randint(1, m)) % m
            if j2 == j:
                j2 = (j + 1) % m
            k2 = (k + random.randint(1, p)) % p
            if k2 == k:
                k2 = (k + 1) % p
            
            # For the term U[i*m+j2, r] * V[j2*p+k2, r] * W[k2*n+i, r]
            # This corresponds to a "cross" term that should be 0
            # unless i=i', j=j', k=k' in the appropriate sense
            
            val = 0.0
            for r in range(rank):
                val += U_arr[i*m+j2, r] * V_arr[j2*p+k2, r] * W_arr[k2*n+i, r]
            
            # This should be 0 unless the indices match in a specific way
            # Actually, the tensor is T[(i,j), (j',k), (k',i')] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
            # So for U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
            # The sum over r should equal delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
            
            # In our case: U uses (i, j2), V uses (j2, k2), W uses (k2, i)
            # So we need delta_{j2,j2} * delta_{k2,k2} * delta_{i,i} = 1*1*1 = 1?
            # Wait, that's not right. Let me reconsider the index mapping.
            
            # The matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
            # Tensor form: T[i,j, j',k, k',i'] = delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
            # Where (i,j) indexes A, (j',k) indexes B, (k',i') indexes C
            
            # With the given conventions:
            # U index: i*m + j (row i, col j of A)
            # V index: j'*p + k (row j', col k of B)  
            # W index: k'*n + i' (col k', row i' of C)
            
            # So T[(i,j), (j',k), (k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
            # This should equal delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
            
            # In our test: i=i, j=j2, j'=j2, k=k2, k'=k2, i'=i
            # So we need delta_{j2,j2} * delta_{k2,k2} * delta_{i,i} = 1
            # That means this should also be 1!
            
            # I see the issue. Let me just verify the full tensor properly.
            pass
        
        return errors == 0
    
    def verify_full(U, V, W):
        """Full verification of the tensor decomposition"""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # T[(i,j), (j',k), (k',i')] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
        # Should equal delta_{j,j'} * delta_{k,k'} * delta_{i,i'}
        
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                expected = 1.0 if (j == jp and k == kp and i == ip) else 0.0
                                actual = 0.0
                                for r in range(rank):
                                    actual += U_arr[i*m+j, r] * V_arr[jp*p+k, r] * W_arr[kp*n+ip, r]
                                if abs(actual - expected) > 1e-10:
                                    return False
        return True
    
    # Search for better decompositions
    # Use structured approaches based on known matrix multiplication algorithms
    
    # Try rank from theoretical lower bound
    # For m x n * n x p, lower bound is max(m*n, n*p, m*p) = max(9, 21, 21) = 21
    # Upper bound from trivial: 9*21 = 189
    
    # Try some structured decompositions
    # For 3x3 * 3x7, we can try to exploit structure
    
    # Approach: Use the fact that 3x3 multiplication can be done in rank 23 (Strassen-like)
    # and extend to 3x7
    
    # Actually, let's try a direct search with constraints for small ranks
    
    # Start with known good ranks and try to improve
    # The trivial rank is 189. Can we do better?
    
    # Try rank 21 (lower bound) - unlikely but let's check
    # Then try increasing ranks with structured patterns
    
    # For now, return the trivial solution as a baseline
    # Future improvements could use:
    # 1. Strassen-like block decomposition
    # 2. Randomized search with verification
    # 3. Structured low-rank approximations
    
    return best_result
# EVOLVE-BLOCK-END
