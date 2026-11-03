import numpy as np
import time
from itertools import product
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the n×m×p matrix multiplication tensor."""
    rank = U.shape[1]
    # Build target tensor T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    # Compute decomposition tensor
    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
    return np.allclose(T_dec, T, atol=tol)

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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

def _subspace_peeling_2x3x6():
    """
    Use subspace peeling to find minimal-rank decomposition for 2x3x6.
    
    The multiplication tensor for 2x3x6 has:
    - U dimension: n*m = 6
    - V dimension: m*p = 18  
    - W dimension: p*n = 12
    
    We'll construct the tensor explicitly and use algebraic peeling.
    """
    n, m, p = 2, 3, 6
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # The tensor can be viewed as a 6x18x12 array
    # We want to find minimal R such that T = sum_r u_r ⊗ v_r ⊗ w_r
    
    # Strategy: Try ranks from 12 (lower bound) up to 36 (schoolbook)
    # Lower bound: max(n*m, m*p, p*n) / max(n,m,p) but more carefully:
    # For matrix multiplication, rank >= max(n*m, m*p, p*n) is not tight
    # Actually rank >= n*m*p / min(n,m,p) is a known bound, but let's just search
    
    # Key insight: The tensor has a special structure. Each output C[i,k] 
    # depends on sum_j A[i,j]*B[j,k]. We can group by output (i,k) pairs.
    
    # Let's try a structured approach:
    # For each (i,k) pair, we need to compute sum_j A[i,j]*B[j,k]
    # There are n*p = 12 such outputs, each requiring m=3 multiplications
    
    # Try to share computation. The key is that A[i,j] appears in p=6 outputs
    # and B[j,k] appears in n=2 outputs.
    
    # Let's try rank 24 first (3 blocks of 8, if 2x3x2 has rank 8)
    # Then try to do better.
    
    # First, let's try to find a rank-8 decomposition for 2x3x2
    n2, m2, p2 = 2, 3, 2
    T2 = np.zeros((n2 * m2, m2 * p2, p2 * n2))
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                T2[i * m2 + j, j * p2 + k, k * n2 + i] = 1.0
    
    # For 2x3x2, try rank 8
    # Use rational arithmetic for exactness
    def _find_rank8_2x3x2():
        """Find rank-8 decomposition for 2x3x2 using structured search."""
        # The tensor T2 has entries at:
        # (i*3+j, j*2+k, k*2+i) = 1 for i in {0,1}, j in {0,1,2}, k in {0,1}
        
        # Explicit positions:
        # i=0,j=0,k=0: (0, 0, 0)
        # i=0,j=0,k=1: (0, 1, 1)
        # i=0,j=1,k=0: (1, 2, 0)
        # i=0,j=1,k=1: (1, 3, 1)
        # i=0,j=2,k=0: (2, 4, 0)
        # i=0,j=2,k=1: (2, 5, 1)
        # i=1,j=0,k=0: (3, 0, 2)
        # i=1,j=0,k=1: (3, 1, 3)
        # i=1,j=1,k=0: (4, 2, 2)
        # i=1,j=1,k=1: (4, 3, 3)
        # i=1,j=2,k=0: (5, 4, 2)
        # i=1,j=2,k=1: (5, 5, 3)
        
        # Try a symmetric construction
        # U is 6x8, V is 6x8, W is 4x8
        
        # Let's try a pattern based on the structure
        # Each rank-1 term: u ⊗ v ⊗ w
        # u has support on some (i,j) pairs
        # v has support on some (j',k) pairs
        # w has support on some (k',i') pairs
        # The product must match T2
        
        # Try random search with rational values
        for seed in range(500):
            np.random.seed(seed)
            
            R = 8
            # Initialize with small rational values
            U = np.random.randint(-2, 3, size=(6, R)) * 0.5
            V = np.random.randint(-2, 3, size=(6, R)) * 0.5
            W = np.random.randint(-2, 3, size=(4, R)) * 0.5
            
            # Optimize
            lr = 0.02
            for it in range(300):
                T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
                diff = T_dec - T2
                loss = np.sum(diff ** 2)
                
                if loss < 1e-14:
                    break
                
                dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
                dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
                dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Round to nearest 0.25 to maintain half-integer property
                U = np.round(U * 4) / 4
                V = np.round(V * 4) / 4
                W = np.round(W * 4) / 4
            
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_dec, T2, atol=1e-12):
                return U, V, W
        
        return None
    
    result_2x3x2 = _find_rank8_2x3x2()
    
    if result_2x3x2 is not None:
        U2, V2, W2 = result_2x3x2
        R2 = U2.shape[1]
        
        # Build 2x3x6 by tiling 3 times
        R_full = 3 * R2
        U_full = np.zeros((n * m, R_full))
        V_full = np.zeros((m * p, R_full))
        W_full = np.zeros((p * n, R_full))
        
        for block in range(3):
            offset = block * R2
            U_full[:, offset:offset+R2] = U2
            
            for j in range(m):
                for k in range(2):
                    V_full[j * p + (block * 2 + k), offset:offset+R2] = V2[j * 2 + k]
            
            for k in range(2):
                for i in range(n):
                    W_full[(block * 2 + k) * n + i, offset:offset+R2] = W2[k * n + i]
        
        if _verify_decomposition(U_full, V_full, W_full, n, m, p):
            return U_full, V_full, W_full, R_full
    
    # If rank-8 for 2x3x2 fails, try rank-9 or higher
    for R2 in [9, 10, 11]:
        def _find_rank_2x3x2(R):
            for seed in range(200):
                np.random.seed(seed)
                
                U = np.random.randint(-2, 3, size=(6, R)) * 0.5
                V = np.random.randint(-2, 3, size=(6, R)) * 0.5
                W = np.random.randint(-2, 3, size=(4, R)) * 0.5
                
                lr = 0.02
                for it in range(300):
                    T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
                    diff = T_dec - T2
                    loss = np.sum(diff ** 2)
                    
                    if loss < 1e-14:
                        break
                    
                    dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
                    dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
                    dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
                    
                    U -= lr * dU
                    V -= lr * dV
                    W -= lr * dW
                    
                    U = np.round(U * 4) / 4
                    V = np.round(V * 4) / 4
                    W = np.round(W * 4) / 4
                
                T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
                if np.allclose(T_dec, T2, atol=1e-12):
                    return U, V, W
            
            return None
        
        result = _find_rank_2x3x2(R2)
        if result is not None:
            U2, V2, W2 = result
            R_full = 3 * R2
            U_full = np.zeros((n * m, R_full))
            V_full = np.zeros((m * p, R_full))
            W_full = np.zeros((p * n, R_full))
            
            for block in range(3):
                offset = block * R2
                U_full[:, offset:offset+R2] = U2
                for j in range(m):
                    for k in range(2):
                        V_full[j * p + (block * 2 + k), offset:offset+R2] = V2[j * 2 + k]
                for k in range(2):
                    for i in range(n):
                        W_full[(block * 2 + k) * n + i, offset:offset+R2] = W2[k * n + i]
            
            if _verify_decomposition(U_full, V_full, W_full, n, m, p):
                return U_full, V_full, W_full, R_full
    
    return None

def _direct_optimization_2x3x6():
    """
    Direct optimization for 2x3x6 with subspace structure.
    """
    n, m, p = 2, 3, 6
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Try ranks from 24 down to 18
    for R in range(24, 17, -1):
        for seed in range(50):
            np.random.seed(seed)
            
            # Initialize with structure: group by output (i,k)
            U = np.zeros((n * m, R))
            V = np.zeros((m * p, R))
            W = np.zeros((p * n, R))
            
            # Each rank-1 term should contribute to specific outputs
            for r in range(R):
                # Choose which (i,j), (j',k), (k',i') this term affects
                i = r % n
                j = (r // n) % m
                k = (r // (n * m)) % p
                
                U[i * m + j, r] = np.random.choice([-1, 1])
                V[j * p + k, r] = np.random.choice([-1, 1])
                W[k * n + i, r] = np.random.choice([-1, 1])
            
            # Optimize
            lr = 0.01
            for it in range(500):
                T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
                diff = T_dec - T
                loss = np.sum(diff ** 2)
                
                if loss < 1e-14:
                    break
                
                dU = 2 * np.einsum('ijk,jr,kr->ir', diff, V, W)
                dV = 2 * np.einsum('ijk,ir,kr->jr', diff, U, W)
                dW = 2 * np.einsum('ijk,ir,jr->kr', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Round to maintain half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
            
            T_dec = np.einsum('ir,jr,kr->ijk', U, V, W)
            if np.allclose(T_dec, T, atol=1e-12):
                return U, V, W, R
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    best_solution = None
    best_rank = n * m * p  # 36
    
    # Try subspace peeling (rank 24 via rank-8 for 2x3x2)
    if time.time() < deadline - 60:
        result = _subspace_peeling_2x3x6()
        if result is not None:
            U, V, W, R = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_solution = (U, V, W, R)
                best_rank = R
    
    # Try direct optimization
    if best_solution is None or best_rank > 24:
        if time.time() < deadline - 30:
            result = _direct_optimization_2x3x6()
            if result is not None:
                U, V, W, R = result
                if _verify_decomposition(U, V, W, n, m, p):
                    if best_solution is None or R < best_rank:
                        best_solution = (U, V, W, R)
                        best_rank = R
    
    # Fall back to schoolbook
    if best_solution is None:
        U, V, W, rank = _schoolbook(n, m, p)
        best_solution = (U, V, W, rank)
        best_rank = rank
    
    U, V, W, R = best_solution
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
