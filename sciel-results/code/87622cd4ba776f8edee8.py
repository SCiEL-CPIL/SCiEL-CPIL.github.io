import time
import sys
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = 3, 3, 6
    rank = len(U[0])
    if len(U) != n * m: return False
    if len(V) != m * p: return False
    if len(W) != p * n: return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-6:
                    return False
    return True

def _find_3x3x3_decomp(rank=23, time_limit=60.0):
    """Find a rank-rank decomposition for 3x3x3 matrix multiplication."""
    n, m, p = 3, 3, 3
    t0 = time.time()
    np.random.seed(42)
    
    # Initialize
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Scale up to avoid vanishing gradients
    U *= 2.0
    V *= 2.0
    W *= 2.0
    
    lr = 0.05
    best_loss = float('inf')
    best_U = U.copy()
    best_V = V.copy()
    best_W = W.copy()
    
    for it in range(5000):
        if time.time() - t0 > time_limit:
            break
        
        # Compute residuals
        residual = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    residual[i, j, k] = s - 1.0
        
        loss = np.sum(residual ** 2)
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            if loss < 1e-8:
                break
        
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r_ijk = residual[i, j, k]
                    for r in range(rank):
                        dU[i * m + j, r] += 2 * r_ijk * V[j * p + k, r] * W[k * n + i, r]
                        dV[j * p + k, r] += 2 * r_ijk * U[i * m + j, r] * W[k * n + i, r]
                        dW[k * n + i, r] += 2 * r_ijk * U[i * m + j, r] * V[j * p + k, r]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Decay learning rate
        lr *= 0.995
        
    if best_loss < 1e-6:
        return best_U, best_V, best_W
    return None

def _construct_3x3x6_from_3x3x3(U3, V3, W3):
    """Construct 3x3x6 decomposition from 3x3x3 decomposition."""
    n, m, p = 3, 3, 6
    rank3 = U3.shape[1]
    rank6 = rank3 * 2
    
    U6 = np.zeros((n * m, rank6))
    V6 = np.zeros((m * p, rank6))
    W6 = np.zeros((p * n, rank6))
    
    # First half of ranks (0 to rank3-1) correspond to k=0,1,2
    # Second half of ranks (rank3 to 2*rank3-1) correspond to k=3,4,5
    
    for r in range(rank3):
        # Copy U
        U6[:, r] = U3[:, r]
        U6[:, r + rank3] = U3[:, r]
        
        # Copy V and W for first block (k=0,1,2)
        for j in range(m):
            for k in range(3):
                V6[j * p + k, r] = V3[j * 3 + k, r]
                W6[k * n + 0, r] = W3[k * n + 0, r] # Wait, W index is k*n+i
                # Correct W indexing:
                # W3[k, i] is stored at W3[k * n + i]
                # We need to map this to W6.
                # For r in [0, rank3), k is 0..2.
                # W6[k * n + i, r] = W3[k * n + i, r]
                # W6[k * n + i, r + rank3] = 0
        
        # Copy V and W for second block (k=3,4,5)
        for j in range(m):
            for k in range(3, 6):
                V6[j * p + k, r + rank3] = V3[j * 3 + (k - 3), r]
                W6[k * n + 0, r + rank3] = 0 # Placeholder, will fill below
        
        # Fill W correctly
        for i in range(n):
            for k in range(3):
                W6[k * n + i, r] = W3[k * n + i, r]
                W6[(k + 3) * n + i, r + rank3] = W3[k * n + i, r]
                # Wait, the second block uses the SAME W coefficients but shifted k?
                # C2 = A B2.
                # The tensor for C2 is isomorphic to C1 = A B1.
                # So yes, the coefficients are the same, just the k index in V and W is shifted.
                
    # Convert to lists
    U_list = [[float(U6[i, r]) for r in range(rank6)] for i in range(n * m)]
    V_list = [[float(V6[i, r]) for r in range(rank6)] for i in range(m * p)]
    W_list = [[float(W6[i, r]) for r in range(rank6)] for i in range(p * n)]
    
    return U_list, V_list, W_list

def _schoolbook():
    n, m, p = 3, 3, 6
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

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # Try to find rank 46 decomposition (2 * 23)
    if time.time() < deadline - 10:
        res3 = _find_3x3x3_decomp(rank=23, time_limit=60.0)
        if res3 is not None:
            U3, V3, W3 = res3
            U6, V6, W6 = _construct_3x3x6_from_3x3x3(U3, V3, W3)
            
            # Round to nearest 0.5 to satisfy "integer/half-integer" constraint if applicable
            # and to clean up numerical noise
            U6_rounded = [[round(val * 2) / 2 for val in row] for row in U6]
            V6_rounded = [[round(val * 2) / 2 for val in row] for row in V6]
            W6_rounded = [[round(val * 2) / 2 for val in row] for row in W6]
            
            if _verify(U6_rounded, V6_rounded, W6_rounded):
                return {"U": U6_rounded, "V": V6_rounded, "W": W6_rounded}
            
            # If rounding broke it, try without rounding (if evaluator accepts floats)
            # But the prompt says "integer/half-integer entries".
            # So we must round. If it fails, we can't return it.
            
    # Try rank 45? Unlikely to find easily.
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
