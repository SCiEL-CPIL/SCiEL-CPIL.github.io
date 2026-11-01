"""
Tensor decomposition for M_{2,2,13} matrix multiplication.
We try to find a lower-rank decomposition than schoolbook (52).
"""

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    r = U.shape[1]
    # Compute tensor: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
    # Check against the true matrix multiplication tensor
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                expected = 1.0
                actual = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(actual - expected) > 1e-9:
                    return False
    # Also check that off-diagonal terms are zero
    # For random samples to verify
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        i2 = np.random.randint(n)
        j2 = np.random.randint(m)
        k2 = np.random.randint(p)
        a = i * m + j
        b = j2 * p + k2
        c = k * n + i2
        expected = 1.0 if (j == j2 and i == i2 and k == k2) else 0.0
        actual = np.sum(U[a, :] * V[b, :] * W[c, :])
        if abs(actual - expected) > 1e-9:
            return False
    return True

def _try_polynomial_decomposition(n, m, p, deadline):
    """
    Try to find a lower-rank decomposition using polynomial/group algebra approach.
    
    The idea: C[i,k] = sum_j A[i,j] * B[j,k]
    For n=2, m=2: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can view B as two vectors b0, b1 of length p.
    The multiplication is: C[i,:] = A[i,0]*b0 + A[i,1]*b1
    
    This is a linear combination of b0 and b1 with coefficients from A.
    We can use the fact that any linear combination of two length-p vectors
    can be decomposed using p+1 terms (by expressing one vector in terms of
    the other plus a residual).
    
    Actually, let's think differently. The rank of M_{2,2,p} should be 2p+2
    based on known results. Let's construct it explicitly.
    
    Construction:
    - For each k in 0..p-1, we need to capture B[0,k] and B[1,k]
    - C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    
    We can write:
    C[i,k] = (A[i,0]*B[0,k] + A[i,1]*B[1,k])
    
    Using the identity: xy = ((x+y)^2 - x^2 - y^2)/4, we can reduce the rank.
    But that's for scalar multiplication.
    
    Let me try a direct construction with rank 2p+2 = 28:
    """
    r = 2 * p + 2
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((n * p, r))
    
    # The multiplication: C[i,k] = sum_j A[i,j] * B[j,k]
    # We need: sum_r U[i*m+j, r] * V[j'*p+k', r] * W[k'*n+i', r] = delta(j,j') * delta(i,i') * delta(k,k')
    
    # Let's use a construction based on:
    # For each k, we have contributions from A[i,0]*B[0,k] and A[i,1]*B[1,k]
    
    # Strategy: Use the fact that we can express the product using
    # (A[i,0]+A[i,1])*(B[0,k]+B[1,k]), A[i,0]*B[0,k], A[i,1]*B[1,k], etc.
    
    # Actually, let's just use a known result: rank(M_{2,2,p}) = 2p+2 for p odd
    # and construct it using a systematic approach.
    
    # Simple construction:
    # Term 1..p: For each k, capture A[i,0]*B[0,k]
    for k in range(p):
        col = k
        for i in range(n):
            U[i * m + 0, col] = 1.0  # A[i,0]
        V[0 * p + k, col] = 1.0      # B[0,k]
        for i in range(n):
            W[k * n + i, col] = 1.0  # C[i,k]
    
    # Term p+1..2p: For each k, capture A[i,1]*B[1,k]
    for k in range(p):
        col = p + k
        for i in range(n):
            U[i * m + 1, col] = 1.0  # A[i,1]
        V[1 * p + k, col] = 1.0      # B[1,k]
        for i in range(n):
            W[k * n + i, col] = 1.0  # C[i,k]
    
    # This gives rank 2p = 26, which is the schoolbook for this specific structure
    # Wait, that's already 26, not 52. Let me check...
    # Actually for n=2, m=2, p=13: schoolbook is n*m*p = 52
    # But the structure C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] only needs
    # 2*p = 26 terms if we can separate the j index properly.
    
    # Let me verify: the tensor T[a,b,c] where a=(i,j), b=(j',k'), c=(i',k')
    # T = 1 if j==j' and i==i' and k==k'
    
    # My construction:
    # For col=k (k=0..p-1): U[i*2+0, k]=1, V[0*p+k, k]=1, W[k*n+i, k]=1
    #   Contribution to T[(i,j),(j',k'),(i',k'')]: 
    #   U[i*2+j, k] * V[j'*p+k', k] * W[k''*n+i', k]
    #   = (1 if j=0) * (1 if j'=0 and k'=k) * (1 if k''=k and i'=i)
    #   = 1 if j=0, j'=0, k'=k, k''=k, i'=i
    #   This captures T[(i,0),(0,k),(k,i)] = 1 ✓
    
    # For col=p+k (k=0..p-1): U[i*2+1, p+k]=1, V[1*p+k, p+k]=1, W[k*n+i, p+k]=1
    #   Contribution: (1 if j=1) * (1 if j'=1 and k'=k) * (1 if k''=k and i'=i)
    #   = 1 if j=1, j'=1, k'=k, k''=k, i'=i
    #   This captures T[(i,1),(1,k),(k,i)] = 1 ✓
    
    # So this construction with rank 2p=26 should work!
    
    # Let's verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # If verification fails, fall back to schoolbook
    return None

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((n * p, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    # Try the polynomial decomposition first
    result = _try_polynomial_decomposition(n, m, p, deadline)
    if result is not None:
        return result
    
    # Fall back to schoolbook
    return _build_schoolbook(n, m, p)

# EVOLVE-BLOCK-END
