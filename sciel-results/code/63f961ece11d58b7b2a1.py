"""
Matrix multiplication 2x3 x 3x3 rank-minimal exact tensor decomposition.

We search for a decomposition of rank r < 18 (schoolbook) using:
  1. A structured ansatz based on known low-rank decompositions for similar sizes.
  2. Numerical refinement via alternating least squares / gradient-based optimization.
  3. Exact rational/half-integer recovery via rounding and verification.

The multiplication tensor T has entries T[i, j, k] = 1 if and only if the
(i,j) entry of A times the (j,k) entry of B contributes to C[i,k].
Under the trace convention W[k*n + i, r], the decomposition is:
  C[i,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
Wait, that's not right. Let me re-read the convention.

Standard matrix multiplication: C[i,k] = sum_j A[i,j] * B[j,k]

The tensor T_{i,j,k} = 1 for all i,j,k (it's the full multiplication tensor).
A rank-r decomposition writes:
  T_{i,j,k} = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]  (or some indexing)

The task says:
  U has shape (n*m, rank) -> U[i*m + j, r]
  V has shape (m*p, rank) -> V[j*p + k, r]
  W has shape (p*n, rank) -> W[k*n + i, r]  (trace convention)

So: C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

This must equal sum_j A[i,j]*B[j,k] for all A, B.
So for each (i,j,k), we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
for all i in [0,n), j in [0,m), k in [0,p).

With n=2, m=3, p=3: 18 equations, each a sum of rank terms.

Known results: The border rank / tensor rank for 2x3 * 3x3 matrix multiplication.
The trivial bound is 18. It's known that for small matrices, the tensor rank can be lower.
For 2x2 * 2x2, rank is 7 (Strassen).
For 2x3 * 3x3, let's think about what's known.

Actually, let me just do a numerical search for rank 12, 13, 14, 15, 16, 17 and see if we can find an exact solution.

Approach:
1. Start with random initialization for rank r.
2. Use alternating optimization: fix U,V and solve for W, etc.
3. After convergence, round to nearest half-integers and verify exactly.
4. Try multiple random restarts.

Given the time budget of 120s, I'll try ranks from 12 upward.
"""

import numpy as np
from fractions import Fraction
import time
import itertools

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly satisfies the multiplication tensor."""
    rank = U.shape[1]
    # Check all (i, j, k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > 1e-10:
                    return False
    return True

def _round_to_half_integer(x, tol=1e-6):
    """Round x to nearest half-integer if close enough."""
    # Half-integers: k/2 for integer k
    # Nearest half-integer: round(2*x)/2
    rounded = round(2 * x) / 2.0
    if abs(x - rounded) < tol:
        return rounded
    return None

def _try_round(U, V, W, tol=1e-6):
    """Try to round all entries to half-integers and verify."""
    U_r = np.zeros_like(U)
    V_r = np.zeros_like(V)
    W_r = np.zeros_like(W)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            val = _round_to_half_integer(U[i, j], tol)
            if val is None:
                return None
            U_r[i, j] = val
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            val = _round_to_half_integer(V[i, j], tol)
            if val is None:
                return None
            V_r[i, j] = val
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            val = _round_to_half_integer(W[i, j], tol)
            if val is None:
                return None
            W_r[i, j] = val
    # Verify exactness with fractions
    n, m, p = DIMENSIONS
    rank = U_r.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U_r[i * m + j, r], 1) * Fraction(V_r[j * p + k, r], 1) * Fraction(W_r[k * n + i, r], 1)
                if s != 1:
                    return None
    return (U_r, V_r, W_r)

def _alternating_optimize(U, V, W, n, m, p, iters=200, lr=0.01):
    """
    Alternating least squares style optimization.
    For fixed V, W: solve for U.
    For fixed U, W: solve for V.
    For fixed U, V: solve for W.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for it in range(iters):
        # Solve for U: for each (i,j), U[i*m+j, :] should satisfy
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
        # This is a linear system in U[i*m+j, :] for each (i,j)
        # Actually it's coupled across k for fixed (i,j).
        # For fixed (i,j): sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1 for k=0..p-1
        # This is a system of p equations in rank unknowns (U[i*m+j, :])
        # If rank >= p, we can solve.
        
        # Let's do gradient descent instead for simplicity and stability.
        # Compute gradient for U:
        # d/dU[i*m+j, r] = sum_k V[j*p+k, r] * W[k*n+i, r] * (1 - sum_r' U[i*m+j, r'] * V[j*p+k, r'] * W[k*n+i, r'])
        
        # This is getting complex. Let me use a simpler approach:
        # Just do coordinate descent or simple GD.
        
        # Actually, let me reformulate. The constraint is:
        # For all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Let me define for each r: u_r[i,j], v_r[j,k], w_r[k,i]
        # Then sum_r u_r[i,j] * v_r[j,k] * w_r[k,i] = 1
        
        # This is a tensor completion / decomposition problem.
        # Let me just use scipy.optimize or manual gradient descent.
        
        # Compute residuals
        resid = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    resid[i, j, k] = s - 1.0
        
        # Gradient for U[i*m+j, r]:
        # sum_k V[j*p+k, r] * W[k*n+i, r] * resid[i,j,k]
        # Gradient for V[j*p+k, r]:
        # sum_i U[i*m+j, r] * W[k*n+i, r] * resid[i,j,k]
        # Gradient for W[k*n+i, r]:
        # sum_j U[i*m+j, r] * V[j*p+k, r] * resid[i,j,k]
        
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        gU[i * m + j, r] += V[j * p + k, r] * W[k * n + i, r] * resid[i, j, k]
                        gV[j * p + k, r] += U[i * m + j, r] * W[k * n + i, r] * resid[i, j, k]
                        gW[k * n + i, r] += U[i * m + j, r] * V[j * p + k, r] * resid[i, j, k]
        
        # Update
        U -= lr * gU
        V -= lr * gV
        W -= lr * gW
        
        # Decay learning rate
        lr *= 0.995
    
    return U, V, W

def _random_init(n, m, p, rank, seed=0):
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # 100s safety margin
    
    # Try ranks from 12 upward
    for rank in range(12, 19):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random restarts
        max_restarts = 50 if rank <= 14 else 20
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
            
            seed = rank * 1000 + restart
            U, V, W = _random_init(n, m, p, rank, seed)
            
            # Run alternating optimization
            U, V, W = _alternating_optimize(U, V, W, n, m, p, iters=300, lr=0.05)
            
            # Try to round
            result = _try_round(U, V, W, tol=1e-5)
            if result is not None:
                U_r, V_r, W_r = result
                # Convert to lists
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                return {"U": U_list, "V": V_list, "W": W_list}
            
            # Try with slightly different tol
            result = _try_round(U, V, W, tol=1e-4)
            if result is not None:
                U_r, V_r, W_r = result
                U_list = [[float(x) for x in row] for row in U_r]
                V_list = [[float(x) for x in row] for row in V_r]
                W_list = [[float(x) for x in row] for row in W_r]
                return {"U": U_list, "V": V_list, "W": W_list}
        
        # If we get here, no exact solution found for this rank
        continue
    
    # Fallback: schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
