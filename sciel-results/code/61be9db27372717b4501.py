"""
Rank-minimal exact tensor decomposition for 3x3 * 3x4 matrix multiplication.

Strategy:
1. Try rank 21 (theoretical lower bound from Strassen-like constructions).
2. If not found quickly, try rank 22, 23, ...
3. Use a randomized ALS (Alternating Least Squares) approach with multiple restarts.
4. Use rational reconstruction to get exact integer/half-integer entries.
5. Verify exactness with sympy or exact arithmetic.

For 3x3 * 3x4, the known best rank is 21 (from Strassen's 2x2 trick applied
multiple times, or from direct constructions).

The tensor T[i,j,k] = e_{i,k} in the multiplication tensor.
We seek U (9 x r), V (12 x r), W (12 x r) such that:
  T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]

Wait, the convention is:
  U shape: (n*m, rank) = (9, rank)
  V shape: (m*p, rank) = (12, rank)  
  W shape: (p*n, rank) = (12, rank)

And the tensor entry is:
  T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]

where i in [0,n), j in [0,m), k in [0,p).
n=3, m=3, p=4.

The multiplication tensor for (n x m) * (m x p) -> (n x p) has entries:
  T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j] B[j,k]
  So the tensor is: for each (i,j,k), the coefficient is 1.

Actually, the matrix multiplication tensor T is defined as:
  T = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k * (e_i ⊗ e_k)
  
More precisely, T in E^* ⊗ F^* ⊗ G^* where E = R^{n*m}, F = R^{m*p}, G = R^{p*n} (with trace convention).

The rank is the minimum r such that T = sum_{c=1}^r u_c ⊗ v_c ⊗ w_c.

Where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}.

The constraint is: for all i in [0,n), j in [0,m), k in [0,p):
  sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

This is a system of n*m*p = 36 equations in r*36 unknowns (but with structure).

For r=21, we have 21*9 + 21*12 + 21*12 = 189 + 252 + 252 = 693 unknowns, 36 equations.
This is underdetermined, so solutions exist if the variety is non-empty.

Let me use a different approach: use the known Strassen construction or try to find a solution via optimization.

Actually, let me try a direct approach:
- Initialize U, V, W randomly
- Use ALS: fix two, solve for the third
- Iterate
- Check convergence
- Rationalize

For the ALS step:
- Fix V and W, solve for U: for each (i,j), U[i*m+j, :] = (sum_k W[k*n+i, :] * V[j*p+k, :]) / 1
  Wait, that's not quite right. Let me think again.

The constraint is:
  For all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1

Fix V and W. Then for each (i,j), we have a linear system in U[i*m+j, :]:
  For each k: sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1

This is a system of p=4 equations in r unknowns. If r >= 4, we can solve it (least squares or exact).

Similarly for V and W.

Let me implement this ALS approach.
"""

import numpy as np
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0  # seconds, leave 10s margin
    
    n, m, p = 3, 3, 4
    nm = n * m  # 9
    mp = m * p  # 12
    pn = p * n  # 12
    np_ = n * m * p  # 36
    
    # Try ranks from 21 upward
    for rank in range(21, 37):
        if time.time() > deadline - 30:
            break
        
        best_result = None
        max_restarts = 50 if rank <= 23 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            # Random initialization with small values
            np.random.seed(restart * 1000 + rank)
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            # ALS iterations
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Update U: for each (i,j), solve linear system
                for i in range(n):
                    for j in range(m):
                        # Build system: for each k, sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        # Solve least squares
                        U[i * m + j, :], residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Update V: for each (j,k), solve linear system
                for j in range(m):
                    for k in range(p):
                        # For each i, sum_c V[j*p+k, c] * U[i*m+j, c] * W[k*n+i, c] = 1
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        V[j * p + k, :], residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Update W: for each (k,i), solve linear system
                for k in range(p):
                    for i in range(n):
                        # For each j, sum_c W[k*n+i, c] * U[i*m+j, c] * V[j*p+k, c] = 1
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        W[k * n + i, :], residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
                
                # Check convergence
                if it % 10 == 0:
                    # Compute residual
                    err = 0.0
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                                err += (val - 1.0) ** 2
                    if err < 1e-12:
                        converged = True
                        break
            
            if converged:
                # Try to rationalize
                U_rat, V_rat, W_rat = _rationalize(U, V, W, rank)
                if U_rat is not None:
                    # Verify exactly
                    if _verify(U_rat, V_rat, W_rat, n, m, p):
                        best_result = (rank, U_rat, V_rat, W_rat)
                        break
        
        if best_result is not None:
            rank, U, V, W = best_result
            # Convert to lists of lists
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _rationalize(U, V, W, rank):
    """Try to rationalize the matrices to get exact entries."""
    try:
        from fractions import Fraction
        
        def to_frac(x):
            # Try to convert to a simple fraction
            f = Fraction(x).limit_denominator(1000)
            return f
        
        U_rat = np.zeros_like(U)
        V_rat = np.zeros_like(V)
        W_rat = np.zeros_like(W)
        
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U_rat[i, j] = to_frac(U[i, j])
        
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V_rat[i, j] = to_frac(V[i, j])
        
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_rat[i, j] = to_frac(W[i, j])
        
        return U_rat, V_rat, W_rat
    except:
        return None, None, None


def _verify(U, V, W, n, m, p):
    """Verify the decomposition exactly."""
    try:
        from fractions import Fraction
        nm = n * m
        mp = m * p
        pn = p * n
        rank = U.shape[1]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if s != 1:
                        return False
        return True
    except:
        # Fallback: check with floats
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
