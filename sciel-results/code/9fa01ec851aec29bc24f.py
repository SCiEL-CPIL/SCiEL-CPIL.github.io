"""
Rank-minimal exact tensor decomposition for 2x3 by 3x8 matrix multiplication.

We search for a rank-r decomposition using:
  U: (n*m, r) = (6, r)
  V: (m*p, r) = (24, r)
  W: (p*n, r) = (16, r)

The tensor T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
where a = i*m+j, b = j*p+k, c = k*n+i.

We use a structured parameterization with random restarts and gradient-free local search
to find low-rank solutions. We verify exactness with rational arithmetic.
"""

import time
import random
import math
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 3, 8]
N, M, P = DIMENSIONS
NM, MP, PN = N * M, M * P, P * N
RANK_SCHOOLBOOK = N * M * P  # 48

def build_target_tensor():
    """Build the matrix multiplication tensor as a dict mapping (a,b,c) -> 1 if valid."""
    target = {}
    for i in range(N):
        for j in range(M):
            for k in range(P):
                a = i * M + j
                b = j * P + k
                c = k * N + i
                target[(a, b, c)] = 1
    return target

def check_decomposition(U, V, W, target, tol=1e-10):
    """
    Check if the decomposition exactly reproduces the target tensor.
    U: (NM, r), V: (MP, r), W: (PN, r)
    For each (a,b,c) in the full grid, compute sum_r U[a,r]*V[b,r]*W[c,r].
    This must equal target.get((a,b,c), 0).
    """
    r = U.shape[1]
    # Use numpy for speed
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # For each (a, b), compute the vector over c: sum_r U[a,r]*V[b,r]*W[:,r]
    # = U[a,:] @ (V[b,:] * W)  element-wise
    # Actually: result_c = sum_r U[a,r] * V[b,r] * W[c,r]
    # = (U[a, :] * V[b, :]) @ W[c, :]
    
    errors = []
    for a in range(NM):
        for b in range(MP):
            # u_a = U[a, :], v_b = V[b, :]
            # product_r = U[a,r] * V[b,r] for each r
            prod = U[a, :] * V[b, :]
            # For each c: sum_r prod[r] * W[c,r]
            # = prod @ W.T  gives (PN,) vector
            result = prod @ W.T  # shape (PN,)
            for c in range(PN):
                expected = target.get((a, b, c), 0)
                if abs(result[c] - expected) > tol:
                    errors.append((a, b, c, result[c], expected))
                    if len(errors) > 5:
                        return False
    return True

def check_decomposition_rational(U_frac, V_frac, W_frac, target):
    """Exact check using Fraction."""
    r = len(U_frac[0])
    for a in range(NM):
        for b in range(MP):
            for c in range(PN):
                val = sum(U_frac[a][rr] * V_frac[b][rr] * W_frac[c][rr] for rr in range(r))
                expected = target.get((a, b, c), 0)
                if val != expected:
                    return False
    return True

def try_convert_to_rational(matrices_float, max_denom=100):
    """
    Try to find rational numbers close to the float values with small denominators.
    Returns list of Fraction matrices or None.
    """
    from fractions import Fraction
    U, V, W = matrices_float
    r = U.shape[1]
    U_f = [[0]*r for _ in range(NM)]
    V_f = [[0]*r for _ in range(MP)]
    W_f = [[0]*r for _ in range(PN)]
    
    for a in range(NM):
        for rr in range(r):
            x = U[a, rr]
            if abs(x) < 1e-12:
                U_f[a][rr] = Fraction(0, 1)
            else:
                best = None
                best_err = float('inf')
                for d in range(1, max_denom + 1):
                    n = round(x * d)
                    err = abs(x - Fraction(n, d))
                    if err < best_err:
                        best_err = err
                        best = Fraction(n, d)
                if best_err > 1e-6:
                    return None
                U_f[a][rr] = best
    for b in range(MP):
        for rr in range(r):
            x = V[b, rr]
            if abs(x) < 1e-12:
                V_f[b][rr] = Fraction(0, 1)
            else:
                best = None
                best_err = float('inf')
                for d in range(1, max_denom + 1):
                    n = round(x * d)
                    err = abs(x - Fraction(n, d))
                    if err < best_err:
                        best_err = err
                        best = Fraction(n, d)
                if best_err > 1e-6:
                    return None
                V_f[b][rr] = best
    for c in range(PN):
        for rr in range(r):
            x = W[c, rr]
            if abs(x) < 1e-12:
                W_f[c][rr] = Fraction(0, 1)
            else:
                best = None
                best_err = float('inf')
                for d in range(1, max_denom + 1):
                    n = round(x * d)
                    err = abs(x - Fraction(n, d))
                    if err < best_err:
                        best_err = err
                        best = Fraction(n, d)
                if best_err > 1e-6:
                    return None
                W_f[c][rr] = best
    return U_f, V_f, W_f

def random_decomposition(r, seed=None):
    """Generate a random rank-r decomposition."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    U = np_rng.standard_normal((NM, r))
    V = np_rng.standard_normal((MP, r))
    W = np_rng.standard_normal((PN, r))
    return U, V, W

def objective(U, V, W, target):
    """
    Compute the squared error of the decomposition.
    Error = sum over all (a,b,c) of (computed - expected)^2
    """
    r = U.shape[1]
    total_err = 0.0
    for a in range(NM):
        for b in range(MP):
            prod = U[a, :] * V[b, :]
            result = prod @ W.T
            for c in range(PN):
                expected = target.get((a, b, c), 0)
                diff = result[c] - expected
                total_err += diff * diff
    return total_err

def local_search(U, V, W, target, max_iters=200, step_size=0.1, seed=42):
    """
    Simple coordinate descent / random perturbation local search.
    """
    rng = random.Random(seed)
    best_err = objective(U, V, W, target)
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        # Randomly pick one matrix and one row to perturb
        which = rng.choice([0, 1, 2])
        if which == 0:
            mat = U
        elif which == 1:
            mat = V
        else:
            mat = W
        
        row = rng.randint(0, mat.shape[0] - 1)
        col = rng.randint(0, mat.shape[1] - 1)
        
        orig = mat[row, col]
        mat[row, col] += rng.gauss(0, step_size)
        
        err = objective(U, V, W, target)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            # Decay step size
            step_size = max(0.01, step_size * 0.95)
        else:
            mat[row, col] = orig
    
    return best_U, best_V, best_W, best_err

def gradient_descent(U, V, W, target, lr=0.01, max_iters=500, seed=42):
    """
    Simple gradient descent on the decomposition parameters.
    """
    np_rng = np.random.default_rng(seed)
    U = U.astype(float)
    V = V.astype(float)
    W = W.astype(float)
    
    for it in range(max_iters):
        # Compute gradients
        # d/dU[a,r] = sum_{b,c} 2*(pred[a,b,c] - target[a,b,c]) * V[b,r] * W[c,r]
        # This is expensive; use mini-batch or just do it all
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        r = U.shape[1]
        # Precompute target as array
        # target_arr[a,b,c] = 1 if (a,b,c) in target else 0
        # But it's sparse, so iterate over non-zero entries
        
        # For efficiency, compute pred for each (a,b) and accumulate
        for a in range(NM):
            for b in range(MP):
                prod = U[a, :] * V[b, :]
                result = prod @ W.T  # (PN,)
                for c in range(PN):
                    expected = target.get((a, b, c), 0)
                    diff = result[c] - expected
                    if diff != 0:
                        # grad_U[a,r] += 2 * diff * V[b,r] * W[c,r]
                        # grad_V[b,r] += 2 * diff * U[a,r] * W[c,r]
                        # grad_W[c,r] += 2 * diff * U[a,r] * V[b,r]
                        for rr in range(r):
                            grad_U[a, rr] += 2 * diff * V[b, rr] * W[c, rr]
                            grad_V[b, rr] += 2 * diff * U[a, rr] * W[c, rr]
                            grad_W[c, rr] += 2 * diff * U[a, rr] * V[b, rr]
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Decay learning rate
        lr *= 0.995
    
    err = objective(U, V, W, target)
    return U, V, W, err

def construct():
    deadline = time.time() + 100  # Leave 20s margin
    target = build_target_tensor()
    
    best_rank = RANK_SCHOOLBOOK
    best_U = best_V = best_W = None
    
    # Try ranks from low to high
    for r in range(10, min(best_rank, 30)):
        if time.time() > deadline:
            break
        
        # Multiple random restarts
        n_restarts = 3 if r <= 15 else 2
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            seed = 42 + restart * 1000 + r
            U, V, W = random_decomposition(r, seed=seed)
            
            # Scale to help convergence
            # Normalize so that typical products are O(1)
            U = U / np.sqrt(r)
            V = V / np.sqrt(r)
            W = W / np.sqrt(r)
            
            # Initial error
            err = objective(U, V, W, target)
            
            # Local search
            U2, V2, W2, err2 = local_search(U, V, W, target, max_iters=100, step_size=0.5, seed=seed)
            
            if err2 < err:
                U, V, W, err = U2, V2, W2, err2
            
            # Gradient descent polish
            U3, V3, W3, err3 = gradient_descent(U, V, W, target, lr=0.05, max_iters=200, seed=seed)
            if err3 < err:
                U, V, W, err = U3, V3, W3, err3
            
            if err < 1e-8:
                # Try to convert to rational
                frac_result = try_convert_to_rational((U, V, W), max_denom=50)
                if frac_result is not None:
                    Uf, Vf, Wf = frac_result
                    if check_decomposition_rational(Uf, Vf, Wf, target):
                        return {"U": Uf, "V": Vf, "W": Wf}
                # If rational conversion fails, check float exactness
                if check_decomposition(U, V, W, target, tol=1e-10):
                    # Round to nearest simple fraction
                    U_rounded = np.round(U, 6)
                    V_rounded = np.round(V, 6)
                    W_rounded = np.round(W, 6)
                    if check_decomposition(U_rounded, V_rounded, W_rounded, target, tol=1e-9):
                        # Convert to Fraction
                        Uf = [[Fraction(int(U_rounded[a, rr] * 1000000), 1000000) for rr in range(r)] for a in range(NM)]
                        Vf = [[Fraction(int(V_rounded[b, rr] * 1000000), 1000000) for rr in range(r)] for b in range(MP)]
                        Wf = [[Fraction(int(W_rounded[c, rr] * 1000000), 1000000) for rr in range(r)] for c in range(PN)]
                        # Reduce fractions
                        Uf = [[x for x in row] for row in Uf]
                        if check_decomposition_rational(Uf, Vf, Wf, target):
                            return {"U": Uf, "V": Vf, "W": Wf}
            
            if err < 1e-4:
                # Very close, try more iterations
                U4, V4, W4, err4 = gradient_descent(U, V, W, target, lr=0.01, max_iters=500, seed=seed+7)
                if err4 < 1e-8:
                    frac_result = try_convert_to_rational((U4, V4, W4), max_denom=100)
                    if frac_result is not None:
                        Uf, Vf, Wf = frac_result
                        if check_decomposition_rational(Uf, Vf, Wf, target):
                            return {"U": Uf, "V": Vf, "W": Wf}
    
    # Fallback: schoolbook
    rank = RANK_SCHOOLBOOK
    U = [[0] * rank for _ in range(NM)]
    V = [[0] * rank for _ in range(MP)]
    W = [[0] * rank for _ in range(PN)]
    column = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][column] = 1
                V[j * P + k][column] = 1
                W[k * N + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
