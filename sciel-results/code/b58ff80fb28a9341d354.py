import time
import numpy as np
from fractions import Fraction
import random

DIMENSIONS = [3, 4, 4]
_DEADLINE = 145  # safety margin before 150s kill

def _verify_exact(U, V, W):
    """Verify exact decomposition using Fraction arithmetic."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Uf = [[Fraction(str(u)) for u in row] for row in U]
    Vf = [[Fraction(str(v)) for v in row] for row in V]
    Wf = [[Fraction(str(w)) for w in row] for row in W]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Uf[i*m+j][c] * Vf[j*p+k][c] * Wf[k*n+i][c]
                if s != 1:
                    return False
    return True

def _snap_to_half(x, tol=1e-6):
    k = round(2 * x)
    if abs(2 * x - k) < tol:
        return k / 2.0
    return None

def _try_snap(U, V, W):
    """Try to snap all entries to half-integers and verify."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    
    Us = [[0.0]*rank for _ in range(n*m)]
    Vs = [[0.0]*rank for _ in range(m*p)]
    Ws = [[0.0]*rank for _ in range(p*n)]
    
    for i in range(n*m):
        for c in range(rank):
            val = _snap_to_half(U[i][c])
            if val is None:
                return None
            Us[i][c] = val
    for i in range(m*p):
        for c in range(rank):
            val = _snap_to_half(V[i][c])
            if val is None:
                return None
            Vs[i][c] = val
    for i in range(p*n):
        for c in range(rank):
            val = _snap_to_half(W[i][c])
            if val is None:
                return None
            Ws[i][c] = val
    
    if _verify_exact(Us, Vs, Ws):
        return Us, Vs, Ws
    return None

def _compute_error(U, V, W, n, m, p):
    """Compute squared error of decomposition."""
    rank = U.shape[1]
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                err += (val - 1.0)**2
    return err

def _als_step(U, V, W, n, m, p):
    """Perform one full cycle of ALS updates."""
    rank = U.shape[1]
    
    # Update U
    for i in range(n):
        for j in range(m):
            M = np.zeros((p, rank))
            for k in range(p):
                for c in range(rank):
                    M[k, c] = V[j*p+k, c] * W[k*n+i, c]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                U[i*m+j, :] = x
            except:
                pass
        
    # Update V
    for j in range(m):
        for k in range(p):
            M = np.zeros((n, rank))
            for i in range(n):
                for c in range(rank):
                    M[i, c] = U[i*m+j, c] * W[k*n+i, c]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                V[j*p+k, :] = x
            except:
                pass
        
    # Update W
    for k in range(p):
        for i in range(n):
            M = np.zeros((m, rank))
            for j in range(m):
                for c in range(rank):
                    M[j, c] = U[i*m+j, c] * V[j*p+k, c]
            try:
                x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                W[k*n+i, :] = x
            except:
                pass

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    start_time = time.time()
    deadline = _DEADLINE
    n, m, p = DIMENSIONS
    
    # Lower bound: max(n*m, m*p, p*n) = 16
    # Try ranks from 16 up to schoolbook (48)
    for rank in range(16, 49):
        if time.time() > deadline - 10:
            break
        
        # Strategy 1: Random Half-Integer Initialization + ALS Refinement
        # We try to find a solution by starting with sparse half-integers
        # and refining with ALS, then snapping.
        
        num_trials = 20
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Randomly initialize with sparse half-integers
            # Density depends on rank; lower rank needs sparser init?
            # Let's try a mix of sparse and dense
            density = 0.3 if rank < 20 else 0.5
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            for _ in range(int(n*m * rank * density)):
                r = np.random.randint(0, n*m)
                c = np.random.randint(0, rank)
                U[r, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            for _ in range(int(m*p * rank * density)):
                r = np.random.randint(0, m*p)
                c = np.random.randint(0, rank)
                V[r, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            for _ in range(int(p*n * rank * density)):
                r = np.random.randint(0, p*n)
                c = np.random.randint(0, rank)
                W[r, c] = np.random.choice([-1.0, -0.5, 0.5, 1.0])
            
            # Run ALS for a few iterations to reduce error
            max_als_iters = 50
            for it in range(max_als_iters):
                _als_step(U, V, W, n, m, p)
                err = _compute_error(U, V, W, n, m, p)
                if err < 1e-10:
                    break
            
            # Try to snap and verify
            result = _try_snap(U.tolist(), V.tolist(), W.tolist())
            if result is not None:
                Us, Vs, Ws = result
                return {"U": Us, "V": Vs, "W": Ws}
            
            # If error is small but not snap-able, try a local search around the solution
            err = _compute_error(U, V, W, n, m, p)
            if err < 1e-6:
                # Try perturbing the solution and snapping again
                for perturb_trial in range(5):
                    U_p = U + 0.1 * np.random.randn(n*m, rank)
                    V_p = V + 0.1 * np.random.randn(m*p, rank)
                    W_p = W + 0.1 * np.random.randn(p*n, rank)
                    
                    # Short ALS refinement
                    for it in range(10):
                        _als_step(U_p, V_p, W_p, n, m, p)
                    
                    result = _try_snap(U_p.tolist(), V_p.tolist(), W_p.tolist())
                    if result is not None:
                        Us, Vs, Ws = result
                        return {"U": Us, "V": Vs, "W": Ws}
        
        # Strategy 2: Pure Random Search on Half-Integers (for very low ranks)
        # This is a shot in the dark for rank 16-18
        if rank <= 18:
            for trial in range(10):
                if time.time() > deadline - 5:
                    break
                
                U = np.zeros((n*m, rank))
                V = np.zeros((m*p, rank))
                W = np.zeros((p*n, rank))
                
                for _ in range(n*m * rank):
                    U[np.random.randint(0, n*m), np.random.randint(0, rank)] = np.random.choice([-1.0, 0.0, 1.0])
                for _ in range(m*p * rank):
                    V[np.random.randint(0, m*p), np.random.randint(0, rank)] = np.random.choice([-1.0, 0.0, 1.0])
                for _ in range(p*n * rank):
                    W[np.random.randint(0, p*n), np.random.randint(0, rank)] = np.random.choice([-1.0, 0.0, 1.0])
                
                # Verify directly without ALS (fast for small rank)
                if _verify_exact(U.tolist(), V.tolist(), W.tolist()):
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook
    U, V, W = _build_schoolbook()
    return {"U": U, "V": V, "W": W}
