"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from math import gcd
from functools import reduce

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def _build_target_tensor():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    T = _build_target_tensor()
    return np.allclose(T_decomp, T, atol=1e-8)

def _to_half_int(x):
    return np.round(x * 2) / 2

def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}

def _gcd_list(xs):
    xs = [x for x in xs if x != 0]
    if not xs:
        return 0
    return reduce(gcd, xs)

def _try_lattice_rank(rank, deadline):
    """
    Search for a rank-R decomposition over half-integers using structured search.
    
    Key insight: for matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1,
    we need U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all (i,j,k).
    
    We use a lattice-based approach: represent entries as 2*x where x is integer,
    then search over integer lattice points.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # We'll do random initialization with half-integer rounding
    # and use alternating least squares with rounding
    
    for trial in range(200):
        if time.time() > deadline - 3:
            return None
        
        random.seed(1000 + trial * 7919)
        np.random.seed(2000 + trial * 104729)
        
        # Initialize with random half-integers
        scale = 1.0
        U = (np.random.randint(-2, 3, size=(n*m, rank)) * 0.5)
        V = (np.random.randint(-2, 3, size=(m*p, rank)) * 0.5)
        W = (np.random.randint(-2, 3, size=(p*n, rank)) * 0.5)
        
        # Alternating minimization with half-integer constraint
        for it in range(50):
            if time.time() > deadline - 3:
                return None
            
            # Update U: for each (i*m+j, r), minimize sum over (k) of (target - ...)
            # T_decomp[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
            # For fixed V, W, the optimal U[i*m+j, r] can be derived.
            
            # Compute current product V*W contribution
            # S[a, c] = sum_r V[b, r] * W[c, r] where b = j*p+k, c = k*n+i
            # Actually: T_decomp[a, b, c] = sum_r U[a,r] V[b,r] W[c,r]
            
            # For each row a = i*m+j of U:
            # Target: T[a, b, c] for all b, c
            # We have: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            
            # This is a linear system in U[a, :] for each (b,c)
            # But it's coupled: U[a,r] appears in all equations for that a.
            
            # Let's do gradient descent with rounding
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            err = T_decomp - T
            
            # Gradient for U
            grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            # Small step
            lr = 0.1 / (it + 1)
            U = U - lr * grad_U
            V = V - lr * grad_V
            W = W - lr * grad_W
            
            # Round to nearest half-integer
            U = _to_half_int(U)
            V = _to_half_int(V)
            W = _to_half_int(W)
            
            # Check convergence
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T_decomp - T)**2)
            if loss < 1e-10:
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        # Final check
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        loss = np.sum((T_decomp - T)**2)
        if loss < 1e-10:
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def _try_structured_rank(rank, deadline):
    """
    Try structured decompositions where U, V, W have specific patterns.
    
    For 2x5x5, we can exploit the small size of n=2.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    # Strategy: use the fact that n=2 is small.
    # W has shape (p*n, rank) = (10, rank)
    # The two blocks of W correspond to i=0 and i=1.
    
    # Try: initialize W with random half-integers, then solve for U and V.
    for trial in range(100):
        if time.time() > deadline - 3:
            return None
        
        np.random.seed(5000 + trial * 31337)
        
        # Random W
        W = (np.random.randint(-3, 4, size=(p*n, rank)) * 0.5)
        
        # For each (i, j, k), we need:
        # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # This is a bilinear system. Let's use alternating least squares.
        U = (np.random.randint(-3, 4, size=(n*m, rank)) * 0.5)
        V = (np.random.randint(-3, 4, size=(m*p, rank)) * 0.5)
        
        for it in range(80):
            if time.time() > deadline - 3:
                return None
            
            # Update U given V, W
            # For each a = i*m+j:
            # We want to find U[a, :] to best satisfy:
            # sum_r U[a,r] * V[j*p+k, r] * W[k*n+i, r] = T[a, j*p+k, k*n+i] for all j,k
            # Note: a = i*m+j, so j is determined by a (j = a % m, i = a // m)
            # But k varies, and the target is T[a, j*p+k, k*n+i]
            
            # Let's reformulate:
            # For fixed i, j (hence a = i*m+j), and varying k:
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            
            # Define: M_k[r] = V[j*p+k, r] * W[k*n+i, r]
            # Then: sum_r U[i*m+j, r] * M_k[r] = 1 for all k
            
            # This is a linear system: M @ U[i*m+j, :]^T = 1 (vector of ones, length p)
            # where M[k, r] = V[j*p+k, r] * W[k*n+i, r]
            
            for i in range(n):
                for j in range(m):
                    a = i * m + j
                    # Build M: (p, rank)
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve M @ x = ones(p) in least squares
                    # x = U[a, :]
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[a, :] = x
                    except:
                        pass
            
            # Round U
            U = _to_half_int(U)
            
            # Similarly update V
            for j in range(m):
                for k in range(p):
                    b = j * p + k
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i
                    # N[i, r] = U[i*m+j, r] * W[k*n+i, r]
                    N = np.zeros((n, rank))
                    for i in range(n):
                        N[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(N, np.ones(n), rcond=None)
                        V[b, :] = x
                    except:
                        pass
            
            # Round V
            V = _to_half_int(V)
            
            # Update W
            for k in range(p):
                for i in range(n):
                    c = k * n + i
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all j
                    # P[j, r] = U[i*m+j, r] * V[j*p+k, r]
                    P = np.zeros((m, rank))
                    for j in range(m):
                        P[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        x, residuals, rank_s, sv = np.linalg.lstsq(P, np.ones(m), rcond=None)
                        W[c, :] = x
                    except:
                        pass
            
            # Round W
            W = _to_half_int(W)
            
            # Check convergence
            T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
            loss = np.sum((T_decomp - T)**2)
            if loss < 1e-10:
                return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
        
        # Final check
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        loss = np.sum((T_decomp - T)**2)
        if loss < 1e-10:
            return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
    
    return None

def _try_perturbation_from_schoolbook(rank, deadline):
    """
    Start from schoolbook (rank 50) and try to reduce to lower rank by
    setting some columns to zero and re-optimizing.
    """
    n, m, p = DIMENSIONS
    T = _build_target_tensor()
    
    if rank >= 50:
        return _schoolbook()
    
    # Start with schoolbook and zero out (50 - rank) random columns
    U_full, V_full, W_full = _schoolbook()['U'], _schoolbook()['V'], _schoolbook()['W']
    U_full = np.array(U_full)
    V_full = np.array(V_full)
    W_full = np.array(W_full)
    
    cols_to_keep = list(range(50))
    random.shuffle(cols_to_keep)
    cols_to_keep = cols_to_keep[:rank]
    
    U = U_full[:, cols_to_keep].copy()
    V = V_full[:, cols_to_keep].copy()
    W = W_full[:, cols_to_keep].copy()
    
    # Optimize
    for it in range(200):
        if time.time() > deadline - 3:
            return None
        
        T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
        err = T_decomp - T
        loss = np.sum(err**2)
        
        if loss < 1e-10:
            return {'U': _to_half_int(U).tolist(), 'V': _to_half_int(V).tolist(), 'W': _to_half_int(W).tolist()}
        
        grad_U = 2 * np.einsum('abc,br,cr->ar', err, V, W)
        grad_V = 2 * np.einsum('abc,ar,cr->br', err, U, W)
        grad_W = 2 * np.einsum('abc,ar,br->cr', err, U, V)
        
        lr = 0.05 / (1 + it * 0.01)
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Occasionally round to half-integers
        if it % 5 == 4:
            U = _to_half_int(U)
            V = _to_half_int(V)
            W = _to_half_int(W)
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    loss = np.sum((T_decomp - T)**2)
    if loss < 1e-10:
        return {'U': _to_half_int(U).tolist(), 'V': _to_half_int(V).tolist(), 'W': _to_half_int(W).tolist()}
    
    return None

def construct():
    start = time.time()
    deadline = start + 115
    best = _schoolbook()
    
    # Try structured search for ranks from 49 down to 25
    # The known minimum rank for 2x5x5 is not well-established, but
    # we know it's at most 50 (schoolbook) and at least max(2,5,5)=5.
    # Realistically, it's likely in the 20-40 range.
    
    # First, try perturbation from schoolbook for ranks 49, 48, 47, ...
    for rank in range(49, 24, -1):
        if time.time() > deadline - 10:
            break
        result = _try_perturbation_from_schoolbook(rank, deadline)
        if result is not None:
            best = result
            break
    
    # If that didn't work, try structured search
    for rank in range(49, 24, -1):
        if time.time() > deadline - 5:
            break
        result = _try_structured_rank(rank, deadline)
        if result is not None:
            best = result
            break
    
    # If still nothing, try lattice search
    for rank in range(49, 24, -1):
        if time.time() > deadline - 3:
            break
        result = _try_lattice_rank(rank, deadline)
        if result is not None:
            best = result
            break
    
    return best
# EVOLVE-BLOCK-END
