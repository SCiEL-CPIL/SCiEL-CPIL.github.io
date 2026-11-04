"""Matrix multiplication 3x4x6 - optimized tensor decomposition search."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol

def _loss_and_grad(x, n, m, p, r):
    """Compute loss and gradient for given parameters."""
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
    W = x[n*m*r + m*p*r :].reshape(p, n, r)
    
    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = T - 1.0
    loss = np.sum(err ** 2)
    
    # Gradients using chain rule
    dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
    dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
    dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2
    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad

def _structured_init_v2(n, m, p, r, seed=42):
    """
    Improved structured initialization that tries to exploit bilinear structure.
    Groups (i,j,k) tuples that share indices.
    """
    rng = np.random.RandomState(seed)
    
    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))
    
    if r >= n * m * p:
        # Use trivial decomposition for first n*m*p factors
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U0[i * m + j, col] = 1.0
                        V0[j * p + k, col] = 1.0
                        W0[k * n + i, col] = 1.0
                        col += 1
        # Add small random perturbations to remaining
        for c in range(n * m * p, r):
            U0[:, c] = rng.randn(n * m) * 0.05
            V0[:, c] = rng.randn(m * p) * 0.05
            W0[:, c] = rng.randn(p * n) * 0.05
    else:
        # For rank reduction: start from trivial and merge nearby terms
        # Assign each factor to cover a subset of (i,j,k) with overlap
        # Use a greedy assignment based on index similarity
        
        # First, create a base assignment
        assignments = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    assignments.append((i, j, k))
        
        # Shuffle and assign to r factors
        rng.shuffle(assignments)
        
        # Each factor covers ceil(n*m*p / r) tuples on average
        tuples_per_factor = len(assignments) / r
        start = 0
        for c in range(r):
            end = min(start + int(tuples_per_factor) + 1, len(assignments))
            for idx in range(start, end):
                i, j, k = assignments[idx]
                U0[i * m + j, c] += 1.0
                V0[j * p + k, c] += 1.0
                W0[k * n + i, c] += 1.0
            start = end
        
        # Add random perturbations to allow optimization to find better solutions
        U0 += rng.randn(n * m, r) * 0.3
        V0 += rng.randn(m * p, r) * 0.3
        W0 += rng.randn(p * n, r) * 0.3
    
    return U0, V0, W0

def _random_init_v2(n, m, p, r, seed=42):
    """Random initialization with better scaling."""
    rng = np.random.RandomState(seed)
    # Scale based on expected contribution per factor
    scale = 1.0 / np.sqrt(max(1, r / (n * m * p)))
    U0 = rng.randn(n * m, r) * scale
    V0 = rng.randn(m * p, r) * scale
    W0 = rng.randn(p * n, r) * scale
    return U0, V0, W0

def _block_diagonal_init(n, m, p, r, seed=42):
    """
    Block diagonal initialization: group by i-index, then j-index.
    This respects the structure that U depends on (i,j), V on (j,k), W on (k,i).
    """
    rng = np.random.RandomState(seed)
    
    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))
    
    # Try to create block structure where factors are grouped
    # by which (i,j) or (j,k) or (k,i) they primarily serve
    
    # Assign factors to blocks based on i
    factors_per_i = r // n
    extra = r % n
    
    col = 0
    for i in range(n):
        block_size = factors_per_i + (1 if i < extra else 0)
        for c in range(block_size):
            # This factor primarily serves row i
            # It can contribute to any (i, j, k) tuple
            for j in range(m):
                for k in range(p):
                    # Weight based on how well this factor fits
                    weight = 1.0 / (m * p)
                    U0[i * m + j, col] += weight
                    V0[j * p + k, col] += weight
                    W0[k * n + i, col] += weight
            col += 1
    
    # Add small random perturbations
    U0 += rng.randn(n * m, r) * 0.1
    V0 += rng.randn(m * p, r) * 0.1
    W0 += rng.randn(p * n, r) * 0.1
    
    return U0, V0, W0

def _polish(U, V, W, n, m, p, r, max_time=5.0):
    """High-precision polish pass."""
    start = time.time()
    x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        result = opt.minimize(
            lambda x: _loss_and_grad(x, n, m, p, r),
            x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 10000}
        )
        
        if time.time() - start > max_time:
            return None
        
        U_opt = result.x[:n*m*r].reshape(n*m, r)
        V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
        
        return U_opt, V_opt, W_opt
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 72
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = trivial_rank
    
    def try_rank(r, max_time_per_attempt=6.0):
        """Try to find a rank-r decomposition using multiple strategies."""
        if time.time() > deadline - 10:
            return None
        
        best_result = None
        best_loss = np.inf
        
        # Generate multiple initialization strategies
        inits = []
        
        # Strategy 1: Structured v2
        U0, V0, W0 = _structured_init_v2(n, m, p, r, seed=42)
        inits.append((U0, V0, W0))
        
        # Strategy 2: Block diagonal
        U0, V0, W0 = _block_diagonal_init(n, m, p, r, seed=42)
        inits.append((U0, V0, W0))
        
        # Strategy 3: Random with different seeds
        for seed in [0, 1, 2]:
            U0, V0, W0 = _random_init_v2(n, m, p, r, seed=seed)
            inits.append((U0, V0, W0))
        
        # Strategy 4: Structured with different seed
        U0, V0, W0 = _structured_init_v2(n, m, p, r, seed=99)
        inits.append((U0, V0, W0))
        
        for U_init, V_init, W_init in inits:
            if time.time() > deadline - 10:
                break
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.minimize(
                    lambda x: _loss_and_grad(x, n, m, p, r),
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 5000}
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    U_opt = result.x[:n*m*r].reshape(n*m, r)
                    V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                    best_result = (U_opt, V_opt, W_opt)
                    
                    # Check if we've found a valid solution
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        # Try to polish for higher precision
                        polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=3.0)
                        if polished is not None:
                            U_p, V_p, W_p = polished
                            if _verify(U_p, V_p, W_p, n, m, p, tol=1e-8):
                                return U_p, V_p, W_p
                        return U_opt, V_opt, W_opt
                        
            except Exception:
                continue
        
        # If we got a good but not perfect solution, try polishing
        if best_result is not None and best_loss < 0.5:
            U_opt, V_opt, W_opt = best_result
            polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=5.0)
            if polished is not None:
                U_p, V_p, W_p = polished
                if _verify(U_p, V_p, W_p, n, m, p, tol=1e-6):
                    return U_p, V_p, W_p
        
        return None
    
    # Try ranks from 71 down to 65
    # We expect that if a lower rank exists, it's likely close to 72
    for r in range(trivial_rank - 1, max(trivial_rank - 10, 60), -1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                break
    
    if best_rank < trivial_rank:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
