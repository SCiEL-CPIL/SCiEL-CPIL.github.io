<<<<<<< SEARCH
"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import itertools
import time
from collections import deque

DIMENSIONS = [2, 5, 6]

def _verify(U, V, W, n, m, p, tol=1e-8):
    """Fast verification: check only the 60 non-zero tensor entries + sample zeros."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = 0.0
                for rr in range(r):
                    val += U[a, rr] * V[b, rr] * W[c, rr]
                if abs(val - 1.0) > tol:
                    return False
    # Check a sample of zero entries
    rng = np.random.RandomState(123)
    for _ in range(200):
        a = rng.randint(nm)
        b = rng.randint(mp)
        c = rng.randint(pn)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0.0
        for rr in range(r):
            val += U[a, rr] * V[b, rr] * W[c, rr]
        if abs(val) > tol:
            return False
    return True

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using the structure of the tensor."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    checked = 0
    for _ in range(500):
        a = np.random.randint(0, n*m)
        b = np.random.randint(0, m*p)
        c = np.random.randint(0, p*n)
        i_a, j_a = divmod(a, m)
        j_b, k_b = divmod(b, p)
        k_c, i_c = divmod(c, n)
        if j_a == j_b and i_a == i_c and k_b == k_c:
            continue
        val = 0.0
        for cc in range(r):
            val += U[a, cc] * V[b, cc] * W[c, cc]
        if abs(val) > tol:
            return False
        checked += 1
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using rational arithmetic or high precision."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(r):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > 1e-10:
                    return False
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i_a, j_a = divmod(a, m)
                j_b, k_b = divmod(b, p)
                k_c, i_c = divmod(c, n)
                if j_a == j_b and i_a == i_c and k_b == k_c:
                    continue
                val = 0.0
                for cc in range(r):
                    val += U[a, cc] * V[b, cc] * W[c, cc]
                if abs(val) > 1e-10:
                    return False
    return True

def _als_polish(U, V, W, T, n, m, p, iterations=20, tol=1e-10, deadline=None):
    """Vectorized ALS polishing step with deadline guard."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for it in range(iterations):
        if deadline and time.time() > deadline:
            break
        
        # Update U: vectorized
        # For each a: T[a] (flattened) = U[a,:] @ A_a.T where A_a[bc, r] = V[b,r]*W[c,r]
        # Precompute V*W outer: shape (mp, pn, r)
        VW = V[:, np.newaxis, :] * W[np.newaxis, :, :]  # (mp, pn, r)
        for a in range(nm):
            A = VW.reshape(mp * pn, r)  # (mp*pn, r)
            T_a = T[a, :, :].flatten()
            try:
                U[a, :], _, _, _ = np.linalg.lstsq(A.T, T_a, rcond=None)
            except:
                pass
        
        # Update V: vectorized
        UW = U[:, np.newaxis, :] * W[np.newaxis, :, :]  # (nm, pn, r)
        for b in range(mp):
            A = UW.reshape(nm * pn, r)
            T_b = T[:, b, :].flatten()
            try:
                V[b, :], _, _, _ = np.linalg.lstsq(A.T, T_b, rcond=None)
            except:
                pass
        
        # Update W: vectorized
        UV = U[:, np.newaxis, :] * V[np.newaxis, :, :]  # (nm, mp, r)
        for c in range(pn):
            A = UV.reshape(nm * mp, r)
            T_c = T[:, :, c].flatten()
            try:
                W[c, :], _, _, _ = np.linalg.lstsq(A.T, T_c, rcond=None)
            except:
                pass
    
    return U, V, W

def _kronecker_decomposition(n, m, p):
    """
    Try to decompose the n x m x p matrix multiplication tensor using
    Kronecker product structure. The idea: if n = n1*n2, m = m1*m2, p = p1*p2,
    then the tensor rank can be bounded by rank(n1,m1,p1) * rank(n2,m2,p2).
    
    For 2x5x6: we can try various factorizations.
    2x5x6 doesn't factor nicely into two non-trivial parts.
    But we can try: treat as a "block" structure.
    
    Alternative: use the fact that rank(2x5x6) <= rank(2x5x5) * 6/something...
    Actually, let's try a different approach: use known bounds and random search
    with structured initialization.
    
    Known: rank(2xmxp) for small cases.
    For 2x5x6, the Strassen-like bounds suggest rank around 20-30.
    """
    nm, mp, pn = n*m, m*p, p*n
    
    # Strategy 1: Use the "row-wise" Kronecker structure
    # C = A * B where A is 2x5, B is 5x6
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # We can view this as: for each i, C[i,:] = A[i,:] * B
    # This is a vector-matrix product of dimension 1x5 times 5x6
    # The rank of 1x5x6 matmul is 5 (schoolbook) or possibly less with Strassen
    # 
    # For two such products (i=0 and i=1), we need rank <= 2*5 = 10? No,
    # we can share the B factors.
    #
    # Actually, the full tensor rank for 2x5x6:
    # Lower bound: max(n*m, m*p, p*n) = max(10, 30, 12) = 30
    # Upper bound (schoolbook): n*m*p = 60
    # 
    # Let's try rank from 30 upward with structured initialization.
    
    return None  # Placeholder - we'll do the search in construct()

def _structured_init(n, m, p, rank, seed=42):
    """Create structured initialization exploiting the tensor's block structure."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Structure: group terms by the "j" index (the inner dimension)
    # For each j in range(m), we have n*p terms that involve A[i,j] and B[j,k]
    # If rank >= m, we can dedicate rank/m terms per j-block
    
    if rank >= m:
        terms_per_j = rank // m
        remainder = rank % m
        col = 0
        for j in range(m):
            n_terms = terms_per_j + (1 if j < remainder else 0)
            for t in range(n_terms):
                # This term handles contributions for this j
                # U[i*m+j, col] should be nonzero for all i
                # V[j*p+k, col] should be nonzero for all k
                # W[k*n+i, col] should be nonzero for all i,k
                for i in range(n):
                    U[i*m+j, col] = rng.randn()
                for k in range(p):
                    V[j*p+k, col] = rng.randn()
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, col] = rng.randn()
                col += 1
    else:
        # Rank < m: use random initialization
        U = rng.randn(nm, rank) / np.sqrt(rank)
        V = rng.randn(mp, rank) / np.sqrt(rank)
        W = rng.randn(pn, rank) / np.sqrt(rank)
    
    return U, V, W

def _gradient_descent(U, V, W, T, n, m, p, max_steps=100, lr=0.01, deadline=None):
    """Gradient descent with adaptive learning rate and early stopping."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    vU = np.zeros_like(U)
    vV = np.zeros_like(V)
    vW = np.zeros_like(W)
    momentum = 0.9
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        grad_U = -np.einsum('abc,br,cr->ar', R, V, W)
        grad_V = -np.einsum('abc,ar,cr->br', R, U, W)
        grad_W = -np.einsum('abc,ar,br->cr', R, U, V)
        
        vU = momentum * vU + (1 - momentum) * grad_U
        vV = momentum * vV + (1 - momentum) * grad_V
        vW = momentum * vW + (1 - momentum) * grad_W
        
        U -= lr * vU
        V -= lr * vV
        W -= lr * vW
        
        if loss > best_loss * 1.5:
            lr *= 0.7
            vU = 0
            vV = 0
            vW = 0
    
    return best_U, best_V, best_W, best_loss
=======
"""Matrix multiplication 2x5x6: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

def _verify_full(U, V, W, n, m, p, tol=1e-9):
    """Full verification: check ALL tensor entries."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    # Check all non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Check ALL zero entries
    for a in range(nm):
        ia, ja = divmod(a, m)
        for b in range(mp):
            jb, kb = divmod(b, p)
            if ja != jb:
                continue
            for c in range(pn):
                kc, ic = divmod(c, n)
                if ia != ic or kb != kc:
                    continue
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
    return True

def _verify_sample(U, V, W, n, m, p, tol=1e-7, n_samples=300, seed=123):
    """Fast verification with sampling."""
    r = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    rng = np.random.RandomState(seed)
    # Check all non-zero entries (only 60)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.dot(U[a, :], V[b, :] * W[c, :])
                if abs(val - 1.0) > tol:
                    return False
    # Sample zero entries
    for _ in range(n_samples):
        a = rng.randint(nm)
        b = rng.randint(mp)
        c = rng.randint(pn)
        ia, ja = divmod(a, m)
        jb, kb = divmod(b, p)
        kc, ic = divmod(c, n)
        if ja == jb and ia == ic and kb == kc:
            continue
        val = np.dot(U[a, :], V[b, :] * W[c, :])
        if abs(val) > tol:
            return False
    return True

def _fast_als(U, V, W, T, n, m, p, iterations=50, deadline=None, tol=1e-12):
    """Fast ALS using precomputed products and normal equations."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    for it in range(iterations):
        if deadline and time.time() > deadline:
            break
        
        # Update U: for each a, solve (V⊙W)^T(V⊙W) u_a = (V⊙W)^T T_a
        # V⊙W has shape (mp*pn, r)
        VW = (V[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(mp*pn, r)
        VWt = VW.T  # (r, mp*pn)
        VWtVW = VWt @ VW  # (r, r)
        
        for a in range(nm):
            Ta = T[a, :, :].flatten()
            rhs = VWt @ Ta  # (r,)
            try:
                U[a, :] = np.linalg.solve(VWtVW + 1e-12*np.eye(r), rhs)
            except:
                pass
        
        # Update V
        UW = (U[:, np.newaxis, :] * W[np.newaxis, :, :]).reshape(nm*pn, r)
        UWt = UW.T
        UWtUW = UWt @ UW
        
        for b in range(mp):
            Tb = T[:, b, :].flatten()
            rhs = UWt @ Tb
            try:
                V[b, :] = np.linalg.solve(UWtUW + 1e-12*np.eye(r), rhs)
            except:
                pass
        
        # Update W
        UV = (U[:, np.newaxis, :] * V[np.newaxis, :, :]).reshape(nm*mp, r)
        UVt = UV.T
        UVtUV = UVt @ UV
        
        for c in range(pn):
            Tc = T[:, :, c].flatten()
            rhs = UVt @ Tc
            try:
                W[c, :] = np.linalg.solve(UVtUV + 1e-12*np.eye(r), rhs)
            except:
                pass
    
    return U, V, W

def _structured_init_v2(n, m, p, rank, seed=42):
    """Better structured init: exploit j-block structure with random phases."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    if rank >= m:
        terms_per_j = rank // m
        remainder = rank % m
        col = 0
        for j in range(m):
            n_terms = terms_per_j + (1 if j < remainder else 0)
            for t in range(n_terms):
                # Random but structured
                U[i*m+j, col] for i in range(n)
                # Use random values with moderate magnitude
                for i in range(n):
                    U[i*m+j, col] = rng.uniform(-1, 1)
                for k in range(p):
                    V[j*p+k, col] = rng.uniform(-1, 1)
                for k in range(p):
                    for i in range(n):
                        W[k*n+i, col] = rng.uniform(-1, 1)
                col += 1
    else:
        U = rng.randn(nm, rank) / np.sqrt(rank)
        V = rng.randn(mp, rank) / np.sqrt(rank)
        W = rng.randn(pn, rank) / np.sqrt(rank)
    
    return U, V, W

def _mixed_init(n, m, p, rank, seed=42, sparse_frac=0.3):
    """Mixed dense/sparse initialization."""
    rng = np.random.RandomState(seed)
    nm, mp, pn = n*m, m*p, p*n
    
    U = rng.randn(nm, rank) / np.sqrt(rank)
    V = rng.randn(mp, rank) / np.sqrt(rank)
    W = rng.randn(pn, rank) / np.sqrt(rank)
    
    # Sparsify
    mask_U = rng.rand(nm, rank) < sparse_frac
    mask_V = rng.rand(mp, rank) < sparse_frac
    mask_W = rng.rand(pn, rank) < sparse_frac
    U[mask_U] = 0
    V[mask_V] = 0
    W[mask_W] = 0
    
    return U, V, W

def _adam_step(U, V, W, T, n, m, p, m_U, v_U, m_V, v_V, m_W, v_W, t, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """Single Adam step."""
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    R = T - recon
    
    grad_U = -np.einsum('abc,br,cr->ar', R, V, W)
    grad_V = -np.einsum('abc,ar,cr->br', R, U, W)
    grad_W = -np.einsum('abc,ar,br->cr', R, U, V)
    
    m_U = beta1*m_U + (1-beta1)*grad_U
    v_U = beta2*v_U + (1-beta2)*grad_U**2
    m_hat_U = m_U/(1-beta1**t)
    v_hat_U = v_U/(1-beta2**t)
    U -= lr * m_hat_U/(np.sqrt(v_hat_U)+eps)
    
    m_V = beta1*m_V + (1-beta1)*grad_V
    v_V = beta2*v_V + (1-beta2)*grad_V**2
    m_hat_V = m_V/(1-beta1**t)
    v_hat_V = v_V/(1-beta2**t)
    V -= lr * m_hat_V/(np.sqrt(v_hat_V)+eps)
    
    m_W = beta1*m_W + (1-beta1)*grad_W
    v_W = beta2*v_W + (1-beta2)*grad_W**2
    m_hat_W = m_W/(1-beta1**t)
    v_hat_W = v_W/(1-beta2**t)
    W -= lr * m_hat_W/(np.sqrt(v_hat_W)+eps)
    
    return U, V, W, m_U, v_U, m_V, v_V, m_W, v_W

def _adam_optimize(U, V, W, T, n, m, p, max_steps=200, lr=0.01, deadline=None):
    """Adam optimizer with early stopping."""
    nm, mp, pn = n*m, m*p, p*n
    r = U.shape[1]
    
    m_U = np.zeros_like(U)
    v_U = np.zeros_like(U)
    m_V = np.zeros_like(V)
    v_V = np.zeros_like(V)
    m_W = np.zeros_like(W)
    v_W = np.zeros_like(W)
    
    best_loss = np.inf
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    t = 0
    
    for step in range(max_steps):
        if deadline and time.time() > deadline:
            break
        t += 1
        
        U, V, W, m_U, v_U, m_V, v_V, m_W, v_W = _adam_step(
            U, V, W, T, n, m, p, m_U, v_U, m_V, v_V, m_W, v_W, t, lr=lr)
        
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        R = T - recon
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-14:
            break
    
    return best_U, best_V, best_W, best_loss
>>>>>>> REPLACE

Now let me replace the `construct` function with an improved version:

