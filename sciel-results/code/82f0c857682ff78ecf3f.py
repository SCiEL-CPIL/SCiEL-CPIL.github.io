"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
from fractions import Fraction

import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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


def _build_target_tensor(n, m, p):
    """Build the target matrix multiplication tensor."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _tensor_from_decomp(U, V, W):
    """Compute tensor T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]."""
    return np.einsum('ar,br,cr->abc', U, V, W)


def _loss_and_grad(U, V, W, T):
    """Compute loss ||T - tensor(U,V,W)||^2 and gradients."""
    T_approx = _tensor_from_decomp(U, V, W)
    diff = T_approx - T
    loss = np.sum(diff * diff)
    
    gU = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    gV = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    gW = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    return loss, gU, gV, gW


def _adam_optimize(U, V, W, T, max_iters=300, lr_init=0.01, deadline=None):
    """Optimize U,V,W using Adam optimizer."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = lr_init
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        loss, gU, gV, gW = _loss_and_grad(U, V, W, T)
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-12:
            break
        
        t = it + 1
        mU = beta1 * mU + (1 - beta1) * gU
        vU = beta2 * vU + (1 - beta2) * gU**2
        mU_hat = mU / (1 - beta1**t)
        vU_hat = vU / (1 - beta2**t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = beta1 * mV + (1 - beta1) * gV
        vV = beta2 * vV + (1 - beta2) * gV**2
        mV_hat = mV / (1 - beta1**t)
        vV_hat = vV / (1 - beta2**t)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = beta1 * mW + (1 - beta1) * gW
        vW = beta2 * vW + (1 - beta2) * gW**2
        mW_hat = mW / (1 - beta1**t)
        vW_hat = vW / (1 - beta2**t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        if it > 100:
            lr *= 0.97
    
    return best_U, best_V, best_W, best_loss


def _snap_to_half_integers(U, V, W, tol=1e-6):
    """Snap values to nearest half-integer if close enough."""
    U_snapped = np.round(U * 2) / 2.0
    V_snapped = np.round(V * 2) / 2.0
    W_snapped = np.round(W * 2) / 2.0
    
    max_diff = max(np.max(np.abs(U - U_snapped)),
                   np.max(np.abs(V - V_snapped)),
                   np.max(np.abs(W - W_snapped)))
    
    if max_diff < tol:
        return U_snapped, V_snapped, W_snapped
    else:
        return U, V, W


def _verify_decomposition(U, V, W, T, tol=1e-6):
    """Check if decomposition exactly matches target tensor."""
    T_approx = _tensor_from_decomp(U, V, W)
    return np.max(np.abs(T_approx - T)) < tol


def _try_strassen_block(n, m, p, deadline):
    """
    Try to exploit block structure. For 3x4x8, we can view the 4x8 part
    as two 4x4 blocks. If we can find a rank-7 decomposition for 4x4x4
    (Strassen gives 7), we might reduce the total rank.
    
    Actually, let's try a different approach: use the known Strassen algorithm
    for 2x2x2 (rank 7) and try to embed it.
    
    For 3x4x8: 
    - We can split the 4 columns into two groups of 2
    - We can split the 8 rows into four groups of 2
    - Each 2x2x2 block can use rank-7 Strassen
    - But the coupling between blocks makes this complex
    
    Let's try a simpler approach: random initialization with structured patterns.
    """
    T = _build_target_tensor(n, m, p)
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    # Try various ranks
    # Lower bound for 3x4x8 is at least max(12, 32, 24) = 32
    # Known: rank(3x4x8) <= 3*4*8 = 96
    # Let's try to find something better
    
    np.random.seed(123)
    
    for r in range(60, 31, -1):
        if time.time() > deadline:
            break
        
        for attempt in range(3):
            if time.time() > deadline:
                break
            
            # Structured initialization: use block patterns
            U_rand = np.random.randn(n * m, r) * 0.5
            V_rand = np.random.randn(m * p, r) * 0.5
            W_rand = np.random.randn(p * n, r) * 0.5
            
            # Add some structure: make some entries zero
            for idx in np.random.choice(U_rand.size, size=int(0.3 * U_rand.size), replace=False):
                U_rand.flat[idx] = 0
            for idx in np.random.choice(V_rand.size, size=int(0.3 * V_rand.size), replace=False):
                V_rand.flat[idx] = 0
            for idx in np.random.choice(W_rand.size, size=int(0.3 * W_rand.size), replace=False):
                W_rand.flat[idx] = 0
            
            U_opt, V_opt, W_opt, loss = _adam_optimize(
                U_rand, V_rand, W_rand, T,
                max_iters=200, lr_init=0.005, deadline=deadline
            )
            
            if loss < 1e-10:
                U_v, V_v, W_v = _snap_to_half_integers(U_opt, V_opt, W_opt)
                if _verify_decomposition(U_v, V_v, W_v, T):
                    best_rank = r
                    best_decomp = {
                        "U": U_v.tolist(),
                        "V": V_v.tolist(),
                        "W": W_v.tolist()
                    }
                    break
        
        if best_rank < n * m * p:
            # Found improvement, try to go lower
            continue
    
    return best_rank, best_decomp


def _try_direct_search(n, m, p, deadline):
    """Direct gradient-based search with multiple restarts and structured inits."""
    T = _build_target_tensor(n, m, p)
    best_rank = n * m * p
    best_decomp = _schoolbook(n, m, p)
    
    np.random.seed(42)
    
    # Try ranks from 95 down to 40
    for r in range(95, 39, -1):
        if time.time() > deadline:
            break
        
        found = False
        for attempt in range(4):
            if time.time() > deadline:
                break
            
            # Different initialization strategies
            if attempt == 0:
                # Pure random
                U_rand = np.random.randn(n * m, r)
                V_rand = np.random.randn(m * p, r)
                W_rand = np.random.randn(p * n, r)
            elif attempt == 1:
                # Sparse random
                U_rand = np.random.randn(n * m, r)
                V_rand = np.random.randn(m * p, r)
                W_rand = np.random.randn(p * n, r)
                mask = np.random.rand(n * m, r) < 0.5
                U_rand *= mask
            elif attempt == 2:
                # Block structure: group by i, j, k
                U_rand = np.zeros((n * m, r))
                V_rand = np.zeros((m * p, r))
                W_rand = np.zeros((p * n, r))
                for col in range(r):
                    i = col % n
                    j = (col // n) % m
                    k = (col // (n * m)) % p
                    U_rand[i * m + j, col] = 1.0
                    V_rand[j * p + k, col] = 1.0
                    W_rand[k * n + i, col] = 1.0
                # Add noise
                U_rand += 0.1 * np.random.randn(n * m, r)
                V_rand += 0.1 * np.random.randn(m * p, r)
                W_rand += 0.1 * np.random.randn(p * n, r)
            else:
                # Half-integer initialization
                U_rand = np.round(np.random.randn(n * m, r) * 2) / 2.0
                V_rand = np.round(np.random.randn(m * p, r) * 2) / 2.0
                W_rand = np.round(np.random.randn(p * n, r) * 2) / 2.0
            
            U_opt, V_opt, W_opt, loss = _adam_optimize(
                U_rand, V_rand, W_rand, T,
                max_iters=150, lr_init=0.01, deadline=deadline
            )
            
            if loss < 1e-10:
                U_v, V_v, W_v = _snap_to_half_integers(U_opt, V_opt, W_opt)
                if _verify_decomposition(U_v, V_v, W_v, T):
                    found = True
                    best_rank = r
                    best_decomp = {
                        "U": U_v.tolist(),
                        "V": V_v.tolist(),
                        "W": W_v.tolist()
                    }
                    break
        
        if found:
            continue
    
    return best_rank, best_decomp


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Try direct search first
    rank, decomp = _try_direct_search(n, m, p, deadline)
    
    return decomp
# EVOLVE-BLOCK-END
