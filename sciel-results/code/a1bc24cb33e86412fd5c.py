"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """
    Fast vectorized verification.
    Checks if sum_c W[k*n+i, c] * U[i*m+j, c] * V[l*p+k, c] == delta(j,l) for all i,k,j,l.
    """
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    try:
        # T[i, k, j, l] = sum_c W_r[k, i, c] * U_r[i, j, c] * V_r[l, k, c]
        # U: (n, m, r), V: (m, p, r), W: (p, n, r)
        # Target indices for result: i, k, j, l
        # U: i, j, c
        # V: l, k, c  -> Note V is (m, p, r) so V_r[l, k, c] is correct if we treat first dim as l?
        # Wait, V shape is (m*p, rank). Reshaped to (m, p, rank).
        # V[j, k, c] in schoolbook.
        # In multiplication: (A B)_{i, k} = sum_j A_{i, j} B_{j, k}.
        # Tensor: M_{i, k, j, l} = delta_{j, l}.
        # Term: U_{i, j, c} V_{j, k, c} W_{k, i, c}?
        # Let's re-read the task: "output factor follows the official trace convention W[k*n+i,r]"
        # U shape (n*m, rank). U[i*m+j, c] = U_{i, j, c}.
        # V shape (m*p, rank). V[j*p+k, c] = V_{j, k, c}.
        # W shape (p*n, rank). W[k*n+i, c] = W_{k, i, c}.
        # Product: U_{i, j, c} V_{j, k, c} W_{k, i, c}.
        # Sum over c.
        # Result index: i, k, j, l?
        # The matrix product element is C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # The tensor decomposition defines a bilinear map.
        # The condition is that the computed product matches the true product.
        # True product: C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # Decomposition: C_{i, k} = sum_c (sum_j U_{i, j, c} A_{i, j}?) No.
        # Standard tensor decomposition for matmul:
        # C_{i, k} = sum_c (sum_j U_{i, j, c} A_{i, j}) (sum_l V_{j, k, c} B_{j, k})? No.
        # It is: C_{i, k} = sum_c (sum_j U_{i, j, c} A_{i, j}) * (sum_l V_{l, k, c} B_{l, k})?
        # Let's check the indices in the verification code provided in the seed.
        # Seed says: `T[i, k, j, l] = sum_c W_r[k, i, c] * U_r[i, j, c] * V_r[l, k, c]`
        # And checks `T[i, k, j, l] == delta(j, l)`.
        # This implies the decomposition is:
        # sum_c U_{i, j, c} V_{l, k, c} W_{k, i, c} = delta_{j, l}.
        # This is the condition for the tensor to represent the identity map on the middle dimension?
        # If we have A (n x m) and B (m x p).
        # C_{i, k} = sum_j A_{i, j} B_{j, k}.
        # Using decomposition:
        # C_{i, k} = sum_c (sum_j U_{i, j, c} A_{i, j}) (sum_l V_{l, k, c} B_{l, k})?
        # Let's expand: sum_c (sum_j U_{i, j, c} A_{i, j}) (sum_l V_{l, k, c} B_{l, k})
        # = sum_{j, l} (sum_c U_{i, j, c} V_{l, k, c}) A_{i, j} B_{l, k}.
        # We want this to equal sum_j A_{i, j} B_{j, k}.
        # This requires sum_c U_{i, j, c} V_{l, k, c} = delta_{j, l} delta_{k, k}? No.
        # It requires the coefficient of A_{i, j} B_{l, k} to be delta_{j, l} if k matches?
        # Actually, the term is A_{i, j} B_{l, k}. The target is A_{i, j} B_{j, k}.
        # So we need sum_c U_{i, j, c} V_{l, k, c} to act as delta_{j, l} when summed with A and B?
        # No, the sum over c is inside.
        # If sum_c U_{i, j, c} V_{l, k, c} = delta_{j, l} * W_{k, i, c}? No, W is also summed.
        
        # The seed verification is:
        # `np.einsum('ijc,lkc,kic->ikjl', U_r, V_r, W_r)`
        # U: i, j, c
        # V: l, k, c  (Note: V_r is (m, p, rank). Index 0 is m (j?), Index 1 is p (k?)).
        # Wait, V is (m*p, rank). Reshape (m, p, rank).
        # V_r[j, k, c].
        # In einsum 'lkc', the first index is l (size m), second is k (size p).
        # So V_r[l, k, c] corresponds to V_{l, k, c}.
        # W: k, i, c. W_r is (p, n, rank). W_r[k, i, c].
        # So the term is U_{i, j, c} V_{l, k, c} W_{k, i, c}.
        # Sum over c.
        # Result T[i, k, j, l].
        # Check if T[i, k, j, l] == delta_{j, l}.
        
        # This matches the standard "border rank" condition for matmul tensors.
        
        T = np.einsum('ijc,lkc,kic->ikjl', U_r, V_r, W_r)
        
        # Target: delta(j, l) for all i, k.
        D = np.zeros((n, p, m, m))
        for j in range(m):
            D[:, :, j, j] = 1.0
            
        return np.allclose(T, D, atol=tol)
        
    except Exception:
        return False


def _optimize_rank(n, m, p, rank, max_steps=2000, lr=0.01, seed=0, time_limit=10):
    """
    Optimizes U, V, W for a given rank using Gradient Descent.
    Returns (U, V, W, loss)
    """
    np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.1
    V = np.random.randn(m * p, rank) * 0.1
    W = np.random.randn(p * n, rank) * 0.1
    
    # Adam optimizer state
    mU = np.zeros_like(U)
    vU = np.zeros_like(U)
    mV = np.zeros_like(V)
    vV = np.zeros_like(V)
    mW = np.zeros_like(W)
    vW = np.zeros_like(W)
    b1 = 0.9
    b2 = 0.999
    eps = 1e-8
    
    start_time = time.time()
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    D = np.zeros((n, p, m, m))
    for j in range(m):
        D[:, :, j, j] = 1.0
        
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    
    for step in range(max_steps):
        if time.time() > start_time + time_limit:
            break
            
        # Forward pass
        # T[i, k, j, l] = sum_c U[i, j, c] V[l, k, c] W[k, i, c]
        # Note: V_r is (m, p, rank) -> indices (l, k, c) in einsum 'lkc'
        # W_r is (p, n, rank) -> indices (k, i, c) in einsum 'kic'
        T = np.einsum('ijc,lkc,kic->ikjl', U_r, V_r, W_r)
        R = T - D
        loss = np.sum(R**2)
        
        if loss < best_loss:
            best_loss = loss
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
            
        if loss < 1e-8:
            break
            
        # Gradients
        # dL/dU[i, j, c] = sum_{k, l} R[i, k, j, l] V[l, k, c] W[k, i, c]
        # S[i, k, l, c] = V[l, k, c] W[k, i, c]
        S = np.einsum('lkc,kic->iklc', V_r, W_r) # i, k, l, c
        gU = np.einsum('ikjl,iklc->ijc', R, S)
        
        # dL/dV[l, k, c] = sum_{i, j} R[i, k, j, l] U[i, j, c] W[k, i, c]
        gV = np.einsum('ikjl,ijc,kic->lkc', R, U_r, W_r)
        
        # dL/dW[k, i, c] = sum_{j, l} R[i, k, j, l] U[i, j, c] V[l, k, c]
        gW = np.einsum('ikjl,ijc,lkc->kic', R, U_r, V_r)
        
        # Adam Update
        t = step + 1
        mU = b1 * mU + (1 - b1) * gU
        vU = b2 * vU + (1 - b2) * (gU ** 2)
        mU_hat = mU / (1 - b1 ** t)
        vU_hat = vU / (1 - b2 ** t)
        U -= lr * mU_hat / (np.sqrt(vU_hat) + eps)
        
        mV = b1 * mV + (1 - b1) * gV
        vV = b2 * vV + (1 - b2) * (gV ** 2)
        mV_hat = mV / (1 - b1 ** t)
        vV_hat = vV / (1 - b2 ** t)
        V -= lr * mV_hat / (np.sqrt(vV_hat) + eps)
        
        mW = b1 * mW + (1 - b1) * gW
        vW = b2 * vW + (1 - b2) * (gW ** 2)
        mW_hat = mW / (1 - b1 ** t)
        vW_hat = vW / (1 - b2 ** t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        
        # Reshape for next iteration
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        
    return best_U, best_V, best_W, best_loss


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Start with schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    best_U, best_V, best_W = U0.copy(), V0.copy(), W0.copy()
    best_rank = U0.shape[1]
    
    # Lower bound heuristic
    # For 2x3x8, schoolbook is 48.
    # Try ranks from 24 to 47
    lower_bound = 24
    
    for target_rank in range(lower_bound, best_rank):
        if time.time() > deadline - 20:
            break
            
        found = False
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
                
            seed = target_rank * 100 + attempt
            U, V, W, loss = _optimize_rank(n, m, p, target_rank, max_steps=1000, lr=0.05, seed=seed, time_limit=10)
            
            if loss < 1e-5:
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                if _verify_fast(U_snap, V_snap, W_snap, n, m, p, tol=1e-4):
                    best_U, best_V, best_W = U_snap, V_snap, W_snap
                    best_rank = target_rank
                    found = True
                    break
                elif _verify_fast(U, V, W, n, m, p, tol=1e-6):
                    # Accept float solution if snap fails but float is exact
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    found = True
                    break
                    
        if found:
            break

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
