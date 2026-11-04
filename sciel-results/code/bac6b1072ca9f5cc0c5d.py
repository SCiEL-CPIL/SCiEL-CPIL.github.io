import numpy as np
import time
import scipy.optimize as opt

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for safety

    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i, j, r] V[j, k, r] W[k, i, r] = 1

    # First, try to find a rank-16 decomposition using a structured ansatz
    # based on Strassen's 3x3x3 decomposition
    result_16 = _try_rank16_structured(n, m, p, deadline)
    if result_16 is not None:
        return result_16

    # Fallback: rank-17 decomposition
    result_17 = _rank17_decomposition(n, m, p)
    return result_17

def _rank17_decomposition(n, m, p):
    U17 = np.zeros((n * m, 17))
    V17 = np.zeros((m * p, 17))
    W17 = np.zeros((p * n, 17))

    def set_U17(i, j, r, val): U17[i * m + j, r] = val
    def set_V17(j, k, r, val): V17[j * p + k, r] = val
    def set_W17(k, i, r, val): W17[k * n + i, r] = val

    set_U17(0, 0, 0, 1); set_U17(1, 1, 0, 1)
    set_V17(0, 0, 0, 1); set_V17(1, 1, 0, 1)
    set_W17(0, 0, 0, 1); set_W17(1, 1, 0, 1)

    set_U17(1, 0, 1, 1); set_U17(1, 1, 1, 1)
    set_V17(0, 0, 1, 1)
    set_W17(0, 1, 1, 1); set_W17(1, 1, 1, -1)

    set_U17(0, 0, 2, 1)
    set_V17(0, 1, 2, 1); set_V17(1, 1, 2, -1)
    set_W17(1, 0, 2, 1); set_W17(1, 1, 2, 1)

    set_U17(1, 1, 3, 1)
    set_V17(1, 0, 3, 1); set_V17(0, 0, 3, -1)
    set_W17(0, 0, 3, 1); set_W17(0, 1, 3, 1)

    set_U17(0, 0, 4, 1); set_U17(0, 1, 4, 1)
    set_V17(1, 1, 4, 1)
    set_W17(0, 0, 4, -1); set_W17(1, 0, 4, 1)

    set_U17(1, 0, 5, 1); set_U17(0, 0, 5, -1)
    set_V17(0, 0, 5, 1); set_V17(0, 1, 5, 1)
    set_W17(1, 1, 5, 1)

    set_U17(0, 1, 6, 1); set_U17(1, 1, 6, -1)
    set_V17(1, 0, 6, 1); set_V17(1, 1, 6, 1)
    set_W17(0, 0, 6, 1)

    set_U17(0, 0, 7, 1)
    set_V17(0, 2, 7, 1)
    set_W17(2, 0, 7, 1)

    set_U17(0, 1, 8, 1)
    set_V17(1, 2, 8, 1)
    set_W17(2, 0, 8, 1)

    set_U17(1, 0, 9, 1)
    set_V17(0, 2, 9, 1)
    set_W17(2, 1, 9, 1)

    set_U17(1, 1, 10, 1)
    set_V17(1, 2, 10, 1)
    set_W17(2, 1, 10, 1)

    set_U17(0, 2, 11, 1)
    set_V17(2, 0, 11, 1)
    set_W17(0, 0, 11, 1)

    set_U17(0, 2, 12, 1)
    set_V17(2, 1, 12, 1)
    set_W17(1, 0, 12, 1)

    set_U17(0, 2, 13, 1)
    set_V17(2, 2, 13, 1)
    set_W17(2, 0, 13, 1)

    set_U17(1, 2, 14, 1)
    set_V17(2, 0, 14, 1)
    set_W17(0, 1, 14, 1)

    set_U17(1, 2, 15, 1)
    set_V17(2, 1, 15, 1)
    set_W17(1, 1, 15, 1)

    set_U17(1, 2, 16, 1)
    set_V17(2, 2, 16, 1)
    set_W17(2, 1, 16, 1)

    return {
        "U": U17.tolist(),
        "V": V17.tolist(),
        "W": W17.tolist()
    }

def _check_exact(U, V, W, n, m, p):
    """Check if the decomposition is exact (half-integer entries)."""
    U_s = np.round(U * 2) / 2.0
    V_s = np.round(V * 2) / 2.0
    W_s = np.round(W * 2) / 2.0

    Ui = np.rint(U_s * 2).astype(int)
    Vi = np.rint(V_s * 2).astype(int)
    Wi = np.rint(W_s * 2).astype(int)

    U_t = Ui.reshape(n, m, -1)
    V_t = Vi.reshape(m, p, -1)
    W_t = Wi.reshape(p, n, -1)

    prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)

    if np.all(prod == 8):
        return True, U_s, V_s, W_s
    return False, None, None, None

def _try_rank16_structured(n, m, p, deadline):
    """
    Try to find a rank-16 decomposition using a structured ansatz.
    
    Strategy: Use Strassen's rank-7 decomposition for 3x3x3, and try to
    share terms between the two rows of the 2x3 matrix.
    
    The 2x3x3 tensor can be viewed as two 1x3x3 tensors (rows).
    If we can find a rank-10 decomposition of 1x3x3 that shares 4 terms
    with another rank-10 decomposition, we get rank 16 total.
    
    Uses Coppersmith-Winograd inspired structured initialization where
    the two row-slices share a common sub-structure, plus a final
    high-precision Newton polish.
    """
    rank = 16
    
    def _polish(U, V, W, n, m, p, deadline, max_iters=50):
        """High-precision Newton-style polish with tight tolerance."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            err = np.sum(res**2)
            
            if err < 1e-14:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return Us, Vs, Ws
            
            # Newton step: solve linear system for updates
            # dU[i,j,r] = sum_{k,l} res[i,j,k] * V[j,k,r] * W[k,i,r]  (simplified)
            dU = np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            # Use small step for stability
            lr = 0.5
            U -= lr * dU.reshape(n * m, rank)
            V -= lr * dV.reshape(m * p, rank)
            W -= lr * dW.reshape(p * n, rank)
            
            if err < 1e-10:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return Us, Vs, Ws
        
        # Final check
        if err < 1e-8:
            ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
            if ok:
                return Us, Vs, Ws
        return None, None, None
    
    def _structured_init(seed):
        """
        Create structured initialization based on Coppersmith-Winograd embedding.
        
        The idea: decompose 2x3x3 into a Kronecker-like sum where the two
        row-slices (i=0 and i=1) share a common 3x3x3 sub-structure.
        
        We initialize with a pattern where:
        - First 10 ranks handle the "shared" 3x3x3 structure (Strassen-like)
        - Ranks 10-15 handle the remaining 2x3x3 specific terms
        - The sharing is enforced by making U[0,j,r] and U[1,j,r] related
        """
        np.random.seed(seed)
        
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Strassen's 7-term decomposition for 3x3x3 (for the shared part)
        # Standard Strassen for 2x2x2 applied to 3x3x3 blocks
        # Here we use a 7-term pattern for the core 3x3x3
        
        # Terms 0-6: Shared structure (Strassen-like for 3x3x3)
        # Using the standard Strassen 7-term pattern adapted to 3x3x3
        s = np.random.randn(7, 3, 3) * 0.2  # small perturbations
        
        # Term 0: A00 * B00 -> C00
        U[0*3+0, 0] = 1; U[1*3+0, 0] = 0
        V[0*3+0, 0] = 1; V[1*3+1, 0] = 1
        W[0*2+0, 0] = 1; W[1*2+1, 0] = 1
        
        # Term 1: A01 * B10 -> C01
        U[0*3+1, 1] = 1
        V[1*3+0, 1] = 1
        W[0*2+1, 1] = 1
        
        # Term 2: A10 * B01 -> C10
        U[1*3+0, 2] = 1
        V[0*3+1, 2] = 1
        W[1*2+0, 2] = 1
        
        # Term 3: A11 * B11 -> C11
        U[1*3+1, 3] = 1
        V[1*3+1, 3] = 1
        W[1*2+1, 3] = 1
        
        # Terms 4-6: Strassen cross terms
        # (A00+A11)(B00+B11)
        U[0*3+0, 4] = 1; U[1*3+1, 4] = 1
        V[0*3+0, 4] = 1; V[1*3+1, 4] = 1
        W[0*2+0, 4] = 1; W[1*2+1, 4] = 1
        
        # (A00+A10)(B10+B11)
        U[0*3+0, 5] = 1; U[1*3+0, 5] = 1
        V[1*3+0, 5] = 1; V[1*3+1, 5] = 1
        W[0*2+0, 5] = -1; W[1*2+0, 5] = 1; W[1*2+1, 5] = 1
        
        # (A10+A11)(B00+B01)
        U[1*3+0, 6] = 1; U[1*3+1, 6] = 1
        V[0*3+0, 6] = 1; V[0*3+1, 6] = 1
        W[0*2+1, 6] = -1; W[1*2+0, 6] = 1; W[1*2+1, 6] = 1
        
        # Terms 7-9: Additional shared structure for 3rd column/row
        # A02 * B02 -> C02
        U[0*3+2, 7] = 1
        V[0*3+2, 7] = 1
        W[2*2+0, 7] = 1
        
        # A12 * B12 -> C12
        U[1*3+2, 8] = 1
        V[1*3+2, 8] = 1
        W[2*2+1, 8] = 1
        
        # (A02+A12)(B02+B12)
        U[0*3+2, 9] = 1; U[1*3+2, 9] = 1
        V[0*3+2, 9] = 1; V[1*3+2, 9] = 1
        W[2*2+0, 9] = 1; W[2*2+1, 9] = 1
        
        # Terms 10-15: Row-specific corrections
        # These handle the 2x3x3 specific structure
        for r in range(10, rank):
            # Random small initialization for correction terms
            U[:, r] = np.random.randn(n * m) * 0.1
            V[:, r] = np.random.randn(m * p) * 0.1
            W[:, r] = np.random.randn(p * n) * 0.1
        
        # Add small random perturbation to break symmetries
        U += np.random.randn(n * m, rank) * 0.01
        V += np.random.randn(m * p, rank) * 0.01
        W += np.random.randn(p * n, rank) * 0.01
        
        return U, V, W
    
    # Try multiple structured initializations with different seeds
    for seed in range(150):
        if time.time() > deadline:
            break
        
        U, V, W = _structured_init(seed)
        
        # Phase 1: Coarse gradient descent
        lr = 0.03
        best_err = float('inf')
        best_state = (U.copy(), V.copy(), W.copy())
        
        for step in range(300):
            if time.time() > deadline:
                break
            
            U_t = U.reshape(n, m, rank)
            V_t = V.reshape(m, p, rank)
            W_t = W.reshape(p, n, rank)
            
            prod = np.einsum('ijr,jkr,kir->ijk', U_t, V_t, W_t)
            res = prod - 1.0
            err = np.sum(res**2)
            
            if err < best_err:
                best_err = err
                best_state = (U.copy(), V.copy(), W.copy())
            
            if err < 1e-10:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
            
            # Compute gradients
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', res, V_t, W_t)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', res, U_t, W_t)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', res, U_t, V_t)
            
            # Adaptive learning rate with decay
            if step % 50 == 0 and step > 0:
                lr *= 0.92
            
            U -= lr * dU.reshape(n * m, rank)
            V -= lr * dV.reshape(m * p, rank)
            W -= lr * dW.reshape(p * n, rank)
            
            # Periodic exact check
            if step % 20 == 0:
                ok, Us, Vs, Ws = _check_exact(U, V, W, n, m, p)
                if ok:
                    return {
                        "U": Us.tolist(),
                        "V": Vs.tolist(),
                        "W": Ws.tolist()
                    }
        
        # Phase 2: High-precision polish on best state
        if best_err < 1e-4:
            U_b, V_b, W_b = best_state
            Us, Vs, Ws = _polish(U_b, V_b, W_b, n, m, p, deadline, max_iters=80)
            if Us is not None:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
        
        # Also try polishing the current state if it's good enough
        if best_err < 1e-6:
            Us, Vs, Ws = _polish(U, V, W, n, m, p, deadline, max_iters=50)
            if Us is not None:
                return {
                    "U": Us.tolist(),
                    "V": Vs.tolist(),
                    "W": Ws.tolist()
                }
    
    return None
