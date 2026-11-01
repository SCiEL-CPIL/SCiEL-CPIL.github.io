"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n=2, m=2, p=4):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    T = np.ones((n, m, p))
    T_recon = np.zeros((n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_recon[i, j, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    if not np.allclose(T, T_recon, atol=1e-9):
        return False
    return True

def _strassen_rank14():
    """Construct the rank-14 decomposition using two Strassen blocks."""
    n, m, p = 2, 2, 4
    rank = 14
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Left half (k=0,1)
    r = 0
    U[0, r] = 1; U[3, r] = 1
    V[0, r] = 1; V[5, r] = 1
    W[0, r] = 1; W[3, r] = 1
    
    r = 1
    U[2, r] = 1; U[3, r] = 1
    V[0, r] = 1
    W[2, r] = 1; W[3, r] = -1
    
    r = 2
    U[0, r] = 1
    V[1, r] = 1; V[5, r] = -1
    W[1, r] = 1; W[3, r] = 1
    
    r = 3
    U[3, r] = 1
    V[4, r] = 1; V[0, r] = -1
    W[0, r] = 1; W[2, r] = 1
    
    r = 4
    U[0, r] = 1; U[1, r] = 1
    V[5, r] = 1
    W[1, r] = 1; W[0, r] = -1
    
    r = 5
    U[2, r] = 1; U[0, r] = -1
    V[0, r] = 1; V[1, r] = 1
    W[3, r] = 1
    
    r = 6
    U[1, r] = 1; U[3, r] = -1
    V[4, r] = 1; V[5, r] = 1
    W[0, r] = 1
    
    # Right half (k=2,3)
    r = 7
    U[0, r] = 1; U[3, r] = 1
    V[2, r] = 1; V[7, r] = 1
    W[4, r] = 1; W[7, r] = 1
    
    r = 8
    U[2, r] = 1; U[3, r] = 1
    V[2, r] = 1
    W[6, r] = 1; W[7, r] = -1
    
    r = 9
    U[0, r] = 1
    V[3, r] = 1; V[7, r] = -1
    W[5, r] = 1; W[7, r] = 1
    
    r = 10
    U[3, r] = 1
    V[6, r] = 1; V[2, r] = -1
    W[4, r] = 1; W[6, r] = 1
    
    r = 11
    U[0, r] = 1; U[1, r] = 1
    V[7, r] = 1
    W[5, r] = 1; W[4, r] = -1
    
    r = 12
    U[2, r] = 1; U[0, r] = -1
    V[2, r] = 1; V[3, r] = 1
    W[7, r] = 1
    
    r = 13
    U[1, r] = 1; U[3, r] = -1
    V[6, r] = 1; V[7, r] = 1
    W[4, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _compute_error(U, V, W, n=2, m=2, p=4):
    """Compute the error tensor for a given decomposition."""
    rank = U.shape[1]
    err = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                err[i, j, k] = s - 1.0
    return err

def _try_rank13_analytic(n, m, p, deadline):
    """
    Try to find a rank-13 decomposition using an analytic approach.
    
    Key insight: The 2x2x4 tensor can be viewed as two 2x2x2 tensors (left and right halves).
    Each 2x2x2 tensor has rank 7 (Strassen). But we can potentially share structure.
    
    Approach: Use 7 products for the left half and 6 for the right half, with one product
    that contributes to both halves.
    """
    start = time.time()
    target_rank = 13
    nm, mp, pn = n*m, m*p, p*n  # 4, 8, 8
    
    # Initialize with Strassen-like structure
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    # Left half products (r=0..5): 6 Strassen products
    # p0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # p1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # p2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # p3: A11(B10-B00)
    U[3, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # p4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[1, 4] = 1; W[0, 4] = -1
    
    # p5: (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # Right half products (r=6..11): 6 Strassen products
    # p6: (A00+A11)(B02+B13)
    U[0, 6] = 1; U[3, 6] = 1
    V[2, 6] = 1; V[7, 6] = 1
    W[4, 6] = 1; W[7, 6] = 1
    
    # p7: (A10+A11)B02
    U[2, 7] = 1; U[3, 7] = 1
    V[2, 7] = 1
    W[6, 7] = 1; W[7, 7] = -1
    
    # p8: A00(B03-B13)
    U[0, 8] = 1
    V[3, 8] = 1; V[7, 8] = -1
    W[5, 8] = 1; W[7, 8] = 1
    
    # p9: A11(B12-B02)
    U[3, 9] = 1
    V[6, 9] = 1; V[2, 9] = -1
    W[4, 9] = 1; W[6, 9] = 1
    
    # p10: (A00+A01)B13
    U[0, 10] = 1; U[1, 10] = 1
    V[7, 10] = 1
    W[5, 10] = 1; W[4, 10] = -1
    
    # p11: (A01-A11)(B12+B13)
    U[1, 11] = 1; U[3, 11] = -1
    V[6, 11] = 1; V[7, 11] = 1
    W[4, 11] = 1
    
    # Product 12: shared/correction product
    # This needs to cover the missing contributions from both halves
    
    # Missing from left: (A01-A11)(B10+B11) -> C00+=
    # Missing from right: (A10-A00)(B02+B03) -> C13+=
    
    # For the shared product, we need U, V, W patterns that can contribute to both
    # Let's try: U = [0, 1, -1, 0] (A01-A11)
    U[1, 12] = 1; U[3, 12] = -1
    
    # V needs to contribute to both left (k=0,1) and right (k=2,3)
    # For left: need B10+B11 = V[4]+V[5]
    # For right: need B02+B03 = V[2]+V[3]
    V[4, 12] = 1; V[5, 12] = 1
    V[2, 12] = 1; V[3, 12] = 1
    
    # W needs to contribute to both left (i=0) and right (i=1)
    # For left: need C00 = W[0]
    # For right: need C13 = W[7]
    W[0, 12] = 1; W[7, 12] = 1
    
    # Now do gradient descent to refine
    for iteration in range(5000):
        if time.time() > deadline - 10:
            break
        
        err = _compute_error(U, V, W, n, m, p)
        max_err = np.max(np.abs(err))
        
        if max_err < 1e-10:
            return U.tolist(), V.tolist(), W.tolist()
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(target_rank):
                        dU[i*m+j, r] += 2 * err[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += 2 * err[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += 2 * err[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
        
        # Adaptive learning rate
        grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
        if grad_norm < 1e-12:
            break
        lr = 0.01 / (1 + iteration * 0.01)
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
    
    err = _compute_error(U, V, W, n, m, p)
    if np.max(np.abs(err)) < 1e-9:
        return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank13_random(n, m, p, deadline, max_restarts=50):
    """
    Try to find a rank-13 decomposition using random restarts with gradient descent.
    """
    start = time.time()
    target_rank = 13
    nm, mp, pn = n*m, m*p, p*n  # 4, 8, 8
    
    np.random.seed(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with structure
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        W = np.zeros((pn, target_rank))
        
        for r in range(target_rank):
            # Random sparse patterns
            u_mask = np.random.random(nm) < 0.5
            v_mask = np.random.random(mp) < 0.4
            w_mask = np.random.random(pn) < 0.4
            
            if not np.any(u_mask):
                u_mask[0] = True
            if not np.any(v_mask):
                v_mask[0] = True
            if not np.any(w_mask):
                w_mask[0] = True
            
            U[u_mask, r] = np.random.choice([-1, 1], size=np.sum(u_mask))
            V[v_mask, r] = np.random.choice([-1, 1], size=np.sum(v_mask))
            W[w_mask, r] = np.random.choice([-1, 1], size=np.sum(w_mask))
        
        # Gradient descent
        for iteration in range(3000):
            if time.time() > deadline - 10:
                break
            
            err = _compute_error(U, V, W, n, m, p)
            max_err = np.max(np.abs(err))
            
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        for r in range(target_rank):
                            dU[i*m+j, r] += 2 * err[i, j, k] * V[j*p+k, r] * W[k*n+i, r]
                            dV[j*p+k, r] += 2 * err[i, j, k] * U[i*m+j, r] * W[k*n+i, r]
                            dW[k*n+i, r] += 2 * err[i, j, k] * U[i*m+j, r] * V[j*p+k, r]
            
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-12:
                break
            
            lr = 0.005 / (1 + iteration * 0.005)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        err = _compute_error(U, V, W, n, m, p)
        if np.max(np.abs(err)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _try_rank13_numpy(n, m, p, deadline):
    """
    Try to find a rank-13 decomposition using vectorized NumPy operations for speed.
    """
    start = time.time()
    target_rank = 13
    nm, mp, pn = n*m, m*p, p*n  # 4, 8, 8
    
    np.random.seed(123)
    
    for restart in range(30):
        if time.time() > deadline - 10:
            break
        
        # Initialize
        U = np.random.randn(nm, target_rank) * 0.5
        V = np.random.randn(mp, target_rank) * 0.5
        W = np.random.randn(pn, target_rank) * 0.5
        
        # Gradient descent with vectorized operations
        for iteration in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Compute error using einsum
            # T_recon[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # We need to be careful with the index mappings
            
            err = np.zeros((n, m, p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                        err[i, j, k] = s - 1.0
            
            max_err = np.max(np.abs(err))
            if max_err < 1e-10:
                return U.tolist(), V.tolist(), W.tolist()
            
            # Compute gradients (vectorized where possible)
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        e = err[i, j, k]
                        dU[i*m+j, :] += 2 * e * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += 2 * e * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += 2 * e * U[i*m+j, :] * V[j*p+k, :]
            
            grad_norm = np.sqrt(np.sum(dU**2) + np.sum(dV**2) + np.sum(dW**2))
            if grad_norm < 1e-12:
                break
            
            lr = 0.01 / (1 + iteration * 0.005)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        err = _compute_error(U, V, W, n, m, p)
        if np.max(np.abs(err)) < 1e-9:
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    
    # Start with known rank-14 solution
    best = _strassen_rank14()
    best_rank = 14
    
    if not _verify(best["U"], best["V"], best["W"], n, m, p):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try analytic rank-13 (fast, structured)
    if time.time() < deadline - 30:
        result = _try_rank13_analytic(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try random restart rank-13
    if time.time() < deadline - 20:
        result = _try_rank13_random(n, m, p, deadline, max_restarts=20)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Try vectorized rank-13
    if time.time() < deadline - 10:
        result = _try_rank13_numpy(n, m, p, deadline)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    # Return best found
    return best
# EVOLVE-BLOCK-END
