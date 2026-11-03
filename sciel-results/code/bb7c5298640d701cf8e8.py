import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Build the target tensor (sparse representation)
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                idx_a.append(a)
                idx_b.append(b)
                idx_c.append(c)
    
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    
    # Target tensor
    T_target = np.zeros((nm, mp, pn))
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    # Try various ranks starting from a reasonable lower bound
    # For 4x4x5, schoolbook is 80. Let's try to find something better.
    # Start with rank 60 and work up if needed.
    
    np.random.seed(42)
    
    best_rank = nm * mp * pn  # fallback
    best_U = None
    best_V = None
    best_W = None
    
    for rank in [60, 64, 68, 72, 76, 80]:
        # Complex random initialization
        U = (np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)) * 0.1
        V = (np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)) * 0.1
        W = (np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)) * 0.1
        
        # Gradient descent with complex values
        lr = 0.005
        for step in range(5000):
            # Compute full tensor
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T_full - T_target
            loss = np.sum(np.abs(residual)**2).real
            
            # Gradients (for complex, we use Wirtinger calculus)
            # dL/dU[a,r] = 2 * conj(residual) contracted with V, W
            dU = 2 * np.einsum('abc,br,cr->ar', residual.conj(), V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', residual.conj(), U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', residual.conj(), U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if step % 1000 == 0:
                pass  # silent
        
        # Take real parts
        U_real = np.real(U)
        V_real = np.real(V)
        W_real = np.real(W)
        
        # Snap to half-integers
        U_half = np.round(U_real * 2) / 2
        V_half = np.round(V_real * 2) / 2
        W_half = np.round(W_real * 2) / 2
        
        # Verify
        T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        error = np.sum((T_half - T_target)**2)
        
        if error < 1e-10:
            best_rank = rank
            best_U = U_half
            best_V = V_half
            best_W = W_half
            break
    
    if best_U is not None:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(n * m)]
        V_sb = [[0] * rank_sb for _ in range(m * p)]
        W_sb = [[0] * rank_sb for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][column] = 1
                    V_sb[j * p + k][column] = 1
                    W_sb[k * n + i][column] = 1
                    column += 1
        return {"U": U_sb, "V": V_sb, "W": W_sb}
