import numpy as np

def construct():
    n, m, p = 4, 4, 5
    nm, mp, pn = n*m, m*p, p*n  # 16, 20, 20
    
    # Create the sparse tensor indices
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
    
    # Target rank is likely 62
    rank = 62
    
    # Try multiple random initializations to escape local minima
    best_result = None
    best_error = np.inf
    
    for seed in range(5):
        np.random.seed(seed)
        # Initialize with random complex values
        U = np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)
        V = np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)
        W = np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)
        
        # Gradient descent
        lr = 0.01
        for step in range(2000):
            # Compute residuals
            T_full = np.einsum('ar,br,cr->abc', U, V, W)
            
            # Target tensor
            T_target = np.zeros((nm, mp, pn), dtype=complex)
            T_target[idx_a, idx_b, idx_c] = 1.0 + 0j
            
            residual = T_full - T_target
            loss = np.sum(np.abs(residual)**2)
            
            # Gradients
            # dL/dU = 2 * sum_{b,c} residual[a,b,c] * conj(V[b,r]) * conj(W[c,r])? 
            # Actually, for complex variables, the gradient of ||T||^2 wrt U is 2 * residual * V^* * W^* contracted?
            # Let's use the standard Wirtinger derivative or just treat real and imaginary parts.
            # Simpler: Just use real-valued gradient descent on the real and imaginary parts separately?
            # Or use the formula: dU[a,r] = 2 * sum_{b,c} residual[a,b,c] * conj(V[b,r]) * conj(W[c,r])
            # Note: numpy einsum handles complex numbers fine.
            
            dU = 2 * np.einsum('abc,br,cr->ar', residual, np.conj(V), np.conj(W))
            dV = 2 * np.einsum('abc,ar,cr->br', residual, np.conj(U), np.conj(W))
            dW = 2 * np.einsum('abc,ar,br->cr', residual, np.conj(U), np.conj(V))
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if loss < 1e-10:
                break
        
        # Snap to half-integers (real part, assuming imaginary part is near 0)
        U_real = np.real(U)
        V_real = np.real(V)
        W_real = np.real(W)
        
        # Check if imaginary parts are small
        if np.max(np.abs(np.imag(U))) > 0.1 or np.max(np.abs(np.imag(V))) > 0.1 or np.max(np.abs(np.imag(W))) > 0.1:
            continue
            
        U_half = np.round(U_real * 2) / 2
        V_half = np.round(V_real * 2) / 2
        W_half = np.round(W_real * 2) / 2
        
        # Verify
        T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
        error = np.sum((T_half - T_target)**2)
        
        if error < 1e-10 and error < best_error:
            best_error = error
            best_result = (U_half, V_half, W_half)
            
    if best_result is not None:
        U_half, V_half, W_half = best_result
        return {"U": U_half.tolist(), "V": V_half.tolist(), "W": W_half.tolist()}
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
