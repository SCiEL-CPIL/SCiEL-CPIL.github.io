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
    
    T_target = np.zeros((nm, mp, pn), dtype=float)
    T_target[idx_a, idx_b, idx_c] = 1.0
    
    # Try ranks from 44 to 50
    # We use complex GD as per directive
    for rank in range(44, 51):
        for restart in range(10):
            np.random.seed(100 + rank * 100 + restart)
            
            # Complex initialization
            U = (np.random.randn(nm, rank) + 1j * np.random.randn(nm, rank)) * 0.5
            V = (np.random.randn(mp, rank) + 1j * np.random.randn(mp, rank)) * 0.5
            W = (np.random.randn(pn, rank) + 1j * np.random.randn(pn, rank)) * 0.5
            
            lr = 0.05
            steps = 3000
            
            for step in range(steps):
                # Compute T_full
                T_full = np.einsum('ar,br,cr->abc', U, V, W)
                
                # Residual
                residual = T_full - T_target
                
                # Gradients
                # dL/dU = 2 * conj(residual) * V * W
                dU = 2 * np.einsum('abc,br,cr->ar', np.conj(residual), V, W)
                dV = 2 * np.einsum('abc,ar,cr->br', np.conj(residual), U, W)
                dW = 2 * np.einsum('abc,ar,br->cr', np.conj(residual), U, V)
                
                # Update
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                # Adaptive learning rate: reduce if loss increases? 
                # Simpler: decay every 1000 steps
                if step % 1000 == 0 and step > 0:
                    lr *= 0.7

            # Take real parts
            U_real = U.real
            V_real = V.real
            W_real = W.real
            
            # Snap to half-integers
            U_half = np.round(U_real * 2) / 2
            V_half = np.round(V_real * 2) / 2
            W_half = np.round(W_real * 2) / 2
            
            # Verify
            T_half = np.einsum('ar,br,cr->abc', U_half, V_half, W_half)
            error = np.max(np.abs(T_half - T_target))
            
            if error < 1e-9:
                return {
                    "U": U_half.tolist(),
                    "V": V_half.tolist(),
                    "W": W_half.tolist()
                }
                
    # Fallback to schoolbook rank 80
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
