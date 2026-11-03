import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 50
    
    # Target ranks to try, starting from lower to higher
    # Known bounds might be higher, but let's try to beat 50.
    # Strassen-like improvements might exist.
    # Let's try ranks 45, 40, 35, 30.
    # If we find one, we return it.
    
    def try_gradient_descent(rank, time_limit=10):
        """
        Attempt to find a decomposition of given rank using gradient descent on a squared error loss.
        Returns U, V, W if found, else None.
        """
        start_time = time.time()
        
        # Initialize with small random values
        # Using symmetric initialization or random normal
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        # Learning rate
        lr = 0.01
        # Decay learning rate
        decay = 0.999
        
        # Precompute indices
        a_idx = []
        b_idx = []
        c_idx = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a_idx.append(i * m + j)
                    b_idx.append(j * p + k)
                    c_idx.append(k * n + i)
        
        a_idx = np.array(a_idx)
        b_idx = np.array(b_idx)
        c_idx = np.array(c_idx)
        
        # Target tensor T is 1 at (a,b,c) and 0 elsewhere?
        # Wait, the tensor T[i*m+j, j*p+k, k*n+i] = 1.
        # There are n*m*p non-zero entries.
        # We want sum_c U[a,c] V[b,c] W[c',c] = 1 for all valid (a,b,c).
        # And we don't care about other entries?
        # No, the decomposition must exactly match the multiplication tensor.
        # The multiplication tensor has 1s at positions (i*m+j, j*p+k, k*n+i).
        # All other entries are 0.
        # So we have n*m*p constraints.
        
        # Loss: Sum over all (a,b,c) in support of (pred - 1)^2 
        #      + Sum over all (a,b,c) NOT in support of (pred - 0)^2?
        # Actually, if we just match the support, the rest might be non-zero.
        # We must ensure the rest are 0.
        # Total entries in U*V*W tensor is (n*m)*(m*p)*(p*n) = (nmp)^2.
        # This is 50^2 = 2500 entries.
        # We have 2500 constraints.
        
        # Let's define the full tensor target.
        # Target[a,b,c] = 1 if (a,b,c) is in support, else 0.
        
        # For efficiency, we can just compute the gradient for the support points first,
        # and then check the full tensor at the end.
        
        # Gradient of L = sum (S - T)^2 w.r.t U, V, W
        # S = sum_c U[a,c] V[b,c] W[c',c]
        # dL/dU[a,c] = 2 * (S - T) * V[b,c] * W[c',c]
        
        # We will iterate.
        for step in range(10000):
            if time.time() - start_time > time_limit:
                break
            
            # Compute predictions for support points
            # S = U[a_idx, :] * V[b_idx, :] * W[c_idx, :]
            # Sum over rank axis
            S = np.sum(U[a_idx, :] * V[b_idx, :] * W[c_idx, :], axis=1)
            T_support = np.ones(len(a_idx))
            
            # Gradient for U
            # dU[a,c] = 2 * (S - 1) * V[b,c] * W[c',c]
            # We need to sum over all (b,c') that pair with a?
            # No, the loss is sum over all (a,b,c).
            # dU[a,c] = 2 * sum_{b,c': (a,b,c') in support} (S_{abc'} - 1) V[b,c] W[c',c]
            
            # This is expensive to compute explicitly for all a,b,c.
            # But n,m,p are small.
            # Let's just compute the full tensor product and extract gradients.
            
            # Full tensor P = sum_c U[:,c] V[:,c] W[:,c]
            # This is an outer product sum.
            # P = np.einsum('ac,bc,dc->abd', U, V, W)
            
            P = np.einsum('ac,bc,dc->abd', U, V, W)
            
            # Target tensor
            # We need to create a target tensor of shape (n*m, m*p, p*n)
            # This is 10 * 25 * 10 = 2500 elements.
            # We can create it once.
            if not hasattr(try_gradient_descent, 'target_tensor'):
                TT = np.zeros((n*m, m*p, p*n))
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            TT[i*m+j, j*p+k, k*n+i] = 1.0
                try_gradient_descent.target_tensor = TT
            
            TT = try_gradient_descent.target_tensor
            
            # Gradient
            # dU = 2 * np.einsum('abd,bc,dc->ac', P - TT, V, W)
            # dV = 2 * np.einsum('abd,ac,dc->bc', P - TT, U, W)
            # dW = 2 * np.einsum('abd,ac,bc->dc', P - TT, U, V)
            
            dU = np.einsum('abd,bc,dc->ac', P - TT, V, W)
            dV = np.einsum('abd,ac,dc->bc', P - TT, U, W)
            dW = np.einsum('abd,ac,bc->dc', P - TT, U, V)
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Optional: Snap to nearest integer/half-integer every N steps?
            # No, that breaks gradient flow.
            
            # Check convergence
            if step % 100 == 0:
                loss = np.sum((P - TT)**2)
                if loss < 1e-6:
                    # Try to snap to integer/half-integer
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    P_snap = np.einsum('ac,bc,dc->abd', U_snap, V_snap, W_snap)
                    if np.allclose(P_snap, TT, atol=1e-9):
                        return U_snap, V_snap, W_snap
            
            lr *= decay
        
        return None

    # Try to find a rank < 50
    # Start from 45 down to 30
    for rank in range(45, 29, -1):
        res = try_gradient_descent(rank, time_limit=5)
        if res:
            U, V, W = res
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
