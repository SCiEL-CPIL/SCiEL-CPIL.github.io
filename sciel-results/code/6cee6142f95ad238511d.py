import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Try to find rank-10 decomposition of 2x2x5
    # If found, use it for 2x4x5 (rank 20)
    # Else, fall back to schoolbook (rank 40)
    
    def find_2x2x5_decomposition(rank, max_iters=10000):
        # U: 4 x rank -> (2, 2, rank)
        # V: 10 x rank -> (2, 5, rank)
        # W: 10 x rank -> (5, 2, rank)
        
        # Random init
        U = np.random.randn(2, 2, rank)
        V = np.random.randn(2, 5, rank)
        W = np.random.randn(5, 2, rank)
        
        # Scale
        U /= np.sqrt(rank)
        V /= np.sqrt(rank)
        W /= np.sqrt(rank)
        
        T_target = np.ones((2, 2, 5))
        
        # Simple gradient descent
        lr = 0.01
        prev_loss = np.inf
        for it in range(max_iters):
            if time.time() > deadline - 5:
                return None
            
            T_comp = np.einsum('ijr,jkr,kir->ijk', U, V, W)
            loss = np.sum((T_comp - T_target)**2)
            
            if loss < 1e-9:
                return U, V, W
                
            # Gradients
            # dL/dU = 2 * (T_comp - T_target) * (V * W)
            # dL/dV = 2 * (T_comp - T_target) * (U * W)
            # dL/dW = 2 * (T_comp - T_target) * (U * V)
            
            diff = T_comp - T_target
            
            # dU[i,j,r] = 2 * sum_{k} diff[i,j,k] * V[j,k,r] * W[k,i,r]
            # dV[j,k,r] = 2 * sum_{i} diff[i,j,k] * U[i,j,r] * W[k,i,r]
            # dW[k,i,r] = 2 * sum_{j} diff[i,j,k] * U[i,j,r] * V[j,k,r]
            
            dU = 2 * np.einsum('ijk,jkr,kir->ijr', diff, V, W)
            dV = 2 * np.einsum('ijk,ijr,kir->jkr', diff, U, W)
            dW = 2 * np.einsum('ijk,ijr,jkr->kir', diff, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if abs(prev_loss - loss) < 1e-12:
                break
            prev_loss = loss
            
        if loss < 1e-9:
            return U, V, W
        return None

    # Try rank 10
    sol = None
    for r in [10, 11, 12]:
        if time.time() > deadline - 10:
            break
        sol = find_2x2x5_decomposition(r)
        if sol:
            break
            
    if sol:
        U1, V1, W1 = sol
        rank_2x2x5 = U1.shape[2]
        total_rank = 2 * rank_2x2x5
        
        # Construct 2x4x5
        n, m, p = 2, 4, 5
        U = np.zeros((n*m, total_rank))
        V = np.zeros((m*p, total_rank))
        W = np.zeros((p*n, total_rank))
        
        # Block 1
        # U1 is (2,2,r). Flatten to (4,r).
        # Index i*2+j
        U[0:4, 0:rank_2x2x5] = U1.reshape(4, rank_2x2x5)
        # V1 is (2,5,r). Flatten to (10,r).
        # Index j*5+k
        V[0:10, 0:rank_2x2x5] = V1.reshape(10, rank_2x2x5)
        # W1 is (5,2,r). Flatten to (10,r).
        # Index k*2+i
        W[0:10, 0:rank_2x2x5] = W1.reshape(10, rank_2x2x5)
        
        # Block 2
        # Same matrices, shifted columns
        U[4:8, rank_2x2x5:total_rank] = U1.reshape(4, rank_2x2x5)
        V[10:20, rank_2x2x5:total_rank] = V1.reshape(10, rank_2x2x5)
        W[0:10, rank_2x2x5:total_rank] = W1.reshape(10, rank_2x2x5)
        
        # Snap to half-integers
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        # Verify
        # ...
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Schoolbook
        n, m, p = 2, 4, 5
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
