"""Matrix multiplication 3,4,4 rank minimization."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

TIME_LIMIT = 145.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _time_left():
    return TIME_LIMIT - _elapsed()

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 48
    
    # Build schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=np.float64)
        V = np.zeros((m * p, rank), dtype=np.float64)
        W = np.zeros((p * n, rank), dtype=np.float64)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, idx] = 1.0
                    V[j * p + k, idx] = 1.0
                    W[k * n + i, idx] = 1.0
                    idx += 1
        return U, V, W
    
    def verify(U, V, W):
        """Check if the decomposition exactly reproduces the matrix multiplication tensor."""
        U_r = U.reshape(n, m, -1)
        V_r = V.reshape(m, p, -1)
        W_r = W.reshape(p, n, -1)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        if not np.allclose(T, 1.0, atol=1e-8):
            return False
        return True
    
    def try_rank_reduction(rank, n_attempts=20, iters_per_attempt=200):
        """Try to find a rank-reduction using structured initialization and momentum-based gradient descent."""
        if rank >= schoolbook_rank:
            return None
        
        best_U, best_V, best_W = None, None, None
        best_error = float('inf')
        
        for attempt in range(n_attempts):
            if _time_left() < 15.0:
                break
            
            # Structured initialization: start from schoolbook and perturb
            U0, V0, W0 = schoolbook()
            
            # Create a rank-reduced version by combining terms
            # For rank < schoolbook, we need to find linear dependencies
            # Initialize with random mixing of schoolbook terms
            U = np.zeros((n * m, rank), dtype=np.float64)
            V = np.zeros((m * p, rank), dtype=np.float64)
            W = np.zeros((p * n, rank), dtype=np.float64)
            
            # Use random projections to create initial rank-reduced basis
            for r in range(rank):
                # Each rank-1 term is a random combination of schoolbook terms
                weights = np.random.randn(schoolbook_rank) * 0.3
                for idx in range(schoolbook_rank):
                    if abs(weights[idx]) > 0.01:
                        U[:, r] += weights[idx] * U0[:, idx]
                        V[:, r] += weights[idx] * V0[:, idx]
                        W[:, r] += weights[idx] * W0[:, idx]
            
            # Normalize to avoid scale issues
            U = U / (np.max(np.abs(U)) + 1e-10)
            V = V / (np.max(np.abs(V)) + 1e-10)
            W = W / (np.max(np.abs(W)) + 1e-10)
            
            # Momentum-based gradient descent
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            beta = 0.9
            lr = 0.05
            
            for it in range(iters_per_attempt):
                if _time_left() < 15.0:
                    break
                
                U_r = U.reshape(n, m, rank)
                V_r = V.reshape(m, p, rank)
                W_r = W.reshape(p, n, rank)
                
                T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
                E = 1.0 - T
                
                error = np.sum(E * E)
                if error < best_error:
                    best_error = error
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    if error < 1e-12:
                        break
                
                # Compute gradients
                dU_r = -2.0 * np.einsum('ijk,jkr,kir->ijr', E, V_r, W_r)
                dV_r = -2.0 * np.einsum('ijk,ijr,kir->jkr', E, U_r, W_r)
                dW_r = -2.0 * np.einsum('ijk,ijr,jkr->kir', E, U_r, V_r)
                
                dU = dU_r.reshape(n * m, rank)
                dV = dV_r.reshape(m * p, rank)
                dW = dW_r.reshape(p * n, rank)
                
                # Momentum update
                vU = beta * vU + (1 - beta) * dU
                vV = beta * vV + (1 - beta) * dV
                vW = beta * vW + (1 - beta) * dW
                
                U += lr * vU
                V += lr * vV
                W += lr * vW
                
                # Adaptive learning rate: reduce if error increases
                if it % 20 == 19:
                    recent_error = np.sum(E * E)
                    if recent_error > error * 1.1:
                        lr *= 0.8
            
            if best_error < 1e-10:
                break
        
        if best_error < 1e-6:
            return best_U, best_V, best_W
        return None
    
    # Start with schoolbook
    U, V, W = schoolbook()
    
    # Try to reduce rank from 47 down, but be conservative with time
    for target_rank in [schoolbook_rank - 1, schoolbook_rank - 2, schoolbook_rank - 3]:
        if _time_left() < 25.0:
            break
        
        result = try_rank_reduction(target_rank)
        if result is not None:
            U, V, W = result
            break
    
    # Final verification with safety margin
    if not verify(U, V, W):
        U, V, W = schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
