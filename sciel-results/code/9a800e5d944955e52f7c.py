"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
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


def _verify(U, V, W, n, m, p):
    """Check exact decomposition: C[i,k] = sum_j A[i,j]B[j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    We check that this equals 1 if (i,j,k) is a valid index, else 0.
    But actually the tensor for matmul is T[i,j,k] = 1 for all valid i,j,k since
    C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i in [n], j in [m], k in [p].
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # Check all indices exactly
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(val - 1.0) > 1e-6:
                    return False
    return True


def _compute_residuals(U, V, W, n, m, p, rank):
    """Compute residuals R[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] - 1"""
    residuals = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                residuals[i, j, k] = np.dot(u, v * w) - 1.0
    return residuals


def _compute_gradients(U, V, W, n, m, p, rank):
    """Compute gradients for the loss function sum R^2"""
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U[i * m + j, :]
                v = V[j * p + k, :]
                w = W[k * n + i, :]
                R = np.dot(u, v * w) - 1.0
                
                # dL/dU = 2*R * (v * w)
                dU[i * m + j, :] += 2 * R * v * w
                # dL/dV = 2*R * (u * w)
                dV[j * p + k, :] += 2 * R * u * w
                # dL/dW = 2*R * (u * v)
                dW[k * n + i, :] += 2 * R * u * v
    
    return dU, dV, dW


def _try_rank(rank, n, m, p, deadline, max_restarts=5):
    """Try to find a rank decomposition using numerical optimization."""
    if time.time() > deadline:
        return None
    
    best_result = None
    best_loss = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Initialize with random values
        U = np.random.randn(n * m, rank) * 0.5
        V = np.random.randn(m * p, rank) * 0.5
        W = np.random.randn(p * n, rank) * 0.5
        
        # Adaptive learning rate
        lr = 0.01
        best_loss_local = float('inf')
        patience = 0
        
        for step in range(2000):
            if time.time() > deadline:
                break
            
            # Compute gradients
            dU, dV, dW = _compute_gradients(U, V, W, n, m, p, rank)
            
            # Update parameters
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Compute loss
            residuals = _compute_residuals(U, V, W, n, m, p, rank)
            loss = np.sum(residuals ** 2)
            
            if loss < best_loss_local:
                best_loss_local = loss
                patience = 0
                
                if best_loss_local < 1e-8:
                    # Found a solution, try to snap to nice values
                    U_snap = np.round(U * 2) / 2
                    V_snap = np.round(V * 2) / 2
                    W_snap = np.round(W * 2) / 2
                    
                    U_list = U_snap.tolist()
                    V_list = V_snap.tolist()
                    W_list = W_snap.tolist()
                    
                    if _verify(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
            else:
                patience += 1
                if patience > 100:
                    # Reduce learning rate
                    lr *= 0.5
                    patience = 0
            
            # Adaptive learning rate decay
            if step % 500 == 0 and step > 0:
                lr *= 0.9
        
        if best_loss_local < best_loss:
            best_loss = best_loss_local
            best_result = (U.copy(), V.copy(), W.copy())
    
    # Final polish: if we got close, try to refine
    if best_result is not None and best_loss < 1e-4:
        U, V, W = best_result
        # High-precision polish
        for step in range(500):
            if time.time() > deadline:
                break
            
            dU, dV, dW = _compute_gradients(U, V, W, n, m, p, rank)
            U -= 0.001 * dU
            V -= 0.001 * dV
            W -= 0.001 * dW
            
            residuals = _compute_residuals(U, V, W, n, m, p, rank)
            loss = np.sum(residuals ** 2)
            
            if loss < 1e-10:
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                
                U_list = U_snap.tolist()
                V_list = V_snap.tolist()
                W_list = W_snap.tolist()
                
                if _verify(U_list, V_list, W_list, n, m, p):
                    return {"U": U_list, "V": V_list, "W": W_list}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # First, try to find a better decomposition
    # The minimum rank for 2x3 * 3x4 is at least max(n*m, m*p, p*n) = max(6, 12, 8) = 12
    # Schoolbook is 24. Let's try ranks from 12 upward.
    
    best_result = _schoolbook()
    
    # Try various ranks
    for rank in range(12, 24):
        if time.time() > deadline:
            break
        result = _try_rank(rank, n, m, p, deadline)
        if result is not None:
            return result
    
    return best_result
# EVOLVE-BLOCK-END
