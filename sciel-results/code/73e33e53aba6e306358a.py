"""Matrix multiplication 3x4x6 rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 4, 6]

def _compute_tensor(U, V, W, n, m, p, rank):
    """Compute T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]"""
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        for j in range(m):
            T[:, j, :] += np.outer(u_c[:, j], v_c[j, :]) * w_c[:, j]
    return T

def _compute_loss_and_grad(U, V, W, n, m, p, rank):
    """Compute loss and gradients for the tensor decomposition."""
    T = np.zeros((n, m, p))
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        for j in range(m):
            T[:, j, :] += np.outer(u_c[:, j], v_c[j, :]) * w_c[:, j]
    
    diff = T - 1.0
    loss = np.sum(diff ** 2)
    
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)
    dW = np.zeros_like(W)
    
    for c in range(rank):
        u_c = U[:, c].reshape(n, m)
        v_c = V[:, c].reshape(m, p)
        w_c = W[:, c].reshape(p, n)
        for j in range(m):
            for i in range(n):
                dU[i * m + j, c] = 2.0 * np.sum(diff[i, j, :] * v_c[j, :] * w_c[:, i])
                for k in range(p):
                    dV[j * p + k, c] += 2.0 * diff[i, j, k] * u_c[i, j] * w_c[k, i]
                    dW[k * n + i, c] += 2.0 * diff[i, j, k] * u_c[i, j] * v_c[j, k]
    
    return loss, dU, dV, dW

def _try_rationalize(U, V, W, n, m, p, rank):
    """Try to find rational representation with small denominators."""
    for denom in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50, 60, 100]:
        U_r = np.round(U * denom) / denom
        V_r = np.round(V * denom) / denom
        W_r = np.round(W * denom) / denom
        T = _compute_tensor(U_r, V_r, W_r, n, m, p, rank)
        if np.max(np.abs(T - 1.0)) < 1e-6:
            return U_r, V_r, W_r
    return None

def _format_result(U, V, W):
    """Convert to required output format."""
    U_list = []
    for row in U:
        r = []
        for x in row:
            f = Fraction(x).limit_denominator(1000000)
            if f.denominator == 1:
                r.append(int(f.numerator))
            else:
                r.append(f)
        U_list.append(r)
    V_list = []
    for row in V:
        r = []
        for x in row:
            f = Fraction(x).limit_denominator(1000000)
            if f.denominator == 1:
                r.append(int(f.numerator))
            else:
                r.append(f)
        V_list.append(r)
    W_list = []
    for row in W:
        r = []
        for x in row:
            f = Fraction(x).limit_denominator(1000000)
            if f.denominator == 1:
                r.append(int(f.numerator))
            else:
                r.append(f)
        W_list.append(r)
    return {"U": U_list, "V": V_list, "W": W_list}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    np.random.seed(12345)
    
    # Try ranks from low to high
    for rank in range(48, 73):
        if time.time() > deadline - 15:
            break
        
        n_restarts = 4
        found = False
        
        for restart in range(n_restarts):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Scale each column so product of norms ~ 1
            for c in range(rank):
                nu = np.sqrt(np.sum(U[:, c] ** 2))
                nv = np.sqrt(np.sum(V[:, c] ** 2))
                nw = np.sqrt(np.sum(W[:, c] ** 2))
                prod = max(nu * nv * nw, 1e-10)
                scale = (1.0 / prod) ** (1.0 / 3.0)
                U[:, c] *= scale
                V[:, c] *= scale
                W[:, c] *= scale
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            best_loss = np.inf
            patience = 0
            
            for step in range(3000):
                if time.time() > deadline - 3:
                    break
                
                loss, dU, dV, dW = _compute_loss_and_grad(U, V, W, n, m, p, rank)
                
                if loss < best_loss:
                    best_loss = loss
                    patience = 0
                else:
                    patience += 1
                
                if patience > 300:
                    break
                if loss < 1e-12:
                    break
                
                # Adaptive LR
                if patience > 50:
                    lr *= 0.9
                elif patience == 0 and lr < 0.2:
                    lr = min(lr * 1.05, 0.2)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Final check
            T = _compute_tensor(U, V, W, n, m, p, rank)
            final_loss = np.sum((T - 1.0) ** 2)
            
            if final_loss < 1e-8:
                # Try rationalization
                rounded = _try_rationalize(U, V, W, n, m, p, rank)
                if rounded:
                    U, V, W = rounded
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    return _format_result(U, V, W)
                else:
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    return _format_result(U, V, W)
        
        if found:
            break
    
    return _format_result(best_U, best_V, best_W)
