import time
import numpy as np
import random
from itertools import product

DIMENSIONS = [3, 3, 4]
START_TIME = time.time()
DEADLINE = START_TIME + 140  # 10s safety margin before 150s limit

def construct():
    n, m, p = DIMENSIONS
    U_rows = n * m  # 9
    V_rows = m * p  # 12
    W_rows = p * n  # 12
    
    # Schoolbook decomposition (rank 36)
    def schoolbook():
        U = np.zeros((U_rows, n*m*p))
        V = np.zeros((V_rows, n*m*p))
        W = np.zeros((W_rows, n*m*p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    def verify(U, V, W, tol=1e-8):
        """Verify that the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    s = np.sum(prod)
                    if abs(s - 1.0) > tol:
                        return False
        return True
    
    def verify_exact(U, V, W):
        """Verify with exact rational arithmetic using fractions."""
        from fractions import Fraction
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(len(U[0])):
                        s += Fraction(U[i*m+j, c]).limit_denominator(1000) \
                            * Fraction(V[j*p+k, c]).limit_denominator(1000) \
                            * Fraction(W[k*n+i, c]).limit_denominator(1000)
                    if s != 1:
                        return False
        return True
    
    def objective(U, V, W):
        """Compute the squared error of the decomposition."""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                    s = np.sum(prod)
                    err += (s - 1.0) ** 2
        return err
    
    def gradient_descent(U, V, W, lr=0.01, max_iter=5000, tol=1e-10):
        """Perform gradient descent to minimize the decomposition error."""
        U = U.copy()
        V = V.copy()
        W = W.copy()
        r = len(U[0])
        
        prev_err = objective(U, V, W)
        
        for it in range(max_iter):
            if time.time() > DEADLINE - 5:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        prod = U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                        s = np.sum(prod) - 1.0
                        # d/dU[i*m+j,c] = 2*s * V[j*p+k,c] * W[k*n+i,c]
                        dU[i*m+j, :] += 2 * s * V[j*p+k, :] * W[k*n+i, :]
                        dV[j*p+k, :] += 2 * s * U[i*m+j, :] * W[k*n+i, :]
                        dW[k*n+i, :] += 2 * s * U[i*m+j, :] * V[j*p+k, :]
            
            # Update
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            cur_err = objective(U, V, W)
            
            if abs(prev_err - cur_err) < tol:
                break
            prev_err = cur_err
            
            # Adaptive learning rate
            if cur_err > prev_err * 1.01:
                lr *= 0.5
            elif it % 100 == 0 and it > 0:
                lr *= 1.1  # Slightly increase if making progress
        
        return U, V, W, cur_err
    
    def snap_to_half(U, V, W):
        """Snap values to nearest half-integer."""
        U = np.round(2 * U) / 2
        V = np.round(2 * V) / 2
        W = np.round(2 * W) / 2
        return U, V, W
    
    def polish(U, V, W, max_iter=1000):
        """High-precision polish with very small learning rate."""
        U, V, W, err = gradient_descent(U, V, W, lr=0.001, max_iter=max_iter, tol=1e-12)
        return U, V, W, err
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 36
    
    # Try to find lower-rank decompositions
    # Strategy: try ranks from 30 down to 25
    # For each rank, try multiple random initializations
    
    ranks_to_try = list(range(30, 24, -1))
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 10:
            break
        
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and time.time() < DEADLINE - 10:
            attempts += 1
            
            # Random initialization
            np.random.seed(attempts * 7919 + target_rank)
            
            U = np.random.randn(U_rows, target_rank) * 0.5
            V = np.random.randn(V_rows, target_rank) * 0.5
            W = np.random.randn(W_rows, target_rank) * 0.5
            
            # Also try structured initialization based on schoolbook
            if attempts % 2 == 0:
                # Use a subset of schoolbook terms
                U = np.zeros((U_rows, target_rank))
                V = np.zeros((V_rows, target_rank))
                W = np.zeros((W_rows, target_rank))
                
                # Pick random (i,j,k) combinations
                indices = []
                used = set()
                for _ in range(target_rank):
                    while True:
                        i = random.randint(0, n-1)
                        j = random.randint(0, m-1)
                        k = random.randint(0, p-1)
                        if (i,j,k) not in used:
                            used.add((i,j,k))
                            break
                    indices.append((i,j,k))
                
                for col, (i,j,k) in enumerate(indices):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                # Add some noise
                U += np.random.randn(U_rows, target_rank) * 0.1
                V += np.random.randn(V_rows, target_rank) * 0.1
                W += np.random.randn(W_rows, target_rank) * 0.1
            
            # Run gradient descent
            U_opt, V_opt, W_opt, err = gradient_descent(U, V, W, lr=0.05, max_iter=3000)
            
            # Check if we found a good solution
            if err < 1e-6:
                # Polish
                U_pol, V_pol, W_pol, err_pol = polish(U_opt, V_opt, W_opt, max_iter=2000)
                
                if err_pol < 1e-8:
                    # Try to snap to half-integers
                    U_snap, V_snap, W_snap = snap_to_half(U_pol, V_pol, W_pol)
                    
                    if verify(U_snap, V_snap, W_snap, tol=1e-6):
                        # Found a valid decomposition!
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = target_rank
                        break  # Move to next (lower) rank
                    
                    # If snapping didn't work, keep the floating point solution
                    # but we need integer/half-integer entries, so try harder
                    # Try different snapping strategies
                    for scale in [1, 2, 4]:
                        U_s = np.round(U_pol * scale) / scale
                        V_s = np.round(V_pol * scale) / scale
                        W_s = np.round(W_pol * scale) / scale
                        if verify(U_s, V_s, W_s, tol=1e-6):
                            best_U, best_V, best_W = U_s, V_s, W_s
                            best_rank = target_rank
                            break
                    else:
                        # Keep floating point, will need to handle conversion
                        best_U, best_V, best_W = U_pol, V_pol, W_pol
                        best_rank = target_rank
                    break
            elif err < 0.01 and target_rank < best_rank:
                # Good progress but not quite there, try to polish more
                U_pol, V_pol, W_pol, err_pol = polish(U_opt, V_opt, W_opt, max_iter=5000)
                if err_pol < 1e-8:
                    U_snap, V_snap, W_snap = snap_to_half(U_pol, V_pol, W_pol)
                    if verify(U_snap, V_snap, W_snap, tol=1e-6):
                        best_U, best_V, best_W = U_snap, V_snap, W_snap
                        best_rank = target_rank
                        break
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-6):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Convert to lists with proper rounding
    U_out = np.round(best_U * 2) / 2
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    # Final check
    if not verify(U_out, V_out, W_out, tol=1e-6):
        U_out = np.round(best_U)  # Try integer rounding
        V_out = np.round(best_V)
        W_out = np.round(best_W)
    
    if not verify(U_out, V_out, W_out, tol=1e-6):
        # Fall back to schoolbook
        U_sb, V_sb, W_sb = schoolbook()
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
