import numpy as np
import time
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    r_school = n * m * p
    
    # Time limit with safety margin
    start_time = time.time()
    max_time = 130  # Evaluator kills at 150s, leave 20s margin
    
    # Schoolbook fallback
    U_school = [[0] * r_school for _ in range(n * m)]
    V_school = [[0] * r_school for _ in range(m * p)]
    W_school = [[0] * r_school for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    fallback = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = r_school
    best_result = fallback
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs T[i,j,k]=1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m + j][r] * V[j*p + k][r] * W[k*n + i][r]
                    if total != 1:
                        return False
        return True
    
    def verify_decomposition_fast(U, V, W, rank):
        """Faster verification using numpy"""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Reshape
        U_3d = U_arr.reshape(n, m, rank)
        V_3d = V_arr.reshape(m, p, rank)
        W_3d = W_arr.reshape(p, n, rank)
        
        # Compute reconstruction
        Rec = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        
        # Check if all entries are 1
        return np.allclose(Rec, 1.0, atol=1e-10)
    
    def try_rank(r, max_restarts=10):
        """Try to find a rank-r decomposition"""
        if time.time() - start_time > max_time:
            return None
        
        for restart in range(max_restarts):
            if time.time() - start_time > max_time:
                return None
            
            # Strategy: Use structured random initialization with some zeros
            # to encourage sparse solutions that might be exact
            
            # Initialize with small random values, but set some to zero
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Add some structure: for each r, try to make one of U,V,W dominant
            for r_idx in range(r):
                choice = np.random.choice(3)
                if choice == 0:
                    U[:, r_idx] *= 2
                    V[:, r_idx] *= 0.5
                    W[:, r_idx] *= 0.5
                elif choice == 1:
                    U[:, r_idx] *= 0.5
                    V[:, r_idx] *= 2
                    W[:, r_idx] *= 0.5
                else:
                    U[:, r_idx] *= 0.5
                    V[:, r_idx] *= 0.5
                    W[:, r_idx] *= 2
            
            # Gradient descent with adaptive learning rate
            lr = 0.05
            prev_loss = float('inf')
            
            for step in range(200):
                if time.time() - start_time > max_time:
                    return None
                
                # Compute reconstruction
                U_3d = U.reshape(n, m, r)
                V_3d = V.reshape(m, p, r)
                W_3d = W.reshape(p, n, r)
                
                Rec = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
                Error = 1.0 - Rec
                loss = np.sum(Error**2)
                
                # Check convergence
                if abs(prev_loss - loss) < 1e-15:
                    break
                prev_loss = loss
                
                # Compute gradients
                # dL/dU[i,j,r] = -2 * sum_{j',k'} Error[i,j',k'] * V[j',k',r] * W[k',i,r]
                # More carefully:
                # Rec[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
                # dRec[i,j,k]/dU[i,j,r] = V[j,k,r] * W[k,i,r]
                # dL/dU[i,j,r] = -2 * sum_{j,k} Error[i,j,k] * V[j,k,r] * W[k,i,r]
                
                # For U:
                grad_U = np.zeros_like(U)
                for r_idx in range(r):
                    # For each (i, j), grad_U[i,j,r_idx] = -2 * sum_k Error[i,j,k] * V[j,k,r_idx] * W[k,i,r_idx]
                    for i in range(n):
                        for j in range(m):
                            s = 0
                            for k in range(p):
                                s += Error[i, j, k] * V_3d[j, k, r_idx] * W_3d[k, i, r_idx]
                            grad_U[i*m + j, r_idx] = -2 * s
                
                # For V:
                grad_V = np.zeros_like(V)
                for r_idx in range(r):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for i in range(n):
                                s += Error[i, j, k] * U_3d[i, j, r_idx] * W_3d[k, i, r_idx]
                            grad_V[j*p + k, r_idx] = -2 * s
                
                # For W:
                grad_W = np.zeros_like(W)
                for r_idx in range(r):
                    for k in range(p):
                        for i in range(n):
                            s = 0
                            for j in range(m):
                                s += Error[i, j, k] * U_3d[i, j, r_idx] * V_3d[j, k, r_idx]
                            grad_W[k*n + i, r_idx] = -2 * s
                
                # Update
                U -= lr * grad_U
                V -= lr * grad_V
                W -= lr * grad_W
                
                # Adaptive learning rate
                if loss < prev_loss:
                    lr = min(lr * 1.05, 0.5)
                else:
                    lr = max(lr * 0.8, 0.001)
            
            # After gradient descent, try to round to half-integers
            # and verify
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify_decomposition_fast(U_rounded, V_rounded, W_rounded, r):
                return {
                    "U": U_rounded.tolist(),
                    "V": V_rounded.tolist(),
                    "W": W_rounded.tolist()
                }
            
            # Also try integer rounding
            U_int = np.round(U)
            V_int = np.round(V)
            W_int = np.round(W)
            
            if verify_decomposition_fast(U_int, V_int, W_int, r):
                return {
                    "U": U_int.tolist(),
                    "V": V_int.tolist(),
                    "W": W_int.tolist()
                }
            
            # Try the unrounded version if it's close enough
            if verify_decomposition_fast(U, V, W, r):
                # Convert to fraction representation
                U_frac = [[float(x) for x in row] for row in U]
                V_frac = [[float(x) for x in row] for row in V]
                W_frac = [[float(x) for x in row] for row in W]
                return {
                    "U": U_frac,
                    "V": V_frac,
                    "W": W_frac
                }
        
        return None
    
    # Try ranks in increasing order
    target_ranks = [35, 38, 40, 42, 45, 48, 50, 52, 55, r_school]
    
    for r in target_ranks:
        if time.time() - start_time > max_time:
            break
        
        if r >= best_rank:
            continue
        
        result = try_rank(r, max_restarts=5)
        if result is not None:
            best_rank = r
            best_result = result
            # If we found a good rank, we can stop or continue
            # For now, continue to try lower ranks
            if r <= 40:
                break
    
    return best_result
