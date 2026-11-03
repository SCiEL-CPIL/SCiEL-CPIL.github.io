import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    num_u = n * m  # 9
    num_v = m * p  # 15
    num_w = p * n  # 15
    
    def get_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(num_u)]
        V = [[0] * rank for _ in range(num_v)]
        W = [[0] * rank for _ in range(num_w)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W):
        """Verify that the decomposition exactly computes matrix multiplication.
        Returns True if valid, False otherwise."""
        rank = len(U[0])
        
        # Check all (i, j, k) combinations
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # The term U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] should sum to 1
                    total = 0
                    for r in range(rank):
                        total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                    if total != 1:
                        return False
        
        # Check that no spurious terms exist
        # For any (i, j, k) and (i', k') where (i', k') != (i, k),
        # the sum should be 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for k2 in range(p):
                            if i2 == i and k2 == k:
                                continue
                            total = 0
                            for r in range(rank):
                                total += U[i * m + j][r] * V[j * p + k][r] * W[k2 * n + i2][r]
                            if total != 0:
                                return False
        return True

    def snap_to_rational(values, max_denom=8):
        """Snap values to nearest fraction with small denominator."""
        snapped = []
        for v in values:
            best_frac = Fraction(0)
            best_dist = float('inf')
            for denom in range(1, max_denom + 1):
                num = round(v * denom)
                frac = Fraction(num, denom)
                dist = abs(v - float(frac))
                if dist < best_dist:
                    best_dist = dist
                    best_frac = frac
            snapped.append(best_frac)
        return snapped

    def search_rank(R, max_iterations=2000):
        """Try to find a rank-R decomposition using vectorized gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        # Precompute index arrays for vectorization
        # We need to map (i, j, k) to flat indices for U, V, W
        # U_idx[i, j] = i * m + j
        # V_idx[j, k] = j * p + k
        # W_idx[k, i] = k * n + i
        
        # Target tensor T of shape (n, m, p, n) ?
        # No, the condition is:
        # sum_r U[i,j,r] V[j,k,r] W[k',i',r] = 1 if k==k' and i==i' else 0.
        # This is equivalent to:
        # For each r, let U_r be vec of U[:,r], V_r, W_r.
        # The tensor is sum_r U_r (outer) V_r (outer) W_r.
        # We want this tensor to match T.
        
        # Let's construct the target tensor T_flat of size num_u * num_v * num_w?
        # No, T is indexed by (i, j, k', i').
        # T[i, j, k', i'] = 1 if k==k' and i==i' ?
        # Wait, the sum is over r. The free indices are i, j, k (from V) and k', i' (from W).
        # The condition is on the sum.
        # Let's define the error as sum over all i,j,k,k',i' of (Sum_r ... - Target)^2.
        # Target[i, j, k, k', i'] = 1 if k==k' and i==i' else 0.
        
        # We can vectorize the "Sum_r" part.
        # Let U be (num_u, R), V be (num_v, R), W be (num_w, R).
        # We want to compute M[i, j, k, k', i'] = sum_r U[i*m+j, r] V[j*p+k, r] W[k'*n+i', r].
        
        # This is a 5D tensor.
        # We can reshape U to (n, m, R), V to (m, p, R), W to (p, n, R).
        # Then M[i, j, k, k', i'] = sum_r U[i,j,r] V[j,k,r] W[k',i',r].
        
        # We can use np.einsum:
        # 'ijr,jkr,kir -> ijkkii' ? No.
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        # We want to contract r.
        # Result shape: (n, m, p, p, n) corresponding to (i, j, k, k', i').
        # Let's call the indices i, j, k for U, V, W parts?
        # U index: i, j.
        # V index: j, k.
        # W index: k', i'.
        # So we sum over r.
        # M = np.einsum('ijr,jkr,kir->ijkkii', U_reshaped, V_reshaped, W_reshaped)
        # Wait, W is indexed by k', i'. So W_reshaped is (p, n, R).
        # The indices in einsum:
        # U: i, j, r
        # V: j, k, r
        # W: k', i', r  -> let's use k2, i2.
        # M[i, j, k, k2, i2] = sum_r U[i,j,r] V[j,k,r] W[k2,i2,r].
        
        # Target T[i, j, k, k2, i2] = 1 if k==k2 and i==i2 else 0.
        
        # We can create T as a numpy array.
        
        # Gradient Descent:
        # Error E = sum (M - T)^2.
        # dE/dU[i,j,r] = 2 * sum_{j,k,k2,i2} (M[i,j,k,k2,i2] - T[i,j,k,k2,i2]) * V[j,k,r] * W[k2,i2,r].
        
        # This can be computed efficiently.
        
        for attempt in range(10):
            if time.time() > deadline - 10:
                return None
            
            U = np.random.uniform(-1, 1, (num_u, R))
            V = np.random.uniform(-1, 1, (num_v, R))
            W = np.random.uniform(-1, 1, (num_w, R))
            
            U_reshaped = U.reshape(n, m, R)
            V_reshaped = V.reshape(m, p, R)
            W_reshaped = W.reshape(p, n, R)
            
            # Target T
            # T[i, j, k, k2, i2]
            T = np.zeros((n, m, p, p, n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k, k, i] = 1.0
                        
            best_error = float('inf')
            
            for iteration in range(max_iterations):
                if time.time() > deadline - 10:
                    break
                
                # Compute M
                M = np.einsum('ijr,jkr,k2ir->ijkk2i2', U_reshaped, V_reshaped, W_reshaped)
                
                error = np.sum((M - T) ** 2)
                
                if error < best_error:
                    best_error = error
                    if error < 1e-10:
                        # Snap and verify
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        U_snapped = []
                        for row in U_list:
                            U_snapped.append([float(x) for x in snap_to_rational(row)])
                        V_snapped = []
                        for row in V_list:
                            V_snapped.append([float(x) for x in snap_to_rational(row)])
                        W_snapped = []
                        for row in W_list:
                            W_snapped.append([float(x) for x in snap_to_rational(row)])
                        
                        U_exact = [[Fraction(str(x)) for x in row] for row in U_snapped]
                        V_exact = [[Fraction(str(x)) for x in row] for row in V_snapped]
                        W_exact = [[Fraction(str(x)) for x in row] for row in W_snapped]
                        
                        if verify_decomposition(U_exact, V_exact, W_exact):
                            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
                
                # Compute Gradients
                # dU[i,j,r] = 2 * sum_{k,k2,i2} (M[i,j,k,k2,i2] - T[i,j,k,k2,i2]) * V[j,k,r] * W[k2,i2,r]
                # Let D = M - T. Shape (n, m, p, p, n).
                D = M - T
                
                # grad_U[i, j, r] = sum_{k, k2, i2} D[i, j, k, k2, i2] * V[j, k, r] * W[k2, i2, r]
                # We can use einsum again.
                # V is (m, p, R), W is (p, n, R).
                # We want to contract k, k2, i2.
                # grad_U = 2 * np.einsum('ijkk2i2,jkr,k2ir->ijr', D, V_reshaped, W_reshaped)
                
                # grad_V[j, k, r] = sum_{i, k2, i2} D[i, j, k, k2, i2] * U[i, j, r] * W[k2, i2, r]
                # grad_V = 2 * np.einsum('ijkk2i2,ijr,k2ir->jkr', D, U_reshaped, W_reshaped)
                
                # grad_W[k2, i2, r] = sum_{i, j, k} D[i, j, k, k2, i2] * U[i, j, r] * V[j, k, r]
                # grad_W = 2 * np.einsum('ijkk2i2,ijr,jkr->k2ir', D, U_reshaped, V_reshaped)
                
                lr = 0.01
                U_reshaped -= lr * np.einsum('ijkk2i2,jkr,k2ir->ijr', D, V_reshaped, W_reshaped)
                V_reshaped -= lr * np.einsum('ijkk2i2,ijr,k2ir->jkr', D, U_reshaped, W_reshaped)
                W_reshaped -= lr * np.einsum('ijkk2i2,ijr,jkr->k2ir', D, U_reshaped, V_reshaped)
                
                # Reshape back to U, V, W
                U = U_reshaped.reshape(num_u, R)
                V = V_reshaped.reshape(num_v, R)
                W = W_reshaped.reshape(num_w, R)
                
                # Noise
                if iteration % 50 == 0:
                    U += np.random.normal(0, 0.01, U.shape)
                    V += np.random.normal(0, 0.01, V.shape)
                    W += np.random.normal(0, 0.01, W.shape)
                    U_reshaped = U.reshape(n, m, R)
                    V_reshaped = V.reshape(m, p, R)
                    W_reshaped = W.reshape(p, n, R)

        return None

    # Try ranks from low to high
    best_result = None
    for R in range(22, 46):
        if time.time() > deadline - 15:
            break
        result = search_rank(R, max_iterations=1000)
        if result is not None:
            best_result = result
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return get_schoolbook()

# EVOLVE-BLOCK-END
