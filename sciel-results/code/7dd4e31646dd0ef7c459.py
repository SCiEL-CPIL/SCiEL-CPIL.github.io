import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    # Build the target tensor T
    # T is a 3D array of shape (n*m, m*p, p*n)
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try to find a lower rank decomposition
    # Known bound for 2x4x6 is 25
    target_rank = 25
    
    start_time = time.time()
    deadline = 120.0 # Leave margin for evaluation overhead
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    
    # Numerical Search
    # We use a simple gradient descent / alternating least squares style approach
    # Minimize || sum u_r v_r w_r^T - T ||_F^2
    
    for attempt in range(5): # 5 restarts
        if time.time() > deadline:
            break
            
        U = np.random.randn(n * m, target_rank)
        V = np.random.randn(m * p, target_rank)
        W = np.random.randn(p * n, target_rank)
        
        # Normalize to prevent scale issues
        for r in range(target_rank):
            U[:, r] /= np.linalg.norm(U[:, r])
            V[:, r] /= np.linalg.norm(V[:, r])
            W[:, r] /= np.linalg.norm(W[:, r])
        
        for step in range(500):
            if time.time() > deadline:
                break
                
            # Compute current tensor
            # T_current = sum_r U[:, r] V[:, r] W[:, r]^T
            # This is expensive to compute fully every step.
            # We can use a trick: for each r, update one factor.
            
            # Update U
            # T = U V W^T
            # For each r, U[:, r] should be the projection of (T - sum_{s!=r} U[:,s] V[:,s] W[:,s]^T) onto span(V[:,r], W[:,r])?
            # Actually, we can fix V and W, and solve for U.
            # T = sum_r u_r v_r^T w_r^T? No, T is 3D.
            # T_{abc} = sum_r U_{ar} V_{br} W_{cr}
            
            # Residual R = T - T_current
            # We want to minimize ||R||^2.
            # This is a tensor completion / decomposition problem.
            
            # Simple Gradient Step:
            # dU = R * (V W^T)
            # dV = R * (U W^T)
            # dW = R * (U V^T)
            
            # Compute T_current
            T_current = np.tensordot(U, np.tensordot(V, W, axes=1), axes=1)
            # Note: np.tensordot(U, V, axes=1) gives shape (n*m, p*n, rank)
            # Then tensordot with W (p*n, rank) on axis 1 gives (n*m, p*n, p*n)? No.
            
            # Correct tensor contraction:
            # T_current[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
            # We can compute this as:
            # Temp = U * V (broadcasting on r) -> (n*m, m*p, rank)
            # T_current = Temp * W (broadcasting on r) -> (n*m, m*p, p*n)
            
            # This is O(n*m*m*p*rank + n*m*m*p*p*n) which is large but rank=25 is small.
            # n*m=8, m*p=24, p*n=12.
            # 8*24*25 = 4800 mults.
            # 8*24*12*25 = 57600 mults.
            # Very fast.
            
            T_current = np.einsum('ar,br,cr->abc', U, V, W)
            R = T - T_current
            
            # Gradient
            dU = np.einsum('abc,br,cr->ar', R, V, W)
            dV = np.einsum('abc,ar,cr->br', R, U, W)
            dW = np.einsum('abc,ar,br->cr', R, U, V)
            
            # Learning rate
            lr = 0.01
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            # Check for exactness periodically
            if step % 50 == 0:
                # Round to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                T_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_check, T, atol=1e-9):
                    # Found exact decomposition
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }
            
            # If error is very small, try harder rounding
            if np.linalg.norm(R) < 1e-3:
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                T_check = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
                if np.allclose(T_check, T, atol=1e-9):
                    return {
                        "U": U_rounded.tolist(),
                        "V": V_rounded.tolist(),
                        "W": W_rounded.tolist()
                    }

    # If no better rank found, return schoolbook
    return best_result

