import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135  # 135s limit to be safe (150s total)
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i else 0
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        # Round to nearest 0.5
        return np.round(arr * 2) / 2
        
    def is_valid_decomposition(U, V, W, T, atol=1e-6):
        """
        Checks if U, V, W (assumed to be snapped or close to snapped)
        form a valid decomposition of T.
        """
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def try_snap_and_validate(U, V, W, T):
        """
        Snaps U, V, W to half-integers and checks if the snapped version is valid.
        Returns (True, U_snap, V_snap, W_snap) if valid, else (False, None, None, None).
        """
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if the snap is close to the current values (to avoid wild jumps)
        # We allow some tolerance here because gradient descent might not be perfectly aligned
        if not (np.allclose(U, U_snap, atol=0.1) and 
                np.allclose(V, V_snap, atol=0.1) and 
                np.allclose(W, W_snap, atol=0.1)):
            return False, None, None, None
            
        if is_valid_decomposition(U_snap, V_snap, W_snap, T):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def polish_solution(U, V, W, max_iter=100, lr_init=0.001):
        """
        High-precision polish pass. 
        """
        lr = lr_init
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            U_new = U - lr * gradU
            V_new = V - lr * gradV
            W_new = W - lr * gradW
            
            loss_new = compute_loss(U_new, V_new, W_new)
            
            if loss_new < prev_loss:
                U, V, W = U_new, V_new, W_new
                prev_loss = loss_new
                lr *= 1.01 # Slight increase if improving
            else:
                lr *= 0.9
                U, V, W = U_new, V_new, W_new # Accept anyway to explore? No, revert.
                U, V, W = U + lr * gradU, V + lr * gradV, W + lr * gradW # Revert
                if lr < 1e-8:
                    break
        
        return U, V, W

    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1.0
                    V[j * p + k][column] = 1.0
                    W[k * n + i][column] = 1.0
                    column += 1
        return {"U": U, "V": V, "W": W}

    best_solution = None
    best_rank = n * m * p  # schoolbook rank as initial best

    # Try ranks from 45 to 49
    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        R = target_rank
        found = False
        
        # Number of restarts: more for lower ranks as they are harder
        num_restarts = 10 if R == 45 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            random.seed(42 + restart * 1000 + R)
            
            U = np.random.randn(U_dim, R) * 0.5
            V = np.random.randn(V_dim, R) * 0.5
            W = np.random.randn(W_dim, R) * 0.5
            
            # Gradient descent
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            max_iter = 200
            no_improve = 0
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Periodic snap check
                if it % 20 == 0:
                    valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        found = True
                        break
                
                # Compute gradient
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                # Update with momentum
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U -= vU
                V -= vV
                W -= vW
                
                # Compute loss
                loss = compute_loss(U, V, W)
                
                if loss < prev_loss - 1e-8:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 20:
                        break
                
                if loss < 1e-6:
                    break
            
            if found:
                break
            
            # Final snap check after optimization
            valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
            if valid:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    # If still not found, return schoolbook
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
