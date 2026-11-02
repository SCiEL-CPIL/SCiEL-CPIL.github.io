import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = DIMENSIONS
    U_dim = n * m  # 10
    V_dim = m * p  # 25
    W_dim = p * n  # 10
    
    # Build the target tensor T
    T = np.zeros((U_dim, V_dim, W_dim))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
    
    def snap_to_half_integers(arr):
        return np.round(arr * 2) / 2
    
    def validate(U, V, W, atol=1e-6):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None
    
    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_dim, rank))
        V = np.zeros((V_dim, rank))
        W = np.zeros((W_dim, rank))
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1.0
                    V[j * p + k, column] = 1.0
                    W[k * n + i, column] = 1.0
                    column += 1
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}
    
    # Try structured decomposition based on Kronecker product ideas
    # The 2x5x5 multiplication can be decomposed by splitting the 5 dimension
    # Let's try to find a low-rank decomposition using algebraic structure
    
    best_solution = None
    best_rank = n * m * p
    
    # Strategy 1: Try to find a decomposition using the structure
    # For 2x5x5, we can try rank values from 45 upward
    
    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        R = target_rank
        found = False
        
        num_restarts = 8 if R <= 46 else 5
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            random.seed(42 + restart * 7919 + R)
            np.random.seed(42 + restart * 7919 + R)
            
            # Initialize using structured approach
            # Each rank position gets contributions from multiple (i,j,k) terms
            U = np.zeros((U_dim, R))
            V = np.zeros((V_dim, R))
            W = np.zeros((W_dim, R))
            
            # Assign each of the 50 terms to a random rank position
            # with some overlap to allow for cancellation
            all_terms = list(range(n*m*p))
            
            # Use a structured initialization: group terms by i
            # For each i, the 25 terms (j,k) can be distributed
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        term_idx = i * (m*p) + j * p + k
                        # Assign to 1-3 rank positions with coefficients
                        num_assignments = random.choice([1, 1, 2, 3])
                        positions = random.sample(range(R), num_assignments)
                        
                        if num_assignments == 1:
                            U[i*m+j, positions[0]] += 1.0
                            V[j*p+k, positions[0]] += 1.0
                            W[k*n+i, positions[0]] += 1.0
                        else:
                            # Distribute with coefficients that sum to 1
                            coeffs = np.random.dirichlet(np.ones(num_assignments))
                            for pos_idx, pos in enumerate(positions):
                                c = coeffs[pos_idx]
                                U[i*m+j, pos] += c
                                V[j*p+k, pos] += c
                                W[k*n+i, pos] += c
            
            # Gradient descent with adaptive learning rate
            lr = 0.005
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            max_iter = 800
            patience = 150
            no_improve = 0
            prev_loss = float('inf')
            best_loss_in_run = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U -= vU
                V -= vV
                W -= vW
                
                loss = compute_loss(U, V, W)
                
                if loss < best_loss_in_run:
                    best_loss_in_run = loss
                    if loss < prev_loss - 1e-10:
                        prev_loss = loss
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                if loss < 1e-8:
                    break
                
                # Adaptive learning rate
                if it > 0 and loss > prev_loss * 1.5:
                    lr *= 0.5
                
                prev_loss = loss
            
            valid, U_snap, V_snap, W_snap = validate(U, V, W)
            if valid:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
                found = True
                break
        
        if found:
            break
    
    if best_solution:
        U, V, W = best_solution
        return {"U": [[float(x) for x in row] for row in U], 
                "V": [[float(x) for x in row] for row in V], 
                "W": [[float(x) for x in row] for row in W]}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
