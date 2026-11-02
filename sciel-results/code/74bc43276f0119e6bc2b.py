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
    
    def validate(U, V, W, atol=1e-6):
        # Check if entries are integer or half-integer
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if snapped matrices are close to original
        if not (np.allclose(U, U_snap, atol=atol) and 
                np.allclose(V, V_snap, atol=atol) and 
                np.allclose(W, W_snap, atol=atol)):
            return False, None, None, None
            
        # Compute tensor with snapped values
        prod = np.einsum('ar,br,cr->abc', U_snap, V_snap, W_snap, optimize=True)
        if np.allclose(prod, T, atol=atol):
            return True, U_snap, V_snap, W_snap
        return False, None, None
    
    def compute_loss(U, V, W):
        # Compute sum_r U[:,r] V[:,r] W[:,r]
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)
    
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
    
    # Known lower bound for 2x5x5 is 45 (from literature on small matrix multiplication)
    # Try to find rank 45 decomposition
    
    best_solution = None
    best_rank = n * m * p  # schoolbook rank as initial best
    
    # Strategy: Use structured initialization based on known algebraic identities
    # For 2x5x5, we can exploit the structure:
    # The tensor has a specific pattern that allows rank reduction
    
    # Approach: Start from schoolbook and use gradient descent with careful initialization
    # Key insight: For small matrices, we can try various rank values and use
    # random restarts with different initialization strategies
    
    ranks_to_try = [45, 46, 47, 48, 49]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 20:
            break
            
        R = target_rank
        found = False
        
        # Multiple restarts with different strategies
        num_restarts = 5 if R == 45 else 3
        
        for restart in range(num_restarts):
            if time.time() > deadline - 15:
                break
            
            random.seed(42 + restart * 1000 + R)
            
            U = np.zeros((U_dim, R))
            V = np.zeros((V_dim, R))
            W = np.zeros((W_dim, R))
            
            # Strategy: Initialize by distributing schoolbook terms across rank positions
            # Each rank position gets a random subset of the 50 schoolbook terms
            
            # Create a random partition/assignment of the 50 terms to R positions
            # Some positions will get more terms than others
            
            all_terms = list(range(n*m*p))
            
            # Randomly assign terms to rank positions
            assignments = [random.choice(range(R)) for _ in range(n*m*p)]
            
            for term_idx, rank_pos in enumerate(assignments):
                i = term_idx // (m*p)
                j = (term_idx // p) % m
                k = term_idx % p
                
                # Add contribution with a coefficient
                # Use coefficients that sum to 1 for each term across ranks
                coeff = 1.0
                U[i*m+j, rank_pos] += coeff
                V[j*p+k, rank_pos] += coeff
                W[k*n+i, rank_pos] += coeff
            
            # Scale to help optimization
            # The above initialization may not be optimal, so let's try a different approach
            
            # Alternative: Use a more structured initialization
            # For 2x5x5, there are known decompositions that exploit the small size
            # Let's try initializing with patterns based on the tensor structure
            
            U2 = np.zeros((U_dim, R))
            V2 = np.zeros((V_dim, R))
            W2 = np.zeros((W_dim, R))
            
            # Initialize each rank position with a random linear combination
            for r in range(R):
                # Pick random (i,j,k) triples and combine them
                num_terms = max(1, int(np.random.chisquare(1) * 2))
                num_terms = min(num_terms, n*m*p)
                selected = random.sample(range(n*m*p), num_terms)
                
                for idx in selected:
                    i = idx // (m*p)
                    j = (idx // p) % m
                    k = idx % p
                    
                    coeff = random.uniform(0.3, 1.0)
                    U2[i*m+j, r] += coeff
                    V2[j*p+k, r] += coeff
                    W2[k*n+i, r] += coeff
            
            # Use the better of the two initializations
            # Try both and see which gives lower loss initially
            loss1 = compute_loss(U, V, W)
            loss2 = compute_loss(U2, V2, W2)
            
            if loss2 < loss1:
                U, V, W = U2, V2, W2
            else:
                # Keep original
                pass
            
            # Gradient descent with momentum
            lr = 0.01
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            max_iter = 500
            patience = 100
            no_improve = 0
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 10:
                    break
                
                # Compute gradient
                # dL/dU = 2 * (prod - T) contracted with V, W
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                
                # Gradient for U: sum over b,c of diff[a,b,c] * V[b,r] * W[c,r]
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
                
                if loss < prev_loss - 1e-10:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > patience:
                    break
                
                if loss < 1e-8:
                    break
            
            # Check if we found a valid solution
            valid, U_snap, V_snap, W_snap = validate(U, V, W)
            if valid:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
                found = True
                break
            
            # Try with the other initialization too
            if not found:
                U3, V3, W3 = U.copy(), V.copy(), W.copy()
                # Reinitialize with the other strategy
                U3 = np.zeros((U_dim, R))
                V3 = np.zeros((V_dim, R))
                W3 = np.zeros((W_dim, R))
                
                for r in range(R):
                    num_terms = max(1, int(np.random.chisquare(1) * 2))
                    num_terms = min(num_terms, n*m*p)
                    selected = random.sample(range(n*m*p), num_terms)
                    
                    for idx in selected:
                        i = idx // (m*p)
                        j = (idx // p) % m
                        k = idx % p
                        
                        coeff = random.uniform(0.3, 1.0)
                        U3[i*m+j, r] += coeff
                        V3[j*p+k, r] += coeff
                        W3[k*n+i, r] += coeff
                
                # Quick gradient descent
                lr = 0.01
                momentum = 0.9
                vU = np.zeros_like(U3)
                vV = np.zeros_like(V3)
                vW = np.zeros_like(W3)
                
                max_iter = 300
                patience = 80
                no_improve = 0
                prev_loss = float('inf')
                
                for it in range(max_iter):
                    if time.time() > deadline - 10:
                        break
                    
                    prod = np.einsum('ar,br,cr->abc', U3, V3, W3, optimize=True)
                    diff = prod - T
                    
                    gradU = np.einsum('abc,br,cr->ar', diff, V3, W3, optimize=True)
                    gradV = np.einsum('abc,ar,cr->br', diff, U3, W3, optimize=True)
                    gradW = np.einsum('abc,ar,br->cr', diff, U3, V3, optimize=True)
                    
                    vU = momentum * vU + lr * gradU
                    vV = momentum * vV + lr * gradV
                    vW = momentum * vW + lr * gradW
                    
                    U3 -= vU
                    V3 -= vV
                    W3 -= vW
                    
                    loss = compute_loss(U3, V3, W3)
                    
                    if loss < prev_loss - 1e-10:
                        prev_loss = loss
                        no_improve = 0
                    else:
                        no_improve += 1
                    
                    if no_improve > patience:
                        break
                    
                    if loss < 1e-8:
                        break
                
                valid, U_snap, V_snap, W_snap = validate(U3, V3, W3)
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
