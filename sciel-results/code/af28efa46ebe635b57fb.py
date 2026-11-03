import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s limit to be safe
    
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
        
    def is_valid_decomposition(U, V, W, T, atol=1e-6):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.allclose(prod, T, atol=atol)
    
    def try_snap_and_validate(U, V, W, T, tol=0.15):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        if not (np.allclose(U, U_snap, atol=tol) and 
                np.allclose(V, V_snap, atol=tol) and 
                np.allclose(W, W_snap, atol=tol)):
            return False, None, None, None
            
        if is_valid_decomposition(U_snap, V_snap, W_snap, T):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def polish_solution(U, V, W, max_iter=500, tol=1e-10):
        """High-precision polish to drive loss below tolerance"""
        lr = 0.005
        momentum = 0.9
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        
        prev_loss = float('inf')
        no_improve = 0
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
                
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adaptive learning rate
            if it > 100:
                lr = 0.002
            if it > 200:
                lr = 0.001
            
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U_new = U - vU
            V_new = V - vV
            W_new = W - vW
            
            loss_new = compute_loss(U_new, V_new, W_new)
            
            if loss_new < prev_loss - 1e-12:
                U, V, W = U_new, V_new, W_new
                prev_loss = loss_new
                no_improve = 0
            else:
                no_improve += 1
                if no_improve > 30:
                    break
            
            if loss_new < tol:
                break
        
        return U, V, W

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

    def winograd_fourier_decomp():
        """
        Winograd scheme using DFT over Z_5 for the 5-dimensional cyclic structure.
        
        For 2x5x5 matrix multiplication, we can use the fact that the index sets
        {0,...,4} form a cyclic group of order 5. The DFT over Z_5 gives us a
        rank-25 decomposition where we share bilinear products between the two
        output rows.
        
        Key insight: C = A @ B where A is 2x5 and B is 5x5.
        C[i,k] = sum_j A[i,j] * B[j,k]
        
        Using Winograd: we can write this as a sequence of:
        1. Input transforms (linear combinations of A and B entries)
        2. Element-wise multiplications (the rank-25 bilinear products)
        3. Output transforms (linear combinations to get C entries)
        
        For the 5x5 part, we use DFT_5 to get the rank-25 decomposition.
        The 2-row structure means we need to handle both rows.
        
        Actually, let me think more carefully. The tensor is:
        T[i,j,k] = delta (corresponding to C[i,k] = sum_j A[i,j] B[j,k])
        
        In our indexing:
        U index: i*m + j = 5i + j (i in 0..1, j in 0..4)
        V index: j*p + k = 5j + k (j in 0..4, k in 0..4)  
        W index: k*n + i = 2k + i (k in 0..4, i in 0..1)
        
        T[u,v,w] = 1 when u = 5i+j, v = 5j+k, w = 2k+i
        
        This is the standard matrix multiplication tensor for 2x5 times 5x5.
        
        Winograd approach: Use the cyclic structure of Z_5.
        The DFT matrix F_5 has entries F_5[a,b] = omega^(a*b) where omega = e^(2*pi*i/5)
        
        For exact rational decomposition, we need to use the minimal polynomial of omega.
        omega satisfies x^4 + x^3 + x^2 + x + 1 = 0
        
        Actually, let me try a different approach. Since we need integer/half-integer
        entries, let me try to construct a rank-25 decomposition directly.
        
        One approach: group the 50 schoolbook terms into 25 pairs by sharing
        intermediate products.
        
        For each j, we have terms:
        C[0,k] += A[0,j] * B[j,k]
        C[1,k] += A[1,j] * B[j,k]
        
        We can combine these using the identity:
        (a+b)(b'+b'') - a*b' - b*b'' = a*b'' + b*b'
        
        But this doesn't directly help for sharing.
        
        Let me try the direct DFT approach with rational coefficients.
        """
        # For now, return schoolbook as fallback
        return schoolbook()

    best_solution = None
    best_rank = n * m * p
    best_loss = float('inf')

    # Try ranks from 45 to 49, prioritizing 45
    ranks_to_try = [45, 45, 46, 47, 48, 49]
    
    for rank_idx, target_rank in enumerate(ranks_to_try):
        if time.time() > deadline - 15:
            break
            
        R = target_rank
        found = False
        
        # More restarts for rank 45
        if R == 45:
            num_restarts = 15
        elif R == 46:
            num_restarts = 8
        else:
            num_restarts = 4
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            # Use structured initialization for first few restarts
            if restart < 5:
                rng = np.random.RandomState(42 + restart * 100 + R)
                U = np.zeros((U_dim, R))
                V = np.zeros((V_dim, R))
                W = np.zeros((W_dim, R))
                
                for r in range(R):
                    for i in range(n):
                        for j in range(m):
                            u_idx = i * m + j
                            if rng.random() < 0.3:
                                U[u_idx, r] = rng.randn() * 0.3
                    
                    for j in range(m):
                        for k in range(p):
                            v_idx = j * p + k
                            if rng.random() < 0.2:
                                V[v_idx, r] = rng.randn() * 0.3
                    
                    for k in range(p):
                        for i in range(n):
                            w_idx = k * n + i
                            if rng.random() < 0.3:
                                W[w_idx, r] = rng.randn() * 0.3
                
                for _ in range(R // 2):
                    u_idx = rng.randint(0, U_dim)
                    v_idx = rng.randint(0, V_dim)
                    w_idx = rng.randint(0, W_dim)
                    r = rng.randint(0, R)
                    U[u_idx, r] += rng.randn() * 0.5
                    V[v_idx, r] += rng.randn() * 0.5
                    W[w_idx, r] += rng.randn() * 0.5
            else:
                rng = np.random.RandomState(42 + restart * 1000 + R)
                U = rng.randn(U_dim, R) * 0.4
                V = rng.randn(V_dim, R) * 0.4
                W = rng.randn(W_dim, R) * 0.4
            
            # Progressive learning rate
            lr = 0.02
            momentum = 0.9
            vU = np.zeros_like(U)
            vV = np.zeros_like(V)
            vW = np.zeros_like(W)
            
            max_iter = 300
            no_improve = 0
            prev_loss = float('inf')
            
            for it in range(max_iter):
                if time.time() > deadline - 8:
                    break
                
                # Check for valid solution periodically
                if it % 10 == 0 and it > 0:
                    valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                    if valid:
                        best_solution = (U_snap, V_snap, W_snap)
                        best_rank = R
                        found = True
                        break
                
                prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
                diff = prod - T
                
                gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
                gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
                gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
                
                # Decay learning rate
                if it > 50:
                    lr = 0.01
                if it > 100:
                    lr = 0.005
                if it > 150:
                    lr = 0.002
                
                vU = momentum * vU + lr * gradU
                vV = momentum * vV + lr * gradV
                vW = momentum * vW + lr * gradW
                
                U -= vU
                V -= vV
                W -= vW
                
                loss = compute_loss(U, V, W)
                
                if loss < prev_loss - 1e-10:
                    prev_loss = loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve > 30:
                        break
                
                if loss < 1e-8:
                    break
            
            if found:
                break
            
            # Try polishing if we got close
            if prev_loss < 1.0:
                U_pol, V_pol, W_pol = polish_solution(U, V, W, max_iter=200)
                valid, U_snap, V_snap, W_snap = try_snap_and_validate(U_pol, V_pol, W_pol, T)
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    best_rank = R
                    found = True
                    break
                # Also check without polishing
                valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                if valid:
                    best_solution = (U_snap, V_snap, W_snap)
                    best_rank = R
                    found = True
                    break
        
        if found:
            break
    
    if best_solution:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        return schoolbook()

# EVOLVE-BLOCK-END
