import numpy as np
import time

DIMENSIONS = [2, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
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

    def try_snap_and_validate(U, V, W, T):
        U_snap = snap_to_half_integers(U)
        V_snap = snap_to_half_integers(V)
        W_snap = snap_to_half_integers(W)
        
        # Check if snap is close to current
        if not (np.allclose(U, U_snap, atol=0.1) and 
                np.allclose(V, V_snap, atol=0.1) and 
                np.allclose(W, W_snap, atol=0.1)):
            return False, None, None, None
            
        if is_valid_decomposition(U_snap, V_snap, W_snap, T):
            return True, U_snap, V_snap, W_snap
        return False, None, None, None

    def compute_loss(U, V, W):
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def polish_solution(U, V, W, T, max_iter=200, lr_init=0.005, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """High-precision polish to find exact half-integer solution"""
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            t += 1
            
            # Periodic snap check
            if it % 5 == 0:
                valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                if valid:
                    return True, U_snap, V_snap, W_snap
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adam update with smaller learning rate for polish
            lr_t = lr_init * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU * gradU)
            U -= lr_t * mU / (np.sqrt(vU) + epsilon)
            
            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV * gradV)
            V -= lr_t * mV / (np.sqrt(vV) + epsilon)
            
            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW * gradW)
            W -= lr_t * mW / (np.sqrt(vW) + epsilon)
            
            loss = compute_loss(U, V, W)
            
            if loss < 1e-8:
                break
            
            if loss < prev_loss - 1e-10:
                prev_loss = loss
            else:
                if it > 50 and loss > prev_loss * 1.05:
                    break

        # Final snap check
        valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
        if valid:
            return True, U_snap, V_snap, W_snap
        
        return False, None, None, None

    def adam_optimize(U, V, W, R, max_iter=500, lr_init=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        mU = np.zeros_like(U)
        vU = np.zeros_like(U)
        mV = np.zeros_like(V)
        vV = np.zeros_like(V)
        mW = np.zeros_like(W)
        vW = np.zeros_like(W)
        
        t = 0
        prev_loss = float('inf')
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            t += 1
            
            # Periodic snap check
            if it % 10 == 0:
                valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
                if valid:
                    return True, U_snap, V_snap, W_snap
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            # Adam update
            lr_t = lr_init * np.sqrt(1 - beta2**t) / (1 - beta1**t)
            
            mU = beta1 * mU + (1 - beta1) * gradU
            vU = beta2 * vU + (1 - beta2) * (gradU * gradU)
            U -= lr_t * mU / (np.sqrt(vU) + epsilon)
            
            mV = beta1 * mV + (1 - beta1) * gradV
            vV = beta2 * vV + (1 - beta2) * (gradV * gradV)
            V -= lr_t * mV / (np.sqrt(vV) + epsilon)
            
            mW = beta1 * mW + (1 - beta1) * gradW
            vW = beta2 * vW + (1 - beta2) * (gradW * gradW)
            W -= lr_t * mW / (np.sqrt(vW) + epsilon)
            
            loss = compute_loss(U, V, W)
            
            if loss < 1e-6:
                break
            
            if loss < prev_loss - 1e-8:
                prev_loss = loss
            else:
                # If stuck, maybe break early? Or continue?
                # Let's continue for a bit
                if it > 100 and loss > prev_loss * 1.1:
                    break

        # Final snap check
        valid, U_snap, V_snap, W_snap = try_snap_and_validate(U, V, W, T)
        if valid:
            return True, U_snap, V_snap, W_snap
        
        # If close but not exact, try polish
        loss = compute_loss(U, V, W)
        if loss < 0.1:
            valid, U_snap, V_snap, W_snap = polish_solution(U, V, W, T)
            if valid:
                return True, U_snap, V_snap, W_snap
        
        return False, None, None, None

    def random_init(U_dim, V_dim, W_dim, R, seed_val):
        rng = np.random.RandomState(seed_val)
        # Initialize with values close to half-integers
        # e.g. random choice from {-1, -0.5, 0, 0.5, 1}
        vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        U = rng.choice(vals, size=(U_dim, R))
        V = rng.choice(vals, size=(V_dim, R))
        W = rng.choice(vals, size=(W_dim, R))
        # Add small noise
        U += rng.randn(U_dim, R) * 0.1
        V += rng.randn(V_dim, R) * 0.1
        W += rng.randn(W_dim, R) * 0.1
        return U, V, W

    def structured_init(U_dim, V_dim, W_dim, R, seed_val):
        """Structured initialization based on matrix multiplication structure"""
        rng = np.random.RandomState(seed_val)
        
        # For matrix multiplication, we can initialize based on the structure
        # U[i*m+j, r] should relate to A[i,j], V[j*p+k, r] to B[j,k], W[k*n+i, r] to C[k,i]
        
        U = np.zeros((U_dim, R))
        V = np.zeros((V_dim, R))
        W = np.zeros((W_dim, R))
        
        # Random assignment of which (i,j,k) triple each rank element handles
        # But allow some sharing
        for r in range(R):
            # Pick a random pattern
            if rng.random() < 0.3:
                # Sparse: only one element in each
                i = rng.randint(n)
                j = rng.randint(m)
                k = rng.randint(p)
                U[i*m+j, r] = rng.choice([-1, 1])
                V[j*p+k, r] = rng.choice([-1, 1])
                W[k*n+i, r] = rng.choice([-1, 1])
            elif rng.random() < 0.6:
                # Medium: a few elements
                for _ in range(rng.randint(2, 4)):
                    i = rng.randint(n)
                    j = rng.randint(m)
                    k = rng.randint(p)
                    U[i*m+j, r] += rng.choice([-1, 0, 1])
                    V[j*p+k, r] += rng.choice([-1, 0, 1])
                    W[k*n+i, r] += rng.choice([-1, 0, 1])
            else:
                # Dense: many elements
                for _ in range(rng.randint(4, 8)):
                    i = rng.randint(n)
                    j = rng.randint(m)
                    k = rng.randint(p)
                    U[i*m+j, r] += rng.choice([-1, -0.5, 0, 0.5, 1])
                    V[j*p+k, r] += rng.choice([-1, -0.5, 0, 0.5, 1])
                    W[k*n+i, r] += rng.choice([-1, -0.5, 0, 0.5, 1])
        
        # Snap to half-integers and add small noise
        U = snap_to_half_integers(U) + rng.randn(U_dim, R) * 0.05
        V = snap_to_half_integers(V) + rng.randn(V_dim, R) * 0.05
        W = snap_to_half_integers(W) + rng.randn(W_dim, R) * 0.05
        
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

    best_solution = None
    best_rank = n * m * p

    # Try ranks from 40 to 49
    ranks_to_try = list(range(40, 50))
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 10:
            break
            
        R = target_rank
        found = False
        
        # Number of restarts
        num_restarts = 8
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                break
            
            seed_val = 42 + restart * 1000 + R
            
            # Alternate between random and structured init
            if restart % 2 == 0:
                U, V, W = random_init(U_dim, V_dim, W_dim, R, seed_val)
            else:
                U, V, W = structured_init(U_dim, V_dim, W_dim, R, seed_val)
            
            found, U_snap, V_snap, W_snap = adam_optimize(U, V, W, R)
            
            if found:
                best_solution = (U_snap, V_snap, W_snap)
                best_rank = R
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
