import time
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    target = np.zeros((n*m, m*p, p*n), dtype=float)
    for c_r in range(n):
        for b in range(m):
            for c_c in range(p):
                i = c_r * m + b
                k = b * p + c_c
                l = c_c * n + c_r
                target[i, k, l] = 1.0

    def verify(U, V, W):
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        R = np.einsum('ic,kc,lc->ikl', Ua, Va, Wa)
        return np.allclose(R, target, atol=1e-9)

    def greedy_prune_random(U, V, W, max_iter=200, seed=None):
        """Aggressive greedy pruning with random column ordering."""
        if seed is not None:
            random.seed(seed)
            
        current_U = np.array(U, dtype=float)
        current_V = np.array(V, dtype=float)
        current_W = np.array(W, dtype=float)
        current_rank = current_U.shape[1]
        
        for iteration in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if current_rank <= 0:
                break
            
            removed_any = False
            # Try removing each column in random order
            cols_to_try = list(range(current_rank))
            random.shuffle(cols_to_try)
            
            for col in cols_to_try:
                if time.time() > deadline - 5.0:
                    break
                
                # Create new arrays without this column
                mask = [i for i in range(current_rank) if i != col]
                U_new = current_U[:, mask]
                V_new = current_V[:, mask]
                W_new = current_W[:, mask]
                
                # Quick check using numpy
                R = np.einsum('ic,kc,lc->ikl', U_new, V_new, W_new)
                
                if np.allclose(R, target, atol=1e-9):
                    current_U, current_V, current_W = U_new, V_new, W_new
                    current_rank -= 1
                    removed_any = True
                    break  # Restart scanning with new random order
            
            if not removed_any:
                break
        
        # Convert back to lists for return
        return current_U.tolist(), current_V.tolist(), current_W.tolist(), current_rank

    def get_schoolbook():
        # Rank n*m*p = 72
        U = [[0] * (n*m*p) for _ in range(n*m)]
        V = [[0] * (n*m*p) for _ in range(m*p)]
        W = [[0] * (n*m*p) for _ in range(p*n)]
        
        col = 0
        for c_r in range(n):
            for b in range(m):
                for c_c in range(p):
                    i = c_r * m + b
                    k = b * p + c_c
                    l = c_c * n + c_r
                    U[i][col] = 1
                    V[k][col] = 1
                    W[l][col] = 1
                    col += 1
        return U, V, W

    def get_strassen_66():
        # Reconstruct the 66 rank solution from the seed
        rank = 66
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        col = 0

        def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A00 = a_r * m + a_c
            A01 = a_r * m + a_c + 1
            A10 = (a_r + 1) * m + a_c
            A11 = (a_r + 1) * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C00 = c_c * n + c_r
            C01 = (c_c + 1) * n + c_r
            C10 = c_c * n + (c_r + 1)
            C11 = (c_c + 1) * n + (c_r + 1)

            U[A00][col] += 1; U[A11][col] += 1
            V[B00][col] += 1; V[B11][col] += 1
            W[C00][col] += 1; W[C11][col] += 1
            col += 1
            U[A10][col] += 1; U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1; W[C11][col] -= 1
            col += 1
            U[A00][col] += 1
            V[B01][col] += 1; V[B11][col] -= 1
            W[C01][col] += 1; W[C11][col] += 1
            col += 1
            U[A11][col] += 1
            V[B10][col] += 1; V[B00][col] -= 1
            W[C00][col] += 1; W[C10][col] += 1
            col += 1
            U[A00][col] += 1; U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1; W[C01][col] += 1
            col += 1
            U[A10][col] += 1; U[A00][col] -= 1
            V[B00][col] += 1; V[B01][col] += 1
            W[C11][col] += 1
            col += 1
            U[A01][col] += 1; U[A11][col] -= 1
            V[B10][col] += 1; V[B11][col] += 1
            W[C00][col] += 1
            col += 1

        def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
            nonlocal col
            A0 = a_r * m + a_c
            A1 = a_r * m + a_c + 1

            B00 = b_r * p + b_c
            B01 = b_r * p + b_c + 1
            B10 = (b_r + 1) * p + b_c
            B11 = (b_r + 1) * p + b_c + 1

            C0 = c_c * n + c_r
            C1 = (c_c + 1) * n + c_r

            U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
            col += 1
            U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
            col += 1
            U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
            col += 1
            U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
            col += 1

        add_1x2_2x2(0, 0, 0, 0, 0, 0)
        add_1x2_2x2(0, 2, 2, 0, 0, 0)
        add_1x2_2x2(0, 0, 0, 2, 0, 2)
        add_1x2_2x2(0, 2, 2, 2, 0, 2)
        add_1x2_2x2(0, 0, 0, 4, 0, 4)
        add_1x2_2x2(0, 2, 2, 4, 0, 4)

        add_strassen(1, 0, 0, 0, 1, 0)
        add_strassen(1, 2, 2, 0, 1, 0)
        add_strassen(1, 0, 0, 2, 1, 2)
        add_strassen(1, 2, 2, 2, 1, 2)
        add_strassen(1, 0, 0, 4, 1, 4)
        add_strassen(1, 2, 2, 4, 1, 4)
        
        return U, V, W

    best_U, best_V, best_W, best_rank = None, None, None, float('inf')
    
    # Generate candidates
    candidates = []
    
    # 1. The 66-rank solution
    U66, V66, W66 = get_strassen_66()
    candidates.append((U66, V66, W66))
    
    # 2. The 72-rank schoolbook solution
    U72, V72, W72 = get_schoolbook()
    candidates.append((U72, V72, W72))
    
    # Prune each candidate using random restarts
    num_restarts = 10
    
    for U, V, W in candidates:
        for restart in range(num_restarts):
            if time.time() > deadline - 10.0:
                break
            
            seed_val = restart * 12345 + 67890
            U_p, V_p, W_p, r_p = greedy_prune_random(U, V, W, max_iter=200, seed=seed_val)
            
            # Verify the pruned decomposition
            if verify(U_p, V_p, W_p):
                if r_p < best_rank:
                    best_rank = r_p
                    best_U, best_V, best_W = U_p, V_p, W_p
            else:
                # If pruning broke it, fall back to original if valid
                if verify(U, V, W):
                    r_orig = len(U[0])
                    if r_orig < best_rank:
                        best_rank = r_orig
                        best_U, best_V, best_W = U, V, W

    # If we didn't find anything better than 66, return 66
    if best_U is None:
        best_U, best_V, best_W = get_strassen_66()
        best_rank = 66

    return {"U": best_U, "V": best_V, "W": best_W}
