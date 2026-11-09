import numpy as np
import time
import random

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank_school = n * m * p
    
    # Initialize schoolbook solution
    U_school = np.zeros((n * m, rank_school), dtype=int)
    V_school = np.zeros((m * p, rank_school), dtype=int)
    W_school = np.zeros((p * n, rank_school), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    deadline = time.time() + 110

    def compute_error(U, V, W):
        """Compute the squared error of the tensor decomposition."""
        # T is the target tensor
        # T[i*m+j, jp*p+k, kp*n+ip] = 1 if i=ip, j=jp, k=kp else 0
        # We compute sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
        # and compare to T.
        
        # Efficiently compute the tensor product
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Result R: (nm, mp, pn)
        # R[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        
        # Use einsum
        # U: (nm, r) -> U[a, k]
        # V: (mp, r) -> V[b, k]
        # W: (pn, r) -> W[c, k]
        # R[a, b, c] = sum_k U[a, k] V[b, k] W[c, k]
        
        # This is a 3-way tensor contraction.
        # We can do it in steps to save memory if needed, but nm=12, mp=28, pn=21.
        # Size is 12*28*21 = 7056. Very small.
        
        # R = np.einsum('ak,bk,ck->abc', U, V, W)
        # This might be slow in pure Python/NumPy for many iterations.
        # Alternative:
        # For each c, add U[:, c, None] * V[:, c, None] * W[:, c, None]
        
        R = np.zeros((n*m, m*p, p*n), dtype=float)
        for c in range(U.shape[1]):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            # Outer product of uc and vc gives (nm, mp)
            # Multiply by wc (pn)
            # R[:, :, :] += np.einsum('a,b,c->abc', uc, vc, wc)
            # Or:
            temp = np.outer(uc, vc) # (nm, mp)
            R += temp[:, :, None] * wc[None, None, :]
            
        # Target Tensor T
        # T[a, b, c] = 1 if indices match
        # a = i*m + j, b = jp*p + k, c = kp*n + ip
        # Match if i=ip, j=jp, k=kp
        # This means:
        # i = a // m, j = a % m
        # jp = b // p, k = b % p
        # kp = c // n, ip = c % n
        # Condition: i==ip, j==jp, k==kp
        
        # We can precompute T
        T = np.zeros((n*m, m*p, p*n), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
                    # Note: Are there other non-zero entries?
                    # The verify function checks:
                    # sum ... = 1 if (i=ip, j=jp, k=kp) else 0
                    # This implies T is 1 ONLY at these positions.
                    # So T is a permutation of the identity vector?
                    # No, it's a set of 84 ones in a 12x28x21 tensor.
        
        error = np.sum((R - T) ** 2)
        return error

    def local_search(U, V, W, r, max_steps=10000):
        """Perform local search to minimize error."""
        current_U = U.copy()
        current_V = V.copy()
        current_W = W.copy()
        
        current_error = compute_error(current_U, current_V, current_W)
        
        if current_error < 1e-6:
            return True, current_U, current_V, current_W
        
        for step in range(max_steps):
            if time.time() > deadline:
                return False, current_U, current_V, current_W
            
            # Choose a random matrix and a random entry to perturb
            mat_choice = random.choice(['U', 'V', 'W'])
            if mat_choice == 'U':
                idx = random.randint(0, n*m - 1)
                col = random.randint(0, r - 1)
                old_val = current_U[idx, col]
                # Try +/- 1
                new_val = old_val + random.choice([-1, 1])
                current_U[idx, col] = new_val
                new_error = compute_error(current_U, current_V, current_W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error < 1e-6:
                        return True, current_U, current_V, current_W
                else:
                    current_U[idx, col] = old_val
                    
            elif mat_choice == 'V':
                idx = random.randint(0, m*p - 1)
                col = random.randint(0, r - 1)
                old_val = current_V[idx, col]
                new_val = old_val + random.choice([-1, 1])
                current_V[idx, col] = new_val
                new_error = compute_error(current_U, current_V, current_W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error < 1e-6:
                        return True, current_U, current_V, current_W
                else:
                    current_V[idx, col] = old_val
                    
            else: # W
                idx = random.randint(0, p*n - 1)
                col = random.randint(0, r - 1)
                old_val = current_W[idx, col]
                new_val = old_val + random.choice([-1, 1])
                current_W[idx, col] = new_val
                new_error = compute_error(current_U, current_V, current_W)
                if new_error < current_error:
                    current_error = new_error
                    if current_error < 1e-6:
                        return True, current_U, current_V, current_W
                else:
                    current_W[idx, col] = old_val
                    
        return False, current_U, current_V, current_W

    # Search for lower rank
    # Start from a reasonable guess, e.g., 70 down to 50
    for r in range(70, 49, -1):
        if time.time() > deadline:
            break
            
        found = False
        for trial in range(20): # Fewer trials for local search
            if time.time() > deadline:
                break
            
            # Initialize with small random integers
            U = np.random.randint(-2, 3, size=(n*m, r))
            V = np.random.randint(-2, 3, size=(m*p, r))
            W = np.random.randint(-2, 3, size=(p*n, r))
            
            success, U_sol, V_sol, W_sol = local_search(U, V, W, r, max_steps=2000)
            
            if success:
                # Verify exactly with integers
                U_int = U_sol.astype(int)
                V_int = V_sol.astype(int)
                W_int = W_sol.astype(int)
                
                # Quick verify
                err = compute_error(U_int, V_int, W_int)
                if err < 1e-6:
                    best_rank = r
                    best_U = U_int
                    best_V = V_int
                    best_W = W_int
                    found = True
                    break # Move to next lower rank
        
        if found:
            # If we found a solution for r, we can try to go lower immediately
            # But the loop goes downwards, so it will naturally try r-1 next
            pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
