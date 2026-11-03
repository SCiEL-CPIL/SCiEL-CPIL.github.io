import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    # Allowed values for half-integer entries
    # We can expand this set if needed, but start with small values
    vals = np.array([-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2])
    
    def compute_loss(U_flat, V_flat, W_flat, r):
        """
        Vectorized loss computation.
        U_flat: shape (n*m, r)
        V_flat: shape (m*p, r)
        W_flat: shape (p*n, r)
        """
        # Reshape to 3D tensors
        # U: (n, m, r)
        # V: (m, p, r)
        # W: (p, n, r)
        U3 = U_flat.reshape(n, m, r)
        V3 = V_flat.reshape(m, p, r)
        W3 = W_flat.reshape(p, n, r)
        
        # Compute T_ijk = sum_r U_ijk_r * V_jkr * W_kir
        # Using einsum: 'ijr,jkr,kir->ijk'
        # Note: The indices in the problem are:
        # U[i, j, r]
        # V[j, k, r]
        # W[k, i, r]
        # Result S[i, j, k]
        S = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        
        # The target tensor for matrix multiplication has 1s at all (i,j,k)
        # because C_ik = sum_j A_ij B_jk, so each term A_ij B_jk contributes 1 to the tensor coefficient.
        # Wait, the tensor T is defined such that T_{i,j,k} = 1 if the term A_{ij} B_{jk} is part of the sum.
        # Yes, for standard matmul, the structure tensor has 1s for all i,j,k.
        diff = S - 1.0
        return np.sum(diff * diff)

    def verify(U_list, V_list, W_list):
        # Final strict verification
        n, m, p = DIMENSIONS
        r = len(U_list[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r_idx in range(r):
                        s += U_list[i*m+j][r_idx] * V_list[j*p+k][r_idx] * W_list[k*n+i][r_idx]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    # Search for minimal rank
    # Start from 30, go up to 35 (36 is schoolbook)
    for target_rank in range(30, 36):
        if time.time() > deadline - 10:
            break
        
        # Multiple restarts
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Initialize randomly
            U_flat = np.random.choice(vals, size=(n * m * target_rank))
            V_flat = np.random.choice(vals, size=(m * p * target_rank))
            W_flat = np.random.choice(vals, size=(p * n * target_rank))
            
            current_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
            best_loss = current_loss
            best_U = U_flat.copy()
            best_V = V_flat.copy()
            best_W = W_flat.copy()
            
            # Local Search (Simulated Annealing / Hill Climbing)
            # Since vectorized loss is fast, we can do many steps
            steps = 2000
            temp = 1.0
            cooling = 0.99
            
            for step in range(steps):
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random variable
                var_type = random.randint(0, 2)
                if var_type == 0:
                    size = n * m * target_rank
                    idx = random.randint(0, size - 1)
                    old_val = U_flat[idx]
                    new_val = random.choice(vals)
                    if new_val == old_val:
                        continue
                    U_flat[idx] = new_val
                    new_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
                elif var_type == 1:
                    size = m * p * target_rank
                    idx = random.randint(0, size - 1)
                    old_val = V_flat[idx]
                    new_val = random.choice(vals)
                    if new_val == old_val:
                        continue
                    V_flat[idx] = new_val
                    new_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
                else:
                    size = p * n * target_rank
                    idx = random.randint(0, size - 1)
                    old_val = W_flat[idx]
                    new_val = random.choice(vals)
                    if new_val == old_val:
                        continue
                    W_flat[idx] = new_val
                    new_loss = compute_loss(U_flat, V_flat, W_flat, target_rank)
                
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_U = U_flat.copy()
                    best_V = V_flat.copy()
                    best_W = W_flat.copy()
                    
                    if best_loss < 1e-9:
                        # Construct lists for verification and return
                        U_list = [[best_U[(i*m+j)*target_rank + r] for r in range(target_rank)] for i in range(n) for j in range(m)]
                        V_list = [[best_V[(j*p+k)*target_rank + r] for r in range(target_rank)] for j in range(m) for k in range(p)]
                        W_list = [[best_W[(k*n+i)*target_rank + r] for r in range(target_rank)] for k in range(p) for i in range(n)]
                        
                        if verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                        found = True
                        break
                
                # Accept or reject
                if new_loss <= current_loss:
                    current_loss = new_loss
                else:
                    # Revert
                    if var_type == 0:
                        U_flat[idx] = old_val
                    elif var_type == 1:
                        V_flat[idx] = old_val
                    else:
                        W_flat[idx] = old_val
                    
                    # Simulated annealing acceptance
                    if random.random() < np.exp((current_loss - new_loss) / temp):
                        if var_type == 0:
                            U_flat[idx] = new_val
                        elif var_type == 1:
                            V_flat[idx] = new_val
                        else:
                            W_flat[idx] = new_val
                        current_loss = new_loss
                
                temp *= cooling
                if temp < 0.01:
                    break
            
            if found:
                break

    return schoolbook()
# EVOLVE-BLOCK-END
