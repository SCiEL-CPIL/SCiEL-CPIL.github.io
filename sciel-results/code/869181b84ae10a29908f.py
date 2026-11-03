import numpy as np
import time
import random

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin for 150s limit
    
    # Schoolbook fallback
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

    # Target ranks: 30, 31, 32, 33, 34, 35
    # Known lower bound is around 27-28, upper bound 36.
    # 30 is a strong candidate for minimal rank.
    
    for target_rank in range(30, 36):
        if time.time() > deadline - 10:
            break
            
        # Simulated Annealing / Local Search
        # Variables: U (n*m*rank), V (m*p*rank), W (p*n*rank)
        # All half-integers.
        
        # Initialize with random half-integers from {-1, -0.5, 0, 0.5, 1}
        # Or better, initialize with a sparse structure similar to schoolbook?
        # Let's try random first.
        
        max_iters = 50000
        temp = 1.0
        cooling = 0.9995
        
        u_vals = [-1, -0.5, 0, 0.5, 1]
        
        # Flattened arrays for fast access
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        
        U_flat = np.array([random.choice(u_vals) for _ in range(size_u)])
        V_flat = np.array([random.choice(u_vals) for _ in range(size_v)])
        W_flat = np.array([random.choice(u_vals) for _ in range(size_w)])
        
        def get_loss(U_f, V_f, W_f):
            # Compute T_ijk = sum_r U_ijk_r * V_jkr * W_kir
            # U index: (i*m+j)*rank + r
            # V index: (j*p+k)*rank + r
            # W index: (k*n+i)*rank + r
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            u_val = U_f[(i*m+j)*target_rank + r]
                            v_val = V_f[(j*p+k)*target_rank + r]
                            w_val = W_f[(k*n+i)*target_rank + r]
                            s += u_val * v_val * w_val
                        diff = s - 1.0
                        loss += diff * diff
            return loss

        current_loss = get_loss(U_flat, V_flat, W_flat)
        best_loss = current_loss
        best_U = U_flat.copy()
        best_V = V_flat.copy()
        best_W = W_flat.copy()
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
                
            # Pick a random variable and new value
            # 1/3 chance U, 1/3 V, 1/3 W
            var_type = random.randint(0, 2)
            if var_type == 0:
                idx = random.randint(0, size_u - 1)
                old_val = U_flat[idx]
                new_val = random.choice(u_vals)
                if new_val == old_val: continue
                U_flat[idx] = new_val
                new_loss = get_loss(U_flat, V_flat, W_flat)
                if new_loss > current_loss:
                    # Revert
                    U_flat[idx] = old_val
                    if random.random() > np.exp((current_loss - new_loss) / temp):
                        continue
                    else:
                        U_flat[idx] = new_val
                        current_loss = new_loss
                else:
                    current_loss = new_loss
            elif var_type == 1:
                idx = random.randint(0, size_v - 1)
                old_val = V_flat[idx]
                new_val = random.choice(u_vals)
                if new_val == old_val: continue
                V_flat[idx] = new_val
                new_loss = get_loss(U_flat, V_flat, W_flat)
                if new_loss > current_loss:
                    V_flat[idx] = old_val
                    if random.random() > np.exp((current_loss - new_loss) / temp):
                        continue
                    else:
                        V_flat[idx] = new_val
                        current_loss = new_loss
                else:
                    current_loss = new_loss
            else:
                idx = random.randint(0, size_w - 1)
                old_val = W_flat[idx]
                new_val = random.choice(u_vals)
                if new_val == old_val: continue
                W_flat[idx] = new_val
                new_loss = get_loss(U_flat, V_flat, W_flat)
                if new_loss > current_loss:
                    W_flat[idx] = old_val
                    if random.random() > np.exp((current_loss - new_loss) / temp):
                        continue
                    else:
                        W_flat[idx] = new_val
                        current_loss = new_loss
                else:
                    current_loss = new_loss
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_U = U_flat.copy()
                best_V = V_flat.copy()
                best_W = W_flat.copy()
                
                if best_loss < 1e-9:
                    # Found exact solution
                    U_list = [[best_U[(i*m+j)*target_rank + r] for r in range(target_rank)] for i in range(n) for j in range(m)]
                    V_list = [[best_V[(j*p+k)*target_rank + r] for r in range(target_rank)] for j in range(m) for k in range(p)]
                    W_list = [[best_W[(k*n+i)*target_rank + r] for r in range(target_rank)] for k in range(p) for i in range(n)]
                    
                    # Final verification
                    valid = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                s = 0
                                for r in range(target_rank):
                                    s += U_list[i*m+j][r] * V_list[j*p+k][r] * W_list[k*n+i][r]
                                if abs(s - 1.0) > 1e-9:
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                    if valid:
                        return {"U": U_list, "V": V_list, "W": W_list}
            
            temp *= cooling
            if temp < 0.01:
                break

    return schoolbook()
# EVOLVE-BLOCK-END
