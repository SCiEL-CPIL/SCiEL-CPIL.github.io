import numpy as np
import time

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T[u, v, w] = 1

    start_time = time.time()
    max_time = 110  # Safety margin
    
    best_solution = None
    best_rank = dim_u * dim_v * dim_w # Schoolbook
    
    # Try ranks from 7 up to 15
    for r in range(7, 16):
        if time.time() - start_time > max_time:
            break
            
        # Run multiple restarts
        for restart in range(200):
            if time.time() - start_time > max_time:
                break
                
            # Initialize U, V, W with random sparse integers {-1, 0, 1}
            U = np.zeros((dim_u, r), dtype=int)
            V = np.zeros((dim_v, r), dtype=int)
            W = np.zeros((dim_w, r), dtype=int)
            
            # Density ~ 0.3 to 0.5
            density = 0.4
            U[np.random.rand(dim_u, r) < density] = np.random.choice([-1, 1], size=np.sum(np.random.rand(dim_u, r) < density))
            # Re-generate masks properly to avoid shape issues in one line
            mask_u = np.random.rand(dim_u, r) < density
            mask_v = np.random.rand(dim_v, r) < density
            mask_w = np.random.rand(dim_w, r) < density
            
            U[mask_u] = np.random.choice([-1, 1], size=np.sum(mask_u))
            V[mask_v] = np.random.choice([-1, 1], size=np.sum(mask_v))
            W[mask_w] = np.random.choice([-1, 1], size=np.sum(mask_w))
            
            # Calculate initial reconstruction and error
            R = np.zeros((dim_u, dim_v, dim_w), dtype=int)
            for c in range(r):
                R += U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
                
            # Error E = sum (R - T)^2
            diff = R - T
            E = np.sum(diff * diff)
            
            if E == 0:
                best_solution = (U, V, W)
                best_rank = r
                break
                
            # Local Search with Delta Error
            # We will perform hill climbing with occasional random jumps (Simulated Annealing lite)
            temperature = 10.0
            cooling_rate = 0.995
            max_steps = 5000
            
            for step in range(max_steps):
                if time.time() - start_time > max_time:
                    break
                    
                # Pick a random matrix, column, row
                mat_id = np.random.randint(3) # 0:U, 1:V, 2:W
                c = np.random.randint(r)
                
                if mat_id == 0:
                    i = np.random.randint(dim_u)
                    old_val = U[i, c]
                    # New value from {-1, 0, 1} \ {old_val}
                    candidates = [x for x in [-1, 0, 1] if x != old_val]
                    new_val = np.random.choice(candidates)
                    
                    # Calculate Delta E
                    # Affected slice: R[i, :, :]
                    # D = (new_val - old_val) * V[:, c] outer W[:, c]
                    delta_val = new_val - old_val
                    D = delta_val * np.outer(V[:, c], W[:, c])
                    E_old_slice = R[i, :, :] - T[i, :, :]
                    # Delta E = sum(2 * E_old_slice * D + D^2)
                    delta_E = 2 * np.sum(E_old_slice * D) + np.sum(D * D)
                    
                    # Acceptance criterion
                    if delta_E < 0:
                        # Accept
                        U[i, c] = new_val
                        R[i, :, :] += D
                        E += delta_E
                    else:
                        # Simulated Annealing accept
                        if np.random.rand() < np.exp(-delta_E / temperature):
                            U[i, c] = new_val
                            R[i, :, :] += D
                            E += delta_E
                            
                    temperature *= cooling_rate
                    if E == 0:
                        break
                        
                elif mat_id == 1:
                    i = np.random.randint(dim_v)
                    old_val = V[i, c]
                    candidates = [x for x in [-1, 0, 1] if x != old_val]
                    new_val = np.random.choice(candidates)
                    
                    delta_val = new_val - old_val
                    D = delta_val * np.outer(U[:, c], W[:, c])
                    # Affected slice: R[:, i, :]
                    E_old_slice = R[:, i, :] - T[:, i, :]
                    delta_E = 2 * np.sum(E_old_slice * D) + np.sum(D * D)
                    
                    if delta_E < 0:
                        V[i, c] = new_val
                        R[:, i, :] += D
                        E += delta_E
                    else:
                        if np.random.rand() < np.exp(-delta_E / temperature):
                            V[i, c] = new_val
                            R[:, i, :] += D
                            E += delta_E
                            
                    temperature *= cooling_rate
                    if E == 0:
                        break
                        
                else: # mat_id == 2
                    i = np.random.randint(dim_w)
                    old_val = W[i, c]
                    candidates = [x for x in [-1, 0, 1] if x != old_val]
                    new_val = np.random.choice(candidates)
                    
                    delta_val = new_val - old_val
                    D = delta_val * np.outer(U[:, c], V[:, c])
                    # Affected slice: R[:, :, i]
                    E_old_slice = R[:, :, i] - T[:, :, i]
                    delta_E = 2 * np.sum(E_old_slice * D) + np.sum(D * D)
                    
                    if delta_E < 0:
                        W[i, c] = new_val
                        R[:, :, i] += D
                        E += delta_E
                    else:
                        if np.random.rand() < np.exp(-delta_E / temperature):
                            W[i, c] = new_val
                            R[:, :, i] += D
                            E += delta_E
                            
                    temperature *= cooling_rate
                    if E == 0:
                        break
                        
            if E == 0:
                best_solution = (U, V, W)
                best_rank = r
                break
                
        if best_solution is not None:
            break

    if best_solution is not None:
        U, V, W = best_solution
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((dim_u, rank), dtype=int)
        V = np.zeros((dim_v, rank), dtype=int)
        W = np.zeros((dim_w, rank), dtype=int)
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    U[u, column] = 1
                    V[v, column] = 1
                    W[w, column] = 1
                    column += 1
        
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
# EVOLVE-BLOCK-END
