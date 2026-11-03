import numpy as np
import time

def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    B_perm = b[:, p]
    M = a @ B_perm
    E = np.diag(a) @ B_perm.T
    D = M - E
    
    current_cost = np.sum(a * b[np.ix_(p, p)])
    best_p = p.copy()
    best_cost = current_cost
    
    # SA Phase
    T = 100.0
    T_min = 1e-5
    cooling_rate = 0.999
    
    steps = 0
    no_improve_steps = 0
    
    while time.time() < deadline - 10.0:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
            
        u = p[i]
        v = p[j]
        
        delta = D[i, v] - D[i, u] + D[j, u] - D[j, v]
        delta += a[i, j] * (2 * b[u, v] - b[u, u] - b[v, v])
        
        accept = False
        if delta < 0:
            accept = True
        else:
            if T > T_min:
                if -delta / T > np.log(np.random.rand()):
                    accept = True
        
        if accept:
            p[i], p[j] = p[j], p[i]
            
            diff_a = a[:, i] - a[:, j]
            diff_b = b[:, v] - b[:, u]
            D += np.outer(diff_a, diff_b)
            
            B_perm[:, i] = b[:, v]
            B_perm[:, j] = b[:, u]
            
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        T *= cooling_rate
        steps += 1
        
        # If stuck for a while, do a quick polish
        if no_improve_steps > 5000:
             # Run a few steps of Polish
             for _ in range(10):
                 if time.time() > deadline - 10.0: break
                 # ... Polish logic ...
             no_improve_steps = 0

    # Final Polish Phase
    # Run until no improvement or time out
    while time.time() < deadline - 1.0:
        D_perm = D[:, p]
        diag_D = np.diag(D_perm)
        A1 = D_perm - diag_D[:, None]
        
        B_p = b[np.ix_(p, p)]
        diag_B = np.diag(B_p)
        C = 2 * B_p - diag_B[:, None] - diag_B[None, :]
        
        Delta = A1 + A1.T + a * C
        Delta[np.diag_indices(n)] = np.inf
        
        min_idx = np.unravel_index(np.argmin(Delta), Delta.shape)
        i, j = min_idx
        delta = Delta[i, j]
        
        if delta >= 0:
            break
        
        u = p[i]
        v = p[j]
        p[i], p[j] = p[j], p[i]
        
        diff_a = a[:, i] - a[:, j]
        diff_b = b[:, v] - b[:, u]
        D += np.outer(diff_a, diff_b)
        
        B_perm[:, i] = b[:, v]
        B_perm[:, j] = b[:, u]
        
        current_cost += delta
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()

    return best_p.tolist()
