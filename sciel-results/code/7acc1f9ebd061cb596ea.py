import numpy as np
import time

def solve(a, b):
    n = len(a)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    
    deadline = time.time() + 145.0
    
    # Initialization
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    B_p = b[:, p]
    cost = np.sum(a * b[p, p])
    
    best_cost = cost
    best_p = p.copy()
    
    T = 5000.0
    cooling = 0.995
    
    diag_a = a.diagonal()
    diag_b = b.diagonal()
    
    while time.time() < deadline:
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        
        u = p[i]
        v = p[j]
        
        # Vectorized delta calculation
        S_i_v = np.dot(a[i, :], B_p[v, :]) + np.dot(a[:, i], B_p[:, v])
        S_i_u = np.dot(a[i, :], B_p[u, :]) + np.dot(a[:, i], B_p[:, u])
        S_j_v = np.dot(a[j, :], B_p[v, :]) + np.dot(a[:, j], B_p[:, v])
        S_j_u = np.dot(a[j, :], B_p[u, :]) + np.dot(a[:, j], B_p[:, u])
        
        corr_i_v = a[i, i] * (b[v, u] + b[u, v]) + a[i, j] * b[v, v] + a[j, i] * b[v, v]
        corr_i_u = a[i, i] * (b[u, u] + b[u, u]) + a[i, j] * b[u, v] + a[j, i] * b[v, u]
        corr_j_v = a[j, j] * (b[v, v] + b[v, v]) + a[j, i] * b[v, u] + a[i, j] * b[u, v]
        corr_j_u = a[j, j] * (b[u, u] + b[u, u]) + a[j, i] * b[u, v] + a[i, j] * b[v, u]
        
        S_i_v_excl = S_i_v - corr_i_v
        S_i_u_excl = S_i_u - corr_i_u
        S_j_v_excl = S_j_v - corr_j_v
        S_j_u_excl = S_j_u - corr_j_u
        
        New_ij = a[i, j] * b[v, u] + a[j, i] * b[u, v]
        Old_ij = a[i, j] * b[u, v] + a[j, i] * b[v, u]
        
        delta = (S_i_v_excl + S_j_u_excl + New_ij) - (S_i_u_excl + S_j_v_excl + Old_ij)
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            p[i], p[j] = p[j], p[i]
            B_p[:, i], B_p[:, j] = B_p[:, j], B_p[:, i]
            cost += delta
            if cost < best_cost:
                best_cost = cost
                best_p = p.copy()
        
        T *= cooling
        if T < 1.0:
            T = 100.0 # Reheat
            
    return best_p.tolist()
