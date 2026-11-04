import numpy as np
import time

def solve(a, b):
    n = len(a)
    # Initial permutation: rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    
    # Convert to list for easier swapping in Python loops if needed, 
    # but NumPy array is fine.
    # Ensure a and b are numpy arrays
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Calculate initial cost
    def calc_cost(p):
        # sum a[i,j] * b[p[i], p[j]]
        # Vectorized: 
        # b_p = b[np.ix_(p, p)]
        # return np.sum(a * b_p)
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = calc_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + 110.0
    
    # 2-Opt search
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over all pairs
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 1, n):
                if time.time() > deadline:
                    break
                
                u = p[i]
                v = p[j]
                
                # Calculate delta
                # Delta = sum_{k != i,j} a[i,k](b[v,p[k]] - b[u,p[k]]) + a[k,i](b[p[k],v] - b[p[k],u]) + 
                #         a[j,k](b[u,p[k]] - b[v,p[k]]) + a[k,j](b[p[k],u] - b[p[k],v]) +
                #         a[i,j](b[v,u] - b[u,v]) + a[j,i](b[u,v] - b[v,u])
                
                # Optimization: Use NumPy for the sum part
                # diff_b = b[v, p] - b[u, p]  # vector of length n
                # term_i = np.dot(a[i, :], diff_b)
                # term_ki = np.dot(a[:, i], diff_b) # assuming b symmetric
                # term_j = np.dot(a[j, :], -diff_b)
                # term_kj = np.dot(a[:, j], -diff_b)
                
                # But we must subtract k=i and k=j terms from the dot products.
                # k=i:
                # a[i,i] * (b[v, p[i]] - b[u, p[i]]) = a[i,i] * (b[v, u] - b[u, u])
                # a[i,i] * (b[p[i], v] - b[p[i], u]) = a[i,i] * (b[u, v] - b[u, u])
                # k=j:
                # a[i,j] * (b[v, p[j]] - b[u, p[j]]) = a[i,j] * (b[v, v] - b[u, v])
                # a[i,j] * (b[p[j], v] - b[p[j], u]) = a[i,j] * (b[v, v] - b[v, u])
                
                # This is getting complicated. The explicit loop might be clearer and fast enough.
                # Let's try the explicit loop first.
                
                delta = 0
                # Pair terms
                delta += a[i, j] * (b[v, u] - b[u, v])
                delta += a[j, i] * (b[u, v] - b[v, u])
                
                for k in range(n):
                    if k == i or k == j:
                        continue
                    pk = p[k]
                    # i with k
                    delta += a[i, k] * (b[v, pk] - b[u, pk])
                    delta += a[k, i] * (b[pk, v] - b[pk, u])
                    # j with k
                    delta += a[j, k] * (b[u, pk] - b[v, pk])
                    delta += a[k, j] * (b[pk, u] - b[pk, v])
                    
                if delta < 0:
                    # Swap
                    p[i] = v
                    p[j] = u
                    current_cost += delta
                    improved = True
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_p = p.copy()
                        
    return best_p.tolist()
