import numpy as np
import time

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Helper to compute cost
    def compute_cost(p):
        # p is a permutation of 0..n-1
        # Cost = sum a[i,j] * b[p[i], p[j]]
        # Vectorized: np.sum(a * b[np.ix_(p, p)])
        # But b[np.ix_(p, p)] creates a full matrix. n=150 is small.
        return np.sum(a * b[np.ix_(p, p)])
    
    # Helper to compute delta for swapping i, j
    # p: current permutation
    # u = p[i], v = p[j]
    # Delta = S - Term_i - Term_j + Interaction
    # S = dot(a[i, :] - a[j, :], (b[v, :] - b[u, :])[p])
    # Term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
    # Term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
    # Interaction = (a[j, i] - a[i, j]) * (b[u, v] - b[v, u])
    
    def get_delta(i, j, p, a, b):
        u = p[i]
        v = p[j]
        
        # Vectorized dot product part
        # a_diff = a[i, :] - a[j, :]
        # b_diff = b[v, :] - b[u, :]
        # S = np.dot(a_diff, b_diff[p])
        
        # To avoid creating temp arrays, we can just do:
        # But slicing is fast enough for n=150.
        
        a_diff = a[i, :] - a[j, :]
        b_diff = b[v, :] - b[u, :]
        
        S = np.dot(a_diff, b_diff[p])
        
        term_i = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        term_j = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        interaction = (a[j, i] - a[i, j]) * (b[u, v] - b[v, u])
        
        return S - term_i - term_j + interaction

    # 2-Opt Local Search
    def local_search(p, current_cost):
        improved = True
        while improved:
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    return p, current_cost
                for j in range(i + 1, n):
                    delta = get_delta(i, j, p, a, b)
                    if delta < 0:
                        # Apply swap
                        p[i], p[j] = p[j], p[i]
                        current_cost += delta
                        improved = True
        return p, current_cost

    best_cost = np.iinfo(np.int64).max
    best_p = None
    
    # Multiple restarts
    # Start with a few random restarts and maybe a greedy one
    # Greedy: match high a-sums to low b-sums? Or random?
    # Let's do 50 random restarts if time permits.
    
    num_restarts = 0
    while time.time() < deadline - 5.0: # Leave 5s for final check
        # Initialize p
        # Random initialization
        p = np.random.permutation(n)
        current_cost = compute_cost(p)
        
        # Run local search
        p, current_cost = local_search(p, current_cost)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
        num_restarts += 1
        
        # If we have done a few and are close to deadline, break
        if num_restarts > 20 and time.time() > deadline - 10.0:
            break

    if best_p is None:
        # Fallback
        p = np.random.permutation(n)
        p, _ = local_search(p, compute_cost(p))
        best_p = p
        best_cost = compute_cost(p)

    return best_p.tolist()
