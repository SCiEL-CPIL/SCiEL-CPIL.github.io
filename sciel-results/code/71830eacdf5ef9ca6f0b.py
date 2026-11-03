import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    # Ensure integer types for precision if needed, but float is okay for deltas
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    
    # Initial Solution: Rank Matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    p = p.astype(np.int64)
    
    # Calculate Initial Cost
    # Cost = sum(a[i,j] * b[p[i], p[j]])
    # Efficient calculation: trace(A * B_perm)
    # B_perm[i,j] = b[p[i], p[j]]
    # We can compute this as sum over i,j
    # Using einsum: np.einsum('ij,ik,jk->', a, b[p[:, None], p[None, :]]) is O(n^3)
    # Or: np.sum(a * b[np.ix_(p, p)])
    
    def compute_cost(p):
        return np.sum(a * b[np.ix_(p, p)])
        
    current_cost = compute_cost(p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    # Tabu Search Parameters
    tabu_length = 20
    k_pairs = 1000 # Number of random pairs to evaluate per step
    iterations = 0
    
    # Tabu list: store (i, j) and the iteration when they were used
    # We use a set for O(1) lookup, and a list to expire old entries
    tabu_set = set()
    tabu_history = [] # list of (i, j, iter)
    
    # Pre-compute row sums of A? Not directly helpful for the delta formula as derived.
    # Delta formula:
    # d = sum_{k != i,j} (a[i,k] + a[j,k]) * (b[p[i], p[k]] - b[p[j], p[k]])
    # + (a[i,j] - a[j,i]) * (b[p[i], p[j]] - b[p[j], p[i]])
    
    # If A and B are symmetric, the last term is 0.
    # Let's assume symmetry for speed, but add a check or just use the full formula if it's not too slow.
    # QAPLIB tho150: A is symmetric, B is symmetric.
    
    while time.time() < deadline:
        iterations += 1
        
        # Generate random pairs
        # We can just pick random i, j
        indices = np.random.choice(n, size=(k_pairs, 2), replace=False)
        
        best_delta = np.inf
        best_i = -1
        best_j = -1
        
        # We need to evaluate deltas for these pairs
        # To avoid python loop overhead, we can try to vectorize the delta calculation for all pairs at once?
        # Delta for pair (i,j):
        # u = p[i], v = p[j]
        # diff_b = b[u, p] - b[v, p]  (vector of length n)
        # sum_a = a[i] + a[j]        (vector of length n)
        # delta = dot(sum_a, diff_b) - correction terms
        
        # Vectorizing across pairs is tricky because p[i] and p[j] vary per pair.
        # However, we can do it in chunks or just accept a python loop if k_pairs is small.
        # With k_pairs=1000, a python loop is ~1000 * (numpy ops).
        # Each numpy op is ~10-100ns? No, overhead is ~1us. 1000 * 1us = 1ms.
        # 150s / 1ms = 150,000 iterations. This is very fast.
        
        for idx in range(k_pairs):
            i, j = indices[idx]
            if i == j:
                continue
            
            # Check Tabu
            key = (i, j) if i < j else (j, i)
            if key in tabu_set:
                # Check Aspiration
                # We need to know if this move improves best_cost.
                # We don't know the delta yet.
                # We can calculate delta first, then check aspiration.
                pass
            
            u = p[i]
            v = p[j]
            
            # Calculate Delta
            # term = sum_{k} (a[i,k] + a[j,k]) * (b[u, p[k]] - b[v, p[k]])
            # We exclude k=i and k=j in the standard formula, but if we include them:
            # For k=i: (a[i,i]+a[j,i]) * (b[u, u] - b[v, u])
            # For k=j: (a[i,j]+a[j,j]) * (b[u, v] - b[v, v])
            # Assuming a[i,i]=0, b[u,u]=0, b[v,v]=0:
            # k=i term: a[j,i] * (-b[v, u])
            # k=j term: a[i,j] * b[u, v]
            # Sum of these two: a[i,j] * (b[u,v] - b[v,u])
            # The full sum includes these.
            # The formula derived earlier: Delta = Sum_{k != i,j} ... + (a[i,j]-a[j,i])(b[u,v]-b[v,u])
            # If symmetric, last term 0.
            # The sum over k != i,j is TotalSum - k=i term - k=j term.
            # TotalSum = dot(a[i]+a[j], b[u,p] - b[v,p])
            # k=i term = (a[i,i]+a[j,i]) * (b[u,u] - b[v,u]) = a[j,i] * (-b[v,u])
            # k=j term = (a[i,j]+a[j,j]) * (b[u,v] - b[v,v]) = a[i,j] * b[u,v]
            
            # So Delta = TotalSum - (a[j,i]*(-b[v,u]) + a[i,j]*b[u,v])
            # Delta = TotalSum + a[j,i]*b[v,u] - a[i,j]*b[u,v]
            
            # If symmetric: a[i,j]=a[j,i], b[u,v]=b[v,u]
            # Delta = TotalSum + a[i,j]*b[u,v] - a[i,j]*b[u,v] = TotalSum
            
            # So if symmetric, Delta = dot(a[i]+a[j], b[u,p] - b[v,p])
            
            # Let's use the symmetric assumption for speed, as tho150 is symmetric.
            # If not symmetric, we should use the general formula.
            # Let's check symmetry once.
            
            if not hasattr(solve, '_symmetric'):
                solve._symmetric = np.allclose(a, a.T) and np.allclose(b, b.T)
                
            if solve._symmetric:
                # delta = np.dot(a[i] + a[j], b[u, p] - b[v, p])
                # This creates temporary arrays.
                # We can do:
                diff = b[u, p] - b[v, p]
                sum_a = a[i] + a[j]
                delta = np.dot(sum_a, diff)
            else:
                # General case
                # TotalSum = dot(a[i]+a[j], b[u,p] - b[v,p])
                # Correction = a[j,i]*b[v,u] - a[i,j]*b[u,v]
                # Delta = TotalSum + Correction
                diff = b[u, p] - b[v, p]
                sum_a = a[i] + a[j]
                total_sum = np.dot(sum_a, diff)
                correction = a[j, i] * b[v, u] - a[i, j] * b[u, v]
                delta = total_sum + correction
            
            # Check if this is a better move than the current best candidate
            if delta < best_delta:
                # Check Tabu
                if key not in tabu_set:
                    best_delta = delta
                    best_i = i
                    best_j = j
                else:
                    # Aspiration: accept if new cost < best_cost
                    new_cost = current_cost + delta
                    if new_cost < best_cost:
                        best_delta = delta
                        best_i = i
                        best_j = j
        
        # If no valid move found (all tabu and no aspiration), we might need to restart or allow a random move
        if best_i == -1:
            # Pick a random non-tabu move, or just break/restart
            # Let's pick a random pair and ignore tabu for this step to escape
            i, j = np.random.choice(n, size=2, replace=False)
            if i == j: continue
            u = p[i]
            v = p[j]
            if solve._symmetric:
                diff = b[u, p] - b[v, p]
                sum_a = a[i] + a[j]
                delta = np.dot(sum_a, diff)
            else:
                diff = b[u, p] - b[v, p]
                sum_a = a[i] + a[j]
                total_sum = np.dot(sum_a, diff)
                correction = a[j, i] * b[v, u] - a[i, j] * b[u, v]
                delta = total_sum + correction
            
            best_i = i
            best_j = j
            best_delta = delta
            key = (i, j) if i < j else (j, i)
        
        # Apply Move
        p[best_i], p[best_j] = p[best_j], p[best_i]
        current_cost += best_delta
        
        # Update Tabu List
        key = (best_i, best_j) if best_i < best_j else (best_j, best_i)
        tabu_set.add(key)
        tabu_history.append((key, iterations))
        
        # Expire old tabu entries
        while tabu_history and iterations - tabu_history[0][1] >= tabu_length:
            old_key, _ = tabu_history.pop(0)
            tabu_set.remove(old_key)
            
        # Update Best
        if current_cost < best_cost:
            best_cost = current_cost
            best_p = p.copy()
            
    return best_p.tolist()
