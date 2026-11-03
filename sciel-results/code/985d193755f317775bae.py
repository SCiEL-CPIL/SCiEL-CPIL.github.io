"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _qap_cost(a, b, p):
    """Compute the QAP cost for permutation p."""
    p_arr = np.asarray(p)
    total = 0
    for i in range(len(a)):
        total += np.dot(a[i], b[p_arr[i]])
    return total


def _delta_swap(a, b, p, i, j):
    """
    Compute the change in cost if we swap locations of facilities i and j.
    Uses vectorized NumPy operations for speed.
    """
    li = p[i]
    lj = p[j]
    
    # Vector for difference in b rows: b[lj, :] - b[li, :]
    vec_i = b[lj, :] - b[li, :]
    
    # Sum for u=i, v != i,j: sum a[i,v] * vec_i[v]
    # Exclude v=i and v=j
    s1 = np.dot(a[i, :], vec_i) - a[i, i]*vec_i[i] - a[i, j]*vec_i[j]
    
    # Sum for u=j, v != i,j: sum a[j,v] * (-vec_i[v])
    s2 = -np.dot(a[j, :], vec_i) + a[j, j]*vec_i[j] + a[j, i]*vec_i[i]
    
    # Sum for u != i,j, v=i: sum a[u,i] * (b[p[u], lj] - b[p[u], li])
    # vec_lj_li[k] = b[k, lj] - b[k, li]
    vec_lj_li = b[:, lj] - b[:, li]
    # gathered[k] = vec_lj_li[p[k]]
    gathered = vec_lj_li[p]
    s3 = np.dot(a[:, i], gathered) - a[i, i]*gathered[i] - a[j, i]*gathered[j]
    
    # Sum for u != i,j, v=j: sum a[u,j] * (b[p[u], li] - b[p[u], lj])
    # = - sum a[u,j] * (b[p[u], lj] - b[p[u], li])
    s4 = -np.dot(a[:, j], gathered) + a[i, j]*gathered[i] + a[j, j]*gathered[j]
    
    # Diagonal and cross terms
    diag_ii = a[i, i] * (b[lj, lj] - b[li, li])
    diag_jj = a[j, j] * (b[li, li] - b[lj, lj])
    cross_ij = a[i, j] * (b[lj, li] - b[li, lj])
    cross_ji = a[j, i] * (b[li, lj] - b[lj, li])
    
    return s1 + s2 + s3 + s4 + diag_ii + diag_jj + cross_ij + cross_ji


def _simulated_annealing(a, b, p, time_limit=110):
    """
    Improve the solution using Simulated Annealing.
    """
    n = len(p)
    p = np.array(p, dtype=np.int64)
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    # Initial temperature
    # Estimate typical delta magnitude
    # Just pick a reasonable starting T
    T = 1000.0
    T_min = 1.0
    cooling_rate = 0.9995
    
    random.seed(42)
    
    while time.time() - start_time < time_limit:
        if T < T_min:
            break
            
        # Pick two random distinct facilities
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        delta = _delta_swap(a, b, p, i, j)
        new_cost = current_cost + delta
        
        if delta < 0:
            # Accept improving move
            p[i], p[j] = p[j], p[i]
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        else:
            # Accept worsening move with probability exp(-delta / T)
            # Avoid overflow
            if delta > 0:
                prob = math.exp(-delta / T)
            else:
                prob = 1.0
                
            if random.random() < prob:
                p[i], p[j] = p[j], p[i]
                current_cost = new_cost
        
        T *= cooling_rate
        
    return best_p.tolist(), best_cost


def solve(a, b):
    n = len(a)
    
    # Start with a random permutation
    p = list(range(n))
    random.seed(42)
    random.shuffle(p)
    
    # Run Simulated Annealing
    p, cost = _simulated_annealing(a, b, p, time_limit=110)
    
    return p
# EVOLVE-BLOCK-END
