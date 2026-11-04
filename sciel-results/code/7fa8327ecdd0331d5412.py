"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from scipy.optimize import linear_sum_assignment

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Precompute for speed
    # a and b are symmetric in QAP standard, but we assume general just in case.
    # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
    
    def calc_cost(p):
        # Vectorized cost calculation
        # b_perm = b[np.ix_(p, p)]
        # return int(np.sum(a * b_perm))
        # This creates a large temporary array. 
        # Alternative: sum(a[i,j] * b[p[i], p[j]])
        # Using einsum or matrix multiplication might be faster for large n, but n=150 is small.
        # Let's stick to the direct one but optimized.
        return int(np.sum(a * b[np.ix_(p, p)]))

    def hungarian_init():
        """
        Initialize using a heuristic based on linearization.
        We can approximate the QAP with a linear assignment problem.
        A common heuristic is to use the sum of row sums or similar.
        Or, use the spectral method briefly to get a good start, then SA.
        Given the directive to use SA, we need a good start.
        Let's use a simple greedy or random start, or a quick spectral init.
        """
        # Spectral initialization (fast, no heavy optimization)
        try:
            w_a, v_a = np.linalg.eigh(a)
            w_b, v_b = np.linalg.eigh(b)
            # Use the eigenvector corresponding to the smallest eigenvalue for minimization
            # (assuming we want to align "low cost" structures)
            # Actually, for minimization of quadratic form, aligning with smallest eigenvalues is often good.
            idx_a = np.argmin(w_a)
            idx_b = np.argmin(w_b)
            vec_a = v_a[:, idx_a]
            vec_b = v_b[:, idx_b]
            
            # Sort facilities by vec_a and locations by vec_b
            order_a = np.argsort(vec_a)
            order_b = np.argsort(vec_b)
            
            p = np.empty(n, dtype=np.int64)
            for i, fac in enumerate(order_a):
                p[fac] = order_b[i]
            
            return p
        except Exception:
            return np.arange(n)

    # Initial solution
    p = hungarian_init()
    current_cost = calc_cost(p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    # Temperature schedule
    T0 = 1000.0  # Initial temperature
    Tend = 1.0   # Final temperature
    # Number of steps
    # We have ~115 seconds.
    # Each step involves:
    # 1. Pick 2 random indices (O(1))
    # 2. Calculate delta cost (O(n) if done naively, but can be O(1) with precomputation? 
    #    Delta for swapping i and j:
    #    Old: a[i,i]b[p[i],p[i]] + a[j,j]b[p[j],p[j]] + sum_{k!=i,j} a[i,k]b[p[i],p[k]] + a[j,k]b[p[j],p[k]] + a[k,i]b[p[k],p[i]] + a[k,j]b[p[k],p[j]]
    #    New: a[i,i]b[p[j],p[i]] ... wait.
    #    Let's derive the delta for swapping locations of facility i and facility j.
    #    Let u = p[i], v = p[j].
    #    Old cost terms involving i or j:
    #    C_old = sum_{k} (a[i,k] b[u, p[k]] + a[k,i] b[p[k], u]) + sum_{k} (a[j,k] b[v, p[k]] + a[k,j] b[p[k], v])
    #    Note: k includes i and j.
    #    Let's split k into {i, j} and rest.
    #    This is O(n) to compute exactly.
    #    For n=150, O(n) is 150 ops. Very fast.
    #    We can do ~100,000 steps in 1 second? Maybe less due to Python overhead.
    #    Let's aim for ~50,000 - 100,000 steps.
    
    # To make it faster, we can precompute row sums or use vectorization for the delta?
    # Delta = New - Old
    # New cost terms involving i or j (with u, v swapped):
    # C_new = sum_{k} (a[i,k] b[v, p[k]] + a[k,i] b[p[k], v]) + sum_{k} (a[j,k] b[u, p[k]] + a[k,j] b[p[k], u])
    # 
    # Let's vectorize the delta calculation.
    # We need to update p[i] and p[j] temporarily or just calculate the difference.
    
    def get_delta(p, i, j):
        u = p[i]
        v = p[j]
        
        # Cost contribution of i in old state:
        # sum_k a[i,k] b[u, p[k]] + sum_k a[k,i] b[p[k], u]
        # Note: a is symmetric? If so, a[i,k]=a[k,i]. b is symmetric?
        # QAPLIB instances usually have symmetric a and b.
        # Assuming symmetry:
        # Old_i = 2 * sum_k a[i,k] b[u, p[k]]
        # Old_j = 2 * sum_k a[j,k] b[v, p[k]]
        # 
        # New_i = 2 * sum_k a[i,k] b[v, p[k]]
        # New_j = 2 * sum_k a[j,k] b[u, p[k]]
        #
        # But we must be careful with the i,j terms themselves.
        # Let's just compute the full delta using vectorized dot products.
        
        # Row i of A
        ai = a[i]
        aj = a[j]
        
        # Columns u and v of B
        # We need dot(ai, b[u, p]) and dot(ai, b[v, p])
        # b[u, p] is the row u of b, permuted by p? No.
        # b[u, p[k]] is element (u, p[k]).
        # So we need the vector B_u = [b[u, p[0]], b[u, p[1]], ...]
        # Which is b[u, p].
        
        B_u = b[u, p]
        B_v = b[v, p]
        
        # Old contributions from i and j:
        # term_i_old = sum_k a[i,k] * b[u, p[k]] = dot(ai, B_u)
        # term_j_old = sum_k a[j,k] * b[v, p[k]] = dot(aj, B_v)
        
        # New contributions (swap u and v):
        # term_i_new = sum_k a[i,k] * b[v, p[k]] = dot(ai, B_v)
        # term_j_new = sum_k a[j,k] * b[u, p[k]] = dot(aj, B_u)
        
        # Wait, this only accounts for the terms where i or j is the FIRST index in A.
        # What about the terms where i or j is the SECOND index in A?
        # Cost = sum_{i,j} a[i,j] b[p[i], p[j]]
        # If a is symmetric, a[i,j] = a[j,i].
        # The term a[i,k] b[p[i], p[k]] and a[k,i] b[p[k], p[i]] are distinct terms in the sum unless we sum over all i,j.
        # The sum is over ALL i,j.
        # So, sum_{k} a[i,k] b[u, p[k]] covers terms where first index is i.
        # Sum_{k} a[k,i] b[p[k], u] covers terms where second index is i.
        # If a is symmetric, these are equal.
        # So Old_i_total = 2 * dot(ai, B_u) ?
        # Let's check the diagonal term k=i.
        # a[i,i] b[u, u].
        # In dot(ai, B_u), k=i term is a[i,i] b[u, p[i]] = a[i,i] b[u, u].
        # In the second part (if we added it), k=i term is a[i,i] b[u, u].
        # So yes, if symmetric, it's 2 * dot(ai, B_u).
        
        # However, we must be careful not to double count the interaction between i and j?
        # The delta is (New_i + New_j) - (Old_i + Old_j).
        # The interaction term a[i,j] b[u, v] is included in both?
        # Old_i includes k=j: a[i,j] b[u, v].
        # Old_j includes k=i: a[j,i] b[v, u].
        # New_i includes k=j: a[i,j] b[v, u].
        # New_j includes k=i: a[j,i] b[u, v].
        # Since a[i,j]=a[j,i] and b[u,v]=b[v,u], these are all the same value.
        # So the cross terms cancel out in the delta?
        # Old_cross = 2 * a[i,j] b[u,v]
        # New_cross = 2 * a[i,j] b[v,u] = 2 * a[i,j] b[u,v]
        # Yes, they cancel.
        
        # So Delta = 2 * (dot(ai, B_v) + dot(aj, B_u) - dot(ai, B_u) - dot(aj, B_v))
        
        delta = 2.0 * (np.dot(ai, B_v) + np.dot(aj, B_u) - np.dot(ai, B_u) - np.dot(aj, B_v))
        
        return delta

    # SA Loop
    steps = 0
    # Estimate steps per second to set a reasonable bound or just run until deadline.
    # Let's run until deadline.
    
    while time.time() < deadline:
        # Temperature decay
        # Exponential decay: T = T0 * exp(-k * steps)
        # Or linear: T = T0 - (T0-Tend) * (steps / total_steps)
        # We don't know total steps. Use exponential.
        # Let's aim for ~50k steps in 100s? 500 steps/s.
        # If we do 1000 steps/s, we need faster decay.
        # Let's try T = T0 * 0.9999^steps ? No, too slow.
        # T = T0 * exp(-steps / 10000)
        
        T = T0 * np.exp(-steps / 10000.0)
        if T < Tend:
            T = Tend
            
        # Generate random move: swap two distinct facilities
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
            
        delta = get_delta(p, i, j)
        
        # Acceptance probability
        if delta < 0:
            accept = True
        else:
            # Avoid overflow in exp
            if delta / T > 500:
                accept = False
            else:
                accept = (random.random() < np.exp(-delta / T))
                
        if accept:
            p[i], p[j] = p[j], p[i]
            current_cost += int(delta)
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                
        steps += 1
        
        # Periodic check to ensure we don't blow time
        if steps % 1000 == 0 and time.time() > deadline - 2.0:
            break

    return best_p.tolist()
# EVOLVE-BLOCK-END
