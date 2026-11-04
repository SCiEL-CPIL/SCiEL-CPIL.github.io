"""QAP solver using Parallel Tempering MCMC with block-reversal moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute exact QAP cost for assignment p."""
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _delta_cost_block_reverse(a, b, p, i, j):
    """
    Compute the change in QAP cost when we reverse the sub-sequence p[i..j].
    p[fac] = loc.
    Reversing facilities i..j: for k in [i, j], facility k gets location p[i+j-k].
    """
    length = j - i + 1
    if length <= 1:
        return 0.0

    S_arr = np.arange(i, j + 1, dtype=np.int64)
    old_locs = p[S_arr]
    new_locs = p[i + j - S_arr]
    
    # Part 1: f1 in S, f2 in S
    A_SS = a[S_arr[:, None], S_arr[None, :]]
    B_old_SS = b[old_locs[:, None], old_locs[None, :]]
    B_new_SS = b[new_locs[:, None], new_locs[None, :]]
    delta = np.sum(A_SS * (B_new_SS - B_old_SS))
    
    # Part 2 & 3: Interactions with not S
    # We need sum over f1 in S, f2 not in S and f1 not in S, f2 in S.
    # Let's compute the contribution of S to the rest and rest to S.
    
    # Efficient way:
    # Delta_ext = sum_{u in S} sum_{v not in S} a[u,v] (b[new_loc[u], p[v]] - b[old_loc[u], p[v]]) 
    #           + sum_{v in S} sum_{u not in S} a[u,v] (b[p[u], new_loc[v]] - b[p[u], old_loc[v]])
    
    # We can compute sum_{v not in S} ... by computing sum_{v all} ... - sum_{v in S} ...
    
    # Term 2: sum_{u in S} [ sum_{v all} a[u,v] b[new_loc[u], p[v]] - sum_{v all} a[u,v] b[old_loc[u], p[v]] ]
    # This is not quite right because p[v] changes for v in S in the "new" world? 
    # No, the formula for delta is based on the OLD permutation p for the "not S" parts.
    # So p[v] is fixed for v not in S.
    
    # Let's stick to the explicit not_S indices for safety and clarity, as N is small.
    not_S_mask = np.ones(len(p), dtype=bool)
    not_S_mask[S_arr] = False
    not_S = np.where(not_S_mask)[0]
    
    if len(not_S) > 0:
        p_notS = p[not_S]
        
        # Part 2: f1 in S, f2 not in S
        A_S_notS = a[S_arr[:, None], not_S[None, :]]
        B_old_S_notS = b[old_locs[:, None], p_notS[None, :]]
        B_new_S_notS = b[new_locs[:, None], p_notS[None, :]]
        delta += np.sum(A_S_notS * (B_new_S_notS - B_old_S_notS))
        
        # Part 3: f1 not in S, f2 in S
        A_notS_S = a[not_S[:, None], S_arr[None, :]]
        B_old_notS_S = b[p_notS[:, None], old_locs[None, :]]
        B_new_notS_S = b[p_notS[:, None], new_locs[None, :]]
        delta += np.sum(A_notS_S * (B_new_notS_S - B_old_notS_S))
        
    return float(delta)


def solve(a, b):
    n = len(a)
    
    # Initial solution: Rank matching
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    
    # Local search refinement on initial solution (swaps)
    p_current = p0.copy()
    cost_current = _qap_cost(a, b, p_current)
    
    # Quick local search to get a better starting point
    for _ in range(100):
        best_delta = 0
        best_i, best_j = -1, -1
        # Random sample of swaps
        for _ in range(10):
            i, j = random.sample(range(n), 2)
            # Calculate delta for swap i, j
            # This is O(N)
            # We can use a helper or inline it
            # For simplicity and speed, we'll just do a few random swaps and keep if better
            p_temp = p_current.copy()
            p_temp[i], p_temp[j] = p_temp[j], p_temp[i]
            c_temp = _qap_cost(a, b, p_temp)
            if c_temp < cost_current:
                cost_current = c_temp
                p_current = p_temp
                best_i, best_j = i, j
                best_delta = c_temp - cost_current # This is negative
                break # Restart loop with new best
    
    # Setup Parallel Tempering
    num_replicas = 4
    # Temperatures: T_0 (low) to T_{R-1} (high)
    # Estimate scale
    sample_delta = 0
    for _ in range(5):
        i = random.randint(0, n-2)
        j = random.randint(i+1, min(i+20, n-1))
        d = _delta_cost_block_reverse(a, b, p_current, i, j)
        if d != 0:
            sample_delta += abs(d)
    sample_delta /= 5
    if sample_delta == 0: sample_delta = 1000
    
    base_temp = sample_delta / 10.0
    temps = [base_temp * (2 ** i) for i in range(num_replicas)]
    # T_0 is lowest
    temps.sort()
    
    # Initialize replicas
    # Replica 0 starts with best known
    # Others start with random permutations or slightly perturbed
    replicas = []
    for r in range(num_replicas):
        if r == 0:
            p_rep = p_current.copy()
            c_rep = cost_current
        else:
            p_rep = np.random.permutation(n).astype(np.int64)
            c_rep = _qap_cost(a, b, p_rep)
        replicas.append({'p': p_rep, 'cost': c_rep, 'temp': temps[r]})
    
    best_cost = cost_current
    best_p = p_current.copy()
    
    start_time = time.time()
    time_limit = 115.0
    
    rng = np.random.default_rng(42)
    
    steps_per_replica = 0
    swap_interval = 20 # Try to swap every 20 steps per replica
    
    while time.time() - start_time < time_limit:
        for r in range(num_replicas):
            rep = replicas[r]
            p = rep['p']
            cost = rep['cost']
            temp = rep['temp']
            
            # Choose move
            # Mix of small and large blocks
            if rng.random() < 0.3:
                length = int(rng.integers(2, 11))
            elif rng.random() < 0.7:
                length = int(rng.integers(10, 51))
            else:
                length = int(rng.integers(50, min(129, n)))
            
            if length > n: length = n
            i = int(rng.integers(0, n - length + 1))
            j = i + length - 1
            
            delta = _delta_cost_block_reverse(a, b, p, i, j)
            
            # Metropolis
            if delta < 0 or rng.random() < np.exp(-delta / temp):
                p[i:j+1] = p[i:j+1][::-1]
                cost += delta
                rep['cost'] = cost
                rep['p'] = p
                
                if cost < best_cost:
                    best_cost = cost
                    best_p = p.copy()
            
            steps_per_replica += 1
            
            # Cooling: Very slow cooling or none? 
            # In parallel tempering, we usually fix temps or cool very slowly.
            # Let's fix temps for simplicity and stability, relying on the exchange.
            # Or slightly cool the lowest temp? No, keep fixed.
        
        # Attempt swaps between adjacent replicas
        for r in range(num_replicas - 1):
            rep_i = replicas[r]
            rep_j = replicas[r+1]
            
            # Acceptance probability for swapping states i and j
            # P = min(1, exp( (C_i/T_i - C_j/T_j) - (C_j/T_i - C_i/T_j) ))
            # = min(1, exp( (C_i - C_j) (1/T_i - 1/T_j) ))
            
            Ci, Ti = rep_i['cost'], rep_i['temp']
            Cj, Tj = rep_j['cost'], rep_j['temp']
            
            exponent = (Ci - Cj) * (1.0/Ti - 1.0/Tj)
            
            if rng.random() < np.exp(exponent):
                # Swap
                rep_i['p'], rep_j['p'] = rep_j['p'].copy(), rep_i['p'].copy()
                rep_i['cost'], rep_j['cost'] = rep_j['cost'], rep_i['cost']
                
                # Update best if the swapped-in state is better
                if rep_i['cost'] < best_cost:
                    best_cost = rep_i['cost']
                    best_p = rep_i['p'].copy()
                if rep_j['cost'] < best_cost:
                    best_cost = rep_j['cost']
                    best_p = rep_j['p'].copy()

    return best_p.tolist()
# EVOLVE-BLOCK-END
