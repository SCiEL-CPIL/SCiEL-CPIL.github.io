"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _qap_cost(a, b, p):
    """Compute the QAP cost for a given permutation p."""
    # Cost = sum_{i,j} a[i,j] * b[p[i], p[j]]
    # We can compute this efficiently using matrix operations
    bp = b[np.ix_(p, p)]
    return int(np.sum(a * bp))


def _rank_matching_init(a, b):
    """Rank-matching initialization: heavy flows to short distances."""
    n = len(a)
    # Sort facilities by descending row sum (heavy first)
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    # Sort locations by ascending row sum (central/short first)
    central = np.argsort(b.sum(axis=1), kind="stable")
    p = np.empty(n, dtype=np.int64)
    p[heavy] = central
    return p


def _local_search_2opt(a, b, p, time_limit=None, start_time=None):
    """
    Local search using 2-opt moves (swap two facilities' locations).
    Uses efficient delta computation.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    
    if start_time is None:
        start_time = time.time()
    
    # Precompute for fast delta evaluation
    # When we swap p[i] and p[j], the change in cost involves:
    # For all k != i, j: a[i,k]*(b[p[j],p[k]] - b[p[i],p[k]]) + a[j,k]*(b[p[i],p[k]] - b[p[j],p[k]])
    # + a[k,i]*(b[p[k],p[j]] - b[p[k],p[i]]) + a[k,j]*(b[p[k],p[i]] - b[p[k],p[j]])
    # + a[i,j]*(b[p[j],p[i]] - b[p[i],p[j]]) + a[j,i]*(b[p[i],p[j]] - b[p[j],p[i]])
    
    best_cost = current_cost
    best_p = p.copy()
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while True:
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        # Randomize the order of pairs to avoid cycling
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(pairs)
        
        for i, j in pairs:
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break
            
            pi = p[i]
            pj = p[j]
            if pi == pj:
                continue
            
            # Compute delta for swapping i and j
            # Terms involving i and j directly
            # a[i,j] * (b[pj, pi] - b[pi, pj]) + a[j,i] * (b[pi, pj] - b[pj, pi])
            # = (a[i,j] - a[j,i]) * (b[pj, pi] - b[pi, pj])
            delta_ij = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
            
            # Terms with k != i, j
            # For facility i moving from pi to pj:
            #   sum_{k!=i,j} a[i,k] * (b[pj, p[k]] - b[pi, p[k]])
            # For facility j moving from pj to pi:
            #   sum_{k!=i,j} a[j,k] * (b[pi, p[k]] - b[pj, p[k]])
            # For location pi now holding j instead of i:
            #   sum_{k!=i,j} a[k,i] * (b[p[k], pj] - b[p[k], pi])
            # For location pj now holding i instead of j:
            #   sum_{k!=i,j} a[k,j] * (b[p[k], pi] - b[p[k], pj])
            
            # Vectorized computation
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            mask[j] = False
            ks = np.where(mask)[0]
            
            if len(ks) > 0:
                pk = p[ks]
                # a[i, ks] * (b[pj, pk] - b[pi, pk])
                term1 = np.dot(a[i, ks], b[pj, pk] - b[pi, pk])
                # a[j, ks] * (b[pi, pk] - b[pj, pk])
                term2 = np.dot(a[j, ks], b[pi, pk] - b[pj, pk])
                # a[ks, i] * (b[pk, pj] - b[pk, pi])
                term3 = np.dot(a[ks, i], b[pk, pj] - b[pk, pi])
                # a[ks, j] * (b[pk, pi] - b[pk, pj])
                term4 = np.dot(a[ks, j], b[pk, pi] - b[pk, pj])
            else:
                term1 = term2 = term3 = term4 = 0
            
            delta = delta_ij + term1 + term2 + term3 + term4
            
            if delta < -1e-9:
                # Perform swap
                p[i], p[j] = pj, pi
                current_cost += delta
                improved = True
                no_improve_count = 0
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
        else:
            no_improve_count = 0
    
    return best_p, best_cost


def _simulated_annealing(a, b, p, time_limit=110.0):
    """
    Simulated Annealing with 2-opt moves.
    """
    n = len(p)
    p = p.copy()
    current_cost = _qap_cost(a, b, p)
    best_cost = current_cost
    best_p = p.copy()
    
    start_time = time.time()
    
    # Temperature schedule
    T_start = 10000.0
    T_end = 1.0
    cooling_rate = 0.9995
    
    T = T_start
    rng = random.Random(42)
    
    while T > T_end:
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random 2-opt move
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        
        pi = p[i]
        pj = p[j]
        if pi == pj:
            continue
        
        # Compute delta
        delta_ij = (a[i, j] - a[j, i]) * (b[pj, pi] - b[pi, pj])
        
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        mask[j] = False
        ks = np.where(mask)[0]
        
        if len(ks) > 0:
            pk = p[ks]
            term1 = np.dot(a[i, ks], b[pj, pk] - b[pi, pk])
            term2 = np.dot(a[j, ks], b[pi, pk] - b[pj, pk])
            term3 = np.dot(a[ks, i], b[pk, pj] - b[pk, pi])
            term4 = np.dot(a[ks, j], b[pk, pi] - b[pk, pj])
        else:
            term1 = term2 = term3 = term4 = 0
        
        delta = delta_ij + term1 + term2 + term3 + term4
        
        # Accept or reject
        if delta < 0 or rng.random() < np.exp(-delta / T):
            p[i], p[j] = pj, pi
            current_cost += delta
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
        
        T *= cooling_rate
    
    return best_p, best_cost


def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s
    
    # Initial solution via rank matching
    p = _rank_matching_init(a, b)
    current_cost = _qap_cost(a, b, p)
    best_p = p.copy()
    best_cost = current_cost
    
    # Phase 1: Local search to polish the initial solution
    time_left = deadline - time.time()
    if time_left > 10:
        p, cost = _local_search_2opt(a, b, p, time_limit=time_left - 5, start_time=start_time)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Phase 2: Simulated Annealing to escape local optima
    time_left = deadline - time.time()
    if time_left > 20:
        p, cost = _simulated_annealing(a, b, p, time_limit=time_left - 5)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Phase 3: Final local search polish
    time_left = deadline - time.time()
    if time_left > 5:
        p, cost = _local_search_2opt(a, b, p, time_limit=time_left - 2, start_time=start_time)
        if cost < best_cost:
            best_cost = cost
            best_p = p.copy()
    
    # Ensure we return before the deadline
    if time.time() > deadline:
        pass  # We should have already returned best_p
    
    return best_p.tolist()
# EVOLVE-BLOCK-END
