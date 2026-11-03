"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import eigh
import time


def _spectral_init(a, b):
    """
    Spectral initialization: align high-flow facilities with high-distance locations
    using principal eigenvectors. This provides a geometrically informed starting point
    that respects the structure of both matrices simultaneously.
    """
    n = a.shape[0]
    
    a_sym = (a + a.T) / 2.0
    b_sym = (b + b.T) / 2.0
    
    try:
        # Principal eigenvectors capture the dominant structure
        eigvals_a, eigvecs_a = eigh(a_sym)
        v_a = eigvecs_a[:, -1]  # Largest eigenvalue eigenvector
        
        eigvals_b, eigvecs_b = eigh(b_sym)
        v_b = eigvecs_b[:, -1]  # Largest eigenvalue eigenvector
        
        # We want to minimize sum a[i,j]*b[p[i],p[j]]
        # High flow pairs should go to low distance pairs
        # So high v_a facilities -> low v_b locations
        facility_order = np.argsort(-v_a)  # High flow centrality first
        location_order = np.argsort(v_b)   # Low distance centrality first
        
        p = np.empty(n, dtype=np.int64)
        p[facility_order] = location_order
        
    except Exception:
        # Fallback: rank by row sums
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central
    
    return p


def _compute_cost(a, b, p):
    """Compute QAP cost: sum_{i,j} a[i,j] * b[p[i],p[j]]"""
    b_perm = b[np.ix_(p, p)]
    return np.sum(a * b_perm)


def _fast_2opt(a, b, p, time_budget=105):
    """
    Fast 2-opt local search with O(1) delta computation per swap.
    
    Delta cost for swapping positions i and j (facilities f_i=p[i], f_j=p[j]):
    We remove contributions involving f_i at loc_i and f_j at loc_j,
    then add contributions with f_i at loc_j and f_j at loc_i.
    
    delta = sum_k [a[i,k]*b[pi,k] - a[i,k]*b[pj,k]] 
          + sum_k [a[k,i]*b[k,pj] - a[k,i]*b[k,pi]]
          + sum_k [a[j,k]*b[pj,k] - a[j,k]*b[pi,k]]
          + sum_k [a[k,j]*b[k,pi] - a[k,j]*b[k,pj]]
          - a[i,j]*b[pi,pj] - a[j,i]*b[pj,pi]
          + a[i,j]*b[pj,pi] + a[j,i]*b[pi,pj]
    """
    start = time.time()
    n = len(p)
    p = p.copy()
    current_cost = _compute_cost(a, b, p)
    
    # Precompute for fast delta: 
    # For each facility f, we need:
    #   out_f[k] = a[f,k]  (flow from f to k)
    #   in_f[k] = a[k,f]   (flow from k to f)
    # For each location l, we need:
    #   dist_to_l[k] = b[l,k] (distance from l to k)
    #   dist_from_l[k] = b[k,l] (distance from k to l)
    
    # Precompute row/col sums of a weighted by b for current assignment
    # Actually, let's precompute the "interaction" terms
    
    # For facility f at location l, its contribution to total cost is:
    #   sum_k a[f,k] * b[l, p[k]] + sum_k a[k,f] * b[p[k], l]
    # But this double counts, so we use half.
    
    # Let's precompute:
    #   S_out[f] = vector where S_out[f][k] = a[f,k]
    #   S_in[f] = vector where S_in[f][k] = a[k,f]
    #   D_to[l] = vector where D_to[l][k] = b[l,k]
    #   D_from[l] = vector where D_from[l][k] = b[k,l]
    
    # For swap of positions i,j with facilities fi=p[i], fj=p[j]:
    # delta = (a[fi,:] * (b[pj,:] - b[pi,:])) @ 1
    #       + (a[:,fi] * (b[:,pj] - b[:,pi])) @ 1
    #       + (a[fj,:] * (b[pi,:] - b[pj,:])) @ 1
    #       + (a[:,fj] * (b[:,pi] - b[:,pj])) @ 1
    #       - a[fi,fj]*b[pi,pj] - a[fj,fi]*b[pj,pi]
    #       + a[fi,fj]*b[pj,pi] + a[fj,fi]*b[pi,pj]
    
    # Simplify:
    # delta = sum_k a[fi,k]*(b[pj,k]-b[pi,k]) + sum_k a[k,fi]*(b[k,pj]-b[k,pi])
    #       + sum_k a[fj,k]*(b[pi,k]-b[pj,k]) + sum_k a[k,fj]*(b[k,pi]-b[k,pj])
    #       + a[fi,fj]*(b[pj,pi]-b[pi,pj]) + a[fj,fi]*(b[pi,pj]-b[pj,pi])
    
    # Precompute a_rows[f] = a[f,:], a_cols[f] = a[:,f]
    # b_rows[l] = b[l,:], b_cols[l] = b[:,l]
    
    a_rows = a  # a[f, k]
    a_cols = a.T  # a_cols[k, f] = a[f, k]... wait, a_cols[f, k] should be a[k, f]
    # Actually a_cols = a.T so a_cols[f,k] = a[k,f]. Good.
    
    b_rows = b  # b[l, k]
    b_cols = b.T  # b_cols[k, l] = b[l, k]
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start
        if elapsed > time_budget:
            break
        
        best_delta = 0
        best_i = -1
        best_j = -1
        
        for i in range(n):
            fi = p[i]
            # Precompute delta contributions for swapping fi with each fj
            # delta_ij = sum_k a[fi,k]*(b[pj,k]-b[pi,k]) 
            #          + sum_k a[k,fi]*(b[k,pj]-b[k,pi])
            #          + sum_k a[fj,k]*(b[pi,k]-b[pj,k])
            #          + sum_k a[k,fj]*(b[k,pi]-b[k,pj])
            #          + a[fi,fj]*(b[pj,pi]-b[pi,pj]) + a[fj,fi]*(b[pi,pj]-b[pj,pi])
            
            # Let's precompute for fixed i:
            # term1[fj] = sum_k a[fi,k] * b[pj,k]  where pj = p[j]
            # term2[fj] = sum_k a[k,fi] * b[k,pj]
            # term3[fj] = sum_k a[fj,k] * b[pi,k]
            # term4[fj] = sum_k a[k,fj] * b[k,pi]
            
            # For fixed i, pi = p[i], fi = p[i]
            pi = p[i]
            
            # Precompute vectors that don't depend on j:
            # v_pi_row = b[pi, :]  (row pi of b)
            # v_pi_col = b[:, pi]  (col pi of b)
            
            for j in range(i + 1, n):
                fj = p[j]
                pj = p[j]
                
                if fj == fi:
                    continue
                
                # Compute delta
                # Part 1: fi moves from pi to pj
                # sum_k a[fi,k]*(b[pj,k] - b[pi,k])
                part1 = np.dot(a[fi, :], b[pj, :]) - np.dot(a[fi, :], b[pi, :])
                
                # sum_k a[k,fi]*(b[k,pj] - b[k,pi])
                part2 = np.dot(a[:, fi], b[:, pj]) - np.dot(a[:, fi], b[:, pi])
                
                # Part 2: fj moves from pj to pi
                # sum_k a[fj,k]*(b[pi,k] - b[pj,k])
                part3 = np.dot(a[fj, :], b[pi, :]) - np.dot(a[fj, :], b[pj, :])
                
                # sum_k a[k,fj]*(b[k,pi] - b[k,pj])
                part4 = np.dot(a[:, fj], b[:, pi]) - np.dot(a[:, fj], b[:, pj])
                
                # Cross terms
                part5 = a[fi, fj] * (b[pj, pi] - b[pi, pj])
                part6 = a[fj, fi] * (b[pi, pj] - b[pj, pi])
                
                delta = part1 + part2 + part3 + part4 + part5 + part6
                
                if delta < best_delta:
                    best_delta = delta
                    best_i = i
                    best_j = j
        
        if best_i >= 0:
            # Apply the best swap
            p[best_i], p[best_j] = p[best_j], p[best_i]
            current_cost += best_delta
            improved = True
        
        elapsed = time.time() - start
        if elapsed > time_budget:
            break
    
    return p, current_cost


def _perturb(p, num_swaps=3):
    """Apply random swaps as perturbation."""
    p = p.copy()
    n = len(p)
    for _ in range(num_swaps):
        i, j = np.random.choice(n, size=2, replace=False)
        p[i], p[j] = p[j], p[i]
    return p


def solve(a, b):
    n = a.shape[0]
    start_time = time.time()
    
    # Spectral initialization
    p_spectral = _spectral_init(a, b)
    
    # Iterated Local Search
    p_best = p_spectral.copy()
    cost_best = _compute_cost(a, b, p_best)
    
    p_current = p_best.copy()
    
    iteration = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed > 108:
            break
        
        # Local search
        p_current, cost_current = _fast_2opt(a, b, p_current, time_budget=108)
        
        if cost_current < cost_best:
            cost_best = cost_current
            p_best = p_current.copy()
        
        # Check time
        elapsed = time.time() - start_time
        if elapsed > 110:
            break
        
        # Perturbation: increase strength over iterations
        num_swaps = min(2 + iteration // 5, 8)
        p_current = _perturb(p_best, num_swaps)
        
        iteration += 1
    
    return p_best.tolist()
# EVOLVE-BLOCK-END
