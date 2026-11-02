"""
Strategy: Spectral/Sidon-set structural design with constrained amplitude optimization.
Instead of generic random search, we:
1. Identify a support set of indices that minimizes the number of off-diagonal 
   autocorrelation collisions (Sidon set / Golomb ruler properties).
2. Optimize the amplitudes on this fixed support to maximize the ratio 
   ||h||_2^2 / (||h||_1 * sup|h*h|).
This decouples the combinatorial structure (support) from the continuous 
amplitude optimization, allowing for a more targeted search.
"""
import numpy as np
import time
from itertools import combinations

N = 50

def _ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * sup|h*h|)"""
    h = np.asarray(h, dtype=float)
    l2_sq = np.sum(h**2)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    # Autoconvolution
    conv = np.convolve(h, h, mode='full')
    sup_conv = np.max(conv)
    if sup_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * sup_conv)

def _collision_count(indices):
    """
    Count the number of pairs (i, j) with i != j such that (i - j) == (k - l)
    for distinct pairs in the support. This relates to the off-diagonal peaks 
    of the autocorrelation. Lower collision count generally implies a flatter 
    autocorrelation tail, which helps the ratio.
    """
    diffs = set()
    count = 0
    idx_list = list(indices)
    for i in range(len(idx_list)):
        for j in range(i + 1, len(idx_list)):
            d = idx_list[i] - idx_list[j]
            if d in diffs:
                count += 1
            else:
                diffs.add(d)
    return count

def _find_sidon_supports(n_support, max_iter=1000, seed=42):
    """
    Find sets of indices with low collision counts (Sidon-like properties).
    Uses a simple greedy + random restart approach.
    """
    np.random.seed(seed)
    best_supports = []
    
    for _ in range(max_iter):
        # Random start
        support = set(np.random.choice(N, size=n_support, replace=False))
        current_collisions = _collision_count(support)
        
        # Local search: swap elements to reduce collisions
        improved = True
        while improved and len(best_supports) < 5:
            improved = False
            # Try swapping one element
            for elem in list(support):
                for new_elem in range(N):
                    if new_elem in support:
                        continue
                    trial = (support - {elem}) | {new_elem}
                    new_collisions = _collision_count(trial)
                    if new_collisions < current_collisions:
                        support = trial
                        current_collisions = new_collisions
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                # Check if this is a good candidate
                if current_collisions < 5: # Threshold for "good" structure
                    # Normalize for comparison
                    if all(len(tuple(sorted(s))) != len(tuple(sorted(b))) for b in best_supports):
                        best_supports.append(tuple(sorted(support)))
                break
    
    return best_supports

def _optimize_amplitudes(support, time_limit):
    """
    Optimize amplitudes on a fixed support using a simple Nelder-Mead-like 
    simplex method adapted for non-negative constraints.
    """
    n_vars = len(support)
    if n_vars == 0:
        return np.zeros(N)
    
    # Initial simplex: uniform + perturbations
    h0 = np.zeros(N)
    for idx in support:
        h0[idx] = 1.0
    
    # Create initial simplex
    simplex = [h0.copy()]
    for i in range(n_vars):
        v = h0.copy()
        v[list(support)[i]] *= 1.5 # Perturb one variable
        simplex.append(v)
    
    best_h = h0.copy()
    best_f = _ratio(h0)
    
    start_time = time.time()
    max_iter = 2000
    
    for _ in range(max_iter):
        if time.time() > time_limit:
            break
        
        # Evaluate simplex
        f_vals = [_ratio(v) for v in simplex]
        
        # Find best and worst
        best_idx = np.argmin(f_vals) # Wait, we want to MAXIMIZE ratio. 
        # Correction: f_vals are ratios. We want highest ratio.
        # Let's use negative ratio for minimization or just track max.
        # Let's stick to maximizing.
        
        # Re-evaluate logic for Maximization
        # Sort by value descending
        sorted_indices = np.argsort(f_vals)[::-1]
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]
        second_worst_idx = sorted_indices[-2]
        
        if f_vals[best_idx] > best_f:
            best_f = f_vals[best_idx]
            best_h = simplex[best_idx].copy()
        
        # Reflection
        centroid = np.mean([simplex[i] for i in range(len(simplex)) if i != worst_idx], axis=0)
        reflection = centroid + (centroid - simplex[worst_idx])
        
        # Enforce non-negativity
        reflection = np.clip(reflection, 0, None)
        
        f_reflection = _ratio(reflection)
        
        if f_reflection >= f_vals[second_worst_idx]:
            if f_reflection > f_vals[best_idx]:
                # Expansion
                expansion = centroid + 2 * (reflection - centroid)
                expansion = np.clip(expansion, 0, None)
                f_expansion = _ratio(expansion)
                if f_expansion > f_reflection:
                    simplex[worst_idx] = expansion
                else:
                    simplex[worst_idx] = reflection
            else:
                simplex[worst_idx] = reflection
        else:
            # Contraction
            if f_reflection < f_vals[worst_idx]:
                # Outside contraction
                contraction = centroid - 0.5 * (simplex[worst_idx] - centroid)
                contraction = np.clip(contraction, 0, None)
                f_contraction = _ratio(contraction)
                if f_contraction >= f_vals[worst_idx]:
                    simplex[worst_idx] = contraction
                else:
                    # Shrink
                    for i in range(len(simplex)):
                        if i != best_idx:
                            simplex[i] = simplex[best_idx] + 0.5 * (simplex[i] - simplex[best_idx])
                            simplex[i] = np.clip(simplex[i], 0, None)
            else:
                # Inside contraction
                contraction = centroid + 0.5 * (simplex[worst_idx] - centroid)
                contraction = np.clip(contraction, 0, None)
                simplex[worst_idx] = contraction
        
        # Check for convergence
        if len(set([round(f, 10) for f in f_vals])) == 1:
            break
            
    return best_h

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    best_h = np.ones(N)
    best_ratio = _ratio(best_h)
    
    # 1. Generate candidate support sets (Sidon-like)
    # We try different sizes of support. Sparse supports often have better ratio 
    # if amplitudes are optimized, but too sparse might limit L2/L1 ratio.
    # Let's try sizes 10 to 25.
    
    candidate_supports = []
    
    # Single spike is a baseline
    for size in [10, 12, 15, 20, 25]:
        supports = _find_sidon_supports(size, max_iter=200, seed=42 + size)
        candidate_supports.extend(supports)
        
        if time.time() > deadline - 20:
            break
            
    # Also include some random supports for diversity
    np.random.seed(123)
    for size in [15, 20]:
        for _ in range(10):
            supp = tuple(sorted(np.random.choice(N, size=size, replace=False)))
            candidate_supports.append(supp)
            
    if time.time() > deadline - 10:
        return list(best_h)
    
    # 2. Optimize amplitudes for each support
    # We allocate time based on remaining budget
    remaining_time = deadline - time.time()
    time_per_support = max(0.5, remaining_time / max(1, len(candidate_supports)))
    
    for supp in candidate_supports:
        if time.time() > deadline - 2:
            break
            
        t_limit = time.time() + time_per_support
        h_opt = _optimize_amplitudes(supp, t_limit)
        
        r = _ratio(h_opt)
        if r > best_ratio:
            best_ratio = r
            best_h = h_opt.copy()
            
    # 3. Final fine-tuning on the best found structure
    # Extract support of best_h
    active_indices = np.where(best_h > 1e-6)[0]
    if len(active_indices) > 0:
        supp_best = tuple(active_indices)
        # Run a final optimization with all remaining time
        t_limit = time.time() + (deadline - time.time() - 1.0)
        if t_limit > time.time():
            h_final = _optimize_amplitudes(supp_best, t_limit)
            r_final = _ratio(h_final)
            if r_final > best_ratio:
                best_ratio = r_final
                best_h = h_final.copy()
                
    return list(best_h)
# EVOLVE-BLOCK-END
