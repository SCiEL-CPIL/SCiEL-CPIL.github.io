import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
import random

N = 50
DELTA = 0.5 / N  # Step size for [-1/4, 1/4]

def _score_raw(h):
    """
    Computes the raw score for a given height vector.
    Score = (sum(h^2)) / (sum(h) * max(conv(h,h)))
    Note: The autoconvolution in the discrete sum context usually implies 
    the peak value of the discrete convolution sum. 
    The task description says "supremum norm of the autoconvolution".
    For step functions on equal intervals, the autoconvolution is a piecewise constant 
    function. The supremum is the maximum value of the discrete convolution sum 
    multiplied by the interval width if we consider the integral definition, 
    or just the max sum if we consider the discrete definition normalized appropriately.
    
    Let's look at the seed program's _score_raw:
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    return (l2_sq / (l1 * autoconv_max))
    
    This matches the definition:
    L2_norm_sq = sum(h_i^2)
    L1_norm = sum(h_i)
    Sup_autoconv = max_i (h*h)_i * delta (if h_i are heights of rectangles)
    Wait, if h_i are heights, the function is f(t) = h_i for t in [x_i, x_i+delta].
    The convolution (f*f)(t) = integral f(s)f(t-s) ds.
    The peak of this convolution for step functions is indeed related to the discrete convolution.
    Specifically, if the steps are aligned, the peak value is delta * sum_{i+j=k} h_i h_j.
    The seed uses DELTA * max(discrete_conv). I will stick to this definition to be consistent 
    with the "seed" behavior which scored 1.1944.
    """
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 == 0:
        return 0.0
    l2_sq = np.sum(h**2)
    h_conv = np.convolve(h, h, mode='full')
    autoconv_max = DELTA * np.max(h_conv)
    if autoconv_max == 0:
        return 0.0
    return (l2_sq / (l1 * autoconv_max))

def _evaluate_candidate(indices, weights):
    """
    Constructs the full vector h from indices and weights, and returns the score.
    indices: array of indices where h is non-zero
    weights: array of positive values for those indices
    """
    h = np.zeros(N)
    h[indices] = weights
    return _score_raw(h)

def _optimize_weights_for_indices(indices):
    """
    Given a fixed set of indices, optimize the weights to maximize the score.
    Score = sum(w_i^2) / (sum(w_i) * max_k sum_{i+j=k} w_i w_j)
    This is a non-linear optimization problem in len(indices) variables.
    We can use scipy.optimize.minimize on the negative score.
    """
    k = len(indices)
    if k == 0:
        return 0.0, np.array([])
    if k == 1:
        # Score is 1 for any single spike (scale invariant)
        return 1.0, np.array([1.0])
    
    def neg_score(w):
        # w must be non-negative
        if np.any(w < 0):
            return 0
        h = np.zeros(N)
        h[indices] = w
        l1 = np.sum(w)
        if l1 == 0: return 0
        l2_sq = np.sum(w**2)
        # Compute conv peak efficiently for small k? 
        # For k up to 10-15, full conv of N=50 is fast enough.
        h_conv = np.convolve(h, h, mode='full')
        autoconv_max = DELTA * np.max(h_conv)
        if autoconv_max == 0: return 0
        return -(l2_sq / (l1 * autoconv_max))
        
    # Initial guess: all ones
    w0 = np.ones(k)
    
    bounds = [(0.0, 10.0)] * k
    try:
        res = minimize(neg_score, w0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        w_opt = res.x
        # Ensure non-negative
        w_opt = np.maximum(w_opt, 0)
        if np.sum(w_opt) == 0:
            return 0.0, np.array([])
        score = -res.fun
        return score, w_opt
    except Exception:
        return 0.0, np.array([])

def _random_indices(k, rng):
    """Generate k distinct random indices from 0 to N-1."""
    return np.sort(rng.choice(N, k, replace=False))

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    best_score = 0.0
    best_h = np.zeros(N)
    
    # 1. Baseline: Uniform
    h_uniform = np.ones(N)
    s = _score_raw(h_uniform)
    best_score = s
    best_h = h_uniform.copy()
    
    # 2. Baseline: Single spike
    h_spike = np.zeros(N)
    h_spike[0] = 1.0
    s = _score_raw(h_spike)
    if s > best_score:
        best_score = s
        best_h = h_spike.copy()
        
    # 3. Baseline: Two spikes
    # Optimal two spikes are usually far apart to minimize conv overlap? 
    # Or close? 
    # Conv of two deltas is delta at i and delta at j and 2delta at i+j? No.
    # h = [1, 0, ..., 1]. Conv has 1 at i, 1 at j, 1 at i+j? 
    # Actually conv of e_i and e_j is e_{i+j}. 
    # If i != j, peaks are 1. If i=j, peak is 1.
    # Score for 2 spikes: L2=2, L1=2, MaxConv=1 (if distinct) or 1 (if same). 
    # Ratio = 2 / (2*1) = 1.
    # Uniform score: L2=N, L1=N, MaxConv ~ N^2/4? 
    # For uniform, conv is triangular. Peak is N. 
    # Ratio = N / (N * N) = 1/N = 0.02. 
    # So sparse is much better.
    
    # Strategy: Search over sparsity k and index sets.
    # For each k, we try several random index sets, then optimize weights.
    
    # We will use a simple evolutionary approach for indices.
    # Population of index sets.
    
    max_k = 15 # Try up to 15 non-zero elements
    min_k = 2
    
    # Pre-compute some good structures?
    # Sidon sets are good because they have distinct sums, minimizing the "collision" in convolution peaks.
    # If sums are distinct, the max conv term is just sum of one product, not a sum of multiple products.
    # This keeps the denominator small relative to L2^2.
    
    def generate_sidon_indices(k, rng):
        """Greedy Sidon set generation."""
        S = []
        sums = set()
        # Start with 0
        S.append(0)
        for i in range(1, N):
            if len(S) >= k:
                break
            valid = True
            new_sums = []
            for s in S:
                val = s + i
                if val in sums:
                    valid = False
                    break
                new_sums.append(val)
            if valid:
                for val in new_sums:
                    sums.add(val)
                S.append(i)
        return np.array(S) if len(S) >= k else None

    # Initial population of index sets
    # We want a diverse set of k values and index patterns.
    
    candidates = []
    
    # Add some known good structures
    # 1. Sidon sets
    for k in range(min_k, max_k + 1):
        for _ in range(5):
            sidon = generate_sidon_indices(k, np.random.default_rng())
            if sidon is not None:
                candidates.append(sidon)
                
    # 2. Random sparse sets
    for k in range(min_k, max_k + 1):
        for _ in range(20):
            idx = _random_indices(k, np.random.default_rng())
            candidates.append(idx)
            
    # 3. Clusters (consecutive indices)
    for k in range(min_k, max_k + 1):
        for _ in range(5):
            start = np.random.randint(0, N - k + 1)
            idx = np.arange(start, start + k)
            candidates.append(idx)

    # Evaluate initial population
    evaluated_indices = []
    for idx in candidates:
        score, weights = _optimize_weights_for_indices(idx)
        if score > best_score:
            best_score = score
            h_new = np.zeros(N)
            h_new[idx] = weights
            best_h = h_new
        evaluated_indices.append((score, idx))
        
    # Sort by score
    evaluated_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Keep top M candidates for evolution
    top_m = 50
    population = [idx for _, idx in evaluated_indices[:top_m]]
    
    # Evolutionary loop
    # We will evolve the index sets.
    # Mutation: Swap an index, or shift an index.
    # Crossover: Take half indices from parent 1, half from parent 2.
    
    num_generations = 20
    pop_size = len(population)
    
    for gen in range(num_generations):
        if time.time() > deadline - 20:
            break
            
        new_population = []
        
        # Elitism: Keep best few
        # Re-evaluate best to be sure
        best_in_pop_score, best_in_pop_idx = _optimize_weights_for_indices(population[0])
        
        # Generate offspring
        for i in range(pop_size):
            if time.time() > deadline - 20:
                break
                
            # Mutation
            if np.random.rand() < 0.7:
                # Point mutation: change one index
                new_idx = population[i].copy()
                if len(new_idx) > 0:
                    pos = np.random.randint(0, len(new_idx))
                    # Replace with random new index
                    new_val = np.random.randint(0, N)
                    # Ensure uniqueness
                    if new_val not in new_idx:
                        new_idx[pos] = new_val
                        new_idx.sort()
                    else:
                        # Try to find a unique one
                        for _ in range(10):
                            new_val = np.random.randint(0, N)
                            if new_val not in new_idx:
                                new_idx[pos] = new_val
                                new_idx.sort()
                                break
            else:
                # Crossover: Mix two parents
                if pop_size > 1:
                    j = np.random.randint(0, pop_size)
                    if j == i:
                        j = (j + 1) % pop_size
                    p1 = population[i]
                    p2 = population[j]
                    # Take roughly half from each
                    len_new = len(p1) # Keep size of p1
                    # Sample from union? Or just take first half of p1 and second half of p2?
                    # Better: Randomly select indices from p1 and p2 to form new set of size len_new
                    # Ensure distinct
                    new_idx = []
                    # Shuffle p1 and p2
                    p1_shuf = p1.copy()
                    p2_shuf = p2.copy()
                    np.random.shuffle(p1_shuf)
                    np.random.shuffle(p2_shuf)
                    
                    # Interleave or random mix
                    # Let's just take random unique indices from the union of p1 and p2
                    union = np.unique(np.concatenate([p1, p2]))
                    if len(union) < len_new:
                         # Not enough unique indices, fill with random
                         missing = len_new - len(union)
                         extra = np.random.choice(N, missing, replace=False)
                         union = np.unique(np.concatenate([union, extra]))
                    
                    # Select len_new from union
                    if len(union) >= len_new:
                        new_idx = np.sort(np.random.choice(union, len_new, replace=False))
                    else:
                        new_idx = np.sort(union)
                        
            # Evaluate offspring
            score, weights = _optimize_weights_for_indices(new_idx)
            new_population.append((score, new_idx))
            
        # Update population
        # Combine parents and offspring, sort by score, keep top pop_size
        all_candidates = [(s, idx) for s, idx in new_population]
        # Add parents back
        for idx in population:
             s, _ = _optimize_weights_for_indices(idx) # Re-eval might be expensive, use cached if possible
             # For simplicity, we'll just re-evaluate the top few parents if time permits
             # Actually, we can just keep the scores from the previous step if we stored them.
             # Let's store scores in the population list.
             pass
             
        # Let's refine the population update logic to be more efficient.
        # We will maintain a list of (score, idx, weights) for the current population.
        
    # Since the above loop logic was getting complex and potentially slow due to re-evaluation, 
    # let's switch to a simpler, more robust hill-climbing / random restart approach 
    # that is easier to bound in time.
    
    # Reset time check
    if time.time() > deadline - 30:
         return list(best_h)
         
    # Alternative Strategy: Direct Random Restart with Local Optimization
    # We will try many random sparse configurations and optimize weights locally.
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        # Randomly choose sparsity k
        k = np.random.randint(min_k, max_k + 1)
        
        # Choose indices
        # Mix of random and structured
        if np.random.rand() < 0.5:
            # Random
            idx = _random_indices(k, np.random.default_rng())
        else:
            # Structured: Sidon-like
            sidon = generate_sidon_indices(k, np.random.default_rng())
            if sidon is not None:
                idx = sidon
            else:
                idx = _random_indices(k, np.random.default_rng())
                
        # Optimize weights
        score, weights = _optimize_weights_for_indices(idx)
        
        if score > best_score:
            best_score = score
            h_new = np.zeros(N)
            h_new[idx] = weights
            best_h = h_new
            
    # Final Polish:
    # Take the best_h found and try to add/remove one element or tweak weights further.
    if time.time() < deadline - 5:
        # Extract non-zero indices from best_h
        non_zero_idx = np.nonzero(best_h)[0]
        if len(non_zero_idx) > 0:
            # Try adding one more random index
            for _ in range(10):
                new_idx_list = list(non_zero_idx)
                new_val = np.random.randint(0, N)
                if new_val not in new_idx_list:
                    new_idx_list.append(new_val)
                    new_idx = np.array(new_idx_list)
                    score, weights = _optimize_weights_for_indices(new_idx)
                    if score > best_score:
                        best_score = score
                        h_new = np.zeros(N)
                        h_new[new_idx] = weights
                        best_h = h_new
                        non_zero_idx = new_idx
                        break # Restart adding if improved? Or just continue?
                
            # Try removing one index
            if len(non_zero_idx) > 2:
                for _ in range(10):
                    remove_pos = np.random.randint(0, len(non_zero_idx))
                    new_idx = np.delete(non_zero_idx, remove_pos)
                    score, weights = _optimize_weights_for_indices(new_idx)
                    if score > best_score:
                        best_score = score
                        h_new = np.zeros(N)
                        h_new[new_idx] = weights
                        best_h = h_new
                        non_zero_idx = new_idx
                        break
                        
    return list(best_h)
