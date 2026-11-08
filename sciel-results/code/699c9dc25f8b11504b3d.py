"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h) over all non-zero shifts.
    
    O_k = sum_i h_i * (1 - h_{i-k}) for shift k (circular).
    We want to minimize max_k O_k over k=1,...,N-1.
    
    Using FFT for efficiency.
    """
    # overlap at shift k: sum_i h[i] * (1 - h[(i-k) % N])
    # = sum_i h[i] - sum_i h[i]*h[(i-k)%N]
    # = N/2 - autocorr[k]
    # where autocorr[k] = sum_i h[i]*h[(i-k)%N]
    
    # Compute circular autocorrelation via FFT
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    # autocorr[k] for k=0,...,N-1; autocorr[0] = sum h^2
    # We need shifts 1..N-1
    max_overlap = N / 2.0 - np.min(autocorr[1:])
    return max_overlap


def _compute_all_overlaps(h):
    """Return all overlaps for shifts 1..N-1."""
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(H * np.conj(H)).real
    overlaps = N / 2.0 - autocorr[1:]
    return overlaps


def _soft_max(x, beta):
    """Soft maximum approximation using log-sum-exp."""
    # Scale to avoid overflow
    x_max = np.max(x)
    if x_max > 500 / beta:
        return x_max + np.log(np.sum(np.exp((x - x_max) * beta))) / beta
    return np.log(np.sum(np.exp(x * beta))) / beta


def _objective(h, beta):
    """Smooth objective: soft-max of overlaps."""
    overlaps = _compute_all_overlaps(h)
    return _soft_max(overlaps, beta)


def _project(h):
    """Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2."""
    h = np.clip(h, 0, 1)
    # Adjust to make sum exactly N/2
    diff = N / 2.0 - np.sum(h)
    if abs(diff) > 1e-12:
        h = h + diff / N
        h = np.clip(h, 0, 1)
        # If clipping changed the sum, do a more careful adjustment
        if abs(np.sum(h) - N / 2.0) > 1e-10:
            # Distribute the remaining difference to elements not at bounds
            diff = N / 2.0 - np.sum(h)
            if diff > 0:
                mask = h < 1.0
            else:
                mask = h > 0.0
            if np.sum(mask) > 0:
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0, 1)
    return h


def _coordinate_descent(h, max_iters=5000, tol=1e-10, deadline=None):
    """
    Coordinate descent to minimize the soft-max objective.
    """
    h = h.copy()
    beta = 50.0
    prev_obj = _objective(h, beta)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            # Try adjusting h[i]
            old_val = h[i]
            # Compute gradient-like direction: try small perturbations
            # Actually, for coordinate descent, we minimize over h[i] given others fixed
            
            # The overlap at shift k: O_k = N/2 - sum_j h[j]*h[(j-k)%N]
            # dO_k/dh[i] = -sum_j [h[j]*d(h[(j-k)%N])/dh[i]]
            # = -h[(i+k)%N] - h[(i-k)%N] for k != 0
            # More carefully: O_k = N/2 - sum_j h[j] * h[(j-k)%N]
            # dO_k/dh[i] = -h[(i+k)%N] (from term j=i in first factor) 
            #              -h[(i-k)%N] (from term where (j-k)%N = i, i.e., j = i+k)
            # Wait, let me redo this.
            # autocorr[k] = sum_j h[j] * h[(j-k)%N]
            # d(autocorr[k])/dh[i] = h[(i+k)%N] + (if i != 0) h[(i-k)%N] ... 
            # Actually for circular: d/dh[i] sum_j h[j] h[(j-k)%N]
            # = h[(i+k)%N] (when j=i) + sum_j h[j] * d/dh[i] h[(j-k)%N]
            # d/dh[i] h[(j-k)%N] = 1 if (j-k)%N = i, i.e., j = (i+k)%N
            # So = h[(i+k)%N] + h[(i+k)%N] = 2*h[(i+k)%N] ... no wait
            # Let me be more careful.
            # autocorr[k] = sum_j h[j] * h[(j-k) mod N]
            # Derivative w.r.t. h[i]:
            # Term 1: when j = i, we get h[(i-k) mod N]
            # Term 2: when (j-k) mod N = i, i.e., j = (i+k) mod N, we get h[(i+k) mod N]
            # So d(autocorr[k])/dh[i] = h[(i-k) mod N] + h[(i+k) mod N]
            # (Note: when k=0, these are the same term counted twice, giving 2*h[i])
            
            # O_k = N/2 - autocorr[k]
            # dO_k/dh[i] = -h[(i-k) mod N] - h[(i+k) mod N]
            
            # For soft-max objective, we need weighted sum of these gradients.
            # This is getting complex. Let's use a simpler approach: 
            # try perturbing h[i] and accept if it improves.
            
            best_val = old_val
            best_obj = prev_obj
            
            # Try a few candidate values
            for candidate in [old_val + 0.01, old_val - 0.01, old_val + 0.001, old_val - 0.001, 0.0, 1.0]:
                candidate = np.clip(candidate, 0, 1)
                if candidate == old_val:
                    continue
                h[i] = candidate
                obj = _objective(h, beta)
                if obj < best_obj:
                    best_obj = obj
                    best_val = candidate
            
            h[i] = best_val
            if best_val != old_val:
                improved = True
        
        if not improved:
            break
        
        # Recompute objective after full pass
        curr_obj = _objective(h, beta)
        if abs(curr_obj - prev_obj) < tol:
            prev_obj = curr_obj
            break
        prev_obj = curr_obj
    
    return _project(h)


def _genetic_algorithm(pop_size=50, max_gens=200, deadline=None):
    """
    Genetic algorithm to search for good binary-like solutions.
    """
    # Initialize population with random balanced binary vectors
    population = []
    for _ in range(pop_size):
        h = np.zeros(N)
        # Choose N/2 positions to be 1
        idx = np.random.choice(N, N // 2, replace=False)
        h[idx] = 1.0
        population.append(h)
    
    # Also include the constant solution
    population.append(np.full(N, 0.5))
    
    best_h = None
    best_score = float('inf')
    
    # Evaluate initial population
    for h in population:
        score = _compute_max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    for gen in range(max_gens):
        if deadline is not None and time.time() > deadline:
            break
        
        # Evaluate all
        scores = [_compute_max_overlap(h) for h in population]
        
        # Update best
        for h, s in zip(population, scores):
            if s < best_score:
                best_score = s
                best_h = h.copy()
        
        # Selection: keep top half
        idx = np.argsort(scores)
        elite = [population[i].copy() for i in idx[:pop_size // 2]]
        
        # Generate new population
        new_pop = list(elite)
        while len(new_pop) < pop_size:
            if deadline is not None and time.time() > deadline:
                break
            # Crossover
            i1, i2 = np.random.randint(0, len(elite), 2)
            parent1 = elite[i1]
            parent2 = elite[i2]
            if np.random.random() < 0.5:
                # Uniform crossover
                mask = np.random.random(N) < 0.5
                child = np.where(mask, parent1, parent2)
            else:
                # Single point crossover
                pt = np.random.randint(1, N)
                child = np.concatenate([parent1[:pt], parent2[pt:]])
            
            # Mutation: flip some bits
            if np.random.random() < 0.7:
                n_flips = np.random.randint(1, 5)
                flip_idx = np.random.choice(N, n_flips, replace=False)
                for fi in flip_idx:
                    child[fi] = 1.0 - child[fi]
            
            # Ensure sum is N/2
            s = np.sum(child)
            if s != N / 2.0:
                # Adjust
                if s > N / 2.0:
                    # Need to turn off some 1s
                    excess = int(s - N / 2.0)
                    ones = np.where(child > 0.5)[0]
                    if len(ones) >= excess:
                        turn_off = np.random.choice(ones, excess, replace=False)
                        child[turn_off] = 0.0
                else:
                    deficit = int(N / 2.0 - s)
                    zeros = np.where(child < 0.5)[0]
                    if len(zeros) >= deficit:
                        turn_on = np.random.choice(zeros, deficit, replace=False)
                        child[turn_on] = 1.0
            
            new_pop.append(child)
        
        population = new_pop[:pop_size]
    
    return best_h, best_score


def _polish(h, deadline=None):
    """
    Final polish using coordinate descent with tighter tolerances.
    """
    h = _project(h)
    beta = 100.0
    prev_obj = _objective(h, beta)
    
    for it in range(2000):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_val = h[i]
            best_val = old_val
            best_obj = prev_obj
            
            # Fine-grained search around current value
            for delta in [0.005, -0.005, 0.002, -0.002, 0.001, -0.001, 0.0005, -0.0005]:
                candidate = np.clip(old_val + delta, 0, 1)
                h[i] = candidate
                obj = _objective(h, beta)
                if obj < best_obj:
                    best_obj = obj
                    best_val = candidate
            
            h[i] = best_val
            if best_val != old_val:
                improved = True
        
        if not improved:
            break
        
        curr_obj = _objective(h, beta)
        if abs(curr_obj - prev_obj) < 1e-12:
            break
        prev_obj = curr_obj
    
    return _project(h)


def construct():
    """
    Returns the step heights for the Erdos minimum overlap problem.
    
    Strategy:
    1. Start with binary balanced sequences (genetic algorithm)
    2. Polish with coordinate descent
    3. Also try continuous optimization from multiple starts
    4. Return the best solution found
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Phase 1: Genetic algorithm on binary sequences
    if time.time() < deadline - 60:
        ga_h, ga_score = _genetic_algorithm(pop_size=40, max_gens=100, deadline=deadline - 30)
        if ga_score < best_score:
            best_score = ga_score
            best_h = ga_h
    
    # Phase 2: Polish the best solution found
    if time.time() < deadline - 20:
        polished_h = _polish(best_h.copy(), deadline=deadline - 5)
        polished_score = _compute_max_overlap(polished_h)
        if polished_score < best_score:
            best_score = polished_score
            best_h = polished_h
    
    # Phase 3: Try a few random continuous starts with coordinate descent
    if time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 10:
                break
            # Random continuous start
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _project(h_rand)
            cd_h = _coordinate_descent(h_rand, max_iters=1000, tol=1e-10, deadline=deadline - 5)
            cd_score = _compute_max_overlap(cd_h)
            if cd_score < best_score:
                best_score = cd_score
                best_h = cd_h
    
    # Final polish
    if time.time() < deadline - 3:
        final_h = _polish(best_h.copy(), deadline=deadline)
        final_score = _compute_max_overlap(final_h)
        if final_score < best_score:
            best_score = final_score
            best_h = final_h
    
    return _project(best_h)
# EVOLVE-BLOCK-END
