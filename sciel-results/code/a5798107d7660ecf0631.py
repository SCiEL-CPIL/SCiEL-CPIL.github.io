"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

# Global state for the cap set solver
_start_time = None
_DEADLINE = 325.0  # Safety margin before 330s kill

def _init_state(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _is_deadline_exceeded():
    return _elapsed() > _DEADLINE

# Cache for precomputed orderings and their metadata
_orderings = {}  # n -> {'order': list, 'rank_map': dict, 'size': int, 'done': bool}

def _fast_greedy_eval(order, n):
    """
    Fast evaluation of greedy cap set size for a given order (list of vectors).
    Returns the size.
    Optimized with local variable bindings.
    """
    cap_set = set()
    forbidden = set()
    size = 0
    
    # Local bindings for speed
    cap_add = cap_set.add
    forb_add = forbidden.add
    cap_contains = cap_set.__contains__
    forb_contains = forbidden.__contains__
    
    for v in order:
        if not forb_contains(v):
            cap_add(v)
            size += 1
            # Update forbidden set
            # For each u in cap_set, the third point w is (-v - u) % 3
            # We iterate over the *current* cap_set including v
            # Note: cap_set is modified during iteration, so we must be careful.
            # However, we only need to add new forbidden points.
            # The points added are lines passing through v and any u in cap_set.
            # Since v is in cap_set, we can just iterate over cap_set.
            # Wait, iterating over a set while adding to another set is fine.
            # But we are adding to `forbidden`.
            for u in cap_set:
                # Calculate w
                # Using tuple comprehension is slow. Unrolling for small n? 
                # n is variable. 
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forb_add(w)
    return size

def _greedy_eval_with_state(order, n, start_idx=0, cap_set=None, forbidden=None, limit_size=None):
    """
    Evaluate greedy from start_idx, optionally continuing from existing state.
    Returns (size, cap_set, forbidden).
    limit_size: If provided, stop early if size + remaining < limit_size (optimization).
    """
    if cap_set is None:
        cap_set = set()
    if forbidden is None:
        forbidden = set()
    
    size = len(cap_set)
    N = len(order)
    
    cap_add = cap_set.add
    forb_add = forbidden.add
    forb_contains = forbidden.__contains__
    
    for idx in range(start_idx, N):
        v = order[idx]
        if not forb_contains(v):
            cap_add(v)
            size += 1
            for u in cap_set:
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forb_add(w)
        
        # Early termination check
        if limit_size is not None:
            remaining = N - idx - 1
            if size + remaining < limit_size:
                break
                
    return size, cap_set, forbidden

def _generate_heuristic_ordering(n, seed=42):
    """
    Generate an ordering based on the heuristic.
    """
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    
    def score_vec(v):
        c0 = sum(1 for x in v if x == 0)
        c1 = sum(1 for x in v if x == 1)
        c2 = sum(1 for x in v if x == 2)
        
        diff_02 = abs(c0 - c2)
        target_1 = n / 3.0
        diff_1 = abs(c1 - target_1)
        
        perturbation = 0.0
        for i, x in enumerate(v):
            perturbation += x * (1.0 / (3.0 ** i))
        
        return diff_02 * diff_02 + 0.5 * diff_1 * diff_1 + 0.0001 * perturbation

    vectors.sort(key=score_vec)
    return vectors

def _extract_features(vectors, n):
    """
    Extract polynomial features for each vector.
    """
    features = []
    for v in vectors:
        c0 = sum(1 for x in v if x == 0)
        c1 = sum(1 for x in v if x == 1)
        c2 = sum(1 for x in v if x == 2)
        
        # Base features
        f = [c0, c1, c2]
        # Squares
        f.extend([c0*c0, c1*c1, c2*c2])
        # Interactions
        f.extend([c0*c1, c0*c2, c1*c2])
        # Sum of coordinates (mod 3 invariant related)
        f.append(sum(v) % 3)
        # Sum of squares of coordinates
        f.append(sum(x*x for x in v))
        # Cubic terms
        f.extend([c0*c0*c1, c0*c0*c2, c1*c1*c2, c0*c1*c2])
        # Normalized counts
        f.extend([c0/n, c1/n, c2/n])
        # Add coordinate-specific features (first 3 coords)
        f.extend(v[:3])
        features.append(f)
    return np.array(features)

def _learned_ordering(n, seed=42):
    """
    1. Generate a few heuristic orderings.
    2. Evaluate them to get ranks.
    3. Train a linear model to predict rank from features.
    4. Use model to sort all vectors.
    5. Evaluate the new ordering.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Get base vectors
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    
    # 2. Generate multiple heuristic orderings with slight random perturbations
    orderings = []
    num_perturbations = 5
    for i in range(num_perturbations):
        def score_vec_perturbed(v, offset=i):
            c0 = sum(1 for x in v if x == 0)
            c1 = sum(1 for x in v if x == 1)
            c2 = sum(1 for x in v if x == 2)
            
            diff_02 = abs(c0 - c2)
            target_1 = n / 3.0
            diff_1 = abs(c1 - target_1)
            
            perturbation = 0.0
            for j, x in enumerate(v):
                perturbation += x * (1.0 / (3.0 ** j))
            
            noise = random.uniform(0, 0.01) * (1 + i*0.1)
            return diff_02 * diff_02 + 0.5 * diff_1 * diff_1 + 0.0001 * perturbation + noise

        order = sorted(vectors, key=score_vec_perturbed)
        orderings.append(order)
        
        if _is_deadline_exceeded():
            break

    if not orderings:
        return None, None

    # 3. Calculate average rank for each vector
    rank_sums = {v: 0 for v in vectors}
    count = len(orderings)
    for order in orderings:
        for rank, v in enumerate(order):
            rank_sums[v] += rank
    
    avg_ranks = {v: rank_sums[v] / count for v in vectors}
    
    # 4. Prepare training data with more features
    X = _extract_features(vectors, n)
    y = np.array([avg_ranks[v] for v in vectors])
    
    try:
        X_b = np.c_[X, np.ones(len(X))]
        # Add regularization to avoid overfitting
        lambda_reg = 0.01
        X_bT = X_b.T
        weights = np.linalg.solve(X_bT @ X_b + lambda_reg * np.eye(X_b.shape[1]), X_bT @ y)
    except Exception:
        try:
            weights, residuals, rank, sv = np.linalg.lstsq(X_b, y, rcond=None)
        except Exception:
            return None, None

    # 5. Generate new ordering using the learned weights
    predictions = X_b @ weights
    
    sorted_items = sorted(zip(predictions, vectors), key=lambda x: x[0])
    new_order = [v for _, v in sorted_items]
    
    # 6. Evaluate
    new_size = _fast_greedy_eval(new_order, n)
    
    # Also evaluate the best of the heuristic orderings for comparison
    best_heur_size = 0
    best_heur_order = orderings[0]
    for o in orderings:
        s = _fast_greedy_eval(o, n)
        if s > best_heur_size:
            best_heur_size = s
            best_heur_order = o
            
    if new_size >= best_heur_size:
        return new_order, new_size
    else:
        return best_heur_order, best_heur_size

def _hill_climb_ordering(n, initial_order, initial_size):
    """
    Improve the ordering using hill climbing with incremental evaluation.
    Uses 2-opt swaps with early termination.
    """
    if n not in _orderings:
        return
    
    state = _orderings[n]
    order = list(state['order'])
    current_size = state['size']
    N = len(order)
    
    start_hc = time.time()
    max_time_hc = 280.0  # Leave 45s buffer for safety and other tasks
    
    attempts_since_improvement = 0
    max_no_improve = 2000  # Lower threshold to allow more restarts or exit
    
    # Precompute position map
    pos_map = {v: i for i, v in enumerate(order)}
    
    while time.time() - start_hc < max_time_hc and attempts_since_improvement < max_no_improve:
        if _is_deadline_exceeded():
            break
        
        # Try a random swap, biased toward closer positions for efficiency
        # Closer swaps are cheaper to evaluate
        if random.random() < 0.5:
            # Swap with a nearby position (distance < 50)
            i = random.randint(0, N - 2)
            max_dist = min(50, N - i - 1)
            j = i + 1 + random.randint(0, max_dist - 1)
        else:
            i, j = random.sample(range(N), 2)
        
        if i > j:
            i, j = j, i
            
        if i == j: continue
        
        vi, vj = order[i], order[j]
        
        # Perform swap
        order[i], order[j] = vj, vi
        
        # Evaluate from position i
        # We need the state before i.
        # To avoid recalculating 0..i-1 every time, we can cache states?
        # Caching states for all i is O(N^2) memory and O(N^2) time to build.
        # N is up to 6561 (n=8). 6561^2 is ~43M. Too much.
        
        # Strategy: Recalculate state 0..i-1. This is O(i * |C|).
        # If i is small, this is fast. If i is large, it's slow.
        # To mitigate, we prefer small i (biased sampling above).
        
        cap_set = set()
        forbidden = set()
        
        # Build state up to i
        for idx in range(i):
            v = order[idx]
            if v not in forbidden:
                cap_set.add(v)
                for u in cap_set:
                    w = tuple((-v[k] - u[k]) % 3 for k in range(n))
                    forbidden.add(w)
        
        # Now evaluate from i to end with the swapped order
        # We can use _greedy_eval_with_state with early termination
        # limit_size = current_size (we only care if we exceed it)
        new_size, _, _ = _greedy_eval_with_state(order, n, i, cap_set, forbidden, limit_size=current_size + 1)
        
        if new_size > current_size:
            current_size = new_size
            state['size'] = current_size
            attempts_since_improvement = 0
            pos_map[vi] = j
            pos_map[vj] = i
        else:
            order[i], order[j] = vi, vj
            attempts_since_improvement += 1
    
    state['order'] = order
    state['rank_map'] = {v: i for i, v in enumerate(order)}
    state['done'] = True

def priority(vector, n):
    global _start_time
    _init_state(vector, n)
    
    if _is_deadline_exceeded():
        c0 = sum(1 for x in vector if x == 0)
        c2 = sum(1 for x in vector if x == 2)
        c1 = sum(1 for x in vector if x == 1)
        return -(abs(c0 - c2)**2 + 0.5*abs(c1 - n/3.0)**2)
    
    if n not in _orderings:
        # Try learned ordering first
        order, size = _learned_ordering(n)
        
        if order is None:
            # Fallback to heuristic
            order = _generate_heuristic_ordering(n)
            size = _fast_greedy_eval(order, n)
            
        _orderings[n] = {
            'order': order,
            'size': size,
            'done': False,
            'rank_map': {v: i for i, v in enumerate(order)}
        }
        
        # Run hill climbing with time budget
        remaining = 325.0 - (time.time() - _start_time)
        if remaining > 10.0:
            _hill_climb_ordering(n, order, size)
    
    rank = _orderings[n]['rank_map'][vector]
    total = 3**n
    return float(total - 1 - rank)

# EVOLVE-BLOCK-END
