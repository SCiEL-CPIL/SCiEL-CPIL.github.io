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
    """
    cap_set = set()
    forbidden = set()
    size = 0
    for v in order:
        if v not in forbidden:
            cap_set.add(v)
            size += 1
            for u in cap_set:
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
    return size

def _greedy_eval_from(order, n, start_idx, cap_set, forbidden):
    """
    Continue greedy evaluation from start_idx with existing state.
    Returns (size, cap_set, forbidden).
    """
    size = len(cap_set)
    for v in order[start_idx:]:
        if v not in forbidden:
            cap_set.add(v)
            size += 1
            for u in cap_set:
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
    return size, cap_set, forbidden

def _eval_swap(order, n, i, j, pre_state):
    """
    Evaluate the effect of swapping positions i and j.
    pre_state = (cap_set_before_i, forbidden_before_i) is the state just before position i.
    Returns the new total size if we swap i and j.
    """
    cap_set, forbidden = pre_state
    # Simulate from position i with the swap
    # Positions i and j are swapped
    # We need to evaluate from i to end with the modified order
    # But we only need to track from i onward
    new_cap_set = set(cap_set)
    new_forbidden = set(forbidden)
    new_size = len(new_cap_set)
    
    # Process positions i through j-1 (original i now has vj, original j has vi)
    # Then position j (now has vi)
    # Then j+1 onward
    
    # Actually, let's just re-evaluate from i with the swapped order
    # This is O((N-i) * |cap_set|) which is still better than full O(N * |cap_set|)
    for idx in range(i, len(order)):
        v = order[idx] if idx != i and idx != j else (order[j] if idx == i else order[i])
        if v not in new_forbidden:
            new_cap_set.add(v)
            new_size += 1
            for u in new_cap_set:
                w = tuple((-v[k] - u[k]) % 3 for k in range(n))
                new_forbidden.add(w)
    return new_size

def _greedy_eval_with_state(order, n, start_idx=0, cap_set=None, forbidden=None):
    """
    Evaluate greedy from start_idx, optionally continuing from existing state.
    Returns (size, cap_set, forbidden).
    """
    if cap_set is None:
        cap_set = set()
    if forbidden is None:
        forbidden = set()
    size = len(cap_set)
    for v in order[start_idx:]:
        if v not in forbidden:
            cap_set.add(v)
            size += 1
            for u in cap_set:
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
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
        # Sum of squares of coordinates
        f.append(c0 + c1 + 4*c2)
        # Cubic terms
        f.extend([c0*c0*c1, c0*c0*c2, c1*c1*c2, c0*c1*c2])
        # Normalized counts
        f.extend([c0/n, c1/n, c2/n])
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
    num_perturbations = 3
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

def _compute_prefix_states(order, n, checkpoints=None):
    """
    Compute greedy state at various checkpoints for faster swap evaluation.
    Returns a dict mapping checkpoint index to (cap_set, forbidden).
    """
    if checkpoints is None:
        checkpoints = [0]
    
    states = {}
    cap_set = set()
    forbidden = set()
    size = 0
    
    cp_set = set(checkpoints)
    cp_idx = 0
    
    for idx, v in enumerate(order):
        if cp_idx < len(checkpoints) and idx == checkpoints[cp_idx]:
            states[idx] = (set(cap_set), set(forbidden))
            cp_idx += 1
        if v not in forbidden:
            cap_set.add(v)
            size += 1
            for u in cap_set:
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
    
    return states, size

def _hill_climb_ordering(n, initial_order, initial_size):
    """
    Improve the ordering using hill climbing with incremental evaluation.
    """
    if n not in _orderings:
        return
    
    state = _orderings[n]
    order = list(state['order'])
    current_size = state['size']
    N = len(order)
    
    start_hc = time.time()
    max_time_hc = 260.0
    
    attempts_since_improvement = 0
    max_no_improve = 5000
    
    # Precompute position map
    pos_map = {v: i for i, v in enumerate(order)}
    
    # We'll do a full eval periodically to get accurate states
    # For swap evaluation, we re-evaluate from min(i,j)
    
    while time.time() - start_hc < max_time_hc and attempts_since_improvement < max_no_improve:
        if _is_deadline_exceeded():
            break
        
        # Try a random swap, biased toward closer positions for efficiency
        if random.random() < 0.3:
            # Swap with a nearby position
            i = random.randint(0, N - 2)
            j = i + 1 + random.randint(0, min(10, N - i - 2))
        else:
            i, j = random.sample(range(N), 2)
        
        if i > j:
            i, j = j, i
        
        vi, vj = order[i], order[j]
        
        # Perform swap
        order[i], order[j] = vj, vi
        
        # Evaluate from position i (the earlier swapped position)
        # We need the state before position i
        # Quick approach: just do a full eval from i
        # First, get state at position i by evaluating 0..i-1
        cap_set = set()
        forbidden = set()
        for idx in range(i):
            v = order[idx]
            if v not in forbidden:
                cap_set.add(v)
                for u in cap_set:
                    w = tuple((-v[k] - u[k]) % 3 for k in range(n))
                    forbidden.add(w)
        
        # Now evaluate from i to end with the swapped order
        new_size = len(cap_set)
        for idx in range(i, N):
            v = order[idx]
            if v not in forbidden:
                cap_set.add(v)
                new_size += 1
                for u in cap_set:
                    w = tuple((-v[k] - u[k]) % 3 for k in range(n))
                    forbidden.add(w)
        
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
