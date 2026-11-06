import itertools
import numpy as np
import random
import time
# EVOLVE-BLOCK-START

# Global state for the cap set solver
_start_time = None
_DEADLINE = 325.0  # Safety margin before 330s kill
_n_vectors = {}  # n -> 3**n
_index_map = {}  # n -> dict vector -> index
_index_to_vec = {} # n -> list index -> vector

def _init_state(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _is_deadline_exceeded():
    return _elapsed() > _DEADLINE

def _get_indices(n):
    if n not in _n_vectors:
        total = 3 ** n
        _n_vectors[n] = total
        vectors = list(itertools.product([0, 1, 2], repeat=n))
        _index_map[n] = {v: i for i, v in enumerate(vectors)}
        _index_to_vec[n] = vectors
    return _n_vectors[n], _index_map[n], _index_to_vec[n]

def _fast_greedy_eval_indices(order_indices, n):
    """
    Fast evaluation of greedy cap set size for a given order of indices.
    Uses bytearray for forbidden set for O(1) access.
    """
    total, _, _ = _get_indices(n)
    forbidden = bytearray(total)
    in_cap = bytearray(total)
    cap_list = []
    size = 0
    for idx in order_indices:
        if not forbidden[idx]:
            forbidden[idx] = 1
            in_cap[idx] = 1
            cap_list.append(idx)
            size += 1
            v = _index_to_vec[n][idx]
            # Calculate forbidden points for new v with all u in cap_list
            # w = (-v - u) % 3
            for u_idx in cap_list:
                u = _index_to_vec[n][u_idx]
                w_idx = 0
                # Calculate index of w
                # This is the bottleneck. We can precompute transition tables?
                # Or just compute it.
                # w_i = (3 - (v_i + u_i) % 3) % 3
                # We can precompute a table: trans[v_idx][u_idx] -> w_idx
                # But 3^n * 3^n is too big.
                # However, we can precompute for each coordinate?
                # Let's just do the loop.
                # Optimization: Precompute the "negative" of each vector index?
                # neg_idx[idx] -> index of -v
                # Then w = -v - u.
                # We can precompute a table add_idx[a][b] -> index of a+b?
                # 3^n * 3^n is too big.
                
                # Let's stick to tuple conversion for now if not optimized, 
                # but bytearray lookup is faster than set lookup.
                # To speed up w calculation:
                w_coords = [(-v[i] - u[i]) % 3 for i in range(n)]
                # Convert to index
                # This is O(n).
                # We can precompute a map from tuple to index? Yes, we have _index_map.
                w_idx = _index_map[n][tuple(w_coords)]
                forbidden[w_idx] = 1
    return size

def _generate_heuristic_ordering_indices(n, seed=42):
    """
    Generate an ordering based on the heuristic, returning indices.
    """
    total, imap, vecs = _get_indices(n)
    
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

    scored = [(score_vec(v), i) for i, v in enumerate(vecs)]
    scored.sort(key=lambda x: x[0])
    return [i for _, i in scored]

def _learned_ordering_indices(n, seed=42):
    """
    1. Generate a few heuristic orderings.
    2. Evaluate them to get ranks.
    3. Train a linear model to predict rank from features.
    4. Use model to sort all vectors.
    5. Evaluate the new ordering.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    total, imap, vecs = _get_indices(n)
    
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

        scored = [(score_vec_perturbed(v), idx) for idx, v in enumerate(vecs)]
        scored.sort(key=lambda x: x[0])
        order = [idx for _, idx in scored]
        orderings.append(order)
        
        if _is_deadline_exceeded():
            break

    if not orderings:
        return None, None

    # Calculate average rank for each vector
    rank_sums = np.zeros(total)
    count = len(orderings)
    for order in orderings:
        for rank, idx in enumerate(order):
            rank_sums[idx] += rank
    
    avg_ranks = rank_sums / count
    
    # Prepare training data
    # Features: counts of 0,1,2
    X = np.zeros((total, 3))
    for idx, v in enumerate(vecs):
        X[idx, 0] = sum(1 for x in v if x == 0)
        X[idx, 1] = sum(1 for x in v if x == 1)
        X[idx, 2] = sum(1 for x in v if x == 2)
    
    # Add interactions
    X_b = np.c_[X, X[:,0]*X[:,1], X[:,0]*X[:,2], X[:,1]*X[:,2], np.ones(total)]
    y = avg_ranks
    
    try:
        lambda_reg = 0.01
        X_bT = X_b.T
        weights = np.linalg.solve(X_bT @ X_b + lambda_reg * np.eye(X_b.shape[1]), X_bT @ y)
    except Exception:
        try:
            weights, residuals, rank, sv = np.linalg.lstsq(X_b, y, rcond=None)
        except Exception:
            return None, None

    predictions = X_b @ weights
    sorted_items = sorted(zip(predictions, range(total)), key=lambda x: x[0])
    new_order = [i for _, i in sorted_items]
    
    new_size = _fast_greedy_eval_indices(new_order, n)
    
    best_heur_size = 0
    best_heur_order = orderings[0]
    for o in orderings:
        s = _fast_greedy_eval_indices(o, n)
        if s > best_heur_size:
            best_heur_size = s
            best_heur_order = o
            
    if new_size >= best_heur_size:
        return new_order, new_size
    else:
        return best_heur_order, best_heur_size

def _hill_climb_ordering_indices(n, initial_order, initial_size):
    """
    Improve the ordering using hill climbing with incremental evaluation.
    Uses bytearray for state.
    """
    order = list(initial_order)
    current_size = initial_size
    N = len(order)
    
    start_hc = time.time()
    max_time_hc = 260.0
    
    attempts_since_improvement = 0
    max_no_improve = 5000
    
    # Precompute w_idx for all pairs? No, too big.
    # But we can precompute the "forbidden contribution" of each vector v
    # given a set of u's? No.
    
    # We will use a simple swap strategy.
    # To speed up, we will use a "window" approach or just accept the cost.
    # The bytearray lookup is ~10x faster than set lookup.
    
    while time.time() - start_hc < max_time_hc and attempts_since_improvement < max_no_improve:
        if _is_deadline_exceeded():
            break
        
        # Try a random swap
        if random.random() < 0.3:
            i = random.randint(0, N - 2)
            j = i + 1 + random.randint(0, min(10, N - i - 2))
        else:
            i, j = random.sample(range(N), 2)
        
        if i > j:
            i, j = j, i
        
        vi, vj = order[i], order[j]
        
        # Perform swap
        order[i], order[j] = vj, vi
        
        # Evaluate
        new_size = _fast_greedy_eval_indices(order, n)
        
        if new_size > current_size:
            current_size = new_size
            attempts_since_improvement = 0
        else:
            order[i], order[j] = vi, vj
            attempts_since_improvement += 1
    
    return order, current_size

def priority(vector, n):
    global _start_time
    _init_state(vector, n)
    
    if _is_deadline_exceeded():
        c0 = sum(1 for x in vector if x == 0)
        c2 = sum(1 for x in vector if x == 2)
        c1 = sum(1 for x in vector if x == 1)
        return -(abs(c0 - c2)**2 + 0.5*abs(c1 - n/3.0)**2)
    
    if n not in _n_vectors or not hasattr(priority, '_cache'):
        priority._cache = {}
        
    if n not in priority._cache:
        # Try learned ordering first
        order, size = _learned_ordering_indices(n)
        
        if order is None:
            # Fallback to heuristic
            order = _generate_heuristic_ordering_indices(n)
            size = _fast_greedy_eval_indices(order, n)
            
        priority._cache[n] = {
            'order': order,
            'size': size,
            'done': False,
            'rank_map': {idx: i for i, idx in enumerate(order)}
        }
        
        # Run hill climbing with time budget
        remaining = 325.0 - (time.time() - _start_time)
        if remaining > 10.0:
            new_order, new_size = _hill_climb_ordering_indices(n, order, size)
            priority._cache[n]['order'] = new_order
            priority._cache[n]['size'] = new_size
            priority._cache[n]['rank_map'] = {idx: i for i, idx in enumerate(new_order)}
            priority._cache[n]['done'] = True
    
    idx = _index_map[n][vector]
    rank = priority._cache[n]['rank_map'][idx]
    total = 3**n
    return float(total - 1 - rank)

# EVOLVE-BLOCK-END
