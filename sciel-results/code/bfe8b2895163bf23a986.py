import random
import time

_PRIORITY_CACHE = {}
_START_TIME = None

def _int_to_tuple(x, n):
    """Convert integer to tuple representation in {0,1,2}^n."""
    result = []
    for _ in range(n):
        result.append(x % 3)
        x //= 3
    return tuple(result)

def _tuple_to_int(v, n):
    """Convert tuple to integer representation."""
    result = 0
    for i, val in enumerate(v):
        result += val * (3 ** i)
    return result

def _compute_weight(v, n):
    """Compute a weight for a vector based on its 'sparseness' and other properties."""
    # Count zeros - sparser vectors are generally better to include first
    zeros = sum(1 for x in v if x == 0)
    # Also consider the sum of elements
    element_sum = sum(v)
    # Higher zeros and lower sum are better
    return (zeros, -element_sum)

def _greedy_cap_set_fast(vectors_int, n, time_limit, start_time=None):
    """
    Greedily build a cap set using integer representations for speed.
    Returns the set of integer representations included.
    """
    if start_time is None:
        start_time = time.time()
    
    S = set()
    forbidden = set()
    
    for v in vectors_int:
        if time.time() - start_time > time_limit:
            break
        
        if v in forbidden:
            continue
        
        S.add(v)
        
        # For each existing a in S, compute the third point on the line
        # and forbid it
        for a in S:
            if a == v:
                continue
            # Convert to tuples for computation (inefficient but correct)
            # Better: work with integers directly
            av = _int_to_tuple(v, n)
            aa = _int_to_tuple(a, n)
            
            # Third point: 2*v - a (mod 3)
            w1 = tuple((2 * av[i] - aa[i]) % 3 for i in range(n))
            w1_int = _tuple_to_int(w1, n)
            forbidden.add(w1_int)
            
            # Third point: 2*a - v (mod 3)
            w2 = tuple((2 * aa[i] - av[i]) % 3 for i in range(n))
            w2_int = _tuple_to_int(w2, n)
            forbidden.add(w2_int)
    
    return S

def _greedy_with_simulated_annealing(n, time_limit, start_time=None):
    """
    Use simulated annealing to find a good ordering, then do greedy.
    """
    if start_time is None:
        start_time = time.time()
    
    total_vectors = 3 ** n
    all_vectors = list(range(total_vectors))
    
    # Initial ordering: sort by weight (sparseness)
    def weight_func(x):
        v = _int_to_tuple(x, n)
        zeros = sum(1 for c in v if c == 0)
        element_sum = sum(v)
        return (zeros, -element_sum)
    
    all_vectors.sort(key=weight_func, reverse=True)
    
    best_set = set()
    best_size = 0
    
    # Multiple restarts with simulated annealing on the ordering
    restart_count = 0
    sa_time_per_restart = max(1.0, time_limit / 10)
    
    while time.time() - start_time < time_limit:
        # Check if we have enough time for another restart
        if time.time() - start_time + sa_time_per_restart > time_limit:
            break
        
        restart_count += 1
        
        # Start from the initial ordering (sorted by sparseness)
        current_order = list(all_vectors)
        
        # Simulated annealing on the ordering
        # We'll swap random pairs and evaluate
        current_temp = 1.0
        cooling_rate = 0.95
        
        # Do a quick greedy to get initial solution
        initial_set = _greedy_cap_set_fast(current_order, n, 2.0, start_time)
        current_size = len(initial_set)
        best_in_sa = initial_set
        best_size_in_sa = current_size
        
        # Simple SA: try random swaps
        sa_steps = min(500, int(time_limit * 2))  # Adjust based on time
        
        for step in range(sa_steps):
            if time.time() - start_time > time_limit - 1.0:
                break
            
            # Random swap
            i = random.randint(0, len(current_order) - 1)
            j = random.randint(0, len(current_order) - 1)
            if i == j:
                continue
            
            # Swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            # Quick greedy evaluation (time-limited)
            eval_time = min(0.5, (time_limit - (time.time() - start_time)) / 2)
            if eval_time < 0.1:
                break
            
            new_set = _greedy_cap_set_fast(current_order, n, eval_time, start_time)
            new_size = len(new_set)
            
            # Accept or reject
            if new_size >= current_size or random.random() < (1.0 - (new_size - current_size) / (1.0 + current_temp * current_size)):
                current_size = new_size
                if new_size > best_size_in_sa:
                    best_size_in_sa = new_size
                    best_in_sa = new_set
            else:
                # Revert swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
            
            current_temp *= cooling_rate
        
        if best_size_in_sa > best_size:
            best_size = best_size_in_sa
            best_set = best_in_sa
    
    return best_set

def _find_best_cap_set(n, time_limit, start_time=None):
    """
    Find the best cap set using multiple strategies.
    """
    if start_time is None:
        start_time = time.time()
    
    total_vectors = 3 ** n
    
    # Strategy 1: Greedy with sparseness ordering
    def weight_func(x):
        v = _int_to_tuple(x, n)
        zeros = sum(1 for c in v if c == 0)
        element_sum = sum(v)
        return (zeros, -element_sum)
    
    all_vectors = list(range(total_vectors))
    all_vectors.sort(key=weight_func, reverse=True)
    
    greedy_time = min(10.0, time_limit * 0.3)
    greedy_set = _greedy_cap_set_fast(all_vectors, n, greedy_time, start_time)
    best_set = greedy_set
    best_size = len(greedy_set)
    
    # Strategy 2: Simulated annealing (if time permits)
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 10.0:
        sa_set = _greedy_with_simulated_annealing(n, remaining_time * 0.7, start_time)
        if len(sa_set) > best_size:
            best_set = sa_set
            best_size = len(sa_set)
    
    # Strategy 3: Random restarts with different orderings
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 5.0:
        for _ in range(5):
            if time.time() - start_time > time_limit - 1.0:
                break
            random.shuffle(all_vectors)
            restart_time = min(2.0, remaining_time / 6)
            restart_set = _greedy_cap_set_fast(all_vectors, n, restart_time, start_time)
            if len(restart_set) > best_size:
                best_set = restart_set
                best_size = len(restart_set)
    
    return best_set

def priority(vector, n):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    if n not in _PRIORITY_CACHE:
        elapsed = time.time() - _START_TIME
        # Reserve 30s for safety margin
        time_limit = max(1.0, 270.0 - elapsed)
        
        random.seed(42)  # Reproducible
        
        best_set = _find_best_cap_set(n, time_limit, _START_TIME)
        _PRIORITY_CACHE[n] = best_set
    
    # Convert vector to integer representation
    vec_int = _tuple_to_int(vector, n)
    
    return 1.0 if vec_int in _PRIORITY_CACHE[n] else 0.0
