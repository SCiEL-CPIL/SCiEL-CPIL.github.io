import time
import random
import itertools
import numpy as np
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

# Global state
_start_time = None
_DEADLINE = 325.0

def _init_state(vector, n):
    global _start_time
    if _start_time is None:
        _start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _is_deadline_exceeded():
    return _elapsed() > _DEADLINE

# Cache for results
_best_set = {}

def _generate_lines(n):
    """Generate all lines in F_3^n as frozensets of 3 vectors."""
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    lines = []
    m = 3**n
    vecs = list(itertools.product([0, 1, 2], repeat=n))
    for i in range(m):
        u = vecs[i]
        for j in range(i+1, m):
            v = vecs[j]
            w = tuple((-u[k] - v[k]) % 3 for k in range(n))
            line_set = frozenset([u, v, w])
            lines.append(line_set)
    return lines

def _solve_ilp(n):
    """Solve ILP for max cap set for small n."""
    if n > 6: 
        return None
        
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    m = len(vectors)
    
    lines = _generate_lines(n)
    
    import scipy.sparse as sp
    
    num_constraints = len(lines)
    row_indices = []
    col_indices = []
    data = []
    
    vec_to_idx = {v: i for i, v in enumerate(vectors)}
    
    for r, line in enumerate(lines):
        for v in line:
            c = vec_to_idx[v]
            row_indices.append(r)
            col_indices.append(c)
            data.append(1.0)
            
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(num_constraints, m))
    
    c = -np.ones(m)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(m)
    cons = LinearConstraint(A, ub=2)
    
    try:
        res = milp(c, constraints=cons, integrality=integrality, bounds=bounds, options={"time_limit": 30})
        if res.status == 0:
            x_sol = res.x
            cap_set = set()
            for i, val in enumerate(x_sol):
                if val > 0.5:
                    cap_set.add(vectors[i])
            return cap_set
        else:
            return None
    except Exception:
        return None

def _greedy_cap_set(order, n):
    """
    Greedily build cap set from ordered list of vectors.
    """
    cap_set = set()
    forbidden = set()
    for v in order:
        if v not in forbidden:
            cap_set.add(v)
            for u in cap_set:
                if u == v: continue
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
    return cap_set

def _local_search_simulated_annealing(n, time_budget, initial_set=None):
    """
    Improve cap set using Simulated Annealing.
    """
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    
    if initial_set is None:
        # Start with a random greedy set
        random.shuffle(vectors)
        current_set = _greedy_cap_set(vectors, n)
    else:
        current_set = set(initial_set)
    
    forbidden = set()
    for v in current_set:
        for u in current_set:
            if u == v: continue
            w = tuple((-v[i] - u[i]) % 3 for i in range(n))
            forbidden.add(w)
            
    best_set = set(current_set)
    best_size = len(current_set)
    
    start_time = time.time()
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while T > min_T and (time.time() - start_time) < time_budget:
        if _is_deadline_exceeded():
            break
            
        # Generate a move
        move_type = random.random()
        
        if move_type < 0.4:
            # Try to add a random vector
            v = random.choice(vectors)
            if v in current_set or v in forbidden:
                continue
            
            # Check if v can be added
            # We know v is not in forbidden, so it doesn't form a line with any pair in current_set?
            # Forbidden contains w such that {u, v, w} is a line.
            # If v is not in forbidden, then for all u in current_set, -u-v is not in current_set.
            # So v can be added.
            current_set.add(v)
            # Update forbidden
            for u in current_set:
                if u == v: continue
                w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                forbidden.add(w)
            
            delta = 1
            # Always accept improvement
            # In SA, we might accept worse, but adding is always better size-wise
            # So we don't need SA for pure addition.
            # SA is useful for swaps.
            
        elif move_type < 0.8:
            # Try to swap: remove one, add one
            if not current_set:
                continue
            v_remove = random.choice(list(current_set))
            current_set.remove(v_remove)
            
            # Rebuild forbidden temporarily? Too slow.
            # Instead, just try to find a valid add.
            # We need to check if any v_add is valid for current_set (without v_remove).
            # Let's just try a few random candidates.
            
            # To speed up, we can't easily check validity without forbidden set.
            # Let's rebuild forbidden for the reduced set.
            forbidden_temp = set()
            for v in current_set:
                for u in current_set:
                    if u == v: continue
                    w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                    forbidden_temp.add(w)
            
            v_add = None
            for _ in range(50):
                cand = random.choice(vectors)
                if cand in current_set or cand in forbidden_temp:
                    continue
                v_add = cand
                break
            
            if v_add is not None:
                current_set.add(v_add)
                # Update forbidden
                for u in current_set:
                    if u == v_add: continue
                    w = tuple((-v_add[i] - u[i]) % 3 for i in range(n))
                    forbidden_temp.add(w)
                forbidden = forbidden_temp
                
                # Size is same, so delta = 0.
                # Accept with probability exp(0/T) = 1?
                # This is a random walk on same-size sets.
                # We want to explore.
                if len(current_set) > best_size:
                    best_set = set(current_set)
                    best_size = len(current_set)
            else:
                # Revert
                current_set.add(v_remove)
                # Rebuild forbidden
                forbidden = set()
                for v in current_set:
                    for u in current_set:
                        if u == v: continue
                        w = tuple((-v[i] - u[i]) % 3 for i in range(n))
                        forbidden.add(w)
                        
        else:
            # Try to remove one (to escape local max? No, we want to maximize)
            # Maybe remove a "bad" vector that blocks many others?
            # This is complex. Stick to add/swap.
            pass
        
        if len(current_set) > best_size:
            best_set = set(current_set)
            best_size = len(current_set)
            
        T *= cooling
        
    return best_set

def _heuristic_greedy_restart(n, time_budget):
    """
    Run multiple greedy iterations with local search improvement.
    """
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    best_set = set()
    best_size = 0
    
    start = time.time()
    
    while time.time() - start < time_budget:
        if _is_deadline_exceeded():
            break
        
        remaining = time_budget - (time.time() - start)
        if remaining < 5:
            break
        ls_time = min(remaining * 0.4, 30.0)
        
        # Random ordering
        w = tuple(random.randint(0, 2) for _ in range(n))
        items = []
        for v in vectors:
            score = sum(v[i] * w[i] for i in range(n))
            noise = random.uniform(0, 1.0)
            items.append((score + noise, v))
        items.sort(key=lambda x: x[0])
        order = [v for _, v in items]
        
        cap_set = _greedy_cap_set(order, n)
        
        # Local search improvement
        if time.time() - start < time_budget - 3:
            cap_set = _local_search_simulated_annealing(n, ls_time, initial_set=cap_set)
        
        if len(cap_set) > best_size:
            best_size = len(cap_set)
            best_set = set(cap_set)
    
    return best_set

def priority(vector, n):
    global _start_time
    _init_state(vector, n)
    
    if _is_deadline_exceeded():
        c0 = sum(1 for x in vector if x == 0)
        c2 = sum(1 for x in vector if x == 2)
        c1 = sum(1 for x in vector if x == 1)
        return -(abs(c0 - c2)**2 + 0.5*abs(c1 - n/3.0)**2)
    
    if n not in _best_set:
        cap_set = None
        
        if n <= 6:
            cap_set = _solve_ilp(n)
            
        if cap_set is None or len(cap_set) == 0:
            remaining = _DEADLINE - _elapsed()
            search_time = max(15.0, min(250.0, remaining - 25.0))
            cap_set = _heuristic_greedy_restart(n, search_time)
            
        _best_set[n] = cap_set
    
    if vector in _best_set[n]:
        return 1.0
    else:
        return 0.0
