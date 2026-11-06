"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools
import numpy as np

# EVOLVE-BLOCK-START

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

_orderings = {}

def _greedy_eval(order, n):
    """
    Fast greedy evaluation.
    order: list of vectors (tuples of 0,1,2)
    """
    cap_set = set()
    forbidden = set()
    size = 0
    for v in order:
        if v not in forbidden:
            cap_set.add(v)
            size += 1
            for u in cap_set:
                w1 = tuple((2*v[i] - u[i]) % 3 for i in range(n))
                w2 = tuple((2*u[i] - v[i]) % 3 for i in range(n))
                forbidden.add(w1)
                forbidden.add(w2)
    return size

def _optimize_order(n, time_budget):
    vectors = list(itertools.product([0, 1, 2], repeat=n))
    N = len(vectors)
    
    # Parameters: w (linear), q (quadratic symmetric)
    # We use a symmetric matrix q, so q[i,j] == q[j,i]
    # Number of params: n + n*(n+1)/2
    
    w = np.random.uniform(-1, 1, n)
    q = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            q[i, j] = random.uniform(-0.5, 0.5)
            q[j, i] = q[i, j]
            
    V = np.array(vectors, dtype=float)
    # Precompute VV features
    # Feature k corresponds to (i,j) with i<=j
    # We just use full matrix product for simplicity, since N is small
    VV = np.einsum('ni,nj->nij', V, V)
    VV_flat = VV.reshape(N, -1)
    
    def get_scores(w, q):
        q_flat = q.flatten()
        return V @ w + VV_flat @ q_flat
    
    def eval_order(order):
        return _greedy_eval(order, n)
    
    # Initial
    scores = get_scores(w, q)
    order_idx = np.argsort(scores)
    order = [vectors[i] for i in order_idx]
    current_size = eval_order(order)
    best_score = current_size
    best_w = w.copy()
    best_q = q.copy()
    best_order = order
    
    start_opt = time.time()
    step_size = 0.5
    no_improve = 0
    
    while time.time() - start_opt < time_budget:
        if _is_deadline_exceeded():
            break
            
        # Perturb
        if random.random() < 0.5:
            i = random.randint(0, n-1)
            delta = random.uniform(-step_size, step_size)
            w[i] += delta
        else:
            i = random.randint(0, n-1)
            j = random.randint(0, n-1)
            delta = random.uniform(-step_size, step_size)
            q[i, j] += delta
            q[j, i] += delta
            
        scores = get_scores(w, q)
        order_idx = np.argsort(scores)
        order = [vectors[i] for i in order_idx]
        new_size = eval_order(order)
        
        if new_size > best_score:
            best_score = new_size
            best_w = w.copy()
            best_q = q.copy()
            best_order = order
            no_improve = 0
            step_size *= 0.95
        else:
            # Revert
            if random.random() < 0.5:
                i = random.randint(0, n-1)
                delta = random.uniform(-step_size, step_size)
                w[i] -= delta
            else:
                i = random.randint(0, n-1)
                j = random.randint(0, n-1)
                delta = random.uniform(-step_size, step_size)
                q[i, j] -= delta
                q[j, i] -= delta
                
            no_improve += 1
            if no_improve > 30:
                # Random restart
                w = np.random.uniform(-1, 1, n)
                q = np.zeros((n, n))
                for i in range(n):
                    for j in range(i, n):
                        q[i, j] = random.uniform(-0.5, 0.5)
                        q[j, i] = q[i, j]
                no_improve = 0
                step_size = 0.5

    return best_order, best_score

def priority(vector, n):
    global _start_time
    _init_state(vector, n)
    
    if _is_deadline_exceeded():
        return -sum(vector)
    
    if n not in _orderings:
        budget = min(300.0, 325.0 - _elapsed() - 5.0)
        if budget < 10:
            budget = 10
        order, size = _optimize_order(n, budget)
        _orderings[n] = {
            'order': order,
            'size': size,
            'rank_map': {v: i for i, v in enumerate(order)}
        }
        
    rank = _orderings[n]['rank_map'][vector]
    total = 3**n
    return float(total - 1 - rank)

# EVOLVE-BLOCK-END
