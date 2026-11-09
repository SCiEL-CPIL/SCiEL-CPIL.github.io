"""Direct bootstrap for R(5,5): extend Paley(37) with smooth-objective SA search."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START

DEADLINE = time.time() + 280


def _build_paley37():
    """Return adjacency bitset list for Paley graph on 37 vertices (vertices 0..36)."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj


def _find_k4_sets(adj, n):
    """Find all K4 cliques in graph with n vertices. Return list of 4-tuples."""
    k4_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            for k in range(j + 1, n):
                if not (common & (1 << k)):
                    continue
                common2 = common & adj[k]
                for l in range(k + 1, n):
                    if common2 & (1 << l):
                        k4_list.append((i, j, k, l))
    return k4_list


def _find_ind4_sets(adj, n):
    """Find all independent 4-sets in graph with n vertices. Return list of 4-tuples."""
    ind4_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i] & (1 << j):
                continue
            non_common = ~(adj[i] | adj[j])
            for k in range(j + 1, n):
                if not (non_common & (1 << k)):
                    continue
                non_common2 = non_common & ~(adj[i] | adj[j] | adj[k])
                for l in range(k + 1, n):
                    if non_common2 & (1 << l):
                        ind4_list.append((i, j, k, l))
    return ind4_list


def _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=10.0):
    """
    Try to add vertex n (new index = n) by choosing edges to 0..n-1.
    Uses SA with NumPy-accelerated cost evaluation.
    Returns new_adj or None if failed.
    """
    new_n = n + 1
    
    # Precompute incidence matrices for K4 and Ind4
    # k4_matrix: (num_k4, n) binary matrix where entry is 1 if vertex is in K4
    # ind4_matrix: (num_ind4, n) binary matrix where entry is 1 if vertex is in Ind4
    
    k4_matrix = np.zeros((len(k4_list), n), dtype=np.int8)
    for idx, k4 in enumerate(k4_list):
        for v in k4:
            k4_matrix[idx, v] = 1
            
    ind4_matrix = np.zeros((len(ind4_list), n), dtype=np.int8)
    for idx, ind4 in enumerate(ind4_list):
        for v in ind4:
            ind4_matrix[idx, v] = 1
            
    # Precompute cost arrays for fast evaluation
    # We want to evaluate cost for a state vector x (length n, 0/1)
    # Cost = sum_{K4} penalty(cnt_ones) + sum_{Ind4} penalty(cnt_zeros)
    # penalty(cnt_ones) = 1 if cnt==4 else exp(-beta*(4-cnt))
    # penalty(cnt_zeros) = 1 if cnt==0 else exp(-beta*cnt)
    
    beta = 8.0 # Sharper penalty
    
    # Precompute penalty tables
    # k4_pen[c] = penalty for c ones in a K4
    k4_pen = np.array([math.exp(-beta * (4 - c)) if c < 4 else 1.0 for c in range(5)])
    # ind4_pen[c] = penalty for c ones in an Ind4 (penalty for c zeros is ind4_pen[4-c] effectively? No.)
    # For Ind4, violation is 0 ones. 
    # Let c be number of ones. Number of zeros is 4-c.
    # Penalty is 1 if 4-c == 0 (c=4) -> wait.
    # Ind4 violation: all 4 bits are 0. So c=0 is full penalty.
    # If c=1, 3 zeros. Partial penalty.
    # Formula: 1 if c==0 else exp(-beta*c)
    ind4_pen = np.array([1.0 if c == 0 else math.exp(-beta * c) for c in range(5)])

    def get_cost(x):
        """
        x: numpy array of shape (n,) with 0/1 values.
        Returns total cost.
        """
        if len(k4_list) == 0 and len(ind4_list) == 0:
            return 0.0
        
        # Count ones in each K4
        if len(k4_list) > 0:
            k4_counts = k4_matrix @ x
            k4_cost = np.sum(k4_pen[k4_counts.astype(np.int64)])
        else:
            k4_cost = 0.0
            
        # Count ones in each Ind4
        if len(ind4_list) > 0:
            ind4_counts = ind4_matrix @ x
            ind4_cost = np.sum(ind4_pen[ind4_counts.astype(np.int64)])
        else:
            ind4_cost = 0.0
            
        return k4_cost + ind4_cost

    def get_hard_violations(x):
        """
        x: numpy array of shape (n,) with 0/1 values.
        Returns count of hard violations (K4s with 4 ones, Ind4s with 0 ones).
        """
        v = 0
        if len(k4_list) > 0:
            k4_counts = k4_matrix @ x
            v += np.sum(k4_counts == 4)
        if len(ind4_list) > 0:
            ind4_counts = ind4_matrix @ x
            v += np.sum(ind4_counts == 0)
        return int(v)

    best_state = None
    best_hard = float('inf')
    best_soft = float('inf')
    
    start_time = time.time()
    
    # Use integer states for bit manipulation speed in SA loop, 
    # but convert to numpy for cost? No, that's slow.
    # Better: Maintain numpy array for cost, but random flip is easy.
    # Or: Use integer state, and precompute contribution of each vertex?
    # Since n is small (<43), and we need speed, let's stick to numpy vectorization 
    # but optimize the flip.
    
    # Actually, for n~40, python loop over constraints is ~1000 ops.
    # Numpy matrix mult is ~40*1000 = 40k ops, but vectorized.
    # Let's try to keep state as numpy array.
    
    max_restarts = 50 # Fewer restarts, but each is faster and longer
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        
        # Initialize with random state
        state = np.random.randint(0, 2, size=n).astype(np.float64)
        cur_soft = get_cost(state)
        cur_hard = get_hard_violations(state)
        
        if cur_hard == 0:
            best_state = state
            break
        
        if cur_hard < best_hard or (cur_hard == best_hard and cur_soft < best_soft):
            best_hard = cur_hard
            best_soft = cur_soft
            best_state = state.copy()
        
        # SA
        temp = 1.0
        cooling = 0.99
        min_temp = 0.01
        steps_per_temp = 100 # More steps per temp since cost is fast
        
        while temp > min_temp and time.time() - start_time < time_limit:
            for _ in range(steps_per_temp):
                v = random.randint(0, n - 1)
                
                # Calculate delta cost efficiently?
                # Delta cost for flipping v:
                # For each K4 containing v:
                #   old_cnt -> new_cnt = old_cnt +/- 1
                #   delta = pen[new_cnt] - pen[old_cnt]
                # For each Ind4 containing v:
                #   similar
                
                # This requires knowing which constraints v is in.
                # Let's precompute lists of constraint indices for each vertex.
                
                # For now, just recompute full cost (vectorized) to be safe and simple.
                # If this is too slow, we can optimize delta later.
                
                old_state = state
                state[v] = 1.0 - state[v] # Flip
                new_soft = get_cost(state)
                delta = new_soft - cur_soft
                
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    cur_soft = new_soft
                    cur_hard = get_hard_violations(state)
                    if cur_hard == 0:
                        best_state = state.copy()
                        break
                    if cur_hard < best_hard or (cur_hard == best_hard and cur_soft < best_soft):
                        best_hard = cur_hard
                        best_soft = cur_soft
                        best_state = state.copy()
                else:
                    state[v] = 1.0 - state[v] # Revert
            
            if best_hard == 0:
                break
            temp *= cooling
        
        if best_hard == 0:
            break
    
    if best_hard == 0:
        new_adj = adj[:] + [0]
        for v in range(n):
            if best_state[v] > 0.5:
                new_adj[new_n] |= (1 << v)
                new_adj[v] |= (1 << new_n)
        return new_adj
    
    return None


def construct():
    adj = _build_paley37()
    n = 37
    
    best_n = 37
    best_adj = adj[:]
    
    # Precompute K4 and independent 4-sets for the base graph
    k4_list = _find_k4_sets(adj, n)
    ind4_list = _find_ind4_sets(adj, n)
    
    # Try to extend the graph by adding vertices one at a time
    for target_n in range(38, 43):
        if time.time() > DEADLINE:
            break
        
        # Recompute k4 and ind4 for current graph
        k4_list = _find_k4_sets(adj, n)
        ind4_list = _find_ind4_sets(adj, n)
        
        # Allocate time for this step
        time_remaining = DEADLINE - time.time()
        if time_remaining < 20:
            break
        
        # Try SA with a time limit
        # Give it most of the remaining time, but keep a buffer
        t_limit = min(60.0, time_remaining - 30)
        new_adj = _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=t_limit)
        
        if new_adj is None and time.time() < DEADLINE - 30:
            # Retry with longer time
            t_limit2 = min(90.0, DEADLINE - time.time() - 20)
            new_adj = _try_add_vertex_sa(adj, n, k4_list, ind4_list, time_limit=t_limit2)
        
        if new_adj is not None:
            adj = new_adj
            n = target_n
            best_n = n
            best_adj = adj[:]
        else:
            break
    
    # Build edge list from best_adj
    n_best = best_n
    edges = []
    for i in range(n_best):
        for j in range(i + 1, n_best):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return n_best, edges

# EVOLVE-BLOCK-END
