import time
import sys
import random
from collections import deque
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 130.0

    if k != 4 or g != 10:
        return _construct_luw(k, g, deadline)

    MOD = 40
    N_A = 4
    N_B = 4
    NUM_VARS = 16

    # Precompute all closed walks of length 4, 6, 8
    # Walk is a list of (var_idx, sign)
    # var_idx = i*4 + j for voltage v[i][j]
    
    def get_closed_walks():
        walks = []
        # Length 4: a0-b1-a1-b2-a0
        # Vars: v[a0,b1](+), v[a1,b1](-), v[a1,b2](+), v[a0,b2](-)
        for a0 in range(4):
            for b1 in range(4):
                for a1 in range(4):
                    for b2 in range(4):
                        walks.append([
                            (a0*4+b1, 1),
                            (a1*4+b1, -1),
                            (a1*4+b2, 1),
                            (a0*4+b2, -1)
                        ])
                        
        # Length 6: a0-b1-a1-b2-a2-b3-a0
        # Vars: v[a0,b1](+), v[a1,b1](-), v[a1,b2](+), v[a2,b2](-), v[a2,b3](+), v[a0,b3](-)
        for a0 in range(4):
            for b1 in range(4):
                for a1 in range(4):
                    for b2 in range(4):
                        for a2 in range(4):
                            for b3 in range(4):
                                walks.append([
                                    (a0*4+b1, 1),
                                    (a1*4+b1, -1),
                                    (a1*4+b2, 1),
                                    (a2*4+b2, -1),
                                    (a2*4+b3, 1),
                                    (a0*4+b3, -1)
                                ])
                                
        # Length 8: a0-b1-a1-b2-a2-b3-a3-b4-a0
        # Vars: v[a0,b1](+), v[a1,b1](-), v[a1,b2](+), v[a2,b2](-), v[a2,b3](+), v[a3,b3](-), v[a3,b4](+), v[a0,b4](-)
        for a0 in range(4):
            for b1 in range(4):
                for a1 in range(4):
                    for b2 in range(4):
                        for a2 in range(4):
                            for b3 in range(4):
                                for a3 in range(4):
                                    for b4 in range(4):
                                        walks.append([
                                            (a0*4+b1, 1),
                                            (a1*4+b1, -1),
                                            (a1*4+b2, 1),
                                            (a2*4+b2, -1),
                                            (a2*4+b3, 1),
                                            (a3*4+b3, -1),
                                            (a3*4+b4, 1),
                                            (a0*4+b4, -1)
                                        ])
                                        if len(walks) > 50000: break
                                    if len(walks) > 50000: break
                                if len(walks) > 50000: break
                            if len(walks) > 50000: break
                        if len(walks) > 50000: break
                    if len(walks) > 50000: break
                if len(walks) > 50000: break
            if len(walks) > 50000: break
        return walks

    all_walks = get_closed_walks()
    num_walks = len(all_walks)
    
    # Convert walks to numpy arrays for fast vectorized evaluation
    # walk_vars[i] = array of var indices for walk i
    # walk_sigs[i] = array of signs for walk i
    walk_vars = np.array([w[0] for w in all_walks])
    walk_sigs = np.array([w[1] for w in all_walks])
    
    # For fast update, we need to know which walks involve which variable
    # var_to_walks[var_idx] = list of walk indices
    var_to_walks = [[] for _ in range(NUM_VARS)]
    for i in range(num_walks):
        for var_idx, sign in all_walks[i]:
            var_to_walks[var_idx].append(i)
    
    # Convert var_to_walks to numpy arrays
    var_to_walks_arr = [np.array(v) for v in var_to_walks]
    
    def count_violations(voltages):
        # Vectorized evaluation
        # voltages is a numpy array of shape (NUM_VARS,)
        # For each walk, compute sum(sign * voltage[var_idx])
        # walk_vars has shape (num_walks, max_len) but variable lengths
        # We need to handle variable lengths
        
        viol_count = 0
        for i in range(num_walks):
            s = 0
            for var_idx, sign in all_walks[i]:
                s += sign * voltages[var_idx]
            if s % MOD == 0:
                viol_count += 1
        return viol_count
    
    # Optimized: Precompute for each walk the list of (var_idx, sign) as separate arrays
    # For fast incremental update
    
    def optimized_min_conflicts(deadline):
        random.seed(42)
        np.random.seed(42)
        
        # Track violation status for each walk
        is_violated = np.zeros(num_walks, dtype=bool)
        var_viol_count = np.zeros(NUM_VARS, dtype=int)
        
        def update_violations(voltages):
            nonlocal is_violated, var_viol_count
            # Reset
            is_violated[:] = False
            var_viol_count[:] = 0
            
            # Compute all walk sums
            for i in range(num_walks):
                s = 0
                for var_idx, sign in all_walks[i]:
                    s += sign * voltages[var_idx]
                if s % MOD == 0:
                    is_violated[i] = True
                    for var_idx, sign in all_walks[i]:
                        var_viol_count[var_idx] += 1
        
        def update_single_var(var_idx, old_val, new_val, voltages):
            nonlocal is_violated, var_viol_count
            # Update walks involving var_idx
            for walk_i in var_to_walks_arr[var_idx]:
                # Recompute sum for this walk
                s = 0
                for vi, sign in all_walks[walk_i]:
                    if vi == var_idx:
                        s += sign * new_val
                    else:
                        s += sign * voltages[vi]
                
                was_violated = is_violated[walk_i]
                is_violated[walk_i] = (s % MOD == 0)
                
                if was_violated and not is_violated[walk_i]:
                    for vi, sign in all_walks[walk_i]:
                        var_viol_count[vi] -= 1
                elif not was_violated and is_violated[walk_i]:
                    for vi, sign in all_walks[walk_i]:
                        var_viol_count[vi] += 1
        
        best_volt = None
        best_total = float('inf')
        
        for restart in range(200):
            if time.time() > deadline - 5:
                break
                
            voltages = np.random.randint(0, MOD, NUM_VARS)
            update_violations(voltages)
            total_viol = np.sum(is_violated)
            
            if total_viol == 0:
                return voltages.tolist()
                
            if total_viol < best_total:
                best_total = total_viol
                best_volt = voltages.tolist()
                
            # Local Search
            max_iters = 50
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                    
                # Find a variable with violations
                candidates = np.where(var_viol_count > 0)[0]
                if len(candidates) == 0:
                    break
                var_idx = candidates[np.random.randint(len(candidates))]
                
                old_val = voltages[var_idx]
                best_new_val = old_val
                min_local_viol = var_viol_count[var_idx]
                
                # Try all values
                for new_val in range(MOD):
                    if new_val == old_val:
                        continue
                    
                    # Temporarily set and evaluate
                    voltages[var_idx] = new_val
                    # Update violations incrementally
                    old_is_violated = is_violated.copy()
                    old_var_viol = var_viol_count.copy()
                    
                    # Recompute affected walks
                    for walk_i in var_to_walks_arr[var_idx]:
                        s = 0
                        for vi, sign in all_walks[walk_i]:
                            if vi == var_idx:
                                s += sign * new_val
                            else:
                                s += sign * voltages[vi]
                        is_violated[walk_i] = (s % MOD == 0)
                    
                    # Update var_viol_count
                    var_viol_count[:] = 0
                    for i in range(num_walks):
                        if is_violated[i]:
                            for vi, sign in all_walks[i]:
                                var_viol_count[vi] += 1
                    
                    current_viol = var_viol_count[var_idx]
                    
                    if current_viol < min_local_viol:
                        min_local_viol = current_viol
                        best_new_val = new_val
                    else:
                        # Revert
                        is_violated[:] = old_is_violated
                        var_viol_count[:] = old_var_viol
                        voltages[var_idx] = old_val
                
                if best_new_val != old_val:
                    voltages[var_idx] = best_new_val
                    # Commit the update
                    old_is_violated = is_violated.copy()
                    old_var_viol = var_viol_count.copy()
                    
                    for walk_i in var_to_walks_arr[var_idx]:
                        s = 0
                        for vi, sign in all_walks[walk_i]:
                            if vi == var_idx:
                                s += sign * best_new_val
                            else:
                                s += sign * voltages[vi]
                        is_violated[walk_i] = (s % MOD == 0)
                    
                    var_viol_count[:] = 0
                    for i in range(num_walks):
                        if is_violated[i]:
                            for vi, sign in all_walks[i]:
                                var_viol_count[vi] += 1
                    
                    total_viol = np.sum(is_violated)
                    if total_viol < best_total:
                        best_total = total_viol
                        best_volt = voltages.tolist()
                        if total_viol == 0:
                            return best_volt
                else:
                    break
        return best_volt

    candidate_volt = optimized_min_conflicts(deadline)
    
    if candidate_volt is not None:
        voltages = [[candidate_volt[i*4+j] for j in range(4)] for i in range(4)]
        n = 320
        edges = []
        for i in range(4):
            for j in range(4):
                volt = voltages[i][j]
                for k in range(40):
                    a_idx = i * 40 + k
                    b_idx = 160 + j * 40 + (k + volt) % 40
                    edges.append((a_idx, b_idx))
                    
        # Verify
        if _verify_girth_exact(n, edges, g, deadline):
            return n, edges
            
    # Fallback
    return _construct_luw(4, 10, deadline)

def _verify_girth_exact(n, edges, g, deadline):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    max_len = g - 1
    
    for start_v in range(n):
        if time.time() > deadline:
            return False
            
        states = set()
        for w in adj[start_v]:
            states.add((w, start_v))
            
        for d in range(2, max_len + 1):
            new_states = set()
            for (v, p) in states:
                for w in adj[v]:
                    if w != p:
                        new_states.add((w, v))
            if any(v == start_v for (v, p) in new_states):
                return False
            states = new_states
            if not states:
                break
        
    return True

def _construct_luw(k, g, deadline):
    q = k
    d_candidates = []
    for d in range(3, 20):
        if d % 2 == 1:
            if d + 5 >= g:
                d_candidates.append(d)
        else:
            if d + 4 >= g:
                d_candidates.append(d)
    d = min(d_candidates) if d_candidates else max(3, g - 5)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rules_cycle = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)]
        rule[c] = rules_cycle[r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side), tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    return len(order), edges
