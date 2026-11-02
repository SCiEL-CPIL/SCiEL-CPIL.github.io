"""Continuous relaxation + simulated annealing for kissing number d=11."""
import math
import numpy as np
import itertools
import time


def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    D = 11

    # Generate all candidates: weight-4 (5280), weight-2 (220), weight-1 (22)
    candidates_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for positions in itertools.combinations(range(D), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            candidates_list.append(v)
    for pos in range(D):
        for s in [1, -1]:
            v = [0] * D
            v[pos] = s
            candidates_list.append(v)

    candidates = np.array(candidates_list, dtype=np.int8)
    n = len(candidates)  # 5522

    # Precompute norms squared
    norms_sq = np.sum(candidates.astype(np.int32) ** 2, axis=1)
    max_norm_sq = int(norms_sq.max())  # 4

    # Precompute dot product matrix
    dot = candidates.astype(np.int32) @ candidates.T.astype(np.int32)

    # Constraint: ||v_i - v_j||^2 >= max_norm_sq
    threshold = (norms_sq[:, np.newaxis] + norms_sq[np.newaxis, :] - max_norm_sq) / 2.0
    compat = (dot <= threshold).astype(np.uint8)
    np.fill_diagonal(compat, 1)

    # conflict[i,j] = 1 if i and j CANNOT both be selected
    conflict = (1 - compat).astype(np.uint8)
    np.fill_diagonal(conflict, 0)

    # Conflict degrees for each node
    conflict_deg = conflict.sum(axis=1).astype(np.int32)

    # Greedy initialization (min-degree)
    def greedy_init():
        remaining = np.ones(n, dtype=bool)
        selected = []
        while remaining.any():
            deg = (conflict * remaining[np.newaxis, :]).sum(axis=1)
            deg[~remaining] = n + 1
            min_deg = deg.min()
            if min_deg >= n:
                break
            mask = (deg == min_deg) & remaining
            picks = np.where(mask)[0]
            if len(picks) == 0:
                break
            new_idx = picks[0]
            selected.append(new_idx)
            remaining = remaining & (conflict[new_idx] == 0)
            remaining[new_idx] = False
        return selected

    init_sel = greedy_init()
    x0 = np.zeros(n, dtype=np.float64)
    for i in init_sel:
        x0[i] = 1.0
    
    best_x = x0.copy()
    best_count = len(init_sel)
    best_viol = 0

    # Continuous relaxation: p in [0,1]^n
    # Objective: maximize sum(p) - Lambda * sum_{i<j} conflict[i,j] * p_i * p_j
    # Use gradient-based SA on p

    Lambda = 50.0
    T_start = 20.0
    T_end = 0.01
    max_steps = 200000
    step_size = 0.1

    p = x0.copy()
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
            
        alpha = (T_end / T_start) ** (1.0 / max_steps)
        T = T_start * (alpha ** step)
        
        # Compute gradient of objective w.r.t. p
        # f(p) = sum(p) - Lambda * sum_{i<j} c_ij p_i p_j
        # df/dp_i = 1 - Lambda * sum_j c_ij p_j
        grad = 1.0 - Lambda * (conflict @ p)
        
        # Gradient ascent with noise
        # Scale gradient to reasonable step
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            grad = grad / grad_norm * step_size
        
        # Add random perturbation scaled by T
        noise = np.random.randn(n) * T * 0.01
        delta_p = grad + noise
        
        # Clip p to [0, 1]
        p_new = np.clip(p + delta_p, 0.0, 1.0)
        
        # Compute objective for p and p_new
        # f(p) = sum(p) - Lambda * 0.5 * sum_{i,j} c_ij p_i p_j
        # The 0.5 accounts for double counting in the full sum
        obj_old = np.sum(p) - 0.5 * Lambda * (p @ (conflict @ p))
        obj_new = np.sum(p_new) - 0.5 * Lambda * (p_new @ (conflict @ p_new))
        
        delta_obj = obj_new - obj_old
        
        accepted = False
        if delta_obj >= 0:
            accepted = True
        else:
            prob = math.exp(-delta_obj / max(T, 1e-10))
            if np.random.random() < prob:
                accepted = True
        
        if accepted:
            p = p_new
            
            # Extract discrete solution: threshold at 0.5
            x_discrete = (p > 0.5).astype(np.uint8)
            count = int(x_discrete.sum())
            
            # Count violations
            viol = int((x_discrete @ conflict @ x_discrete) / 2)
            
            if viol == 0 and count > best_count:
                best_count = count
                best_viol = 0
                best_x = x_discrete.copy()
            elif viol < best_viol:
                best_viol = viol
                best_x = x_discrete.copy()
    
    # Final local search: try to add any compatible node
    if best_viol == 0:
        x_final = best_x.astype(np.uint8)
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if x_final[i] == 0:
                    if int(np.dot(conflict[i], x_final)) == 0:
                        x_final[i] = 1
                        changed = True
        best_x = x_final
        best_count = int(x_final.sum())
    
    # If still have violations, do a greedy cleanup
    if best_viol > 0:
        x_clean = np.zeros(n, dtype=np.uint8)
        # Sort by degree, pick greedily
        order = np.argsort(conflict_deg)
        for i in order:
            if int(np.dot(conflict[i], x_clean)) == 0:
                x_clean[i] = 1
        best_x = x_clean
        best_count = int(x_clean.sum())

    if best_count == 0:
        v = np.zeros(D, dtype=np.int64)
        v[0] = 1
        v[1] = 1
        return v.reshape(1, -1)

    result = candidates[best_x.astype(bool)]
    return result.astype(np.int64)
