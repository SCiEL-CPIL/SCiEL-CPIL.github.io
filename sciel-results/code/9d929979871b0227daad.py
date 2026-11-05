import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    n = 16
    points = list(range(n))
    
    # Generate all triples
    all_triples = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                all_triples.append((i, j, k))
                
    num_triples = len(all_triples)
    
    # Precompute masks for triples
    triple_masks = [0] * num_triples
    for i, t in enumerate(all_triples):
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks[i] = mask
        
    # Precompute conflicts
    # conflict[i] is a set of indices j such that triples i and j share >= 2 points
    pair_to_triples = {}
    for i, t in enumerate(all_triples):
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            if p not in pair_to_triples:
                pair_to_triples[p] = []
            pair_to_triples[p].append(i)
            
    conflicts = [set() for _ in range(num_triples)]
    for p, idxs in pair_to_triples.items():
        # All triples in idxs conflict with each other
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                conflicts[idxs[i]].add(idxs[j])
                conflicts[idxs[j]].add(idxs[i])
                
    # Helper: Calculate independence number (alpha)
    def calc_alpha(system_indices):
        if not system_indices:
            return n
            
        sys_masks = [triple_masks[i] for i in system_indices]
        best = 0
        
        def is_independent(mask):
            for m in sys_masks:
                if (mask & m) == m:
                    return False
            return True
            
        def backtrack(point, current_mask, current_count):
            nonlocal best
            if point == n:
                if current_count > best:
                    best = current_count
                return
                
            # Pruning
            if current_count + (n - point) <= best:
                return
                
            # Option 1: Exclude point
            backtrack(point + 1, current_mask, current_count)
            
            # Option 2: Include point
            new_mask = current_mask | (1 << point)
            if is_independent(new_mask):
                backtrack(point + 1, new_mask, current_count + 1)
                
        backtrack(0, 0, 0)
        return best

    # MILP Formulation
    # Variables: x_i for i in range(num_triples), binary.
    # Objective: Maximize sum(x_i)
    # Constraints: For each pair of points (u, v), sum(x_i for i where (u,v) in triple_i) <= 1
    
    # Build constraint matrix A for MILP
    # Rows correspond to pairs (u, v) with u < v.
    # Columns correspond to triples.
    # A[row, col] = 1 if triple col contains the pair corresponding to row.
    
    # Map pair (u, v) to row index
    pair_index = {}
    row = 0
    for u in range(n):
        for v in range(u + 1, n):
            pair_index[(u, v)] = row
            row += 1
            
    num_pairs = row
    A = np.zeros((num_pairs, num_triples), dtype=int)
    
    for i, t in enumerate(all_triples):
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            r = pair_index[p]
            A[r, i] = 1
            
    # Objective: c = -1 (minimize -sum x, equivalent to maximize sum x)
    c = np.ones(num_triples)
    
    # Constraints: A @ x <= 1
    # scipy.optimize.milp requires LinearConstraint(A, lb, ub)
    # We want A @ x <= 1, so lb = -inf, ub = 1
    constraints = LinearConstraint(A, lb=-np.inf, ub=1)
    integrality = np.ones(num_triples)
    bounds = Bounds(lb=0, ub=1)
    
    # Solve MILP
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        if res.status == 0:
            x = res.x
            system_indices = [i for i in range(num_triples) if x[i] > 0.5]
            current_alpha = calc_alpha(system_indices)
            best_alpha = current_alpha
            best_system = list(system_indices)
        else:
            # Fallback to greedy if MILP fails or times out (though milp doesn't have explicit timeout in this context easily without callback, but it's fast for this size usually)
            system_indices = []
            used_pairs = set()
            for i, t in enumerate(all_triples):
                pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
                conflict = False
                for p in pairs:
                    if p in used_pairs:
                        conflict = True
                        break
                if not conflict:
                    system_indices.append(i)
                    for p in pairs:
                        used_pairs.add(p)
            current_alpha = calc_alpha(system_indices)
            best_alpha = current_alpha
            best_system = list(system_indices)
    except Exception:
        system_indices = []
        used_pairs = set()
        for i, t in enumerate(all_triples):
            pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
            conflict = False
            for p in pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                system_indices.append(i)
                for p in pairs:
                    used_pairs.add(p)
        current_alpha = calc_alpha(system_indices)
        best_alpha = current_alpha
        best_system = list(system_indices)

    # Local Search to improve upon MILP result
    # The MILP maximizes the number of triples, which is a good proxy for minimizing alpha,
    # but different packings with the same number of triples can have different alphas.
    # We perform swaps to try to reduce alpha further.
    
    active_indices = list(best_system)
    used_pairs = set()
    for i in active_indices:
        t = all_triples[i]
        used_pairs.add((t[0], t[1]))
        used_pairs.add((t[0], t[2]))
        used_pairs.add((t[1], t[2]))
        
    # Precompute list of available triples for faster random selection?
    # Just use range(len(all_triples))
    
    steps = 0
    max_steps = 20000
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Try to swap a random active triple with a random inactive one
        if not active_indices:
            break
            
        remove_idx = random.choice(active_indices)
        t_rem = all_triples[remove_idx]
        rem_pairs = [(t_rem[0], t_rem[1]), (t_rem[0], t_rem[2]), (t_rem[1], t_rem[2])]
        
        # Tentatively remove
        for p in rem_pairs:
            used_pairs.remove(p)
        active_indices.remove(remove_idx)
        
        # Try to find a replacement
        swapped = False
        # Sample a few candidates to avoid O(N) scan every time
        for _ in range(20):
            candidate_idx = random.choice(range(num_triples))
            if candidate_idx in active_indices:
                continue
            t_add = all_triples[candidate_idx]
            add_pairs = [(t_add[0], t_add[1]), (t_add[0], t_add[2]), (t_add[1], t_add[2])]
            conflict = False
            for p in add_pairs:
                if p in used_pairs:
                    conflict = True
                    break
            if not conflict:
                # Accept swap tentatively
                active_indices.append(candidate_idx)
                for p in add_pairs:
                    used_pairs.add(p)
                swapped = True
                break
                
        if swapped:
            new_alpha = calc_alpha(active_indices)
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_system = list(active_indices)
            else:
                # Revert swap
                last_idx = active_indices.pop()
                t_add = all_triples[last_idx]
                for p in [(t_add[0], t_add[1]), (t_add[0], t_add[2]), (t_add[1], t_add[2])]:
                    used_pairs.remove(p)
                # Re-add removed
                active_indices.append(remove_idx)
                for p in rem_pairs:
                    used_pairs.add(p)
        else:
            # Revert removal
            active_indices.append(remove_idx)
            for p in rem_pairs:
                used_pairs.add(p)

    # Return the best solution found
    result = [list(all_triples[i]) for i in best_system]
    return result
