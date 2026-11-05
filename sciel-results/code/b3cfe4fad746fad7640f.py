import math
import time
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin for 90s limit
    
    N = 256
    n_nodes = 255
    numbers = list(range(2, 257))
    
    # Objective: Maximize sum(1/x_i)
    # milp minimizes c @ x, so we set c = -1/x_i
    c = np.array([-1.0 / x for x in numbers])
    
    # Build conflict pairs
    conflict_pairs = []
    for i in range(n_nodes):
        a = i + 2
        for j in range(i + 1, n_nodes):
            b = j + 2
            g = math.gcd(a, b)
            if a * b <= 256 * g:
                conflict_pairs.append((i, j))
                
    E = len(conflict_pairs)
    
    if E == 0:
        return numbers
        
    # Build conflict sets for fast lookup
    conflict_sets = [set() for _ in range(n_nodes)]
    for i, j in conflict_pairs:
        conflict_sets[i].add(j)
        conflict_sets[j].add(i)
    
    # Build constraint matrix A (sparse)
    # Each constraint is x_i + x_j <= 1
    row_idx = []
    col_idx = []
    data = []
    
    for k, (i, j) in enumerate(conflict_pairs):
        row_idx.append(k)
        col_idx.append(i)
        data.append(1)
        
        row_idx.append(k)
        col_idx.append(j)
        data.append(1)
        
    A = sp.csr_matrix((data, (row_idx, col_idx)), shape=(E, n_nodes))
    b = np.ones(E)
    
    bounds = Bounds(0, 1)
    integrality = np.ones(n_nodes)
    constraints = LinearConstraint(A, lb=-np.inf, ub=b)
    
    best_result = None
    best_score = 0.0
    
    def validate_and_update(result):
        nonlocal best_result, best_score
        if not result:
            return
        score = sum(1.0 / x for x in result)
        # Check validity
        valid = True
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if math.lcm(result[i], result[j]) <= N:
                    valid = False
                    break
            if not valid:
                break
        if valid and score > best_score:
            best_score = score
            best_result = list(result)
    
    # Try MIP solver
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": 30})
        
        if res.status in [0, 2, 3]:  # Optimal, MIP gap, or other success codes
            x = res.x
            selected_indices = [i for i in range(n_nodes) if x[i] > 0.5]
            result = [idx + 2 for idx in selected_indices]
            validate_and_update(result)
    except Exception:
        pass
    
    # Multi-start greedy + enhanced local search
    def greedy_select(order):
        selected = []
        selected_set = set()
        for i in order:
            if not conflict_sets[i].intersection(selected_set):
                selected.append(i)
                selected_set.add(i)
        return selected, selected_set
    
    def local_search(selected, selected_set, unselected, current_sum, max_time):
        improved = True
        while improved and time.time() < max_time:
            improved = False
            
            # Try single swaps
            for si in range(len(selected)):
                old_idx = selected[si]
                best_new = None
                best_gain = 0.0
                old_val = 1.0 / numbers[old_idx]
                
                for new_idx in unselected:
                    # Check if new_idx is compatible with all selected except old_idx
                    compatible = True
                    for sj in range(len(selected)):
                        if sj == si:
                            continue
                        if selected[sj] in conflict_sets[new_idx]:
                            compatible = False
                            break
                    if not compatible:
                        continue
                    new_val = 1.0 / numbers[new_idx]
                    gain = new_val - old_val
                    if gain > best_gain:
                        best_gain = gain
                        best_new = new_idx
                
                if best_new is not None and best_gain > 1e-15:
                    selected[si] = best_new
                    selected_set.remove(old_idx)
                    selected_set.add(best_new)
                    unselected.remove(best_new)
                    unselected.append(old_idx)
                    current_sum += best_gain
                    improved = True
            
            # Try 2-for-1 moves: remove two, add one if it fits
            for si in range(len(selected)):
                for sj in range(si + 1, len(selected)):
                    old1 = selected[si]
                    old2 = selected[sj]
                    best_new = None
                    best_gain = 0.0
                    old_val = 1.0 / numbers[old1] + 1.0 / numbers[old2]
                    
                    for new_idx in unselected:
                        # Check if new_idx is compatible with all selected except old1 and old2
                        compatible = True
                        for sk in range(len(selected)):
                            if sk == si or sk == sj:
                                continue
                            if selected[sk] in conflict_sets[new_idx]:
                                compatible = False
                                break
                        if not compatible:
                            continue
                        new_val = 1.0 / numbers[new_idx]
                        gain = new_val - old_val
                        if gain > best_gain:
                            best_gain = gain
                            best_new = new_idx
                    
                    if best_new is not None and best_gain > 1e-15:
                        # Remove both, add new
                        selected.remove(old1)
                        selected.remove(old2)
                        selected_set.remove(old1)
                        selected_set.remove(old2)
                        selected.append(best_new)
                        selected_set.add(best_new)
                        unselected.remove(best_new)
                        unselected.append(old1)
                        unselected.append(old2)
                        current_sum += best_gain
                        improved = True
                        break  # Restart si loop
                if improved:
                    break
            
            # Try 1-for-2 moves: remove one, add two if they fit
            for si in range(len(selected)):
                old_idx = selected[si]
                old_val = 1.0 / numbers[old_idx]
                
                best_pair = None
                best_gain = 0.0
                
                for ni in range(len(unselected)):
                    new1 = unselected[ni]
                    # Check new1 compatible with all except old_idx
                    compatible1 = True
                    for sj in range(len(selected)):
                        if sj == si:
                            continue
                        if selected[sj] in conflict_sets[new1]:
                            compatible1 = False
                            break
                    if not compatible1:
                        continue
                    
                    for nj in range(ni + 1, len(unselected)):
                        new2 = unselected[nj]
                        # Check new2 compatible with all except old_idx, and with new1
                        compatible2 = True
                        for sj in range(len(selected)):
                            if sj == si:
                                continue
                            if selected[sj] in conflict_sets[new2]:
                                compatible2 = False
                                break
                        if not compatible2:
                            continue
                        if new1 in conflict_sets[new2]:
                            continue
                        
                        new_val = 1.0 / numbers[new1] + 1.0 / numbers[new2]
                        gain = new_val - old_val
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (new1, new2)
                
                if best_pair is not None and best_gain > 1e-15:
                    new1, new2 = best_pair
                    selected.remove(old_idx)
                    selected_set.remove(old_idx)
                    selected.append(new1)
                    selected.append(new2)
                    selected_set.add(new1)
                    selected_set.add(new2)
                    unselected.remove(new1)
                    unselected.remove(new2)
                    unselected.append(old_idx)
                    current_sum += best_gain
                    improved = True
                    break  # Restart si loop
            
            # Try adding any compatible unselected element
            added = False
            for new_idx in list(unselected):
                if not conflict_sets[new_idx].intersection(selected_set):
                    selected.append(new_idx)
                    selected_set.add(new_idx)
                    unselected.remove(new_idx)
                    current_sum += 1.0 / numbers[new_idx]
                    improved = True
                    added = True
                    break
            
        return selected, selected_set, current_sum
    
    # Try multiple greedy orderings
    orderings = []
    
    # Ordering 1: by weight descending (smaller number = higher weight)
    order1 = list(range(n_nodes))
    orderings.append(order1)
    
    # Ordering 2: by number of conflicts ascending (less constrained first)
    order2 = sorted(range(n_nodes), key=lambda i: len(conflict_sets[i]))
    orderings.append(order2)
    
    # Ordering 3: by weight / conflict_degree (heuristic)
    order3 = sorted(range(n_nodes), key=lambda i: (1.0 / numbers[i]) / (len(conflict_sets[i]) + 1), reverse=True)
    orderings.append(order3)
    
    for order in orderings:
        if time.time() > deadline - 10.0:
            break
        
        selected, selected_set = greedy_select(order)
        unselected = [i for i in range(n_nodes) if i not in selected_set]
        current_sum = sum(1.0 / numbers[i] for i in selected)
        
        # Local search
        selected, selected_set, current_sum = local_search(
            selected, selected_set, unselected, current_sum, deadline - 5.0
        )
        
        validate_and_update([numbers[i] for i in selected])
    
    if not best_result:
        return [2]
        
    return best_result
