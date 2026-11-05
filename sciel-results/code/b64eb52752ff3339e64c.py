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
    
    # Try MIP solver
    try:
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds,
                   options={"time_limit": 40})
        
        if res.status in [0, 2, 3]:  # Optimal, MIP gap, or other success codes
            x = res.x
            selected_indices = [i for i in range(n_nodes) if x[i] > 0.5]
            result = [idx + 2 for idx in selected_indices]
            
            if result:
                # Validate and score
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
                if valid:
                    best_result = result
                    best_score = score
    except Exception:
        pass
    
    # Fallback / Improvement: Greedy + Local Search
    # Build conflict sets for fast lookup
    conflict_sets = [set() for _ in range(n_nodes)]
    for i, j in conflict_pairs:
        conflict_sets[i].add(j)
        conflict_sets[j].add(i)
        
    # Greedy: sort by weight descending (x ascending)
    selected = []
    selected_set = set()
    
    for i in range(n_nodes):
        num = i + 2
        if not conflict_sets[i].intersection(selected_set):
            selected.append(i)
            selected_set.add(i)
            
    current_sum = sum(1.0 / numbers[i] for i in selected)
    
    if current_sum > best_score:
        best_score = current_sum
        best_result = [numbers[i] for i in selected]
        
    # Local Search: try swaps
    unselected = [i for i in range(n_nodes) if i not in selected_set]
    
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
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
                # Perform the swap
                selected[si] = best_new
                selected_set.remove(old_idx)
                selected_set.add(best_new)
                unselected.remove(best_new)
                unselected.append(old_idx)
                current_sum += best_gain
                improved = True
                if current_sum > best_score:
                    best_score = current_sum
                    best_result = [numbers[i] for i in selected]
    
    # Try adding more elements if possible (though greedy usually maximizes cardinality early, 
    # swapping might have left room? No, swapping keeps size constant. 
    # But maybe we can add a new element if it doesn't conflict?)
    # The greedy phase already added all possible non-conflicting elements in order.
    # However, after swaps, the set composition changed.
    # Let's try to add any unselected element that fits.
    for new_idx in unselected:
        if not conflict_sets[new_idx].intersection(selected_set):
            selected.append(new_idx)
            selected_set.add(new_idx)
            unselected.remove(new_idx)
            current_sum += 1.0 / numbers[new_idx]
            if current_sum > best_score:
                best_score = current_sum
                best_result = [numbers[i] for i in selected]
            # Restart adding loop since adding one might allow others? 
            # Actually, if it didn't conflict before, it won't start conflicting.
            # But adding it might block others. We just want max sum.
            # Since we're adding positive weights, we should add all compatible ones.
            # But the order matters for future additions? No, if A and B are both compatible with current set,
            # and compatible with each other, order doesn't matter.
            # If they are NOT compatible with each other, we can only pick one.
            # The simple loop above picks the first one it finds.
            # To be thorough, we should probably re-run the "add all compatible" logic or 
            # just rely on the fact that the greedy phase did the heavy lifting.
            # Given time constraints, let's just do one pass.
            break # To avoid O(n^2) complexity in this final phase, just one pass.
            
    if not best_result:
        return [2]
        
    return best_result
