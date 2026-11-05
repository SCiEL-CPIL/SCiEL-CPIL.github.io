"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s kill
    
    n = 256
    elements = list(range(1, 257))
    
    # Precompute conflict matrix: conflict[i][j] = True if (a*b) % (a+b) == 0
    # where a = elements[i], b = elements[j]
    # We want to find maximum independent set in this graph.
    # Conflict graph: edge between i,j if they CANNOT both be in the set.
    
    # Use bitset representation for speed
    # conflict_mask[i] is a bitset where bit j is set if i and j conflict
    
    # First, let's build the conflict structure
    # For efficiency, we'll work with numpy arrays
    
    a = np.array(elements, dtype=np.int64)
    
    # Build conflict matrix as a set of pairs
    # conflict[i][j] = 1 if a[i]*a[j] % (a[i]+a[j]) == 0
    # This is O(n^2) = 256^2 = 65536 checks, very fast
    
    conflict = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in range(i+1, n):
            ai = a[i]
            aj = a[j]
            if (ai * aj) % (ai + aj) == 0:
                conflict[i][j] = 1
                conflict[j][i] = 1
    
    # Now we need to find maximum independent set.
    # This is equivalent to maximum clique in the complement graph,
    # or we can formulate as MILP:
    # Maximize sum(x_i)
    # Subject to: x_i + x_j <= 1 for all (i,j) in conflict edges
    #             x_i in {0,1}
    
    # Count conflict edges
    conflict_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if conflict[i][j]:
                conflict_edges.append((i, j))
    
    m = len(conflict_edges)
    
    # Build MILP
    # c = -1 for all variables (we minimize -sum(x))
    c = -np.ones(n)
    
    # Constraints: x_i + x_j <= 1 for each conflict edge
    if m > 0:
        A = np.zeros((m, n), dtype=np.int8)
        for idx, (i, j) in enumerate(conflict_edges):
            A[idx, i] = 1
            A[idx, j] = 1
        lb = np.full(m, 0, dtype=np.int8)
        ub = np.full(m, 1, dtype=np.int8)
        constraints = LinearConstraint(A, lb, ub)
    else:
        constraints = LinearConstraint(np.zeros((1, n)), 0, 1)
    
    integrality = np.ones(n, dtype=np.int8)
    bounds = Bounds(0, 1)
    
    # Try MILP
    try:
        result = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if result.status == 0 and result.x is not None:
            selected = [elements[i] for i in range(n) if result.x[i] > 0.5]
            if len(selected) > 0:
                # Verify the solution
                valid = True
                sel_set = set(selected)
                for idx_i in range(len(selected)):
                    for idx_j in range(idx_i+1, len(selected)):
                        ai = selected[idx_i]
                        aj = selected[idx_j]
                        if (ai * aj) % (ai + aj) == 0:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    return selected
    except Exception:
        pass
    
    # Fallback: greedy approach with local search
    # Greedy: pick elements in order of fewest conflicts first
    degree = np.sum(conflict, axis=1)
    
    # Sort by degree (ascending) - pick low-degree vertices first
    order = np.argsort(degree)
    
    selected = []
    selected_set = set()
    for idx in order:
        if time.time() > deadline - 5:
            break
        # Check if this vertex conflicts with any already selected
        can_add = True
        for sel_idx in selected_set:
            if conflict[idx, sel_idx]:
                can_add = False
                break
        if can_add:
            selected.append(elements[idx])
            selected_set.add(idx)
    
    # Local search: try to improve by swapping
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            current_set = set(selected)
            current_idx_set = selected_set.copy()
            
            # Try adding one element
            for idx in range(n):
                if idx in current_idx_set:
                    continue
                can_add = True
                for sel_idx in current_idx_set:
                    if conflict[idx, sel_idx]:
                        can_add = False
                        break
                if can_add:
                    selected.append(elements[idx])
                    current_idx_set.add(idx)
                    improved = True
                    break
            
            # Try swapping: remove one, add another that was previously blocked
            if not improved:
                for rem_idx in list(current_idx_set):
                    # Try to add something that conflicts with rem_idx
                    for add_idx in range(n):
                        if add_idx in current_idx_set:
                            continue
                        # Check if add_idx only conflicts with rem_idx among selected
                        only_conflicts_rem = True
                        for sel_idx in current_idx_set:
                            if sel_idx != rem_idx and conflict[add_idx, sel_idx]:
                                only_conflicts_rem = False
                                break
                        if only_conflicts_rem:
                            # Do the swap
                            selected.remove(elements[rem_idx])
                            current_idx_set.remove(rem_idx)
                            selected.append(elements[add_idx])
                            current_idx_set.add(add_idx)
                            improved = True
                            break
                    if improved:
                        break
    
    if len(selected) == 0:
        return [1]
    
    return selected
# EVOLVE-BLOCK-END
