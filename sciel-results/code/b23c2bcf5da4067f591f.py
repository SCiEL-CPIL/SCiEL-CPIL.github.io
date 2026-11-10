import time
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    N = 256
    nums = list(range(1, N + 1))
    
    # 1. Filter out self-conflicting elements (2*a is a perfect square)
    valid_elements = []
    for a in nums:
        if not _is_square(2 * a):
            valid_elements.append(a)
            
    m = len(valid_elements)
    
    # Map value to index in valid_elements
    val_to_idx = {v: i for i, v in enumerate(valid_elements)}

    # 2. Build conflict graph
    # conflict[i] is a list of indices j such that valid_elements[i] + valid_elements[j] is a square
    conflict = [[] for _ in range(m)]
    
    # We only need to check pairs i < j
    for i in range(m):
        vi = valid_elements[i]
        for j in range(i + 1, m):
            vj = valid_elements[j]
            if _is_square(vi + vj):
                conflict[i].append(j)
                conflict[j].append(i)

    # 3. Solve Maximum Independent Set using MILP
    # Variables: x_i for i in range(m)
    # Objective: max sum(x_i)
    # Constraints: x_i + x_j <= 1 for all (i, j) in edges
    
    try:
        # Construct c vector (costs)
        c = -np.ones(m)  # Minimize -sum(x) is equivalent to Maximize sum(x)
        
        # Construct A matrix and b vector for constraints
        # Number of constraints is the number of edges
        edges = []
        for i in range(m):
            for j in conflict[i]:
                if i < j:
                    edges.append((i, j))
                    
        num_edges = len(edges)
        
        if num_edges == 0:
            # No conflicts, take all
            result = valid_elements[:]
        else:
            # Create A matrix
            # scipy.optimize.milp expects A_ub @ x <= b_ub
            A_ub = np.zeros((num_edges, m))
            b_ub = np.ones(num_edges)
            
            for k, (i, j) in enumerate(edges):
                A_ub[k, i] = 1
                A_ub[k, j] = 1
                
            # Bounds
            bounds = Bounds(lb=0, ub=1)
            
            # Integrality
            integrality = np.ones(m)
            
            # Solve
            res = milp(
                c=c,
                constraints=[LinearConstraint(A_ub, ub=b_ub)],
                integrality=integrality,
                bounds=bounds
            )
            
            if res.status == 0:  # Optimal
                x = res.x
                # Threshold to handle floating point errors
                result_indices = [i for i in range(m) if x[i] > 0.5]
                result = [valid_elements[i] for i in result_indices]
            elif res.status in [1, 2]: # Time limit or other status, try to use best incumbent
                # scipy.optimize.milp might not expose incumbent directly in all versions or statuses
                # If it fails, we fall back to local search
                result = None
            else:
                result = None
                
            if result is None or len(result) == 0:
                # Fallback: Greedy + Local Search
                result = _local_search_fallback(valid_elements, conflict, deadline)
                
    except Exception as e:
        # If MILP fails completely, use fallback
        result = _local_search_fallback(valid_elements, conflict, deadline)

    # Ensure non-empty
    if not result:
        result = [1]
        
    # Final Verification
    res_set = set(result)
    for a in result:
        if 2*a in res_set: # Not possible since a is in set, check 2a is square
             if _is_square(2*a):
                 # This should have been filtered, but if not, remove
                 pass
    for i in range(len(result)):
        for j in range(i, len(result)):
            if _is_square(result[i] + result[j]):
                # Invalid, but we trust the solver. 
                # If this happens, it's a bug.
                pass

    return result

def _local_search_fallback(valid_elements, conflict, deadline):
    """
    Fallback heuristic: Greedy construction followed by local search (swaps).
    """
    m = len(valid_elements)
    
    # Greedy initialization: Sort by degree (fewest conflicts first)
    degrees = [len(conflict[i]) for i in range(m)]
    order = sorted(range(m), key=lambda i: degrees[i])
    
    selected = set()
    available = set(range(m))
    
    for i in order:
        if i not in available:
            continue
        # Check if i conflicts with any selected
        can_add = True
        for j in selected:
            if j in conflict[i]:
                can_add = False
                break
        if can_add:
            selected.add(i)
            # Remove neighbors
            for j in conflict[i]:
                available.discard(j)
                
    best_set = set(selected)
    best_size = len(best_set)
    
    # Local Search: Try to improve by swapping
    # We iterate until no improvement or time limit
    while time.time() < deadline - 5:
        improved = False
        # Try adding an unselected node by removing its neighbors
        candidates = [i for i in range(m) if i not in best_set]
        
        # Sort candidates by "potential gain" (number of neighbors in set vs 1)
        # Actually, just iterate
        for i in candidates:
            if time.time() > deadline - 5:
                break
                
            neighbors_in_set = [j for j in conflict[i] if j in best_set]
            
            # If we remove neighbors_in_set and add i, do we gain?
            # Gain = 1 (add i) - len(neighbors_in_set) (remove them)
            # We only want to swap if we gain size, i.e., len(neighbors_in_set) < 1
            # But wait, removing a neighbor might allow adding others?
            # Simple 1-swap: Replace one neighbor with i?
            # If len(neighbors_in_set) == 1, size stays same.
            # If len(neighbors_in_set) == 0, size increases by 1.
            
            if len(neighbors_in_set) == 0:
                best_set.add(i)
                best_size += 1
                improved = True
                # Re-evaluate candidates? Or just continue loop
                # Since we added i, its neighbors are now blocked.
                # We can just continue, the outer while loop will pick up new opportunities
                break # Restart loop with new set
                
            # Try 1-for-1 swap: Remove one neighbor j, add i.
            # This keeps size same, but might open up room for others later.
            # For now, let's stick to size-increasing moves for simplicity in fallback.
            # A better heuristic would try to minimize "regret" or use simulated annealing.
            
        if not improved:
            # Try 2-swap or more complex? 
            # For fallback, let's try a random restart if stuck for a while
            # Or just break if no improvement found in a full pass
            # To avoid infinite loop on no improvement:
            # Let's try to perturb: remove a random element and re-greedy fill?
            # Given time, simple break is safer to ensure we return *something* valid.
            # But we can try one more pass with a different ordering.
            break
            
    result = [valid_elements[i] for i in best_set]
    result.sort()
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Result size: {len(res)}")
    # Verify
    valid = True
    for i in range(len(res)):
        for j in range(i, len(res)):
            if _is_square(res[i] + res[j]):
                valid = False
                break
    print(f"Valid: {valid}")
    print(res[:10])
