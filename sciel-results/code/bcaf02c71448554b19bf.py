import time
import numpy as np
from itertools import combinations

def construct():
    """
    Constructs the Steiner System S(5, 6, 12) using MILP via scipy.
    
    The problem is formulated as:
    Maximize sum(x_i)
    Subject to:
    A * x <= 1  (where A is the incidence matrix of 5-subsets in 6-sets)
    x_i in {0, 1}
    
    This guarantees the optimal 132 blocks if a solver is available.
    """
    start_time = time.time()
    deadline = start_time + 110.0
    
    n_points = 12
    block_size = 6
    sub_size = 5
    
    # Generate all 6-subsets (blocks)
    all_blocks = list(combinations(range(n_points), block_size))
    num_blocks = len(all_blocks)
    
    # Generate all 5-subsets
    all_subs = list(combinations(range(n_points), sub_size))
    num_subs = len(all_subs)
    
    # Create incidence matrix A (num_subs x num_blocks)
    # A[i, j] = 1 if sub i is contained in block j, else 0
    # We can build this efficiently by iterating blocks
    A = np.zeros((num_subs, num_blocks), dtype=np.int8)
    
    # Map each 5-subset to an index for fast lookup
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}
    
    for j, b in enumerate(all_blocks):
        for s in combinations(b, sub_size):
            A[sub_to_idx[s], j] = 1
    
    # Formulate MILP
    # Objective: maximize sum(x) -> minimize -sum(x)
    c = -np.ones(num_blocks)
    
    # Constraints: A * x <= 1
    # scipy.optimize.milp expects constraints in form: lb <= A_ub @ x <= ub
    # Here A_ub = A, lb = 0, ub = 1
    A_ub = A
    b_ub = np.ones(num_subs)
    b_lb = np.zeros(num_subs)
    
    # Integrality: x is binary
    integrality = np.ones(num_blocks)
    
    # Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * num_blocks
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraint = LinearConstraint(A_ub, b_lb, b_ub)
        bounds_obj = Bounds(0, 1)
        
        # Solve
        res = milp(c=c, constraints=constraint, integrality=integrality, bounds=bounds_obj)
        
        if res.status == 0: # Optimal
            # Extract solution
            x = res.x
            # Due to floating point, threshold at 0.5
            selected_blocks = []
            for i, val in enumerate(x):
                if val > 0.5:
                    selected_blocks.append(all_blocks[i])
            
            # Verify solution validity and count
            if len(selected_blocks) == 132:
                # Quick verification: check no 5-subset is repeated
                covered = set()
                valid = True
                for b in selected_blocks:
                    for s in combinations(b, 5):
                        if s in covered:
                            valid = False
                            break
                        covered.add(s)
                    if not valid:
                        break
                
                if valid and len(covered) == 792:
                    return selected_blocks
                else:
                    # If verification fails (shouldn't happen with MILP), fall through to greedy
                    pass
            
            # If optimal solution wasn't 132 or verification failed, try to return what we have if it's valid
            # But the task asks to maximize, and 132 is the max.
            # If we didn't get 132, we might have a suboptimal result due to solver issues.
            # Let's check if the current result is a valid packing.
            if selected_blocks:
                covered = set()
                valid = True
                for b in selected_blocks:
                    for s in combinations(b, 5):
                        if s in covered:
                            valid = False
                            break
                        covered.add(s)
                    if not valid:
                        break
                if valid:
                    return selected_blocks

    except Exception as e:
        # Fallback if SciPy fails or is not available
        pass

    # Fallback: Greedy Heuristic
    # This is a safety net. It won't reach 132, but provides a valid packing.
    covered = set()
    greedy_blocks = []
    
    # Sort blocks to ensure determinism
    # A better greedy might sort by potential, but simple lexicographic is fine for fallback
    for b in all_blocks:
        # Check if any 5-subset of b is already covered
        conflict = False
        for s in combinations(b, 5):
            if s in covered:
                conflict = True
                break
        
        if not conflict:
            greedy_blocks.append(b)
            # Add all 5-subsets of b to covered
            for s in combinations(b, 5):
                covered.add(s)
            
        # Check time
        if time.time() > deadline:
            break
            
    return greedy_blocks

if __name__ == "__main__":
    # For local testing
    import time
    t0 = time.time()
    sol = construct()
    print(f"Time: {time.time() - t0:.2f}s")
    print(f"Blocks: {len(sol)}")
    # Verify
    covered = set()
    valid = True
    for b in sol:
        for s in combinations(b, 5):
            if s in covered:
                valid = False
                break
            covered.add(s)
        if not valid:
            break
    print(f"Valid: {valid}, Covered: {len(covered)}")
