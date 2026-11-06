import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 80.0 # 10s margin
    
    n = 100
    target = 200
    
    # Initial ILP: Maximize sum(x)
    # c = -1 for all
    c = -np.ones(n)
    
    # Integrality
    integrality = np.ones(n)
    
    # Bounds
    bounds = Bounds(lb=0, ub=1)
    
    # List of forbidden subsets (as lists of indices)
    forbidden_subsets = []
    
    # We will iterate
    best_solution = None
    best_cardinality = 0
    
    # Initial constraint: None
    # But milp needs constraints. We can start with empty or dummy.
    # If no constraints, it will pick all 1s.
    
    while time.time() < deadline:
        # Build constraints matrix
        if not forbidden_subsets:
            A = np.empty((0, n))
            b = np.empty(0)
            constraints = LinearConstraint(A, -np.inf, b)
        else:
            # Each row corresponds to a forbidden subset
            # Constraint: sum(x_i for i in subset) <= len(subset) - 1
            A = np.zeros((len(forbidden_subsets), n))
            b = np.zeros(len(forbidden_subsets))
            for k, subset in enumerate(forbidden_subsets):
                for idx in subset:
                    A[k, idx] = 1
                b[k] = len(subset) - 1
            constraints = LinearConstraint(A, -np.inf, b)
            
        # Solve
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)
        
        if not res.success:
            break
            
        x = res.x
        # x should be 0 or 1
        x = (x > 0.5).astype(int)
        
        cardinality = np.sum(x)
        
        # Check for violation
        violation = find_violation(x, target, n)
        
        if violation is None:
            # Valid solution
            if cardinality > best_cardinality:
                best_cardinality = cardinality
                best_solution = x.copy()
            break
        else:
            # Add constraint
            forbidden_subsets.append(violation)
            
            # Optional: Check if we are stuck or taking too long
            # If we have many constraints, it might be slow.
            # But n=100 is small.
            
            # Update best if this solution (even if invalid) has high cardinality?
            # No, we need a valid solution.
            
            # If the current x is the same as a previously rejected one, we might loop.
            # But each new constraint cuts off the current solution.
            # So it should terminate.
            
    if best_solution is None:
        # Fallback to greedy if ILP fails or times out
        return greedy_construct()
        
    return list(np.where(best_solution == 1)[0] + 1)

def find_violation(x, target, n):
    """
    Returns a list of indices (0-based) summing to target, or None.
    x is a numpy array of 0/1.
    """
    # Get items
    items = []
    for i in range(n):
        if x[i] == 1:
            items.append(i) # value is i+1
            
    if not items:
        return None
        
    # DP to find subset sum
    # dp[s] = (prev_sum, item_index) or None
    # We need to reconstruct the set, so we store parent pointers.
    
    # Since target is 200, we can use a list of size target+1
    # parent[s] = (prev_s, item_idx)
    
    parent = [None] * (target + 1)
    reachable = [False] * (target + 1)
    reachable[0] = True
    
    # We iterate through items.
    # To allow backtracking, we need to know which item was used to reach each sum.
    # Standard 0/1 knapsack DP for subset sum.
    
    # We process items one by one.
    # To reconstruct, we can store for each sum s, the last item added.
    # But we need to be careful with the order.
    
    # Alternative: Store the set of indices? No, too big.
    # Store parent pointer in a 2D array? Or just re-run DP with backtracking?
    
    # Let's use a dictionary for the path: path[s] = (prev_s, item_idx)
    # But we need to ensure we don't use an item twice.
    # The standard DP `for item in items: for s from target down to val: ...` ensures distinct items.
    
    # We'll store `prev_sum` and `item_idx` for each sum reached.
    # `parent[s]` = (s - val, idx)
    
    # Initialize
    parent = [None] * (target + 1)
    reachable = [False] * (target + 1)
    reachable[0] = True
    
    # We need to track which items are used.
    # The backtracking will give us the items.
    
    for idx in items:
        val = idx + 1
        if val > target:
            continue
            
        # Iterate backwards to avoid using the same item multiple times
        for s in range(target, val - 1, -1):
            if not reachable[s] and reachable[s - val]:
                reachable[s] = True
                parent[s] = (s - val, idx)
                
        if reachable[target]:
            # Backtrack
            subset = []
            s = target
            while s > 0:
                prev_s, idx = parent[s]
                subset.append(idx)
                s = prev_s
            return subset
            
    return None

def greedy_construct():
    selected, reachable = [], {0}
    for x in range(100, 0, -1):
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected
