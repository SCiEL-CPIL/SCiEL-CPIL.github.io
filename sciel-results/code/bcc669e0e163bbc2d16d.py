"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = 60.0  # Keep well under the 90s limit
    
    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    
    # Map number to index
    num_to_idx = {num: i for i, num in enumerate(numbers)}
    
    # Objective: Maximize sum(1/x) -> Minimize sum(-1/x)
    c = [-1.0 / num for num in numbers]
    
    # Constraints: x_i + x_j <= 1 for all conflicting pairs
    # A conflict exists if lcm(i, j) <= N
    A_rows = []
    b_lower = []
    b_upper = []
    
    # Precompute conflicts
    # This is O(N^2) which is ~65k checks, very fast.
    for i in range(n):
        for j in range(i + 1, n):
            a = numbers[i]
            b = numbers[j]
            if math.lcm(a, b) <= N:
                # Constraint: x_i + x_j <= 1
                row = [0.0] * n
                row[i] = 1.0
                row[j] = 1.0
                A_rows.append(row)
                b_lower.append(0.0)
                b_upper.append(1.0)
    
    # If no conflicts, we can just pick all? No, lcm(x,x) is not considered for distinct pairs, 
    # but if there are no conflicts between distinct pairs, we can pick all.
    # However, the problem states "for all distinct pairs".
    
    if not A_rows:
        # No conflicts, pick all numbers? 
        # Wait, if there are no conflicts, any subset is valid. 
        # To maximize sum, we pick all.
        return numbers
    
    A = [A_rows]
    
    # Bounds: 0 <= x_i <= 1
    integrality = [1] * n
    
    # Solve MIP
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, b_lower, b_upper),
            integrality=integrality,
            bounds=Bounds(lb=0, ub=1)
        )
        
        # Check time
        if time.time() > deadline:
            raise TimeoutError("MIP took too long")
            
        if res.status == 0:  # Optimal
            # Extract solution
            x = res.x
            selected = []
            for i, val in enumerate(x):
                if val > 0.5: # It's integer, so 1.0
                    selected.append(numbers[i])
            
            # Verify solution validity just in case
            valid = True
            for i in range(len(selected)):
                for j in range(i + 1, len(selected)):
                    if math.lcm(selected[i], selected[j]) <= N:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return selected
            else:
                # Fallback if solver returned invalid (should not happen)
                pass
                
    except Exception as e:
        # Fallback to greedy heuristic if MIP fails
        pass

    # Greedy Fallback:
    # Sort by weight (1/x) descending, i.e., x ascending.
    # Pick x if it doesn't conflict with already picked.
    selected_greedy = []
    for x in numbers: # numbers are already sorted ascending
        # Check conflict with all in selected_greedy
        conflict = False
        for y in selected_greedy:
            if math.lcm(x, y) <= N:
                conflict = True
                break
        if not conflict:
            selected_greedy.append(x)
            
    if time.time() > deadline:
         # Emergency return
         return selected_greedy if selected_greedy else [2]
         
    return selected_greedy if selected_greedy else [2]

# EVOLVE-BLOCK-END
