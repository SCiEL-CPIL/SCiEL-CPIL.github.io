import time
import random
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit (5s buffer)
    
    n = 512
    
    # 1. Generate all 4-term arithmetic progressions (4-APs) in [1, n]
    # A 4-AP is defined by (a, a+d, a+2d, a+3d)
    # We need to identify all such tuples to create constraints.
    # To save memory and time, we can generate them on the fly or store as a list of indices.
    # Number of 4-APs is roughly sum_{d=1}^{n/3} (n - 3d) ~ n^2/6 ~ 43k for n=512. This is small.
    
    ap_list = []
    # a is the first term, d is the difference
    # a + 3d <= n => a <= n - 3d
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            # The four terms are a, a+d, a+2d, a+3d
            # Indices in 0-based array: a-1, a+d-1, a+2d-1, a+3d-1
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
            
    m = len(ap_list)
    
    if m == 0:
        # If no APs, we can take all numbers
        return list(range(1, n + 1))
        
    # 2. Formulate ILP
    # Variables: x_i in {0, 1} for i in 0..n-1
    # Maximize: sum(x_i)
    # Subject to: For each AP (i1, i2, i3, i4), x_i1 + x_i2 + x_i3 + x_i4 <= 3
    
    # Objective coefficients
    c = -np.ones(n) # Minimize -sum(x) to maximize sum(x)
    
    # Constraints matrix A and bounds
    # We can build the sparse matrix directly or use dense if m*n is small enough.
    # m ~ 43k, n = 512. m*n ~ 22 million floats. 22M * 8 bytes = 176 MB.
    # This might be too large for dense memory in some environments, but SciPy handles sparse better.
    # Let's use sparse matrix for safety and speed.
    
    import scipy.sparse as sp
    
    # We need a matrix A of shape (m, n) where A[k, j] = 1 if element j is in AP k.
    # Since each AP has exactly 4 elements, we can build the sparse matrix efficiently.
    
    row_indices = []
    col_indices = []
    data = []
    
    for k, (i1, i2, i3, i4) in enumerate(ap_list):
        row_indices.extend([k, k, k, k])
        col_indices.extend([i1, i2, i3, i4])
        data.extend([1, 1, 1, 1])
        
    A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(m, n))
    
    # Constraints: A @ x <= 3
    # LinearConstraint(A, -np.inf, 3 * np.ones(m))
    
    # Bounds for variables: 0 <= x_i <= 1
    bounds = Bounds(lb=0, ub=1)
    
    # Integrality: x_i must be integer (0 or 1)
    integrality = np.ones(n)
    
    # 3. Solve
    try:
        res = milp(
            c=c,
            constraints=[LinearConstraint(A, -np.inf, 3 * np.ones(m))],
            integrality=integrality,
            bounds=bounds
        )
        
        if time.time() > deadline:
            raise TimeoutError("Solver took too long")
            
        if res.status != 0:
            # If solver fails, fall back to a greedy heuristic
            return greedy_fallback(n, start_time, deadline)
            
        # Extract solution
        # res.x is the array of decisions
        # Due to floating point, we need to round
        x_sol = np.round(res.x).astype(int)
        
        # Ensure validity (just in case of numerical issues, though binary should be exact)
        # The constraint ensures <= 3, so no 4-AP is fully selected.
        
        selected_indices = np.where(x_sol == 1)[0] + 1 # Convert to 1-based
        
        if len(selected_indices) == 0:
             # Fallback if empty
             return greedy_fallback(n, start_time, deadline)
             
        return sorted(selected_indices)
        
    except Exception as e:
        # Fallback if SciPy fails or times out
        return greedy_fallback(n, start_time, deadline)

def greedy_fallback(n, start_time, deadline):
    """
    Fallback strategy: Greedy with local search.
    Used if ILP fails or times out.
    """
    # Simple greedy: add numbers if they don't form a 4-AP with existing ones
    selected = set()
    in_set = [False] * (n + 1)
    
    # Order: Try adding all numbers from 1 to n.
    # To improve, we can randomize or use a heuristic order.
    # For speed, we'll just do a fast greedy pass.
    
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if can_add_fast(x, in_set, n):
            selected.add(x)
            in_set[x] = True
            
    # Local search: Try to swap or add more?
    # Given time constraints, a simple pass is better than nothing.
    # But we can do a few passes of adding if possible? No, greedy is monotonic for addition.
    # We can try to remove elements to allow adding others?
    # Let's stick to the greedy set for the fallback.
    
    return sorted(selected)

def can_add_fast(x, in_set, n):
    """
    Check if x can be added to the set defined by in_set without creating a 4-AP.
    """
    # Check if x is part of any 4-AP with existing elements.
    # x can be term 1, 2, 3, or 4.
    
    # Case 1: x is 1st term (x, x+d, x+2d, x+3d)
    max_d = (n - x) // 3
    for d in range(1, max_d + 1):
        if in_set[x + d] and in_set[x + 2 * d] and in_set[x + 3 * d]:
            return False
            
    # Case 2: x is 2nd term (x-d, x, x+d, x+2d)
    max_d = min(x - 1, (n - x) // 2)
    for d in range(1, max_d + 1):
        if in_set[x - d] and in_set[x + d] and in_set[x + 2 * d]:
            return False
            
    # Case 3: x is 3rd term (x-2d, x-d, x, x+d)
    max_d = min((x - 1) // 2, n - x)
    for d in range(1, max_d + 1):
        if in_set[x - 2 * d] and in_set[x - d] and in_set[x + d]:
            return False
            
    # Case 4: x is 4th term (x-3d, x-2d, x-d, x)
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if in_set[x - 3 * d] and in_set[x - 2 * d] and in_set[x - d]:
            return False
            
    return True

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
