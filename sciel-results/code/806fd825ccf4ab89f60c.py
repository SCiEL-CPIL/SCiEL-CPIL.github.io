"""Low discrepancy on all arithmetic progressions - MILP Global Optimization via scipy.optimize.milp."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s/90s kill
    
    n = 256
    
    # Build constraint matrix: each row is one arithmetic progression
    # Variable x_j in {0,1} represents sign_j = 2*x_j - 1
    # For AP with indices idx, sum = sum_{j in idx} (2*x_j - 1)
    # We want: -D <= sum <= D for all APs
    # => -D <= sum_{j in idx} 2*x_j - |idx| <= D
    # => |idx| - D <= sum_{j in idx} 2*x_j <= |idx| + D
    
    # Collect all APs
    ap_indices_list = []
    for d in range(1, n + 1):
        for i in range(min(d, n)):
            idx = list(range(i, n, d))
            if len(idx) > 0:
                ap_indices_list.append(idx)
    
    n_aps = len(ap_indices_list)
    
    # Build constraint matrix A (n_aps x n) where A[k,j] = 2 if j in ap_k else 0
    # Using sparse-friendly approach: build as dense since n=256 is small
    A = np.zeros((n_aps, n), dtype=np.float64)
    b_lower = np.zeros(n_aps, dtype=np.float64)
    b_upper = np.zeros(n_aps, dtype=np.float64)
    
    for k, idx in enumerate(ap_indices_list):
        for j in idx:
            A[k, j] = 2.0
        L = len(idx)
        b_lower[k] = -L  # |idx| - D where D is the variable we minimize
        b_upper[k] = L   # |idx| + D
    
    # Objective: minimize D (the maximum discrepancy bound)
    # c = [0, 0, ..., 0, 1] where last variable is D
    c = np.zeros(n + 1, dtype=np.float64)
    c[n] = 1.0  # Minimize D
    
    # Variables: x_0..x_{n-1} in {0,1}, D in [0, n]
    # Constraints:
    # For each AP k: b_lower[k] - A[k,:]@x <= 0  and  A[k,:]@x - b_upper[k] <= 0
    # But we need D in the constraints: |idx| - D <= 2*sum(x_j for j in idx) <= |idx| + D
    # Rearranging:
    # 2*sum(x_j) - |idx| - D <= 0   =>  A[k,:]@x - D <= |idx|
    # -2*sum(x_j) + |idx| - D <= 0  =>  -A[k,:]@x - D <= -|idx|
    
    # Full constraint matrix: [A, -1] for upper, [-A, -1] for lower
    # Upper: A@x - D <= b_upper  =>  [A, -1] @ [x; D] <= b_upper
    # Lower: -A@x - D <= -b_lower  =>  [-A, -1] @ [x; D] <= -b_lower
    
    A_full = np.zeros((2 * n_aps, n + 1), dtype=np.float64)
    b_full = np.zeros(2 * n_aps, dtype=np.float64)
    
    for k in range(n_aps):
        # Upper constraint: A[k,:]@x - D <= b_upper[k]
        A_full[2*k, :n] = A[k, :]
        A_full[2*k, n] = -1.0
        b_full[2*k] = b_upper[k]
        
        # Lower constraint: -A[k,:]@x - D <= -b_lower[k]
        A_full[2*k+1, :n] = -A[k, :]
        A_full[2*k+1, n] = -1.0
        b_full[2*k+1] = -b_lower[k]
    
    # Integer variables: x_0..x_{n-1} are binary (integer)
    # D is continuous
    integrality = np.zeros(n + 1, dtype=np.int8)
    integrality[:n] = 1  # Binary variables
    integrality[n] = 0   # Continuous
    
    # Bounds: x_j in [0, 1], D in [0, n]
    lb = np.zeros(n + 1, dtype=np.float64)
    ub = np.ones(n + 1, dtype=np.float64)
    ub[n] = float(n)
    
    bounds = Bounds(lb, ub)
    
    # Linear constraint: A_full @ vars <= b_full
    lin_cons = LinearConstraint(A_full, -np.inf, b_full)
    
    try:
        result = milp(
            c=c,
            constraints=[lin_cons],
            integrality=integrality,
            bounds=bounds
        )
        
        if result.success and result.x is not None:
            # Extract signs
            x = result.x[:n]
            signs = np.where(x >= 0.5, 1, -1).astype(np.int16)
            
            # Verify and return
            elapsed = time.time() - start_time
            if elapsed < deadline:
                return signs.tolist()
        
        # If MILP failed or timed out, fall back to a quick heuristic
        # Use Thue-Morse sequence as a good structured initial guess
        thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int16)
        return thue_morse.tolist()
        
    except Exception:
        # Fallback: Thue-Morse
        thue_morse = np.array([1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)], dtype=np.int16)
        return thue_morse.tolist()

# EVOLVE-BLOCK-END
