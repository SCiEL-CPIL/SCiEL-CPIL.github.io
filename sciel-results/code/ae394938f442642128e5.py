import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

def construct():
    n = 128
    target_low = 128
    target_high = 256
    targets = list(range(target_low, target_high + 1))
    
    # Precompute pairs for each target sum
    pairs = {s: [] for s in targets}
    for s in targets:
        start = max(0, s - 128)
        end = min(128, s // 2)
        for a in range(start, end + 1):
            b = s - a
            if b >= a and b <= 128:
                pairs[s].append((a, b))
                
    all_pairs = []
    pair_to_sum = {}
    for s in targets:
        for p in pairs[s]:
            all_pairs.append(p)
            pair_to_sum[p] = s
            
    num_elements = 129
    num_pairs = len(all_pairs)
    pair_indices = {p: i for i, p in enumerate(all_pairs)}
    
    # Variables:
    # 0..128: x_i
    # 129..128+num_pairs: z_j
    
    total_vars = num_elements + num_pairs
    
    c = np.zeros(total_vars)
    integrality = np.ones(total_vars)
    bounds = Bounds(lb_var=np.zeros(total_vars), ub_var=np.ones(total_vars))
    
    def solve_for_K(K):
        # Constraints
        # 1. Sum constraints: 2 per target
        # 2. Linearization: 3 per pair
        num_constraints = 2 * len(targets) + 3 * num_pairs
        
        A = lil_matrix((num_constraints, total_vars))
        lb = np.zeros(num_constraints)
        ub = np.zeros(num_constraints)
        
        row_idx = 0
        
        # Sum constraints
        for s in targets:
            # >= 1
            for p in pairs[s]:
                idx = pair_indices[p]
                A[row_idx, 129 + idx] = 1.0
            lb[row_idx] = 1.0
            ub[row_idx] = np.inf
            row_idx += 1
            
            # <= K
            for p in pairs[s]:
                idx = pair_indices[p]
                A[row_idx, 129 + idx] = 1.0
            lb[row_idx] = -np.inf
            ub[row_idx] = K
            row_idx += 1
            
        # Linearization constraints
        for j, p in enumerate(all_pairs):
            a, b = p
            # z_j <= x_a  =>  -x_a + z_j <= 0
            A[row_idx, a] = -1.0
            A[row_idx, 129 + j] = 1.0
            lb[row_idx] = -np.inf
            ub[row_idx] = 0.0
            row_idx += 1
            
            # z_j <= x_b  =>  -x_b + z_j <= 0
            A[row_idx, b] = -1.0
            A[row_idx, 129 + j] = 1.0
            lb[row_idx] = -np.inf
            ub[row_idx] = 0.0
            row_idx += 1
            
            # z_j >= x_a + x_b - 1  =>  -x_a - x_b + z_j <= -1
            A[row_idx, a] = -1.0
            A[row_idx, b] = -1.0
            A[row_idx, 129 + j] = 1.0
            lb[row_idx] = -np.inf
            ub[row_idx] = -1.0
            row_idx += 1
            
        res = milp(c=c, constraints=LinearConstraint(A.tocsr(), lb, ub), integrality=integrality, bounds=bounds)
        
        if res.status == 0:
            x_vals = res.x[:num_elements]
            A_set = set(int(i) for i in range(num_elements) if x_vals[i] > 0.5)
            return A_set
        else:
            return None
            
    best_A = None
    # Search for minimal K
    # The current best is ~16. Try 1 to 20.
    for K in range(1, 21):
        A_set = solve_for_K(K)
        if A_set is not None:
            # Verification
            mults = {s: 0 for s in targets}
            A_list = sorted(A_set)
            for i, a in enumerate(A_list):
                for j in range(i, len(A_list)):
                    b = A_list[j]
                    s = a + b
                    if target_low <= s <= target_high:
                        mults[s] += 1
            if all(m > 0 for m in mults.values()):
                best_A = A_set
                break
                
    if best_A is None:
        # Fallback
        # Try higher K
        for K in range(21, 40):
            A_set = solve_for_K(K)
            if A_set is not None:
                best_A = A_set
                break
        if best_A is None:
            best_A = set(range(64, 129))
            
    return sorted(best_A)
