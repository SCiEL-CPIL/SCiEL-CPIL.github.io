"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
import math
from scipy.optimize import minimize

N = 256

def _build_ap_data():
    """Build all APs and precompute position-to-AP mapping."""
    ap_starts = []
    ap_steps = []
    ap_lengths = []
    
    for d in range(1, N + 1):
        for a in range(1, N + 1):
            max_l = (N - a) // d + 1
            for l in range(1, max_l + 1):
                ap_starts.append(a - 1)
                ap_steps.append(d)
                ap_lengths.append(l)
    
    num_aps = len(ap_starts)
    
    # Build pos_to_ap: for each position, which APs contain it
    pos_to_ap = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        start = ap_starts[ap_idx]
        d = ap_steps[ap_idx]
        l = ap_lengths[ap_idx]
        for i in range(l):
            pos = start + i * d
            pos_to_ap[pos].append(ap_idx)
    
    pos_to_ap_np = [np.array(lst, dtype=np.int32) if lst else np.empty(0, dtype=np.int32) 
                    for lst in pos_to_ap]
    
    return ap_starts, ap_steps, ap_lengths, pos_to_ap_np, num_aps

# Precompute at module load time
_ap_starts, _ap_steps, _ap_lengths, _pos_to_ap_np, _num_aps = _build_ap_data()

def _compute_initial_sums(s):
    """Compute sums for all APs efficiently using prefix sums per (step, residue)."""
    S = np.empty(_num_aps, dtype=np.int16)
    
    # Group APs by step
    aps_by_step = [[] for _ in range(N + 1)]
    for ap_idx in range(_num_aps):
        d = _ap_steps[ap_idx]
        aps_by_step[d].append(ap_idx)
    
    for d in range(1, N + 1):
        if not aps_by_step[d]:
            continue
        
        # Build prefix sums for each residue class mod d
        prefix = {}
        for r in range(d):
            positions = list(range(r, N, d))
            if not positions:
                continue
            vals = s[positions]
            ps = np.zeros(len(vals) + 1, dtype=np.int16)
            np.cumsum(vals, out=ps[1:])
            prefix[r] = ps
        
        # Compute each AP sum with step d
        for ap_idx in aps_by_step[d]:
            start = _ap_starts[ap_idx]
            l = _ap_lengths[ap_idx]
            r = start % d
            idx_in_residue = start // d
            ps = prefix[r]
            S[ap_idx] = ps[idx_in_residue + l] - ps[idx_in_residue]
    
    return S

def _build_hist(S):
    """Build histogram of |S| values and return (hist, max_abs)."""
    hist = np.zeros(N + 1, dtype=np.int32)
    abs_S = np.abs(S)
    np.add.at(hist, abs_S, 1)
    return hist, int(np.max(abs_S))

def _gradient_descent_init(seed=42, max_iter=2000):
    """Use gradient descent on continuous relaxation to find good initial point."""
    rng = random.Random(seed)
    
    # Start with random continuous values in [-1, 1]
    x0 = np.array([rng.uniform(-1, 1) for _ in range(N)], dtype=np.float64)
    
    # Precompute AP data for fast gradient computation
    # For each AP, we need the sum of x over positions in the AP
    # Gradient of sum(x^2) w.r.t. x_i = 2 * x_i * (number of APs containing i) 
    # but actually we want gradient of sum over all APs of (sum_{j in AP} x_j)^2
    
    # More efficient: for each AP, the contribution to gradient at position i is
    # 2 * (sum_{j in AP} x_j) * 1_{i in AP}
    # So grad_i = 2 * sum_{AP containing i} (sum_{j in AP} x_j)
    
    # Precompute: for each position, which APs contain it (already have _pos_to_ap_np)
    # And for each AP, what's its sum
    
    def objective_and_grad(x):
        # Compute all AP sums
        ap_sums = np.zeros(_num_aps, dtype=np.float64)
        
        for d in range(1, N + 1):
            for r in range(d):
                positions = list(range(r, N, d))
                if not positions:
                    continue
                vals = x[positions]
                ps = np.zeros(len(vals) + 1, dtype=np.float64)
                np.cumsum(vals, out=ps[1:])
                
                # Find APs with this step and residue
                for ap_idx in range(_num_aps):
                    if _ap_steps[ap_idx] == d and _ap_starts[ap_idx] % d == r:
                        start = _ap_starts[ap_idx]
                        l = _ap_lengths[ap_idx]
                        idx_in_residue = start // d
                        ap_sums[ap_idx] = ps[idx_in_residue + l] - ps[idx_in_residue]
        
        # Objective: sum of squares of AP sums
        obj = np.sum(ap_sums ** 2)
        
        # Gradient: for each position i, grad_i = 2 * sum_{AP containing i} ap_sum[AP]
        grad = np.zeros(N, dtype=np.float64)
        for i in range(N):
            for ap_idx in _pos_to_ap_np[i]:
                grad[i] += 2 * ap_sums[ap_idx]
        
        return obj, grad
    
    # Use L-BFGS-B for efficiency
    from scipy.optimize import L_BFGS_B
    
    def obj_only(x):
        return objective_and_grad(x)[0]
    
    def grad_only(x):
        return objective_and_grad(x)[1]
    
    res = minimize(obj_only, x0, jac=grad_only, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'gtol': 1e-4})
    
    # Round to nearest -1 or +1
    x_opt = res.x
    signs = np.where(x_opt >= 0, 1, -1).astype(np.int8)
    
    return signs

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_seq = None
    best_score = float('inf')
    
    rng = random.Random(42)
    
    start_temp = 5.0
    end_temp = 0.001
    cooling_rate = 0.995
    max_steps_per_restart = 15000
    
    # Try several initializations: gradient descent + random
    num_initializations = 6
    
    for init_idx in range(num_initializations):
        if time.time() >= deadline:
            break
        
        if init_idx == 0:
            # Gradient descent initialization
            try:
                s = _gradient_descent_init(seed=42)
            except:
                s = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
        elif init_idx == 1:
            # Alternating pattern
            s = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
        elif init_idx == 2:
            # Periodic pattern with period 3
            s = np.array([1 if (i % 3) != 1 else -1 for i in range(N)], dtype=np.int8)
        else:
            rng_init = random.Random(42 + init_idx * 1000)
            s = np.array([1 if rng_init.random() < 0.5 else -1 for _ in range(N)], dtype=np.int8)
        
        S = _compute_initial_sums(s)
        hist, current_max = _build_hist(S)
        
        if current_max < best_score:
            best_score = current_max
            best_seq = s.copy()
        
        # Simulated annealing
        temp = start_temp
        steps_in_restart = 0
        
        while time.time() < deadline and steps_in_restart < max_steps_per_restart:
            j = rng.randint(0, N - 1)
            
            old_val = s[j]
            delta = -2 * old_val
            
            ap_idx_list = _pos_to_ap_np[j]
            
            old_max = current_max
            max_new_abs_in_affected = 0
            
            for ap_idx in ap_idx_list:
                old_sum = S[ap_idx]
                new_sum = old_sum + delta
                
                old_abs = abs(old_sum)
                new_abs = abs(new_sum)
                
                hist[old_abs] -= 1
                hist[new_abs] += 1
                
                S[ap_idx] = new_sum
                
                if new_abs > max_new_abs_in_affected:
                    max_new_abs_in_affected = new_abs
            
            # Update max
            if hist[old_max] == 0:
                if max_new_abs_in_affected > old_max:
                    new_max = max_new_abs_in_affected
                else:
                    scan_result = old_max - 1
                    while scan_result > 0 and hist[scan_result] == 0:
                        scan_result -= 1
                    new_max = max(max_new_abs_in_affected, scan_result)
            else:
                new_max = max(old_max, max_new_abs_in_affected)
            
            cost_diff = new_max - old_max
            
            if cost_diff <= 0:
                accept = True
            else:
                if temp > 0:
                    p = math.exp(-cost_diff / temp)
                    accept = (rng.random() < p)
                else:
                    accept = False
            
            if accept:
                s[j] = -old_val
                current_max = new_max
                if current_max < best_score:
                    best_score = current_max
                    best_seq = s.copy()
            else:
                # Revert
                for ap_idx in ap_idx_list:
                    new_sum = S[ap_idx]
                    old_sum = new_sum - delta
                    
                    old_abs = abs(old_sum)
                    new_abs = abs(new_sum)
                    
                    hist[new_abs] -= 1
                    hist[old_abs] += 1
                    
                    S[ap_idx] = old_sum
            
            temp *= cooling_rate
            steps_in_restart += 1
            
            if temp < end_temp:
                break
        
        # Final local search: try flipping each position and keep if improves
        if time.time() < deadline - 2.0:
            improved = True
            while improved and time.time() < deadline - 1.0:
                improved = False
                for idx in range(N):
                    if time.time() >= deadline - 1.0:
                        break
                    
                    old_sign = s[idx]
                    s[idx] = -old_sign
                    delta = -2 * old_sign
                    
                    ap_idx_list = _pos_to_ap_np[idx]
                    old_max = current_max
                    max_new_abs_in_affected = 0
                    
                    for ap_idx in ap_idx_list:
                        old_sum = S[ap_idx]
                        new_sum = old_sum + delta
                        
                        old_abs = abs(old_sum)
                        new_abs = abs(new_sum)
                        
                        hist[old_abs] -= 1
                        hist[new_abs] += 1
                        
                        S[ap_idx] = new_sum
                        
                        if new_abs > max_new_abs_in_affected:
                            max_new_abs_in_affected = new_abs
                    
                    if hist[old_max] == 0:
                        if max_new_abs_in_affected > old_max:
                            new_max = max_new_abs_in_affected
                        else:
                            scan_result = old_max - 1
                            while scan_result > 0 and hist[scan_result] == 0:
                                scan_result -= 1
                            new_max = max(max_new_abs_in_affected, scan_result)
                    else:
                        new_max = max(old_max, max_new_abs_in_affected)
                    
                    if new_max < best_score:
                        best_score = new_max
                        best_seq = s.copy()
                        current_max = new_max
                        improved = True
                    else:
                        # Revert
                        for ap_idx in ap_idx_list:
                            new_sum = S[ap_idx]
                            old_sum = new_sum - delta
                            
                            old_abs = abs(old_sum)
                            new_abs = abs(new_sum)
                            
                            hist[new_abs] -= 1
                            hist[old_abs] += 1
                            
                            S[ap_idx] = old_sum
    
    if best_seq is None:
        # Fallback: alternating pattern
        best_seq = np.array([1 if i % 2 == 0 else -1 for i in range(N)], dtype=np.int8)
    
    return [int(x) for x in best_seq]
# EVOLVE-BLOCK-END
