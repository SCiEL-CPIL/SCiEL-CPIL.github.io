"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
import sys
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit

    # 1. Precompute Arithmetic Progressions
    # We focus on APs of length >= 2.
    # To keep the MILP tractable, we select a "critical subset" of APs.
    # The worst discrepancies usually arise from short APs (length 2, 3) or long APs with small steps.
    # Let's include all APs of length 2 and 3, and a sample of longer ones.
    # Actually, for n=256, the number of APs is manageable but large (~15k-30k).
    # MILP with 256 variables and ~10k-20k constraints might be slow if not careful.
    # However, scipy's milp (HiGHS) is quite fast for small binaries.
    
    # Let's refine the AP selection.
    # All APs of length 2: (i, j) with i<j. ~32k pairs.
    # All APs of length 3: (i, i+d, i+2d).
    # All APs of length >= 4 with step 1 (contiguous blocks).
    
    # To ensure we don't miss the global max, we should ideally include all.
    # But 32k constraints might push the time limit.
    # Let's try including all APs of length <= 4 and all step-1 APs.
    # And maybe some random longer ones.
    
    # Actually, let's just generate all APs of length >= 2.
    # Count:
    # step=1: 255+254+...+1 = ~32k
    # step=2: 127+... ~8k
    # Total is roughly sum_{d=1}^{128} (256-d)^2/d approx? No.
    # Number of APs with step d is sum_{L=2}^{floor((255)/d)+1} (256 - L*d + 1).
    # This is roughly O(n^3). For n=256, it's around 10^5-10^6?
    # Let's estimate.
    # d=1: 255*128 ~ 32k
    # d=2: 127*64 ~ 8k
    # d=3: 85*28 ~ 2.4k
    # Sum is dominated by small d. Total < 50k.
    # 50k constraints x 256 vars. This is a sparse matrix. HiGHS should handle it.
    
    ap_indices = []
    for d in range(1, n // 2 + 1):
        max_start = n - 2 * d
        for start in range(1, max_start + 1, 1):
            # Generate AP: start, start+d, ...
            terms = []
            cur = start
            while cur <= n:
                terms.append(cur - 1)
                cur += d
            if len(terms) >= 2:
                ap_indices.append(terms)
                
    # Optimization: We don't need to check APs that are subsets of others?
    # No, the sum of a subset can be larger in magnitude than the superset if signs cancel in the superset.
    # So we must check all.
    
    num_aps = len(ap_indices)
    # Build sparse matrix A for constraints: sum(signs in AP) <= D and -sum(signs in AP) <= D
    # x_i in {-1, 1}. Let y_i = (x_i + 1) / 2 in {0, 1}.
    # x_i = 2*y_i - 1.
    # sum(2*y_i - 1) <= D  =>  2*sum(y_i) - len(AP) <= D  =>  sum(y_i) <= (D + len(AP)) / 2
    # -sum(2*y_i - 1) <= D => -2*sum(y_i) + len(AP) <= D => sum(y_i) >= (len(AP) - D) / 2
    
    # We will bisect on D.
    # Lower bound for D: 1 (since singleton APs have sum +/-1, wait, task says "including one" in description 
    # "any start, positive step and length including one". 
    # "minimize the maximum absolute signed sum on ANY arithmetic progression ... including one"
    # If length 1 is included, max sum is at least 1.
    # If we ignore length 1, min max sum could be 0 for some n? No, for n=256, can we have sum 0 for all APs?
    # Thue-Morse has discrepancy O(sqrt(n)).
    # Let's assume D_min = 1.
    
    # Build the constraint matrix A for y variables.
    # A[i, j] = 1 if position j is in AP i, else 0.
    # We need A and -A for the two inequalities.
    # Actually, we can just use A and bounds.
    # Constraint: (len(AP) - D) / 2 <= A @ y <= (len(AP) + D) / 2
    
    # Let's build the sparse matrix efficiently.
    # Using scipy.sparse
    from scipy.sparse import csr_matrix
    
    rows = []
    cols = []
    vals = []
    ap_lengths = []
    
    for i, idx in enumerate(ap_indices):
        L = len(idx)
        ap_lengths.append(L)
        for j in idx:
            rows.append(i)
            cols.append(j)
            vals.append(1)
            
    A = csr_matrix((vals, (rows, cols)), shape=(num_aps, n))
    ap_lengths = np.array(ap_lengths)
    
    def check_feasibility(D):
        # Returns solution y if feasible, else None
        # Constraints:
        # A @ y <= (ap_lengths + D) / 2
        # -A @ y <= (ap_lengths - D) / 2  => A @ y >= (D - ap_lengths) / 2
        
        c = np.zeros(n) # Minimize 0
        lb = (D - ap_lengths) / 2.0
        ub = (ap_lengths + D) / 2.0
        
        # LinearConstraint(A, lb, ub)
        # Note: scipy milp expects constraints in form A @ x <= b or A @ x >= b or bounds.
        # LinearConstraint(A, lb, ub) means lb <= A @ x <= ub.
        
        cons = LinearConstraint(A, lb, ub)
        integrality = np.ones(n)
        bounds = Bounds(0, 1)
        
        # HiGHS options
        options = {
            'time_limit': 10.0, # Limit per MILP call to avoid hanging
            'output_flag': False
        }
        
        try:
            res = milp(c=c, constraints=cons, integrality=integrality, bounds=bounds, options=options)
            if res.status == 0: # Optimal
                # y is binary
                y = res.x
                # Convert to signs
                signs = 2 * y - 1
                return signs
            else:
                return None
        except Exception:
            return None

    # Bisection on D
    # Max possible D is n (sum of all).
    # Min possible D is 1.
    
    # To speed up, we can use a heuristic initial guess or just bisect.
    # Range [1, 10] is likely.
    
    best_signs = None
    best_D = None
    
    # Try a few specific D values first to get a quick result, then bisect?
    # Or just bisect from 1 to 10.
    
    low = 1
    high = 10 # Unlikely to be > 10 for n=256 with good search, but let's be safe.
    
    # Actually, let's just try D=1, 2, 3... until feasible.
    # Since D is integer (sums are integers), we can iterate.
    
    # But MILP might be slow. Let's try D=3 first?
    # Thue-Morse discrepancy for n=256 is roughly sqrt(256)=16? No, Thue-Morse is O(sqrt(n)) for APs?
    # Actually, Thue-Morse has low discrepancy for APs.
    # Let's check Thue-Morse max discrepancy quickly.
    
    def thue_morse_signs(n):
        signs = np.zeros(n, dtype=np.int8)
        for i in range(n):
            cnt = bin(i).count('1')
            signs[i] = 1 if cnt % 2 == 0 else -1
        return signs
    
    tm = thue_morse_signs(n)
    
    # Compute max discrepancy for TM
    def compute_max_disc(signs):
        max_d = 0
        for idx in ap_indices:
            s = np.sum(signs[idx])
            if abs(s) > max_d:
                max_d = abs(s)
        return max_d
    
    # This is O(num_aps * avg_len). Might be slow in Python loop.
    # Use vectorized sum for TM check.
    # A @ tm
    tm_disc = np.abs(A @ tm)
    tm_max = np.max(tm_disc)
    
    # Use tm_max as upper bound for bisection
    high = int(tm_max)
    low = 1
    
    # If high is small, we are good.
    # If high is large, we need to search.
    
    # Let's try to find the minimal D.
    # We will test D values from low to high.
    # But MILP is expensive.
    # Maybe we can use the TM solution as a starting point for a local search?
    # The prompt asks for a DIFFERENT strategy. MILP is different.
    
    # Let's try D = high, high-1, ... until feasible.
    # Or bisection. Bisection is safer for time.
    
    # However, if D is small (e.g. 3), and infeasible, we move up.
    # If D is large, we might find it fast.
    
    # Let's do a linear scan from low upwards, but with a time limit per call.
    # If we exceed time, return best found (or TM if no MILP result).
    
    current_best = tm.copy()
    current_best_max = tm_max
    
    # Try D values
    for D in range(low, high + 1):
        if time.time() > deadline - 5.0:
            break
            
        # Check if D is feasible
        # Optimization: If D >= current_best_max, we don't need to solve, we already have a solution.
        # But we want to MINIMIZE D. So we stop at the first feasible D.
        
        signs = check_feasibility(D)
        if signs is not None:
            # Verify the discrepancy of the found solution
            # The MILP ensures |sum| <= D for the included APs.
            # We should verify on ALL APs just in case of numerical issues or if we missed some?
            # We included all APs.
            # Let's compute the actual max discrepancy to be sure.
            actual_max = np.max(np.abs(A @ signs))
            
            if actual_max < current_best_max:
                current_best_max = actual_max
                current_best = signs.copy()
            
            # Since we are scanning from low to high, the first feasible D is the minimum.
            # So we can break.
            break
            
    # If we didn't find a better solution via MILP, or if we ran out of time,
    # we can try a quick local search polish on the best solution found (TM or MILP).
    
    # Local Search Polish
    # We will use the best_signs found so far.
    
    # Precompute pos_to_aps for fast updates
    pos_to_aps = [[] for _ in range(n)]
    for i, idx in enumerate(ap_indices):
        for j in idx:
            pos_to_aps[j].append(i)
            
    # Convert A to dense for fast dot products? No, A is sparse.
    # For local search, we need to update sums.
    # ap_sums = A @ signs
    # If we flip sign at pos p, ap_sums[ap_id] changes by +/- 2.
    
    signs = current_best.astype(np.int8)
    ap_sums = (A @ signs).astype(np.int32)
    current_max = np.max(np.abs(ap_sums))
    
    # Greedy Polish
    # Iterate through positions, flip if it reduces max.
    # To speed up, we can keep a histogram of abs sums?
    # Or just recompute max? Recomputing max is O(num_aps).
    # num_aps is ~50k. 256 * 50k = 12.8M ops per pass.
    # In Python, this is slow.
    # Use numpy for the check?
    
    # Let's do a few passes of greedy.
    improved = True
    passes = 0
    while improved and passes < 5 and time.time() < deadline - 2.0:
        improved = False
        passes += 1
        # Shuffle order to avoid bias?
        # For simplicity, sequential.
        for i in range(n):
            if time.time() > deadline - 2.0:
                break
                
            old_sign = signs[i]
            # Flip
            signs[i] = -old_sign
            
            # Update sums
            # ap_sums[pos_to_aps[i]] += -2 * old_sign
            # Since old_sign is +/-1, delta is -2*old_sign = +/-2
            delta = -2 * old_sign
            affected = pos_to_aps[i]
            ap_sums[affected] += delta
            
            new_max = np.max(np.abs(ap_sums))
            
            if new_max < current_max:
                current_max = new_max
                improved = True
            else:
                # Revert
                signs[i] = old_sign
                ap_sums[affected] -= delta
                
    # Final check
    final_max = np.max(np.abs(A @ signs))
    if final_max < current_best_max:
        current_best = signs
        current_best_max = final_max
        
    return [int(s) for s in current_best]

# EVOLVE-BLOCK-END
