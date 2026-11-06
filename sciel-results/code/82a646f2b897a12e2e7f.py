import time
import numpy as np
from scipy.optimize import minimize

def construct():
    n = 256
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    # Precompute all 3-APs and 4-APs
    # 3-AP: (a, a+d, a+2d)
    ap3s = []
    for d in range(1, n // 2 + 1):
        for a in range(0, n - 2 * d):
            ap3s.append((a, a + d, a + 2 * d))
    
    # 4-AP: (a, a+d, a+2d, a+3d)
    ap4s = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap4s.append((a, a + d, a + 2 * d, a + 3 * d))
    
    ap3_array = np.array(ap3s, dtype=np.int32)
    ap4_array = np.array(ap4s, dtype=np.int32)
    
    M3 = len(ap3_array)
    M4 = len(ap4_array)
    
    # Penalty weight for 4-APs. Must be high enough to effectively forbid them.
    # Max possible 3-APs is ~ n^2/4. Let's use a large penalty.
    PENALTY = 1000.0

    def objective_and_grad(x):
        """
        Objective to minimize: -(# 3-APs) + PENALTY * (# 4-APs)
        x: numpy array of shape (n,) in [0,1]
        """
        # # 3-APs: sum of products x[a]*x[b]*x[c] for each AP
        # Vectorized: np.sum(x[ap3_array].prod(axis=1))
        # This creates a large temporary array (M3,), which is fine for M3 ~ 10k.
        
        # # 4-APs: sum of products x[a]*x[b]*x[c]*x[d]
        # Vectorized: np.sum(x[ap4_array].prod(axis=1))
        
        # To be more efficient and avoid huge temporaries if n was larger, 
        # we could use dot products, but for n=256, M3~10k, M4~2k, this is fast.
        
        # Note: We want to MAXIMIZE 3-APs, so we minimize -3APs.
        # We want to MINIMIZE 4-APs (ideally 0), so we add +PENALTY * 4APs.
        
        # Optimization: 
        # 3AP count = sum_{i,j,k in AP} x_i x_j x_k
        # 4AP count = sum_{i,j,k,l in AP} x_i x_j x_k x_l
        
        # Let's compute them.
        # Using np.prod on the sliced array is straightforward.
        
        # Potential issue: x might have values slightly outside [0,1] during optimization steps?
        # COBYLA respects bounds.
        
        p3 = np.prod(x[ap3_array], axis=1)
        p4 = np.prod(x[ap4_array], axis=1)
        
        score_3 = np.sum(p3)
        score_4 = np.sum(p4)
        
        val = -score_3 + PENALTY * score_4
        return val

    # Initial guesses
    # 1. All 1s (will have many 4-APs, but optimizer should push them down)
    # 2. Random
    # 3. Sparse random
    
    best_val = float('inf')
    best_x = None
    
    # We will run COBYLA from a few starting points.
    # COBYLA is a derivative-free method suitable for constrained optimization.
    # Bounds: [0, 1] for each x_i
    
    bounds = [(0, 1)] * n
    
    # Starting points
    starts = [
        np.ones(n),
        np.random.rand(n),
        np.random.choice([0, 1], n, p=[0.5, 0.5]).astype(float),
        np.random.choice([0, 1], n, p=[0.8, 0.2]).astype(float) # Sparse
    ]
    
    for i, x0 in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
            
        try:
            res = minimize(objective_and_grad, x0, method='COBYLA', bounds=bounds, 
                           options={'maxiter': 2000, 'rhobeg': 1.0})
            
            if res.success or res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception as e:
            pass
    
    # If COBYLA didn't find a good solution, or if we want to ensure a valid discrete solution,
    # we should round the continuous solution and verify/repair.
    
    if best_x is None:
        # Fallback: random valid set
        # Greedy construction
        in_S = np.zeros(n, dtype=bool)
        for v in range(n):
            # Check 4-AP validity
            valid = True
            # Check all 4-APs containing v
            # This is slow, but fallback should be rare.
            # Actually, let's just return a simple known valid set if optimization fails completely.
            # But let's try a quick greedy.
            pass
        
        # Simple fallback: even numbers? No, 2,4,6,8 is a 4-AP.
        # Primes? 2,3,5,7 (diff 1,2,2) no. 3,5,7 (2). 5,7,9 (2). 
        # Let's just use a random set and check.
        
        # For now, if best_x is None, return [1]
        return [1]
        
    # Round to binary
    # Threshold 0.5
    x_bin = (best_x > 0.5).astype(int)
    
    # Verify validity and count
    # If invalid, we might need to repair.
    # Given the high penalty, it's likely valid.
    
    # Let's do a quick local repair if invalid.
    # Check 4-APs
    # For each 4-AP, if all 4 are 1, we must remove one.
    # Which one to remove? The one that minimizes loss of 3-APs.
    
    # Precompute 3-AP membership for repair
    ap3_membership = [[] for _ in range(n)]
    for idx, ap in enumerate(ap3_array):
        for v in ap:
            ap3_membership[v].append(idx)
            
    # Convert to arrays
    ap3_mem_arrays = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in ap3_membership]
    
    # Check for violations
    # Iterate through 4-APs
    # If a 4-AP is active, resolve it.
    # This is a greedy repair.
    
    # To make repair efficient, we can loop until no violations.
    # But we must be careful with time.
    
    # Let's count 3-APs and 4-APs first.
    
    def count_3aps_discrete(x):
        if M3 == 0: return 0
        return np.sum(np.all(x[ap3_array] == 1, axis=1))
        
    def count_4aps_discrete(x):
        if M4 == 0: return 0
        return np.sum(np.all(x[ap4_array] == 1, axis=1))
        
    current_3 = count_3aps_discrete(x_bin)
    current_4 = count_4aps_discrete(x_bin)
    
    # If current_4 > 0, repair.
    # We will iterate.
    # For each violating 4-AP, remove the element that is in the fewest 3-APs (among those in the 4-AP).
    
    max_iter = 100
    for _ in range(max_iter):
        if time.time() - start_time > time_limit:
            break
            
        # Find first violating 4-AP
        # To be faster, we can check all at once?
        # violations = np.where(np.all(x_bin[ap4_array] == 1, axis=1))[0]
        # If empty, break.
        
        violation_mask = np.all(x_bin[ap4_array] == 1, axis=1)
        if not np.any(violation_mask):
            break
            
        # Pick the first one
        v_idx = np.argmax(violation_mask)
        ap = ap4_array[v_idx]
        
        # Find element in ap with min 3-AP degree
        min_deg = float('inf')
        min_v = ap[0]
        
        for v in ap:
            # Count 3-APs containing v that are active in current x_bin
            mem = ap3_mem_arrays[v]
            if len(mem) == 0:
                deg = 0
            else:
                # Check if other 2 elements are in x_bin
                # x_bin is 0/1 int array
                # We need to check if the other 2 are 1.
                # Since v is 1, sum of x_bin[ap3_array[mem]] should be 3.
                counts = np.sum(x_bin[ap3_array[mem]], axis=1)
                deg = np.sum(counts == 3)
                
            if deg < min_deg:
                min_deg = deg
                min_v = v
                
        # Remove min_v
        x_bin[min_v] = 0
        
    # Final check and count
    final_3 = count_3aps_discrete(x_bin)
    final_4 = count_4aps_discrete(x_bin)
    
    # If still invalid (should not happen), fallback
    if final_4 > 0:
        # Force validity by removing elements from any remaining 4-APs
        # This is brute force
        for idx in range(M4):
            if time.time() - start_time > time_limit:
                break
            if np.all(x_bin[ap4_array[idx]] == 1):
                # Remove first element
                x_bin[ap4_array[idx][0]] = 0
        
        final_3 = count_3aps_discrete(x_bin)
        
    # If empty, return [1]
    if not np.any(x_bin):
        return [1]
        
    result = [i + 1 for i in range(n) if x_bin[i]]
    return result

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
