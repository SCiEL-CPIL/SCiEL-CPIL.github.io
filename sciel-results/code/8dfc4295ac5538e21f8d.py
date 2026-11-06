import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin before 90s kill
    
    n = 512
    k = 4
    
    # Precompute all valid 4-term APs in [1, n]
    # An AP is defined by (start, diff). 
    # We need to check if start, start+d, start+2d, start+3d are all in the set.
    # We will represent the set as a boolean array `S` of length n+1 (index 0 unused).
    # A set is invalid if there exists an AP such that S[start] & S[start+d] & S[start+2d] & S[start+3d] is True.
    
    # To speed up checking, we can precompute the indices for each AP.
    # However, checking all APs every time might be slow if done in Python loops.
    # Let's use a vectorized approach with NumPy.
    
    # List of all (start, d) pairs
    ap_starts = []
    ap_diffs = []
    for d in range(1, (n - 1) // 3 + 1):
        # start can go from 1 to n - 3d
        max_start = n - 3 * d
        if max_start >= 1:
            starts = np.arange(1, max_start + 1)
            ap_starts.append(starts)
            ap_diffs.append(np.full(len(starts), d))
    
    # Flatten into a single array of indices for the 4 terms
    # term1, term2, term3, term4
    all_t1 = np.concatenate(ap_starts)
    all_t2 = np.concatenate([s + d for s, d in zip(ap_starts, ap_diffs)])
    all_t3 = np.concatenate([s + 2*d for s, d in zip(ap_starts, ap_diffs)])
    all_t4 = np.concatenate([s + 3*d for s, d in zip(ap_starts, ap_diffs)])
    
    # The total number of APs
    num_aps = len(all_t1)
    
    def is_valid(S):
        """
        S is a boolean array of length n+1.
        Returns True if no 4-term AP is fully contained in S.
        """
        if not S.any():
            return True
        # Vectorized check: 
        # valid_ap_mask = S[t1] & S[t2] & S[t3] & S[t4]
        # If any is True, there is a 4-AP.
        # We can break early if we find one, but vectorized check is fast enough for N=512.
        # num_aps is roughly sum_{d=1}^{170} (512 - 3d) ~ 170*512 - 3*170^2/2 ~ 87k - 43k ~ 44k checks.
        # NumPy handles 44k boolean ops in microseconds.
        
        # Optimization: Only check if potential.
        # We can just do the full check.
        check = S[all_t1] & S[all_t2] & S[all_t3] & S[all_t4]
        return not check.any()
    
    def count_aps(S):
        """Count the number of 4-APs in S."""
        check = S[all_t1] & S[all_t2] & S[all_t3] & S[all_t4]
        return np.sum(check)
    
    # Initial state: Random sparse set to avoid immediate conflicts, or greedy.
    # Let's start with a random set of density ~ 1/4 to give SA room.
    # Actually, a better start is a greedy set.
    
    def greedy_init():
        S = np.zeros(n + 1, dtype=bool)
        # Try to add elements in random order
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        for x in candidates:
            S[x] = True
            if not is_valid(S):
                S[x] = False
        return S
    
    # Run greedy init
    S = greedy_init()
    best_S = S.copy()
    best_size = np.sum(S)
    
    # Simulated Annealing
    # Temperature starts high, cools down.
    # Move: pick a random element to add or remove.
    # If size increases and valid, accept.
    # If size decreases, reject (we want to maximize size).
    # If size same, accept with prob exp(-delta_E / T). But we don't have energy, we have validity.
    # Standard approach for Max-Cardinality Constraint Satisfaction:
    # 1. If current state is invalid, try to fix it (remove elements involved in APs).
    # 2. If current state is valid, try to add elements.
    # 3. If adding fails, try swapping (remove one, add one) or double-swap (remove two, add two).
    
    # Let's use a simpler "Hill Climbing with Random Restarts" or "SA on Validity"
    # Actually, for this specific problem, "Local Search on Valid Sets" is best.
    # Strategy:
    # 1. Maintain a valid set S.
    # 2. Try to add a random element x. If valid, keep it.
    # 3. If stuck, try to remove a random element y and add two random elements x1, x2. If valid, keep it.
    # 4. If stuck, try to remove a random element y and add one random element x (neutral move) to explore, 
    #    but this risks getting stuck in smaller sets. 
    #    Better: "Kick" - remove a random subset of k elements and re-run greedy on the freed spots + random new spots.
    
    # Let's implement a robust SA-like loop:
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute for fast validity check of single additions?
    # Adding x creates APs if x is part of an AP where other 3 are in S.
    # We can precompute for each x, the list of APs it participates in.
    
    # For simplicity and speed, we rely on the vectorized is_valid(S) which is ~0.1ms.
    # This allows ~1000-5000 moves per second.
    
    while time.time() < deadline - 5:
        if not is_valid(S):
            # This should not happen if we maintain validity, but just in case
            # Remove random elements until valid
            while not is_valid(S) and np.sum(S) > 0:
                idxs = np.where(S)[0]
                if len(idxs) == 0: break
                rem = random.choice(idxs)
                S[rem] = False
        
        # Try to improve
        improved = False
        
        # Phase 1: Try to add random elements
        unselected = np.where(~S)[0]
        unselected = unselected[unselected > 0] # Exclude 0
        
        if len(unselected) > 0:
            # Try a batch of random additions
            for _ in range(100):
                if time.time() > deadline - 5:
                    break
                x = random.choice(unselected)
                S[x] = True
                if is_valid(S):
                    best_S = S.copy()
                    best_size = np.sum(S)
                    improved = True
                    # Update unselected list lazily or just re-evaluate next time
                    # For efficiency in this loop, we just break and restart the loop
                    break
                else:
                    S[x] = False
        
        if improved:
            continue
            
        # Phase 2: Try swaps (Remove 1, Add 2) to increase size
        selected = np.where(S)[0]
        selected = selected[selected > 0]
        
        if len(selected) > 0 and len(unselected) >= 2:
            for _ in range(50):
                if time.time() > deadline - 5:
                    break
                # Remove one
                y = random.choice(selected)
                S[y] = False
                
                # Add two
                # Pick two from unselected (excluding y, which is now unselected)
                # Note: unselected list is stale, so filter
                candidates = [x for x in unselected if x != y]
                if len(candidates) < 2:
                    S[y] = True
                    continue
                
                x1, x2 = random.sample(candidates, 2)
                S[x1] = True
                S[x2] = True
                
                if is_valid(S):
                    best_S = S.copy()
                    best_size = np.sum(S)
                    improved = True
                    break
                else:
                    S[x1] = False
                    S[x2] = False
                    S[y] = True # Restore
            
        if improved:
            continue
            
        # Phase 3: Neutral moves (Remove 1, Add 1) to escape local optima
        # This allows changing the configuration without changing size
        if len(selected) > 0 and len(unselected) > 0:
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                y = random.choice(selected)
                x = random.choice([u for u in unselected if u != y])
                
                S[y] = False
                S[x] = True
                
                # Accept if valid (size same)
                if is_valid(S):
                    # Accept with some probability to allow exploration? 
                    # For now, accept if valid.
                    best_S = S.copy() # Size is same, but config changed
                    # Do not update best_size
                    break
                else:
                    S[x] = False
                    S[y] = True
        
        # If stuck, kick: remove a few random elements and let greedy fill them back
        if not improved:
            # Remove 5 random elements
            if len(selected) >= 5:
                kicks = random.sample(selected, 5)
                for k in kicks:
                    S[k] = False
                # Re-greedy on the empty spots + some random new spots?
                # Just let the next iterations try to add.
                # To be more aggressive, re-run greedy on the whole set? No, too slow.
                # Just let the loop continue, it will try to add.

        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
    # Final check and return
    if not is_valid(best_S):
        # Fallback: return a valid subset of best_S
        # Remove elements until valid
        while not is_valid(best_S) and np.sum(best_S) > 0:
            idxs = np.where(best_S)[0]
            if len(idxs) == 0: break
            rem = random.choice(idxs)
            best_S[rem] = False
            
    result = np.where(best_S)[0]
    result = result[result > 0]
    return sorted(result.tolist())

# Test run
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    S = set(res)
    valid = True
    for a in S:
        for d in range(1, (512 - a) // 3 + 1):
            if (a+d) in S and (a+2*d) in S and (a+3*d) in S:
                valid = False
                break
        if not valid:
            break
    print(f"Valid: {valid}")
