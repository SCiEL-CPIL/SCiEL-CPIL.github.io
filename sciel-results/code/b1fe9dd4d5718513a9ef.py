import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit

    N_LOW = 128
    N_HIGH = 256
    RANGE_SIZE = N_HIGH - N_LOW + 1
    MAX_VAL = 128
    N_ELEMENTS = 129

    # Precompute for each sum s in [128, 256], the list of pairs (i, j) with i <= j
    # such that i + j = s and 0 <= i, j <= 128
    # We store them as flat arrays of indices for fast numpy operations
    pairs_for_sum = [None] * (N_HIGH + 1)
    for s in range(N_LOW, N_HIGH + 1):
        pairs = []
        i_min = max(0, s - 128)
        i_max = min(128, s // 2)
        for i in range(i_min, i_max + 1):
            j = s - i
            if j <= 128 and j >= 0:
                pairs.append((i, j))
        if pairs:
            arr = np.array(pairs, dtype=np.int32).flatten()
            pairs_for_sum[s] = arr
        else:
            pairs_for_sum[s] = np.array([], dtype=np.int32)

    def calc_reps(mask):
        """
        Calculate representation counts for all sums in [128, 256].
        mask: numpy array of 0s and 1s of length 129.
        Returns: numpy array of length 129 (indices 0..128 corresponding to sums 128..256).
        """
        reps = np.zeros(RANGE_SIZE, dtype=np.int32)
        for idx, s in enumerate(range(N_LOW, N_HIGH + 1)):
            pairs = pairs_for_sum[s]
            if len(pairs) == 0:
                continue
            p = pairs.reshape(-1, 2)
            # Get mask values for both elements in each pair
            m_vals = mask[p]
            # Product is 1 only if both are in A
            prods = m_vals[:, 0] * m_vals[:, 1]
            reps[idx] = np.sum(prods)
        return reps

    def get_objective(reps):
        """
        Objective: Minimize peak multiplicity, but heavily penalize uncovered sums.
        If any sum is uncovered (rep == 0), return a large penalty.
        Otherwise, return the max rep.
        """
        uncovered_count = np.sum(reps == 0)
        if uncovered_count > 0:
            # Penalty encourages covering first
            return 1000 + uncovered_count * 100 + int(np.max(reps))
        else:
            return int(np.max(reps))

    def mask_to_list(mask):
        return [i for i in range(N_ELEMENTS) if mask[i] == 1]

    # --- Phase 1: Greedy Balanced Set Cover ---
    # Goal: Start with an empty set and add elements that cover the most "deficient" sums.
    # "Deficient" means sums with low current representation count.
    
    def greedy_balanced_cover():
        mask = np.zeros(N_ELEMENTS, dtype=np.int32)
        reps = calc_reps(mask)
        
        # We want to cover all sums. 
        # Strategy: Iteratively pick an element x that maximizes the sum of "gains" 
        # for uncovered or low-rep sums.
        
        # To make it "balanced", we prioritize sums that are currently 0 or close to the current min.
        # Let's define a weight for each sum s: 
        # If reps[s] == 0, weight is high.
        # If reps[s] > 0, weight is lower, proportional to 1/(reps[s]+1) to flatten peaks.
        
        # Actually, a simpler greedy: 
        # 1. Ensure coverage of all sums.
        # 2. Then refine.
        
        # Let's try a greedy that adds elements to cover the most uncovered sums first.
        # If tie, pick the one that adds to the sum with the currently lowest rep > 0 (or 0).
        
        # Initialize with 0 and 128 to help cover extremes? 
        # 0+128=128, 128+128=256.
        # Let's start empty and let greedy decide.
        
        for step in range(N_ELEMENTS):
            if time.time() > deadline - 10:
                break
                
            best_elem = -1
            best_score = -1
            
            # Calculate potential gain for each candidate element not in A
            # Gain = sum over all sums s where x participates in a pair with y in A
            # of weight(s)
            
            # For efficiency, we can just iterate candidates and compute reps delta.
            # Since N is small (129), and we do this ~129 times, it's fast enough.
            
            for c in range(N_ELEMENTS):
                if mask[c] == 1:
                    continue
                
                # Simulate adding c
                trial_mask = mask.copy()
                trial_mask[c] = 1
                trial_reps = calc_reps(trial_mask)
                
                # Score: 
                # 1. How many new sums are covered (reps went from 0 to >0)?
                new_covered = np.sum((trial_reps > 0) & (reps == 0))
                
                # 2. Among sums already covered, how well does it balance?
                # We want to avoid increasing peaks. 
                # Let's compute the max rep in trial_reps.
                max_rep = int(np.max(trial_reps))
                
                # We prefer high new_covered, and low max_rep.
                # Score = new_covered * 1000 - max_rep
                score = new_covered * 1000 - max_rep
                
                if score > best_score:
                    best_score = score
                    best_elem = c
            
            if best_elem == -1:
                break
                
            mask[best_elem] = 1
            reps = calc_reps(mask)
            
            # Check if all covered
            if np.all(reps > 0):
                # We can stop if we think we are good, or continue to balance?
                # Let's stop coverage phase and move to refinement.
                # But maybe we can keep adding if it reduces max_rep? 
                # No, adding elements usually increases reps. 
                # We should stop adding once covered, and start swapping/removing.
                break

        return mask

    # --- Phase 2: Local Search Refinement ---
    # Goal: Minimize max(reps) while keeping all reps > 0.
    # Moves:
    # 1. Remove an element if it doesn't cause any sum to drop to 0.
    # 2. Swap an element x in A with y not in A if it reduces max_rep or keeps it same but improves variance.
    
    def local_search_refine(mask, time_limit=10.0):
        current_mask = mask.copy()
        current_reps = calc_reps(current_mask)
        current_obj = get_objective(current_reps)
        
        if np.any(current_reps == 0):
            # Must fix coverage first
            # Use a simple greedy repair
            while np.any(current_reps == 0) and time.time() < time_limit:
                uncovered = [s for s in range(RANGE_SIZE) if current_reps[s] == 0]
                if not uncovered:
                    break
                # Pick a random uncovered sum
                s_idx = random.choice(uncovered)
                s_val = N_LOW + s_idx
                pairs = pairs_for_sum[s_val]
                if len(pairs) == 0:
                    continue
                # Find a pair (i, j) where at least one is not in A, 
                # and adding that one covers the sum.
                # Actually, if sum is 0, NO pair has both in A.
                # We need to add an element that forms a valid pair with an existing element.
                # Or add two elements? 
                # Let's find an element c such that there exists a in A with c+a=s.
                added = False
                candidates = set()
                p = pairs.reshape(-1, 2)
                for i, j in p:
                    if current_mask[i] == 1 and current_mask[j] == 0:
                        candidates.add(j)
                    elif current_mask[j] == 1 and current_mask[i] == 0:
                        candidates.add(i)
                    # If both 0, we might need to add both, but let's try single add first.
                
                if candidates:
                    # Pick one that minimizes new max rep
                    best_c = None
                    best_max_rep = float('inf')
                    for c in candidates:
                        trial_mask = current_mask.copy()
                        trial_mask[c] = 1
                        trial_reps = calc_reps(trial_mask)
                        if np.all(trial_reps > 0):
                            mr = int(np.max(trial_reps))
                            if mr < best_max_rep:
                                best_max_rep = mr
                                best_c = c
                    if best_c is not None:
                        current_mask[best_c] = 1
                        current_reps = calc_reps(current_mask)
                        added = True
                if not added:
                    # Add a random element from the union of pairs for all uncovered sums
                    all_cands = set()
                    for s in uncovered:
                        p = pairs_for_sum[N_LOW + s].reshape(-1, 2)
                        for i, j in p:
                            all_cands.add(i)
                            all_cands.add(j)
                    if all_cands:
                        c = random.choice(list(all_cands))
                        current_mask[c] = 1
                        current_reps = calc_reps(current_mask)
        
        # Now assume all covered. Optimize peak.
        best_mask = current_mask.copy()
        best_obj = get_objective(current_reps)
        best_reps = current_reps.copy()
        
        T = 5.0
        cooling = 0.995
        no_improve = 0
        
        while time.time() < time_limit:
            if T < 0.1:
                break
                
            new_mask = current_mask.copy()
            move_type = random.random()
            
            if move_type < 0.5:
                # Swap
                present = np.where(current_mask == 1)[0]
                absent = np.where(current_mask == 0)[0]
                if len(present) == 0 or len(absent) == 0:
                    continue
                i = random.choice(present)
                j = random.choice(absent)
                new_mask[i] = 0
                new_mask[j] = 1
            else:
                # Remove (if safe) or Add (if helps balance)
                if random.random() < 0.7:
                    # Try remove
                    present = np.where(current_mask == 1)[0]
                    if len(present) > 1:
                        i = random.choice(present)
                        new_mask[i] = 0
                else:
                    # Try add
                    absent = np.where(current_mask == 0)[0]
                    if len(absent) > 0:
                        j = random.choice(absent)
                        new_mask[j] = 1
            
            new_reps = calc_reps(new_mask)
            new_obj = get_objective(new_reps)
            
            delta = new_obj - current_obj
            
            accept = False
            if delta < 0:
                accept = True
                no_improve = 0
            else:
                if delta > 100:
                    prob = 0.0
                else:
                    prob = np.exp(-delta / T)
                if random.random() < prob:
                    accept = True
                    no_improve += 1
            
            if accept:
                current_mask = new_mask
                current_reps = new_reps
                current_obj = new_obj
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_mask = current_mask.copy()
                    best_reps = current_reps.copy()
                    no_improve = 0
            
            T *= cooling
            
            if no_improve > 500:
                # Restart search from best
                current_mask = best_mask.copy()
                current_reps = best_reps.copy()
                current_obj = best_obj
                T = 5.0
                no_improve = 0
                
        return best_mask

    # Main Execution
    best_mask = None
    best_obj = float('inf')
    
    # Run multiple rounds of Greedy + Refine
    num_rounds = 5
    time_per_round = (deadline - start_time - 5) / num_rounds
    
    for r in range(num_rounds):
        if time.time() > deadline - 5:
            break
            
        # Greedy Init
        mask = greedy_balanced_cover()
        
        # Refine
        # Allocate remaining time for this round
        round_deadline = time.time() + time_per_round
        refined_mask = local_search_refine(mask, time_limit=round_deadline - time.time())
        
        reps = calc_reps(refined_mask)
        obj = get_objective(reps)
        
        if obj < best_obj:
            best_obj = obj
            best_mask = refined_mask.copy()
            
        # If we found a valid solution with low peak, we might want to try more aggressively
        # But for now, just keep the best.

    # Final check and fallback
    if best_mask is None:
        # Fallback: Dense set
        mask = np.zeros(N_ELEMENTS, dtype=np.int32)
        for i in range(64, 129):
            mask[i] = 1
        best_mask = mask
    else:
        # Ensure validity
        reps = calc_reps(best_mask)
        if np.any(reps == 0):
            # Repair
            # Simple greedy repair
            while np.any(reps == 0) and time.time() < deadline - 1:
                uncovered = [s for s in range(RANGE_SIZE) if reps[s] == 0]
                if not uncovered:
                    break
                s_idx = random.choice(uncovered)
                s_val = N_LOW + s_idx
                pairs = pairs_for_sum[s_val].reshape(-1, 2)
                added = False
                for i, j in pairs:
                    if best_mask[i] == 1 and best_mask[j] == 0:
                        best_mask[j] = 1
                        added = True
                        break
                    elif best_mask[j] == 1 and best_mask[i] == 0:
                        best_mask[i] = 1
                        added = True
                        break
                if not added:
                    # Add both from a random pair
                    i, j = pairs[0]
                    best_mask[i] = 1
                    best_mask[j] = 1
                reps = calc_reps(best_mask)
                
            # If still invalid, use fallback
            if np.any(reps == 0):
                mask = np.zeros(N_ELEMENTS, dtype=np.int32)
                for i in range(64, 129):
                    mask[i] = 1
                best_mask = mask

    return mask_to_list(best_mask)
