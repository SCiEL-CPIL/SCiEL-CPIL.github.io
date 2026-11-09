import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    N_LOW = 128
    N_HIGH = 256
    MAX_VAL = 128
    
    # Precompute valid pairs for each sum in [128, 256]
    # pairs[s] is a list of (a, b) such that a+b=s, a<=b, 0<=a,b<=128
    pairs_for_sum = {}
    for s in range(N_LOW, N_HIGH + 1):
        min_a = max(0, s - 128)
        max_a = s // 2
        pairs = []
        for a in range(min_a, max_a + 1):
            b = s - a
            if b <= 128:
                pairs.append((a, b))
        pairs_for_sum[s] = pairs

    def evaluate_peak(A_list):
        """Calculate peak multiplicity and coverage."""
        A_set = set(A_list)
        indicator = np.zeros(MAX_VAL + 1, dtype=np.int32)
        for a in A_set:
            indicator[a] = 1
        
        conv = np.convolve(indicator, indicator, mode='full')
        s_vals = np.arange(N_LOW, N_HIGH + 1, dtype=np.int32)
        conv_slice = conv[N_LOW:N_HIGH+1]
        
        even_mask = (s_vals % 2 == 0)
        counts = np.zeros(len(s_vals), dtype=np.int32)
        
        counts[~even_mask] = conv_slice[~even_mask] // 2
        
        if np.any(even_mask):
            even_s = s_vals[even_mask]
            mids = even_s // 2
            valid_mid = (mids >= 0) & (mids <= MAX_VAL)
            raw = conv_slice[even_mask].copy()
            raw[valid_mid] += indicator[mids[valid_mid]]
            counts[even_mask] = raw // 2
        
        if not np.all(counts > 0):
            return float('inf'), False, counts
        
        return int(np.max(counts)), True, counts

    def try_repair(A_set, counts):
        """
        Given a set A and its counts, try to fix uncovered sums by adding elements.
        Returns (success, new_A, new_peak)
        """
        if time.time() > deadline - 2:
            return False, A_set, float('inf')
        
        A_set = set(A_set)
        current_counts = counts.copy()
        
        # Find uncovered sums
        uncovered = [s for i, s in enumerate(range(N_LOW, N_HIGH + 1)) if current_counts[i] == 0]
        
        if not uncovered:
            return True, A_set, int(np.max(current_counts))
        
        # Try to repair by adding elements
        # Heuristic: For each uncovered sum, find a pair that adds the fewest new high-multiplicity issues
        # We will greedily pick elements that cover multiple uncovered sums if possible, 
        # or just cover one with minimal side effects.
        
        # To keep it simple and fast, we'll just iterate through uncovered sums and add a random valid pair element
        # that is not already in A, hoping it helps. 
        # Actually, a better repair: 
        # If s is uncovered, we need at least one (a,b) in A. 
        # We can add a or b. 
        # Let's add elements that are part of many potential pairs for uncovered sums.
        
        # Frequency of elements in potential pairs for uncovered sums
        potential_counts = defaultdict(int)
        for s in uncovered:
            for a, b in pairs_for_sum[s]:
                potential_counts[a] += 1
                potential_counts[b] += 1
        
        # Sort candidates by potential coverage
        candidates = sorted(potential_counts.keys(), key=lambda x: potential_counts[x], reverse=True)
        
        # Add top candidates until covered or time runs out
        added = 0
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if x in A_set:
                continue
            
            # Check if adding x helps
            # We just add it and re-evaluate locally? 
            # For speed, let's just add the top few.
            A_set.add(x)
            added += 1
            
            # Quick check: did we cover all?
            # Re-evaluating full peak is expensive, but necessary for correctness.
            # We'll do a fast check on coverage only first.
            # Actually, let's just re-evaluate peak at the end of repair.
            if added > 10: # Limit repairs to avoid explosion
                break
        
        new_peak, is_covered, new_counts = evaluate_peak(list(A_set))
        if is_covered:
            return True, A_set, new_peak
        else:
            return False, A_set, float('inf')

    best_peak = float('inf')
    best_A = None

    # Initial state: Full set [0, 128] is valid and has peak 65.
    # We want to prune it.
    
    # Strategy: 
    # 1. Start with A = {0, ..., 128}
    # 2. Calculate peak.
    # 3. Identify elements that are "redundant" for maintaining coverage while reducing peak.
    #    An element x is redundant if removing it doesn't uncover any sum AND doesn't increase peak.
    #    Actually, removing an element might uncover a sum. 
    #    We want to remove elements that contribute to high multiplicity sums but are not the *only* representation for any sum.
    
    A = set(range(0, 129))
    current_peak, is_covered, current_counts = evaluate_peak(list(A))
    
    if is_covered:
        best_peak = current_peak
        best_A = list(A)
    
    # Greedy Removal Phase
    # We will try to remove elements one by one.
    # To be systematic, we'll iterate through a few random orders or specific heuristics.
    
    # Heuristic: Remove elements that are "central" (high degree) but have alternatives.
    # Or simply: Try removing each element, if coverage holds and peak doesn't increase, keep it removed.
    # This is O(N^2) evaluations if done naively, but N=129 is small.
    # Evaluation is O(N^2) with numpy convolution? No, convolution is fast.
    # 129 elements. 129 removals * 129 checks? No, just 129 removals sequentially.
    # Each removal requires re-evaluating peak (O(1) with precomputed? No, need re-conv).
    # Convolution of size 129 is very fast (~microseconds).
    # So 129 steps is negligible.
    
    # Let's do a multi-pass greedy removal with randomization to escape local minima.
    
    for restart in range(5):
        if time.time() > deadline - 15:
            break
            
        random.seed(100 + restart)
        A = set(range(0, 129))
        current_peak, is_covered, current_counts = evaluate_peak(list(A))
        
        if not is_covered:
            continue
            
        # Sort elements to try removing
        # Prioritize removing elements that are "expensive" (contribute to many high-multiplicity sums)
        # Or just random order.
        elements_to_try = list(A)
        random.shuffle(elements_to_try)
        
        removed_any = True
        while removed_any and time.time() < deadline - 10:
            removed_any = False
            random.shuffle(elements_to_try)
            for x in elements_to_try:
                if x not in A:
                    continue
                
                # Try removing x
                A.remove(x)
                new_peak, new_covered, new_counts = evaluate_peak(list(A))
                
                if new_covered:
                    # Accept if peak is lower or equal
                    # We prefer lower peak. If equal, we keep it removed to reduce set size (might help future removals?)
                    # Actually, smaller set usually means lower or equal peak.
                    if new_peak <= current_peak:
                        current_peak = new_peak
                        current_counts = new_counts
                        # Update best
                        if current_peak < best_peak:
                            best_peak = current_peak
                            best_A = list(A)
                        removed_any = True
                    else:
                        # Peak increased, revert
                        A.add(x)
                else:
                    # Coverage lost, revert
                    A.add(x)
        
        # Repair Phase: 
        # If we got stuck with a high peak, maybe we can swap?
        # Or if coverage was lost in a previous failed removal, we don't have a valid state.
        # The above loop maintains validity.
        
        # Local Search: Swap
        if time.time() < deadline - 5:
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                if len(A) <= 2:
                    break
                
                to_remove = random.choice(list(A))
                to_add = random.randint(0, 128)
                
                if to_remove == to_add or to_add in A:
                    continue
                
                A.remove(to_remove)
                A.add(to_add)
                
                new_peak, new_covered, new_counts = evaluate_peak(list(A))
                
                if new_covered and new_peak < current_peak:
                    current_peak = new_peak
                    current_counts = new_counts
                    if current_peak < best_peak:
                        best_peak = current_peak
                        best_A = list(A)
                else:
                    # Revert
                    A.remove(to_add)
                    A.add(to_remove)

    if best_A is None:
        best_A = list(range(0, 129))
        
    return best_A
