"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin

    def generate_vectors(weight):
        """Generate all vectors of given weight with +/-1 entries."""
        candidates = []
        # Use combinations of positions for non-zero entries
        for positions in itertools.combinations(range(D), weight):
            for signs in itertools.product([1, -1], repeat=weight):
                v = [0] * D
                for pos, s in zip(positions, signs):
                    v[pos] = s
                candidates.append(v)
        return np.array(candidates, dtype=np.int8)

    def solve_for_weight(weight, time_limit):
        """Find a large compatible set for vectors of a specific weight."""
        if time.time() > time_limit:
            return np.array([], dtype=np.int64).reshape(0, D)
        
        candidates = generate_vectors(weight)
        n = len(candidates)
        if n == 0:
            return np.array([], dtype=np.int64).reshape(0, D)
        
        # Compute dot products
        # Norm squared is 'weight'
        # Condition: dot <= weight // 2
        limit = weight // 2
        
        dot_matrix = candidates @ candidates.T
        # conflict[i,j] is True if dot > limit
        conflict = (dot_matrix > limit)
        
        best_set = []
        best_len = 0
        
        # Greedy with random restarts
        n_restarts = 500
        for r in range(n_restarts):
            if time.time() > time_limit:
                break
            
            # Random order
            order = np.random.permutation(n)
            
            selected = []
            available = np.ones(n, dtype=bool)
            
            for idx in order:
                if not available[idx]:
                    continue
                selected.append(idx)
                # Remove incompatible
                available[conflict[idx]] = False
                available[idx] = False
                
                # Early termination if no more available
                if not np.any(available):
                    break
            
            if len(selected) > best_len:
                best_len = len(selected)
                best_set = selected
                
                # If we found a very large set, we can stop early or continue
                # For weight 5, max possible is bounded by kissing number ~500?
                # If we hit 400+, we are doing well.
        
        # Local Search on best_set
        if best_set and time.time() < time_limit - 5.0:
            best_set = local_search(best_set, conflict, candidates, time_limit - 2.0)
            
        return candidates[best_set].astype(np.int64)

    def local_search(selected_indices, conflict, candidates, time_limit):
        """Improve the set by adding compatible vectors and swapping."""
        selected_set = set(selected_indices)
        n = len(candidates)
        improved = list(selected_indices)
        improved_set = set(improved)
        
        # 1. Try to add any compatible vector
        # Compute max dot with selected for all candidates
        if len(improved) > 0:
            sel_arr = np.array(improved, dtype=np.int32)
            max_dot = np.max(candidates[sel_arr] @ candidates.T, axis=0)
        else:
            max_dot = np.zeros(n, dtype=np.int32)
            
        # Addable: max_dot <= limit
        # We need the limit. We can infer it from candidates norm.
        # All candidates have same norm squared = weight.
        # limit = weight // 2
        # We passed candidates, so we can compute weight = sum(candidates[0]**2)
        weight = np.sum(candidates[0]**2)
        limit = weight // 2
        
        addable = np.where(max_dot <= limit)[0]
        addable = [i for i in addable if i not in improved_set]
        
        # Greedy add
        for idx in addable:
            if time.time() > time_limit:
                break
            if idx in improved_set:
                continue
            # Re-check compatibility with all current improved (should be safe if max_dot was updated, 
            # but adding one changes max_dot for others? No, max_dot is max over selected.
            # If max_dot[idx] <= limit, it is compatible with ALL selected.
            # So we can just add it.
            improved.append(idx)
            improved_set.add(idx)
            # Update max_dot for future? 
            # We are iterating a static list. We should probably just do a full pass or 
            # update max_dot incrementally.
            # For simplicity and safety, we'll do a few passes of "add if compatible".
            # But the max_dot array is now stale.
            # Let's just do a simple check for the next few additions.
            # Actually, if we add idx, it might conflict with some other addable j.
            # So we need to be careful.
            # Let's just do a simple loop:
            # Check if idx is compatible with all in improved.
            # This is O(K) per check.
            # Since we are adding, K is growing.
            # It's better to recompute max_dot periodically or just check.
            # Given K is small (<1000), checking is fast.
            pass # The above logic is flawed for multiple additions. 
                 # Let's stick to the single-pass add based on initial max_dot, 
                 # which is safe because if it was compatible with ALL, it's compatible.
                 # Wait, if I add A, and B was compatible with all original, 
                 # is B compatible with A? Yes, because A was in the original set? 
                 # No, A is NEW.
                 # So B might conflict with A.
                 # So we cannot just add all addable.
                 # We must check pairwise.
        
        # Corrected Local Search:
        # We have a set S. We want to add more.
        # Let's just try to add one by one, checking against current S.
        # We'll do this in a loop.
        
        current_improved = list(selected_indices)
        current_set = set(current_improved)
        
        # Precompute dot matrix for the candidates? 
        # We have 'conflict' which is a boolean matrix.
        # conflict[i,j] is True if NOT compatible.
        
        # 1. Add phase
        # Try to add any candidate not in current_set that is compatible with all in current_set.
        # To speed up, we can maintain a list of "candidates_to_try".
        
        # Let's do a few iterations of "try to add best candidate"
        # Best candidate: one with max "room" (min max_dot).
        
        if len(current_improved) == 0:
            return current_improved
            
        # Compute max_dot for all candidates against current_improved
        sel_arr = np.array(current_improved, dtype=np.int32)
        max_dot = np.max(candidates[sel_arr] @ candidates.T, axis=0)
        
        # Iterate
        for _ in range(200):
            if time.time() > time_limit:
                break
            
            # Find compatible candidates (max_dot <= limit)
            compat = np.where(max_dot <= limit)[0]
            compat = [i for i in compat if i not in current_set]
            
            if not compat:
                break
            
            # Pick the one with smallest max_dot (most compatible)
            best_idx = min(compat, key=lambda i: max_dot[i])
            
            # Add it
            current_improved.append(best_idx)
            current_set.add(best_idx)
            
            # Update max_dot
            # max_dot = max(max_dot, dot(best_idx, all))
            max_dot = np.maximum(max_dot, candidates[best_idx] @ candidates.T)
            
        # 2. Swap phase
        # Try to remove one and add two.
        # This is expensive.
        # Let's skip for now if time is tight, or do simple swaps.
        
        return current_improved

    # Main execution
    best_result = None
    best_score = 0
    
    # Try Weight 4 (Baseline)
    res4 = solve_for_weight(4, deadline - 30.0)
    if len(res4) > best_score:
        best_score = len(res4)
        best_result = res4
        
    # Try Weight 5
    if time.time() < deadline - 30.0:
        res5 = solve_for_weight(5, deadline - 30.0)
        if len(res5) > best_score:
            best_score = len(res5)
            best_result = res5
            
    # Try Weight 6
    if time.time() < deadline - 10.0:
        res6 = solve_for_weight(6, deadline - 5.0)
        if len(res6) > best_score:
            best_score = len(res6)
            best_result = res6

    if best_result is None or len(best_result) == 0:
        # Fallback
        v = [0] * D
        v[0] = 1
        v[1] = 1
        return np.array([v], dtype=np.int64)
        
    return best_result

# EVOLVE-BLOCK-END
