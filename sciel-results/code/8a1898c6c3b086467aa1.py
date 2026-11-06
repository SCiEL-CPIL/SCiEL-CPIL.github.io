import random
import time
from math import lcm
from functools import reduce

# LCM of 1..22
L = reduce(lcm, range(1, 23))
# Values v_i = L // i
V = [L // i for i in range(1, 23)]

def is_dissociated(indices):
    """
    Check if the subset of indices is dissociated.
    indices: list of indices (0-based) into V.
    Returns True if all subset sums of V[i] for i in indices are distinct.
    """
    if not indices:
        return True
    
    k = len(indices)
    # For small k, use bitmask generation
    if k <= 14:
        sums = set()
        sums.add(0)
        for idx in indices:
            val = V[idx]
            # Check collision with existing sums
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True
    else:
        # For larger k, we can't do full check efficiently, so we use a probabilistic check
        # But since we're aiming for size <= 12, this branch won't be hit often
        # Let's just do the same but with a note
        sums = {0}
        for idx in indices:
            val = V[idx]
            new_sums = set()
            for s in sums:
                ns = s + val
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = [0, 1, 2, 4, 6, 10, 12, 16, 18]  # Corresponds to 1,2,3,5,7,11,13,17,19
    best_size = len(best_set)
    
    all_indices = list(range(22))
    
    def indices_to_numbers(indices):
        return [i + 1 for i in indices]
    
    # Strategy: 
    # 1. Greedy construction with random orderings
    # 2. Local improvement by trying to add missing elements
    # 3. Local improvement by trying swaps (replace one element with another)
    
    # Phase 1: Randomized Greedy
    max_restarts = 50000  # High bound, but we'll break on time
    for _ in range(max_restarts):
        if time.time() > deadline:
            break
        
        random.shuffle(all_indices)
        current_set = []
        
        for idx in all_indices:
            # Only try to add if current size < best_size or current_size + 1 could exceed best
            # Actually, always try to build a full set, then compare
            if is_dissociated(current_set + [idx]):
                current_set.append(idx)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
    
    # Phase 2: Local improvement - add missing elements
    if time.time() < deadline - 3.0:
        best_set_set = set(best_set)
        missing = [i for i in all_indices if i not in best_set_set]
        for m in missing:
            if time.time() > deadline - 3.0:
                break
            if is_dissociated(best_set + [m]):
                best_set.append(m)
                best_size += 1
                best_set_set.add(m)
    
    # Phase 3: Local improvement - swap elements to try to grow
    # Try replacing one element with a missing one, then try to add more
    if time.time() < deadline - 3.0:
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            best_set_set = set(best_set)
            missing = [i for i in all_indices if i not in best_set_set]
            
            for j, old_idx in enumerate(best_set):
                if time.time() > deadline - 3.0:
                    break
                for new_idx in missing:
                    # Try swapping old_idx with new_idx
                    trial = best_set.copy()
                    trial[j] = new_idx
                    if is_dissociated(trial):
                        # Now try to add any missing element to this trial
                        trial_set = set(trial)
                        for m in missing:
                            if m == new_idx:
                                continue
                            if is_dissociated(trial + [m]):
                                trial.append(m)
                                break  # Found one, accept this swap
                        # Also try the swap without adding
                        if len(trial) > best_size:
                            best_set = trial
                            best_size = len(trial)
                            improved = True
                            break
                        elif len(trial) == best_size and is_dissociated(trial):
                            # Keep track but don't necessarily replace unless it allows future growth
                            # For simplicity, only replace if strictly better
                            pass
                if improved:
                    break
    
    return indices_to_numbers(best_set)
