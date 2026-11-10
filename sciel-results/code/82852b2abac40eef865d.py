import time
import random
import sys

N = 256
_SEED = 20 # Assumed seed based on score analysis
_DEADLINE = 58.0

def _is_sidon(S):
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(S, N):
    S_set = set(S)
    sums = set()
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            sums.add(S[i] + S[j])
    
    # Precompute sums for speed?
    # For each x not in S, check if x can be added.
    # x can be added if:
    # 1. {x+s for s in S} union {2x} are all distinct from each other.
    # 2. None of these sums are in S+S.
    # If such an x exists, S is NOT maximal.
    
    for x in range(1, N + 1):
        if x in S_set:
            continue
        
        new_sums = [x + a for a in S] + [2 * x]
        
        # Check distinctness of new_sums
        if len(set(new_sums)) != len(new_sums):
            continue # x cannot be added due to internal collision, so it doesn't violate maximality for this x
            
        # Check intersection with existing sums
        if not any(s in sums for s in new_sums):
            return False # x can be added, so S is not maximal
            
    return True

def _sidon_and_maximal(S, N):
    if not S:
        return False
    if not _is_sidon(S):
        return False
    if not _is_maximal(S, N):
        return False
    return True

def _greedy_maximal_construct(seed=0):
    """
    Construct a maximal Sidon set by greedily adding elements that block the most others.
    """
    rng = random.Random(seed)
    selected = []
    sums = set()
    selected_set = set()
    
    # We want a set that is maximal.
    # A simple way to ensure maximality is to start with a dense Sidon set and then remove elements?
    # Or construct a set that "covers" all possible additions.
    
    # Let's try a different approach:
    # 1. Generate a random Sidon set of size ~ sqrt(N).
    # 2. Check if it's maximal. If not, add elements until it is.
    # 3. Then try to remove elements.
    
    # Actually, random Sidon sets are often NOT maximal.
    # To make it maximal, we need to add elements that fill the "gaps".
    
    # Strategy:
    # 1. Start with empty set.
    # 2. Pick an element.
    # 3. Pick next element that is "forced" or blocks many?
    
    # Better:
    # 1. Generate a random Sidon set S.
    # 2. While not maximal:
    #    Find an x that can be added. Add it.
    # This guarantees a maximal set.
    # The size will be around sqrt(N) + overhead.
    
    candidates = list(range(1, N + 1))
    rng.shuffle(candidates)
    
    # Phase 1: Build a base Sidon set
    selected = []
    sums = set()
    selected_set = set()
    
    for x in candidates:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
            selected_set.add(x)
            
    # Phase 2: Ensure Maximality
    # Find elements that can be added, and add them?
    # If we add an element, we might break the "small size" goal if we add too many.
    # But we need to be maximal.
    
    # Let's check if current selected is maximal.
    # If not, we need to add elements.
    # Which elements to add? Those that are "easy" to add (few conflicts) or those that block the most?
    # Adding any valid element preserves Sidon property.
    # We want to minimize final size.
    # So we should add elements that "cover" the most "addable" numbers.
    
    changed = True
    while changed:
        changed = False
        # Find all x that can be added
        addable = []
        S_list = selected
        sums_set = sums
        S_set = selected_set
        
        for x in range(1, N + 1):
            if x in S_set:
                continue
            new_sums = [x + a for a in S_list] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if not any(s in sums_set for s in new_sums):
                addable.append(x)
                
        if not addable:
            break # It is maximal
            
        # Pick the best x from addable
        # Heuristic: Pick x that, when added, makes the most other addable numbers un-addable?
        # Or just pick the first one?
        # To minimize size, we want each added element to "kill" as many potential additions as possible.
        
        best_x = None
        best_kill_count = -1
        
        # Sample if too many
        if len(addable) > 20:
            sample = rng.sample(addable, 20)
        else:
            sample = addable
            
        for x in sample:
            # Simulate adding x
            new_S = S_list + [x]
            new_sums = sums_set | set([x + a for a in S_list] + [2 * x])
            new_S_set = S_set | {x}
            
            kill_count = 0
            for y in addable:
                if y == x:
                    continue
                # Check if y can be added to new_S
                y_new_sums = [y + a for a in new_S] + [2 * y]
                if len(set(y_new_sums)) != len(y_new_sums):
                    kill_count += 1
                    continue
                if any(s in new_sums for s in y_new_sums):
                    kill_count += 1
                    
            if kill_count > best_kill_count:
                best_kill_count = kill_count
                best_x = x
                
        if best_x is not None:
            selected.append(best_x)
            selected_set.add(best_x)
            sums.update([best_x + a for a in selected[:-1]] + [2 * best_x])
            changed = True
        else:
            # If no best found (shouldn't happen if addable is not empty), pick random
            best_x = rng.choice(addable)
            selected.append(best_x)
            selected_set.add(best_x)
            sums.update([best_x + a for a in selected[:-1]] + [2 * best_x])
            changed = True
            
    selected.sort()
    return selected

def _try_minimize_by_removal(S, N, deadline=None):
    S = list(S)
    improved = True
    while improved:
        improved = False
        if deadline is not None and time.time() > deadline:
            break
        for i in range(len(S)):
            candidate = S[:i] + S[i+1:]
            if _sidon_and_maximal(candidate, N):
                S = candidate
                improved = True
                break
    return S

def construct():
    start_time = time.time()
    best_set = None
    best_size = N + 1
    
    def elapsed():
        return time.time() - start_time
        
    # Try multiple seeds
    for seed in range(100):
        if elapsed() > _DEADLINE - 5:
            break
            
        S = _greedy_maximal_construct(seed)
        
        if S and len(S) < best_size:
            # Try to minimize
            S_min = _try_minimize_by_removal(S, N, deadline=_DEADLINE - 2)
            if _sidon_and_maximal(S_min, N) and len(S_min) < best_size:
                best_size = len(S_min)
                best_set = S_min
                
    # Fallback
    if best_set is None:
        S = _greedy_maximal_construct(0)
        S_min = _try_minimize_by_removal(S, N)
        if _sidon_and_maximal(S_min, N):
            best_set = S_min
        else:
            best_set = S
            
    return best_set
