"""B2[2] set in 1..1000: maximize size via randomized greedy + local fill, with deadline guard."""
import time
import random

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

def _elapsed():
    return time.time() - _START_TIME

def _valid_insertion(counts, x, chosen_set):
    """Return None if invalid, else the list of sums to update and their new counts (delta dict)."""
    for y in chosen_set:
        s = x + y
        if counts.get(s, 0) >= 2:
            return None
    if counts.get(2 * x, 0) >= 2:
        return None
    # valid; produce delta
    delta = {}
    for y in chosen_set:
        s = x + y
        delta[s] = delta.get(s, 0) + 1
    delta[2 * x] = delta.get(2 * x, 0) + 1
    return delta

def _remove_element(counts, x, chosen_set):
    """Remove x from set and update counts. Returns the delta to reverse if needed."""
    delta = {}
    for y in chosen_set:
        s = x + y
        delta[s] = delta.get(s, 0) - 1
    delta[2 * x] = delta.get(2 * x, 0) - 1
    
    for s, d in delta.items():
        counts[s] += d
        if counts[s] == 0:
            del counts[s]
    
    chosen_set.remove(x)
    return delta

def _try_add_all(counts, chosen_set, candidates):
    """Try to add all possible candidates from the list. Returns number of added."""
    added = 0
    for x in candidates:
        if x in chosen_set:
            continue
        delta = _valid_insertion(counts, x, chosen_set)
        if delta is not None:
            for s, d in delta.items():
                counts[s] = counts.get(s, 0) + d
            chosen_set.add(x)
            added += 1
    return added

def _local_search(counts, chosen_set, universe_set, time_limit):
    """Perform local search (add and kick) to maximize size."""
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Try to add any element
        # Iterate through universe to find addable elements
        # To speed up, we can just iterate through all non-chosen elements
        non_chosen = [x for x in universe_set if x not in chosen_set]
        random.shuffle(non_chosen) # Randomize to avoid bias
        
        added_any = False
        for x in non_chosen:
            if time.time() - start_time > time_limit:
                break
            delta = _valid_insertion(counts, x, chosen_set)
            if delta is not None:
                for s, d in delta.items():
                    counts[s] = counts.get(s, 0) + d
                chosen_set.add(x)
                added_any = True
                improved = True
        
        if added_any:
            continue # Restart adding process as set changed
            
        # If no additions, try a kick (remove one, add others)
        # Only do this if we haven't improved in a while or randomly to escape local optima
        # Simple strategy: try removing each element and see if we can add more than 1 back
        
        # Optimization: Only try kicks if we are stuck. 
        # To save time, we might limit kicks or do them less frequently.
        # Given the small size (1000), we can afford to check a few kicks.
        
        # Select a random element to remove
        if chosen_set:
            x_remove = random.choice(list(chosen_set))
            
            # Save state
            old_counts = dict(counts)
            old_chosen = set(chosen_set)
            
            # Remove x_remove
            _remove_element(counts, x_remove, chosen_set)
            
            # Try to add as many as possible
            non_chosen = [x for x in universe_set if x not in chosen_set]
            # Sort by some heuristic? Or random?
            # Random is safer for diversity.
            random.shuffle(non_chosen)
            
            added = _try_add_all(counts, chosen_set, non_chosen)
            
            # Compare sizes
            if len(chosen_set) > len(old_chosen):
                improved = True
                # Keep the new state
            else:
                # Revert
                counts.clear()
                counts.update(old_counts)
                chosen_set.clear()
                chosen_set.update(old_chosen)

    return len(chosen_set)

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = 145.0
    random.seed(0)
    
    universe = list(range(1, 1001))
    universe_set = set(universe)
    
    best_size = 0
    best_set = set()
    
    # We will run a few iterations of:
    # 1. Random Greedy to get a start
    # 2. Local Search to improve
    
    # Budget allocation:
    # Spend ~10% time on initial diverse builds, 90% on local search for the best.
    # Or interleave.
    
    # Let's do 5 diverse initial builds.
    # Then take the best and do extensive local search.
    # If time remains, do local search on the 2nd best, etc.
    
    initial_candidates = []
    
    time_for_initial = 10.0 # 10 seconds for initial builds
    
    while time.time() - _START_TIME < time_for_initial:
        # Random Greedy
        order = universe[:]
        random.shuffle(order)
        # Bias towards larger numbers? 
        # B2 sets often have better density if spread out.
        # Let's try a mix.
        
        counts = {}
        chosen_set = set()
        
        for x in order:
            if time.time() - _START_TIME > time_for_initial:
                break
            delta = _valid_insertion(counts, x, chosen_set)
            if delta is not None:
                for s, d in delta.items():
                    counts[s] = counts.get(s, 0) + d
                chosen_set.add(x)
                
        initial_candidates.append((len(chosen_set), counts, chosen_set))
        
    if not initial_candidates:
        # Fallback if time ran out immediately
        counts = {}
        chosen_set = set()
        for x in universe:
             delta = _valid_insertion(counts, x, chosen_set)
             if delta is not None:
                 for s, d in delta.items():
                     counts[s] = counts.get(s, 0) + d
                 chosen_set.add(x)
        return sorted(chosen_set)

    # Sort candidates by size
    initial_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Now, take the top candidate and improve it
    # We can also improve the others if time permits
    
    for size, counts, chosen_set in initial_candidates:
        if time.time() - _START_TIME > _DEADLINE - 5.0:
            break
            
        # Copy state for local search
        c_counts = dict(counts)
        c_chosen = set(chosen_set)
        
        # Run local search with remaining time
        remaining = _DEADLINE - 5.0 - (time.time() - _START_TIME)
        if remaining > 1.0:
            new_size = _local_search(c_counts, c_chosen, universe_set, remaining)
            
            if new_size > best_size:
                best_size = new_size
                best_set = set(c_chosen)
                
            # If we improved significantly, we might want to stop and restart local search on the new best?
            # Or just continue to next candidate.
            # Since local search is stochastic, running it on the best is usually sufficient.
            # However, if the first local search got stuck, the second one (on a slightly different start) might do better.
            # So we continue the loop.

    if best_set:
        return sorted(best_set)
    else:
        # Fallback to the best initial
        return sorted(initial_candidates[0][2])
