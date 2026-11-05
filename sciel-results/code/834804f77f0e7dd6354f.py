import random
import time

def construct():
    N = 180
    vertices = list(range(1, N + 1))
    
    # Precompute forbidden triples
    pairs = {i: [] for i in vertices}
    
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            num = a * b
            den = b - a
            if num % den == 0:
                c = num // den
                if c <= N and c != a and c != b:
                    pairs[a].append((b, c))
                    pairs[b].append((a, c))
                    pairs[c].append((a, b))
    
    def is_valid_addition(v, current_set):
        for (u, w) in pairs[v]:
            if u in current_set and w in current_set:
                return False
        return True

    def greedy_construct(order):
        current_set = set()
        for v in order:
            if is_valid_addition(v, current_set):
                current_set.add(v)
        return current_set

    def local_search(initial_set):
        current_set = set(initial_set)
        improved = True
        while improved:
            improved = False
            # 1. Try to add any vertex
            for v in vertices:
                if v not in current_set and is_valid_addition(v, current_set):
                    current_set.add(v)
                    improved = True
                    # Restart the loop to check for more additions
                    break 
            if improved:
                continue
                
            # 2. Try swaps: remove one, add more
            # We want to find a v in current_set such that removing it allows adding > 1 new vertices
            # Or at least maintaining size while allowing future improvements?
            # Let's try: for each v in current_set, remove it, then greedily fill.
            # This is expensive if done for all v every time.
            # Limit to a few iterations or random selection.
            
            candidates_to_remove = list(current_set)
            random.shuffle(candidates_to_remove)
            
            best_new_set = None
            best_size = len(current_set)
            
            for v in candidates_to_remove:
                # Remove v
                trial_set = current_set - {v}
                
                # Greedily add from the rest
                # We can use a random order or sorted order.
                # To be fast, just try adding any valid vertex from the complement.
                added = 0
                for u in vertices:
                    if u not in trial_set and u != v:
                        if is_valid_addition(u, trial_set):
                            trial_set.add(u)
                            added += 1
                    elif u == v:
                        # Maybe we can add v back? No, we removed it to free up space.
                        # But if it doesn't conflict with the newly added ones, we might add it back.
                        pass
                
                # Check if we can add v back
                if is_valid_addition(v, trial_set):
                    trial_set.add(v)
                    
                if len(trial_set) > best_size:
                    best_size = len(trial_set)
                    best_new_set = trial_set
            
            if best_new_set is not None:
                current_set = best_new_set
                improved = True

        return current_set

    best_set = set()
    best_size = 0
    
    start_time = time.time()
    
    while time.time() - start_time < 50:
        random.shuffle(vertices)
        current_set = greedy_construct(vertices)
        
        # Apply local search
        if time.time() - start_time < 45: # Leave time for finishing
            current_set = local_search(current_set)
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    return sorted(best_set)
