import time
import random

def _score(S):
    """Calculate the number of distinct d > 0 such that a, a+d, a+2d in S."""
    S_set = set(S)
    valid_ds = set()
    for a in S:
        for d in range(1, 257):  # d max is 511/2 = 255.5, so up to 255
            if a + d in S_set and a + 2*d in S_set:
                valid_ds.add(d)
                break  # only need one witness per (a,d)
    # But we need ALL distinct d, so we can't break early per a.
    # Better: iterate all pairs to find d, check if progression exists
    valid_ds = set()
    elems = sorted(S)
    n = len(elems)
    for i in range(n):
        x = elems[i]
        for j in range(i + 1, n):
            y = elems[j]
            d = y - x
            if x + 2*d in S_set:
                valid_ds.add(d)
    return len(valid_ds)

def _generate_candidate_moves(S, max_candidates=200):
    """Generate candidate (remove, add) pairs that are likely to create progressions."""
    S_set = set(S)
    elems = sorted(S)
    candidates = []
    
    # For each pair (x, y) in S, consider adding x-d or y+d where d = y-x
    # This could complete a progression
    for i in range(len(elems)):
        x = elems[i]
        for j in range(i + 1, len(elems)):
            y = elems[j]
            d = y - x
            # Adding x-d would create (x-d, x, y) if x-d not in S
            if x - d >= 1 and x - d not in S_set:
                # Pick a random element to remove (not x or y)
                if len(candidates) < max_candidates:
                    remove_elem = random.choice([e for e in elems if e != x and e != y]) if len(elems) > 2 else None
                    if remove_elem is not None:
                        candidates.append((remove_elem, x - d))
            # Adding y+d would create (x, y, y+d) if y+d not in S
            if y + d <= 512 and y + d not in S_set:
                if len(candidates) < max_candidates:
                    remove_elem = random.choice([e for e in elems if e != x and e != y]) if len(elems) > 2 else None
                    if remove_elem is not None:
                        candidates.append((remove_elem, y + d))
    
    # Also add some random moves for exploration
    for _ in range(30):
        if len(candidates) >= max_candidates:
            break
        remove_elem = random.choice(elems)
        add_elem = random.randint(1, 512)
        if add_elem not in S_set:
            candidates.append((remove_elem, add_elem))
    
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s kill
    
    best_set = None
    best_score = 0
    
    # Try several random restarts
    while time.time() < deadline:
        # Random initialization
        current_set = set(random.sample(range(1, 513), 32))
        current_score = _score(current_set)
        
        if current_score > best_score:
            best_score = current_score
            best_set = set(current_set)
            
        # Hill climbing phase with time slice
        restart_deadline = min(time.time() + 5.0, deadline)
        
        while time.time() < restart_deadline:
            # Generate candidate moves
            moves = _generate_candidate_moves(current_set)
            
            if not moves:
                # Fallback to random move
                elem_to_replace = random.choice(list(current_set))
                new_val = random.randint(1, 512)
                if new_val in current_set:
                    continue
                new_set = set(current_set)
                new_set.remove(elem_to_replace)
                new_set.add(new_val)
                new_score = _score(new_set)
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                continue
            
            # Shuffle and try moves
            random.shuffle(moves)
            improved = False
            for remove_elem, add_elem in moves:
                if add_elem in current_set:
                    continue
                
                new_set = set(current_set)
                new_set.discard(remove_elem)
                if len(new_set) != 31:
                    continue
                new_set.add(add_elem)
                
                new_score = _score(new_set)
                
                if new_score > current_score:
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_set = set(current_set)
                    break
                elif new_score == current_score and random.random() < 0.1:
                    # Occasionally accept neutral moves to escape plateaus
                    current_set = new_set
                    current_score = new_score
                    improved = True
                    break
            
            if not improved:
                # No improvement found, break out of inner loop
                break
            
            # Check time
            if time.time() >= restart_deadline:
                break
    
    # Return the best set found
    if best_set is None:
        return list(range(1, 33))
    return list(best_set)
