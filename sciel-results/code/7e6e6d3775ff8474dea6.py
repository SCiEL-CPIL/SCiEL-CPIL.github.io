import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin
    
    N = 256
    
    # 1. Identify bad vertices (2a is a perfect square)
    bad_vertices = set()
    for m in range(1, 20):
        a = 2 * m * m
        if a <= N:
            bad_vertices.add(a)
            
    candidates = [i for i in range(1, N+1) if i not in bad_vertices]
    cand_set = set(candidates)
    
    # 2. Build conflict masks
    # squares up to 256+256=512
    squares = [k*k for k in range(1, 23)] # 22^2 = 484, 23^2 = 529
    squares_set = set(squares)
    
    conflict = [0] * (N+1)
    for i in candidates:
        mask = 0
        for s in squares_set:
            j = s - i
            if 1 <= j <= N:
                if j in cand_set:
                    mask |= (1 << j)
        conflict[i] = mask

    # 3. Helper: Greedy Fill
    def greedy_fill(current_mask, current_set):
        # Shuffle candidates for diverse maximal sets
        cand_list = candidates[:]
        random.shuffle(cand_list)
        for v in cand_list:
            if current_mask & (1 << v):
                continue
            if conflict[v] & current_mask:
                continue
            current_mask |= (1 << v)
            current_set.add(v)
        return current_mask, current_set

    # 4. Initial Solution
    best_mask = 0
    best_set = set()
    best_size = 0
    
    init_mask, init_set = greedy_fill(0, set())
    best_mask = init_mask
    best_set = init_set.copy()
    best_size = len(init_set)
    
    current_mask = init_mask
    current_set = init_set
    current_size = len(init_set)
    
    # 5. Iterated Local Search
    while time.time() < deadline:
        # Try to add any possible vertex (Single Flip)
        improved = False
        # We can check all candidates not in set
        # Optimization: Keep a list of "blocked" vertices? 
        # For N=256, linear scan is fine.
        for v in candidates:
            if current_mask & (1 << v):
                continue
            if not (conflict[v] & current_mask):
                # Add v
                current_mask |= (1 << v)
                current_set.add(v)
                current_size += 1
                improved = True
                # Don't break, try to add more? 
                # Adding one might block others, but might not block all.
                # Actually, if we add v, we should restart the check or continue?
                # Continuing is fine, we just check remaining candidates.
                # But we might have skipped some earlier candidates that are now valid?
                # No, adding v only adds conflicts, it never removes them.
                # So if a candidate was invalid before, it remains invalid.
                # So we can just continue the loop.
        
        if improved:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
                best_set = current_set.copy()
            continue
        
        # If no single addition possible, we are at a local maximum.
        # Perform a kick: remove k random vertices
        k = random.randint(1, 4)
        if current_size < k:
            k = current_size
        
        if k > 0:
            # Remove k random vertices
            to_remove = random.sample(list(current_set), k)
            for v in to_remove:
                current_mask &= ~(1 << v)
                current_set.remove(v)
                current_size -= 1
            
            # Refill
            current_mask, current_set = greedy_fill(current_mask, current_set)
            current_size = len(current_set)
            
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
                best_set = current_set.copy()
            
            # Acceptance criterion: 
            # If new size > old size (before kick), accept.
            # If new size == old size, accept with prob?
            # If new size < old size, accept with prob exp(-delta/T)?
            # For simplicity, let's just keep the best found. 
            # But to escape local optima, we might need to accept worse temporarily.
            # However, "kick + refill" usually produces a set that is comparable or better 
            # if the kick was well-chosen. 
            # Let's stick to keeping the best so far to ensure monotonic improvement of the best record.
            # The current state might be worse, but the next kick will be from this new state.
            # This is effectively a random walk on the space of maximal independent sets.

    return list(best_set)
