import random
import time
import math

N = 16
FULL_MASK = (1 << N) - 1
ALL_SUBSETS = 1 << N

def _update_count(count, mask, delta):
    """Update count array for all supersets of mask."""
    rest = FULL_MASK ^ mask
    sub = rest
    while True:
        count[mask | sub] += delta
        if sub == 0:
            break
        sub = (sub - 1) & rest

def _calc_alpha(count):
    """Calculate independence number from count array."""
    best = 0
    # We can optimize by checking from high bit counts, but linear scan is okay for 65536
    for S in range(ALL_SUBSETS):
        if count[S] == 0:
            c = S.bit_count()
            if c > best:
                best = c
    return best

def _is_linear(triples):
    m = len(triples)
    for i in range(m):
        a, b, c = triples[i]
        for j in range(i + 1, m):
            x, y, z = triples[j]
            if len({a, b, c} & {x, y, z}) > 1:
                return False
    return True

def _init_system(rng):
    """Create a random linear system."""
    triples = []
    points = list(range(N))
    attempts = 0
    max_attempts = 2000
    while attempts < max_attempts:
        attempts += 1
        t = tuple(sorted(rng.sample(points, 3)))
        ok = True
        for e in triples:
            if len(set(t) & set(e)) > 1:
                ok = False
                break
        if ok:
            triples.append(t)
    return triples

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    rng = random.Random(42)
    
    # Initial solution
    cur_triples = _init_system(rng)
    
    # Initialize count array
    count = [0] * ALL_SUBSETS
    for t in cur_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        _update_count(count, mask, 1)
        
    cur_alpha = _calc_alpha(count)
    best_alpha = cur_alpha
    best_triples = list(cur_triples)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    # Precompute all possible triples as masks and point tuples
    all_triples = []
    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                all_triples.append(((1 << i) | (1 << j) | (1 << k), (i, j, k)))
    
    # Map mask to index in all_triples for quick lookup? 
    # Or just keep track of which triples are in the current set.
    # Current set is small, so linear search for "is this triple in current set" is fine.
    # But for "can we add this triple", we need to check linearity.
    
    # To check linearity quickly when adding a triple:
    # We can precompute for each point, the list of triples containing it?
    # Or just iterate over current triples (size < 50) and check intersection.
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
            
        # Generate a neighbor
        r = rng.random()
        new_triples = list(cur_triples)
        delta_count = 0 # Not used directly, we update count array
        
        if r < 0.3:
            # Add a random triple
            # Pick a random triple from all_triples that is not in cur_triples
            # and is linear with cur_triples
            # To avoid O(M) check for every candidate, we can just pick random and check
            added = False
            for _ in range(10): # Try 10 times
                mask, t_pts = rng.choice(all_triples)
                # Check if already in cur_triples
                if any(cur_triples[i] == t_pts for i in range(len(cur_triples))):
                    continue
                # Check linearity
                ok = True
                for e in cur_triples:
                    if len(set(t_pts) & set(e)) > 1:
                        ok = False
                        break
                if ok:
                    new_triples.append(t_pts)
                    # Update count
                    _update_count(count, mask, 1)
                    added = True
                    break
            
            if not added:
                # Try removing instead if add fails? Or just skip
                # Let's try remove
                if cur_triples:
                    idx = rng.randrange(len(cur_triples))
                    mask, t_pts = all_triples[idx] # Wait, index in all_triples is not same as index in cur_triples
                    # We need to get the mask of the triple to remove
                    t_to_remove = cur_triples[idx]
                    mask = (1 << t_to_remove[0]) | (1 << t_to_remove[1]) | (1 << t_to_remove[2])
                    new_triples.pop(idx)
                    _update_count(count, mask, -1)
                else:
                    continue
                    
        elif r < 0.6:
            # Remove a random triple
            if cur_triples:
                idx = rng.randrange(len(cur_triples))
                t_to_remove = cur_triples[idx]
                mask = (1 << t_to_remove[0]) | (1 << t_to_remove[1]) | (1 << t_to_remove[2])
                new_triples.pop(idx)
                _update_count(count, mask, -1)
            else:
                continue
                
        else:
            # Swap: Remove one, add one
            if not cur_triples:
                continue
            idx = rng.randrange(len(cur_triples))
            t_to_remove = cur_triples[idx]
            mask_remove = (1 << t_to_remove[0]) | (1 << t_to_remove[1]) | (1 << t_to_remove[2])
            
            # Find a triple to add
            added = False
            for _ in range(10):
                mask_add, t_pts = rng.choice(all_triples)
                # Check if t_pts is t_to_remove
                if t_pts == t_to_remove:
                    continue
                # Check if t_pts is in other triples
                # And check linearity with other triples
                ok = True
                for j, e in enumerate(cur_triples):
                    if j == idx:
                        continue
                    if len(set(t_pts) & set(e)) > 1:
                        ok = False
                        break
                if ok:
                    # Perform swap
                    new_triples[idx] = t_pts
                    _update_count(count, mask_remove, -1)
                    _update_count(count, mask_add, 1)
                    added = True
                    break
            
            if not added:
                # Revert? No, we didn't modify count yet if we failed to find add?
                # Wait, in the loop above, we only modify count if we find a valid add.
                # So if not added, count is unchanged, cur_triples unchanged.
                continue

        # Calculate new alpha
        new_alpha = _calc_alpha(count)
        
        # Accept/Reject
        if new_alpha < cur_alpha:
            # Improvement
            cur_alpha = new_alpha
            cur_triples = new_triples
            if cur_alpha < best_alpha:
                best_alpha = cur_alpha
                best_triples = list(cur_triples)
        else:
            # Accept with probability exp(-(new_alpha - cur_alpha) / T)
            # Since alpha is integer, delta is >= 1
            delta = new_alpha - cur_alpha
            if delta == 0:
                cur_alpha = new_alpha
                cur_triples = new_triples
            else:
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    cur_alpha = new_alpha
                    cur_triples = new_triples
        
        T *= cooling
        
    return [list(t) for t in best_triples]
