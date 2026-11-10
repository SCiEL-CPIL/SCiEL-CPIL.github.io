"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _is_sidon_and_maximal(selected, N):
    """Check if selected is a valid Sidon set and inclusion-maximal in [1,N]."""
    if len(selected) == 0:
        return False
    selected_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    # Check maximality: no missing element can be added
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        added_sums = set()
        ok = True
        for a in selected:
            s1 = x + a
            if s1 in sums or s1 in added_sums:
                ok = False
                break
            added_sums.add(s1)
        if not ok:
            continue
        s2 = 2 * x
        if s2 not in sums and s2 not in added_sums:
            return False
    return True

def _greedy_max_blocking(N, rng=None, tie_break='max', weight_random=0.0):
    """
    Max-blocking greedy: at each step, among all valid candidates,
    pick the one that blocks the most other valid candidates.
    
    tie_break: 'max' = always pick max-block, 'random' = random among max-block
    weight_random: if >0, add random noise to block count for diversity
    """
    selected = []
    used_sums = set()
    candidates = set(range(1, N + 1))
    
    while candidates:
        # Find all valid candidates
        valid = []
        for x in candidates:
            new_sums = {2 * x} | {x + s for s in selected}
            if not new_sums & used_sums:
                valid.append(x)
        
        if not valid:
            break
        
        # For each valid candidate, count how many other valid candidates it blocks
        best_x = None
        best_blocks = -1
        best_ties = []
        
        for x in valid:
            added_sums = {2 * x} | {x + s for s in selected}
            blocks = 0
            for y in valid:
                if y == x:
                    continue
                # Check if y would be blocked by adding x
                # y is currently valid, so y's sums don't collide with used_sums
                # Adding x introduces: 2x, x+s for s in selected
                # y's new sums would be: 2y, y+s for s in selected, y+x
                # We need to check if any of y's sums collide with added_sums
                # OR if y+x collides with used_sums (since y+x is a new sum not checked yet)
                
                blocked = False
                # Check 2y against added_sums
                if 2 * y in added_sums:
                    blocked = True
                else:
                    # Check y+s for s in selected against added_sums
                    for s in selected:
                        if y + s in added_sums:
                            blocked = True
                            break
                    if not blocked:
                        # Check y+x against used_sums
                        # (y+x is not in added_sums since y!=x and y not in selected)
                        if y + x in used_sums:
                            blocked = True
                
                if blocked:
                    blocks += 1
            
            # Apply random weighting if desired
            score = blocks
            if weight_random > 0:
                score += rng.random() * weight_random
            
            if score > best_blocks:
                best_blocks = score
                best_x = x
                best_ties = [x]
            elif score == best_blocks:
                best_ties.append(x)
        
        # Select the best candidate
        if tie_break == 'random' and len(best_ties) > 1:
            chosen = rng.choice(best_ties)
        else:
            chosen = best_x
        
        # Add chosen to selected
        selected.append(chosen)
        used_sums.update({2 * chosen} | {chosen + s for s in selected[:-1]})
        candidates.discard(chosen)
    
    return selected

def _greedy_with_order(order, N):
    """Greedy construction using a specific element ordering."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def construct():
    deadline = time.time() + 55.0  # 55s safety margin before 60s kill
    best_size = N + 1
    best_set = None
    rng = random.Random(42)
    
    # Strategy 1: Max-blocking greedy (pure) - this is the key improvement
    candidate = _greedy_max_blocking(N, rng, tie_break='max')
    if _is_sidon_and_maximal(candidate, N):
        best_size = len(candidate)
        best_set = candidate
    
    # Strategy 2: Max-blocking with random tie-breaking among max-block candidates
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        candidate = _greedy_max_blocking(N, rng, tie_break='random')
        if len(candidate) < best_size and _is_sidon_and_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:
                break
    
    # Strategy 3: Max-blocking with random weighting for diversity
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        candidate = _greedy_max_blocking(N, rng, tie_break='max', weight_random=1.0)
        if len(candidate) < best_size and _is_sidon_and_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:
                break
    
    # Strategy 4: Random ordering greedy (fallback/diversity)
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_with_order(order, N)
        if len(candidate) < best_size and _is_sidon_and_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:
                break
    
    if best_set is None:
        # Fallback
        best_set = _greedy_max_blocking(N, rng, tie_break='max')
    
    return best_set
# EVOLVE-BLOCK-END
