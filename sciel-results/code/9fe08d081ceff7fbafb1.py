"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time

def _is_sidon_and_maximal_bitmask(selected, N):
    """Check if selected is a valid Sidon set and inclusion-maximal in [1,N] using bitmasks."""
    if len(selected) == 0:
        return False
    
    selected_set = set(selected)
    used_sums_mask = 0
    selected_mask = 0
    
    # Build masks
    # Note: Python ints are arbitrary precision, so shifts are fine.
    # Max sum is 2*256 = 512. Bit 512 is fine.
    
    for x in selected:
        # Check internal consistency first? 
        # If we build incrementally, we assume input is valid or we check at end.
        # Let's just build and check collisions.
        pass
        
    # Re-build properly to ensure validity
    used_sums_mask = 0
    valid = True
    for i, a in enumerate(selected):
        # Sums with previous elements
        for j in range(i):
            b = selected[j]
            s = a + b
            bit = 1 << s
            if used_sums_mask & bit:
                valid = False
                break
            used_sums_mask |= bit
        if not valid:
            break
        # Sum with self
        s2 = 2 * a
        bit2 = 1 << s2
        if used_sums_mask & bit2:
            valid = False
            break
        used_sums_mask |= bit2
    
    if not valid:
        return False
        
    # Check maximality
    for x in range(1, N + 1):
        if x in selected_set:
            continue
            
        # Calculate sums introduced by x
        # new_sums = {2*x} U {x+s | s in selected}
        # Bitmask for {x+s | s in selected} is (selected_mask << x)
        # But we need selected_mask. Let's construct it.
        # Actually, we can just compute the mask for x's sums directly from selected list?
        # Or keep selected_mask.
        
        # Let's optimize: Precompute selected_mask once.
        # I'll refactor the function to take selected and compute masks efficiently.
        
    # This implementation is getting messy. Let's do it cleanly.
    
    selected_mask = 0
    for x in selected:
        selected_mask |= (1 << x)
        
    for x in range(1, N + 1):
        if selected_mask & (1 << x):
            continue
            
        # new_sums_mask = (1 << (2*x)) | (selected_mask << x)
        # Note: selected_mask has bits at positions s. Shifting left by x moves bit s to s+x.
        # This correctly generates bits for x+s.
        
        new_sums_mask = (1 << (2 * x)) | (selected_mask << x)
        
        # If any of these sums are already in used_sums_mask, x cannot be added.
        # If new_sums_mask & used_sums_mask == 0, then x CAN be added.
        # If it CAN be added, the set is NOT maximal.
        
        if (new_sums_mask & used_sums_mask) == 0:
            return False
            
    return True

def _greedy_max_blocking_bitmask(N, rng=None, tie_break='max', weight_random=0.0):
    """
    Max-blocking greedy using bitmasks for O(1) validity and blocking checks.
    """
    selected = []
    used_sums_mask = 0
    selected_mask = 0
    candidates = list(range(1, N + 1))
    
    while candidates:
        # Find all valid candidates
        valid = []
        for x in candidates:
            # new_sums_mask for x
            new_sums_mask = (1 << (2 * x)) | (selected_mask << x)
            if (new_sums_mask & used_sums_mask) == 0:
                valid.append(x)
        
        if not valid:
            break
        
        # For each valid candidate, count how many other valid candidates it blocks
        best_x = None
        best_blocks = -1
        best_ties = []
        
        for x in valid:
            # Mask of sums added by x
            added_sums_mask = (1 << (2 * x)) | (selected_mask << x)
            
            blocks = 0
            for y in valid:
                if y == x:
                    continue
                
                # Check if y is blocked by adding x.
                # y is currently valid, so its sums don't collide with used_sums_mask.
                # Adding x changes used_sums to used_sums_mask | added_sums_mask.
                # y's potential sums are: {2*y} U {y+s | s in selected} U {y+x}
                # Mask for y's potential sums:
                # (1 << (2*y)) | (selected_mask << y) | (1 << (y+x))
                
                y_potential_mask = (1 << (2 * y)) | (selected_mask << y) | (1 << (y + x))
                
                # y is blocked if any of its potential sums collide with added_sums_mask
                # Note: We don't need to check against used_sums_mask because y is valid, 
                # so y_potential_mask & used_sums_mask == 0.
                # We only need to check if y_potential_mask & added_sums_mask != 0.
                
                if (y_potential_mask & added_sums_mask) != 0:
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
        # Update masks
        new_sums_mask = (1 << (2 * chosen)) | (selected_mask << chosen)
        used_sums_mask |= new_sums_mask
        selected_mask |= (1 << chosen)
        candidates.remove(chosen)
        
    return selected

def _greedy_with_order_bitmask(order, N):
    """Greedy construction using a specific element ordering, bitmask optimized."""
    selected = []
    used_sums_mask = 0
    selected_mask = 0
    for x in order:
        new_sums_mask = (1 << (2 * x)) | (selected_mask << x)
        if (new_sums_mask & used_sums_mask) == 0:
            selected.append(x)
            used_sums_mask |= new_sums_mask
            selected_mask |= (1 << x)
    return selected

def _local_search_improve(selected, N, rng, deadline):
    """
    Try to reduce the size of a maximal Sidon set by removing elements and
    re-adding others to maintain maximality. Returns improved set or original.
    """
    if not selected:
        return selected
    
    selected_set = set(selected)
    selected_mask = 0
    for x in selected:
        selected_mask |= (1 << x)
    
    # Compute used_sums_mask
    used_sums_mask = 0
    for i, a in enumerate(selected):
        for j in range(i):
            used_sums_mask |= (1 << (a + selected[j]))
        used_sums_mask |= (1 << (2 * a))
    
    improved = list(selected)
    improved_set = set(improved)
    improved_mask = selected_mask
    improved_used_sums = used_sums_mask
    
    # Try removing each element one at a time
    for i in range(len(improved) - 1, -1, -1):
        if time.time() > deadline:
            break
        
        x = improved[i]
        
        # Temporarily remove x
        temp_improved = improved[:i] + improved[i+1:]
        temp_set = improved_set - {x}
        temp_mask = improved_mask & ~(1 << x)
        
        # Recompute used_sums without x
        temp_used_sums = 0
        for a in temp_improved:
            for b in temp_improved:
                if a <= b:
                    temp_used_sums |= (1 << (a + b))
        
        # Check if temp is still Sidon (it should be, since it's a subset)
        # Now check if we can re-add some element to make it maximal again
        # Find all elements that can be added
        addable = []
        for y in range(1, N + 1):
            if y in temp_set:
                continue
            new_sums = (1 << (2 * y)) | (temp_mask << y)
            if (new_sums & temp_used_sums) == 0:
                addable.append(y)
        
        if not addable:
            # Set is still maximal without x - great, we can keep it removed
            improved = temp_improved
            improved_set = temp_set
            improved_mask = temp_mask
            improved_used_sums = temp_used_sums
        else:
            # We need to add at least one element to maintain maximality
            # Try to find an element that blocks the most other addable elements
            best_y = None
            best_blocks = -1
            for y in addable:
                # Count how many other addable elements y blocks
                blocks = 0
                y_sums = (1 << (2 * y)) | (temp_mask << y)
                for z in addable:
                    if z == y:
                        continue
                    z_sums = (1 << (2 * z)) | (temp_mask << z)
                    # z is blocked if z_sums intersects with y_sums or temp_used_sums
                    # But z was addable, so z_sums & temp_used_sums == 0
                    # So z is blocked if z_sums & y_sums != 0
                    if (z_sums & y_sums) != 0:
                        blocks += 1
                
                if blocks > best_blocks:
                    best_blocks = blocks
                    best_y = y
            
            if best_y is not None and len(addable) > 1:
                # Only remove x if we can replace it with something that blocks more
                # Actually, we're trying to REDUCE size, so we want to NOT add anything
                # But if we must add something, we can't reduce size
                # So skip this removal
                pass
            # If addable has exactly 1 element, we could swap x for that element
            # but that doesn't reduce size. So skip.
    
    return improved

def construct():
    deadline = time.time() + 55.0  # 55s safety margin before 60s kill
    best_size = N + 1
    best_set = None
    rng = random.Random(42)
    
    # Strategy 1: Max-blocking greedy (pure) - this is the key improvement
    candidate = _greedy_max_blocking_bitmask(N, rng, tie_break='max')
    if _is_sidon_and_maximal_bitmask(candidate, N):
        # Try local search improvement
        improved = _local_search_improve(candidate, N, rng, deadline)
        if len(improved) < len(candidate) and _is_sidon_and_maximal_bitmask(improved, N):
            candidate = improved
        best_size = len(candidate)
        best_set = candidate
    
    # Strategy 2: Max-blocking with random tie-breaking among max-block candidates
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        candidate = _greedy_max_blocking_bitmask(N, rng, tie_break='random')
        if len(candidate) < best_size:
            # Try local search improvement
            improved = _local_search_improve(candidate, N, rng, deadline)
            if len(improved) < len(candidate) and _is_sidon_and_maximal_bitmask(improved, N):
                candidate = improved
            if len(candidate) < best_size and _is_sidon_and_maximal_bitmask(candidate, N):
                best_size = len(candidate)
                best_set = candidate
                if best_size <= 16:
                    break
    
    # Strategy 3: Max-blocking with random weighting for diversity
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        candidate = _greedy_max_blocking_bitmask(N, rng, tie_break='max', weight_random=1.0)
        if len(candidate) < best_size:
            # Try local search improvement
            improved = _local_search_improve(candidate, N, rng, deadline)
            if len(improved) < len(candidate) and _is_sidon_and_maximal_bitmask(improved, N):
                candidate = improved
            if len(candidate) < best_size and _is_sidon_and_maximal_bitmask(candidate, N):
                best_size = len(candidate)
                best_set = candidate
                if best_size <= 16:
                    break
    
    # Strategy 4: Random ordering greedy (fallback/diversity)
    for attempt in range(2000):
        if time.time() > deadline:
            break
        
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_with_order_bitmask(order, N)
        if len(candidate) < best_size:
            # Try local search improvement
            improved = _local_search_improve(candidate, N, rng, deadline)
            if len(improved) < len(candidate) and _is_sidon_and_maximal_bitmask(improved, N):
                candidate = improved
            if len(candidate) < best_size and _is_sidon_and_maximal_bitmask(candidate, N):
                best_size = len(candidate)
                best_set = candidate
                if best_size <= 16:
                    break
    
    if best_set is None:
        # Fallback
        best_set = _greedy_max_blocking_bitmask(N, rng, tie_break='max')
    
    return best_set
# EVOLVE-BLOCK-END
