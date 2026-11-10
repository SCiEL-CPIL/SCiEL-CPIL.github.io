"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256
# EVOLVE-BLOCK-START

def _is_maximal_sidon(selected, N):
    """Check if no missing element can be added without breaking Sidon property."""
    sel_set = set(selected)
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        for j in range(i, n):
            s = a + selected[j]
            sums.add(s)
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x breaks Sidon
        # x can be added if 2x not in sums and x+s not in sums for all s in selected
        if 2 * x in sums:
            continue
        blocked = False
        for a in selected:
            if x + a in sums:
                blocked = True
                break
        if not blocked:
            # x could be added, so not maximal
            return False
    return True

def _greedy_maximal(rng, N):
    """Generate a maximal Sidon set using greedy saturation in random order."""
    order = list(range(1, N + 1))
    rng.shuffle(order)
    
    selected = []
    sel_set = set()
    sums = set()
    
    for x in order:
        added = []
        ok = True
        for a in selected:
            s = x + a
            if s in sums or s in added:
                ok = False
                break
            added.append(s)
        s = 2 * x
        if not ok or s in sums or s in added:
            continue
        selected.append(x)
        sel_set.add(x)
        sums.update(added)
        sums.add(s)
        
    return selected

def _shrink_maximal(selected, N, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    
    while time.time() < deadline:
        improved = False
        # Shuffle to avoid bias
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            x = selected[idx]
            candidate = selected[:idx] + selected[idx+1:]
            
            # Check if candidate is still maximal
            # Note: candidate is always a valid Sidon set because it's a subset of a valid one.
            if _is_maximal_sidon(candidate, N):
                selected = candidate
                improved = True
                break
        
        if not improved:
            break
            
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5-second safety margin
    
    best = None
    best_size = float('inf')
    
    # We will run multiple random restarts
    # Each restart: generate a maximal set, then shrink it.
    
    num_restarts = 2000 # We will be limited by time anyway
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        rng = random.Random(i * 12345 + 67890)
        
        # Generate initial maximal set
        sel = _greedy_maximal(rng, N)
        
        if not sel:
            continue
            
        # Shrink it
        sel = _shrink_maximal(sel, N, deadline)
        
        if len(sel) < best_size:
            best = sel
            best_size = len(sel)
            
            # If we get very small, we can stop early?
            # Theoretical lower bound is roughly sqrt(N) = 16.
            # If we hit 16, it's likely optimal.
            if best_size <= 16:
                break

    if best is None:
        # Fallback: standard ascending greedy
        sel = _greedy_maximal(random.Random(0), N)
        return sel
        
    return best

# EVOLVE-BLOCK-END
