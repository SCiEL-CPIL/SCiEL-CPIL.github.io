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
            sums.add(a + selected[j])
    
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        # Check if adding x breaks Sidon
        # x can be added if 2x not in sums and x+a not in sums for all a in selected
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
        # Check if x can be added
        # 1. 2x not in sums
        if 2 * x in sums:
            continue
        # 2. x + a not in sums for all a in selected
        # Also need to ensure new sums (x+a) are distinct from each other?
        # If x+a1 == x+a2 => a1==a2, which is true since a's are distinct.
        # If x+a == 2x => a==x, but x is not in selected yet.
        # So we just need to check against existing sums.
        
        valid = True
        for a in selected:
            if x + a in sums:
                valid = False
                break
        
        if valid:
            selected.append(x)
            sel_set.add(x)
            sums.add(2 * x)
            for a in selected[:-1]: # selected[:-1] is the set before adding x
                sums.add(x + a)
                
    return selected

def _shrink_maximal(selected, N, deadline):
    """Try to remove elements while maintaining maximality."""
    selected = list(selected)
    
    # Greedy removal loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(selected)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                return selected
            x = selected[idx]
            candidate = selected[:idx] + selected[idx+1:]
            
            # Check if candidate is still maximal
            # Note: candidate is always a valid Sidon set because it's a subset of a valid one.
            if _is_maximal_sidon(candidate, N):
                selected = candidate
                improved = True
                # Restart the loop with the new smaller set to try removing more
                break
        
    return selected

def construct():
    start_time = time.time()
    # Keep a safety margin. The evaluator kills at 60s.
    # Let's aim to finish by 55s to be safe.
    deadline = start_time + 55.0
    
    best = None
    best_size = float('inf')
    
    # Since _greedy_maximal is very fast, we can do many restarts.
    # _shrink_maximal is the bottleneck, but it's also fast for small sets.
    
    num_restarts = 5000 # We will be limited by time anyway
    
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
