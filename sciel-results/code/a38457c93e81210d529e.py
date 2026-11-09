"""Stronger algorithm for reciprocal weight of 3-progression-free sets on [1,256]."""
import time
import random
from fractions import Fraction
from itertools import combinations

# EVOLVE-BLOCK-START

def _is_3free(selected):
    """Check if a set has no 3-term arithmetic progression."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a < b:
                c = 2*b - a
                if c in s and c > b:
                    return False
    return True

def _is_3free_fast(selected):
    """Faster check using set membership."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a < b:
                c = 2*b - a
                if c > 256:
                    continue
                if c in s:
                    return False
    return True

def _recip_sum(selected):
    """Compute exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_by_reciprocal(n=256):
    """Greedy: add elements in order of decreasing 1/x (i.e., increasing x is wrong; 
    we want largest reciprocals first = smallest x first), but check 3-free constraint."""
    selected = []
    selected_set = set()
    # Try adding in order of smallest x (largest reciprocal) first
    for x in range(1, n+1):
        # Check if adding x creates a 3-AP
        ok = True
        for a in selected:
            # a, x, c where c = 2*x - a
            c = 2*x - a
            if c > x and c <= n and c in selected_set:
                ok = False
                break
            # a, b, x where b = (a+x)/2
            if (a + x) % 2 == 0:
                b = (a + x) // 2
                if a < b < x and b in selected_set:
                    ok = False
                    break
            # x, b, c where b = x + d, c = x + 2d for some d
            # This is covered by the first case when we consider pairs (a,x)
        if ok:
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_randomized(n=256, seed=0):
    """Greedy with randomization: try multiple random orderings weighted by reciprocal."""
    rng = random.Random(seed)
    best = None
    best_val = Fraction(0, 1)
    
    for trial in range(50):
        # Create weighted random ordering: prefer smaller numbers (larger reciprocals)
        # but with some randomness
        candidates = list(range(1, n+1))
        # Assign weights: weight = 1/x with some noise
        weights = [1.0/x for x in candidates]
        # Add noise
        noisy_weights = [w * (0.5 + rng.random()) for w in weights]
        
        # Sort by noisy weight descending (largest reciprocal first)
        order = sorted(candidates, key=lambda x: noisy_weights[x-1], reverse=True)
        
        selected = []
        selected_set = set()
        for x in order:
            ok = True
            for a in selected:
                c = 2*x - a
                if c > x and c <= n and c in selected_set:
                    ok = False
                    break
                if (a + x) % 2 == 0:
                    b = (a + x) // 2
                    if a < b < x and b in selected_set:
                        ok = False
                        break
            if ok:
                selected.append(x)
                selected_set.add(x)
        
        val = _recip_sum(selected)
        if val > best_val:
            best_val = val
            best = selected
    
    return best if best else [1]

def _local_search(start_set, n=256, max_iters=10000, seed=0):
    """Local search: try swapping elements to improve reciprocal sum."""
    rng = random.Random(seed)
    current = list(start_set)
    current_set = set(current)
    current_val = _recip_sum(current)
    best = list(current)
    best_val = current_val
    
    for it in range(max_iters):
        # Pick a random element to remove and a random candidate to add
        if not current:
            break
        remove_idx = rng.randrange(len(current))
        removed = current[remove_idx]
        
        # Find a candidate to add
        candidates = [x for x in range(1, n+1) if x not in current_set]
        if not candidates:
            break
        # Prefer candidates with large reciprocal (small x)
        # Weighted random choice
        weights = [1.0/x for x in candidates]
        total_w = sum(weights)
        r = rng.random() * total_w
        cum = 0
        add_candidate = candidates[0]
        for c in candidates:
            cum += 1.0/c
            if cum >= r:
                add_candidate = c
                break
        
        # Try the swap
        new_set = current_set - {removed} | {add_candidate}
        new_list = list(new_set)
        
        # Check 3-free
        if _is_3fast(new_list):
            new_val = _recip_sum(new_list)
            if new_val > current_val:
                current = new_list
                current_set = new_set
                current_val = new_val
                if current_val > best_val:
                    best_val = current_val
                    best = list(current)
    
    return best

def _is_3fast(selected):
    """Fast 3-free check."""
    s = set(selected)
    for a in selected:
        for b in selected:
            if a < b:
                c = 2*b - a
                if c > 256:
                    continue
                if c in s:
                    return False
    return True

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s limit
    
    # Start with greedy by reciprocal (smallest x first)
    best_set = _greedy_by_reciprocal(256)
    best_val = _recip_sum(best_set)
    
    # Try randomized greedy
    while time.time() - start_time < deadline - 5:
        for seed in range(20):
            if time.time() - start_time > deadline - 5:
                break
            candidate = _greedy_randomized(256, seed=seed)
            if candidate:
                val = _recip_sum(candidate)
                if val > best_val:
                    best_val = val
                    best_set = candidate
    
    # Local search from best
    while time.time() - start_time < deadline - 2:
        for seed in range(10):
            if time.time() - start_time > deadline - 2:
                break
            candidate = _local_search(best_set, 256, max_iters=5000, seed=seed)
            if candidate:
                val = _recip_sum(candidate)
                if val > best_val:
                    best_val = val
                    best_set = candidate
    
    return best_set

# EVOLVE-BLOCK-END
