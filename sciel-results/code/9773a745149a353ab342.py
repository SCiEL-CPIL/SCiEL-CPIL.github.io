"""Systematic local search for reciprocal weight of 3-progression-free sets on [1,256]."""
import time
import random
from fractions import Fraction

# EVOLVE-BLOCK-START

def _can_add(x, S_set, S_list):
    """Check if x can be added to S without creating a 3-AP. O(K) time."""
    for a in S_list:
        if a == x:
            continue
        # Case 1: x is the middle term. a, x, c. c = 2x - a.
        c = 2 * x - a
        if 1 <= c <= 256 and c != x and c in S_set:
            return False
        # Case 2: x is an endpoint. a, b, x or x, b, a. b = (a+x)/2.
        s_sum = a + x
        if s_sum % 2 == 0:
            b = s_sum // 2
            if b != x and b in S_set:
                return False
    return True

def _recip_sum(selected):
    """Compute exact sum of reciprocals."""
    total = Fraction(0, 1)
    for x in selected:
        total += Fraction(1, x)
    return total

def _greedy_by_reciprocal(n=256):
    """Greedy: add elements in order of decreasing 1/x (i.e., smallest x first)."""
    selected = []
    selected_set = set()
    for x in range(1, n+1):
        if _can_add(x, selected_set, selected):
            selected.append(x)
            selected_set.add(x)
    return selected

def _greedy_randomized(n=256, seed=0, trials=100):
    """Greedy with randomization: try multiple random orderings weighted by reciprocal."""
    rng = random.Random(seed)
    best = None
    best_val = Fraction(0, 1)
    
    for trial in range(trials):
        candidates = list(range(1, n+1))
        weights = [1.0/x for x in candidates]
        noisy_weights = [w * (0.5 + rng.random()) for w in weights]
        order = sorted(candidates, key=lambda x: noisy_weights[x-1], reverse=True)
        
        selected = []
        selected_set = set()
        for x in order:
            if _can_add(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
        
        val = _recip_sum(selected)
        if val > best_val:
            best_val = val
            best = selected
    
    return best if best else [1]

def _local_search_systematic(start_set, n=256, max_iters=5000, deadline=None):
    """
    Systematic local search: at each step, try ALL single-element removals and 
    ALL valid single-element additions to find the best improvement.
    This is more thorough than random sampling.
    """
    current = list(start_set)
    current_set = set(current)
    current_val = _recip_sum(current)
    best = list(current)
    best_val = current_val
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Try all possible single-element removals
        best_improvement = Fraction(0, 1)
        best_remove = None
        best_add = None
        
        for i, removed in enumerate(current):
            # Remove 'removed' from the set
            temp_set = current_set - {removed}
            temp_list = [x for x in current if x != removed]
            
            # Try all possible additions
            for add_candidate in range(1, n+1):
                if add_candidate in current_set and add_candidate != removed:
                    continue
                if add_candidate == removed:
                    continue
                    
                if _can_add(add_candidate, temp_set, temp_list):
                    # Valid swap
                    delta = Fraction(1, add_candidate) - Fraction(1, removed)
                    if delta > best_improvement:
                        best_improvement = delta
                        best_remove = removed
                        best_add = add_candidate
        
        if best_remove is not None and best_improvement > 0:
            # Apply the best swap
            current = [x for x in current if x != best_remove]
            current_set = current_set - {best_remove}
            current.append(best_add)
            current_set.add(best_add)
            current_val += best_improvement
            
            if current_val > best_val:
                best_val = current_val
                best = list(current)
        else:
            # No improvement found
            break
    
    return best

def construct():
    """Main entry point: find a 3-free subset of [1,256] maximizing sum of reciprocals."""
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit (evaluator kills at 90s, but we want to be safe)
    
    # Start with greedy by reciprocal
    best_set = _greedy_by_reciprocal(256)
    best_val = _recip_sum(best_set)
    
    # Try randomized greedy to get diverse starting points
    seed = 0
    while time.time() < deadline - 20:
        candidate = _greedy_randomized(256, seed=seed, trials=20)
        if candidate:
            val = _recip_sum(candidate)
            if val > best_val:
                best_val = val
                best_set = candidate
        seed += 1
    
    # Systematic local search from best
    while time.time() < deadline - 2:
        candidate = _local_search_systematic(best_set, 256, max_iters=1000, deadline=deadline - 2)
        if candidate:
            val = _recip_sum(candidate)
            if val > best_val:
                best_val = val
                best_set = candidate
            else:
                break
    
    return best_set

# EVOLVE-BLOCK-END
