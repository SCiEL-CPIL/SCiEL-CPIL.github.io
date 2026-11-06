"""Rewritten program: backtracking search over integer Sidon sets with 0,
covering differences 1..30, minimizing the maximum element."""
# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = time.time() + 148.0  # safety margin before 150s kill
_TARGET_MASK = (1 << 30) - 1

# ---------------------------------------------------------------------------
# Fast helpers
# ---------------------------------------------------------------------------

def _is_sidon(s):
    """True iff all positive differences are distinct."""
    n = len(s)
    seen = set()
    for i in range(n):
        si = s[i]
        for j in range(i + 1, n):
            d = s[j] - si
            if d in seen:
                return False
            seen.add(d)
    return True

def _covers_1_to_30(s):
    d = set()
    for i in range(len(s)):
        si = s[i]
        for j in range(i + 1, len(s)):
            d.add(s[j] - si)
    for x in range(1, 31):
        if x not in d:
            return False
    return True

def _valid(s):
    if len(s) < 1 or len(s) > 32:
        return False
    if 0 not in s:
        return False
    if max(s) > 10**6:
        return False
    if not _is_sidon(s):
        return False
    if not _covers_1_to_30(s):
        return False
    return True

# ---------------------------------------------------------------------------
# Greedy baseline (fast, deterministic)
# ---------------------------------------------------------------------------

def _greedy_construct():
    a = [0]
    def diffs_list(s):
        return [y - x for i, x in enumerate(s) for y in s[i + 1:]]
    for d in range(1, 31):
        if d in diffs_list(sorted(a)):
            continue
        found = False
        candidates = []
        for v in a:
            for x in (v + d, v - d):
                if x >= 0 and x not in a:
                    candidates.append(x)
        candidates = sorted(set(candidates))
        for x in candidates:
            trial = sorted(a + [x])
            dl = diffs_list(trial)
            if len(set(dl)) == len(dl):
                a = trial
                found = True
                break
        if not found:
            y = max(a) + 1
            while True:
                trial = sorted(a + [y, y + d])
                dl = diffs_list(trial)
                if len(set(dl)) == len(dl):
                    a = trial
                    break
                y += 1
    return a

# ---------------------------------------------------------------------------
# Backtracking search
# ---------------------------------------------------------------------------

def _backtracking_search(D, time_limit):
    """
    Search for a valid Sidon set in [0, D] covering 1..30.
    Returns the set if found, None otherwise.
    """
    best_sol = None
    nodes = [0]
    
    # Precompute which differences are "hard" to cover (larger values)
    # We'll weight improvements by how hard the difference is to place
    
    def dfs(current_set, covered_mask, used_diffs_mask):
        nonlocal best_sol
        if time.time() > time_limit:
            return False
        
        nodes[0] += 1
        
        if covered_mask == _TARGET_MASK:
            best_sol = current_set[:]
            return True
            
        if len(current_set) >= 32:
            return False
            
        last_val = current_set[-1]
        
        # Strong pruning: count uncovered differences
        uncovered = _TARGET_MASK & ~covered_mask
        popcount_uncovered = bin(uncovered).count("1")
        if popcount_uncovered == 0:
            best_sol = current_set[:]
            return True
        
        remaining_slots = 32 - len(current_set)
        if remaining_slots <= 0:
            return False
        
        # Each new element x adds differences x-y for all y in current_set
        # Plus differences between new elements (but we add one at a time)
        # Max new differences from one element: len(current_set) (to existing) 
        # but only those in 1..30 count. Use a tighter bound.
        # Actually, with k existing elements, a new element adds k differences.
        # But we also need differences among future elements.
        # Conservative bound: each element can cover at most (current_size + 1) new diffs in 1..30
        # because it connects to all existing + potentially to one future element
        max_new_per_elem = len(current_set) + 1
        if popcount_uncovered > remaining_slots * max_new_per_elem:
            return False
        
        # Additional pruning: check if remaining diameter can accommodate
        # The largest uncovered difference must fit in [0, D]
        # Find the largest uncovered d
        for d in range(30, 0, -1):
            if uncovered & (1 << (d-1)):
                if d > D:
                    return False
                break
        
        # Generate candidates x > last_val
        candidates = []
        for x in range(last_val + 1, D + 1):
            valid = True
            new_diffs = []
            for y in current_set:
                d = x - y
                if d > 0 and (used_diffs_mask >> d) & 1:
                    valid = False
                    break
                if d > 0:
                    new_diffs.append(d)
            
            if not valid:
                continue
                
            new_mask = covered_mask
            for d in new_diffs:
                if 1 <= d <= 30:
                    new_mask |= (1 << (d-1))
                    
            improvement = bin(new_mask ^ covered_mask).count("1")
            if improvement == 0:
                continue
            
            # Weight: prefer covering larger differences first (harder to place later)
            weight = 0
            for d in new_diffs:
                if 1 <= d <= 30 and (uncovered >> (d-1)) & 1:
                    weight += d  # Larger d = harder to cover = higher priority
            
            candidates.append((weight, improvement, x, new_diffs, new_mask))
            
        if not candidates:
            return False
            
        # Sort: prefer higher weight (covers hard diffs), then higher improvement, then smaller x
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        
        # Limit candidates to top few for speed
        max_candidates = min(len(candidates), 20)
        
        for w, imp, x, new_diffs, new_mask in candidates[:max_candidates]:
            new_set = current_set + [x]
            new_used_mask = used_diffs_mask
            for d in new_diffs:
                if 1 <= d <= 60:  # Only track diffs up to 60 for efficiency
                    new_used_mask |= (1 << d)
            
            if dfs(new_set, new_mask, new_used_mask):
                return True
                
        return False

    if dfs([0], 0, 0):
        return best_sol
    return None

# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------

def construct():
    t0 = time.time()
    
    # Get a baseline solution to ensure we have something valid
    baseline = _greedy_construct()
    best_val = max(baseline)
    best_set = baseline
    
    # Try to find a smaller diameter using backtracking
    # Start from a lower bound.
    # We need at least 9 elements.
    # A safe lower bound for diameter is 50.
    # We iterate D upwards.
    
    # To save time, we can skip D values if we are close to deadline
    # But backtracking is fast for small D.
    
    # Start from a lower bound.
    # A Sidon set covering 1..30 needs at least ~9 elements.
    # With k elements, we have k*(k-1)/2 differences.
    # Need at least 30 differences, so k*(k-1)/2 >= 30 => k >= 9.
    # The minimum diameter for a (v,k,lambda) difference set is related to k.
    # For small sets, diameter is at least ~30-40.
    
    start_D = 40
    end_D = best_val
    
    # Adaptive time per D: give more time to smaller D values
    # Use exponential decay: smaller D gets much more time
    for D in range(start_D, end_D + 1):
        if time.time() > _DEADLINE - 2:
            break
            
        # Smaller D gets significantly more time
        # D=40: ~15s, D=50: ~10s, D=60: ~6s, D=70+: ~3s
        if D <= 45:
            per_d_time = 15.0
        elif D <= 50:
            per_d_time = 10.0
        elif D <= 55:
            per_d_time = 7.0
        elif D <= 60:
            per_d_time = 5.0
        elif D <= 70:
            per_d_time = 3.5
        else:
            per_d_time = 2.5
        
        t_start = time.time()
        limit = t_start + per_d_time
        
        sol = _backtracking_search(D, limit)
        
        if sol:
            # This is the minimum diameter found so far
            return sol
            
    # If no better solution found, return baseline
    return best_set

# EVOLVE-BLOCK-END
