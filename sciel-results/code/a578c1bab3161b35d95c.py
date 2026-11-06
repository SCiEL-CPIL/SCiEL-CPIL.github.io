import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _precompute_aps():
    """
    Precompute all 3-term arithmetic progressions in 1..512.
    Returns:
    ap_by_elem: list of lists. ap_by_elem[x] contains tuples (d, e1, e2) 
                such that {x, e1, e2} is a 3-AP with difference d.
    """
    ap_by_elem = [[] for _ in range(513)]
    # d ranges from 1 to 255
    for d in range(1, 256):
        # a ranges such that a+2d <= 512
        max_a = 512 - 2 * d
        for a in range(1, max_a + 1):
            b = a + d
            c = a + 2 * d
            # Add to each element's list
            # For element a: AP is (a, b, c), diff d
            ap_by_elem[a].append((d, b, c))
            # For element b: AP is (a, b, c), diff d
            ap_by_elem[b].append((d, a, c))
            # For element c: AP is (a, b, c), diff d
            ap_by_elem[c].append((d, a, b))
    return ap_by_elem

# Global precomputed data
AP_BY_ELEM = _precompute_aps()

def _init_state(S):
    """
    Initialize state for a given set S (list of 32 ints).
    Returns:
    S_set: set of elements
    d_count: array of counts for each d
    score: number of d with count > 0
    """
    S_set = set(S)
    d_count = np.zeros(256, dtype=np.int32)
    
    # Count active APs
    # Iterate through elements in S
    # For each element, check its APs
    # To avoid double counting, we can just iterate all APs?
    # No, iterating elements and checking if other 2 are in S is O(N * deg).
    # 32 * 500 = 16k ops. Fast.
    
    for x in S:
        for d, e1, e2 in AP_BY_ELEM[x]:
            if e1 in S_set and e2 in S_set:
                # This AP is active
                # We need to ensure we don't count the same AP multiple times.
                # An AP is defined by (a, d). 
                # In AP_BY_ELEM[x], the same AP appears for x, x+d, x+2d.
                # So if we iterate all x in S, we will count each active AP 3 times.
                # We need to count each AP once.
                # How?
                # We can just check if the *smallest* element of the AP is in S?
                # Or just divide by 3? No, if only 2 elements are in S, it's not active.
                # If all 3 are in S, it's active.
                # It will be counted exactly 3 times (once for each element).
                # So we can sum counts and divide by 3?
                # Let's verify.
                # AP (1, 2, 3), d=1.
                # In AP_BY_ELEM[1]: (1, 2, 3)
                # In AP_BY_ELEM[2]: (1, 1, 3) -> Wait, my tuple is (d, e1, e2).
                # For x=1: (1, 2, 3). e1=2, e2=3.
                # For x=2: (1, 1, 3). e1=1, e2=3.
                # For x=3: (1, 1, 2). e1=1, e2=2.
                # Yes, each active AP is counted 3 times.
                
                d_count[d] += 1
    
    # Since each active AP is counted 3 times, we divide by 3
    d_count //= 3
    score = np.sum(d_count > 0)
    return S_set, d_count, score

def _delta_score_add(S_set, d_count, score, x):
    """
    Calculate delta score if we add element x to S_set.
    Assumes x not in S_set.
    Returns (delta_score, new_d_count_changes)
    """
    delta = 0
    # We need to know which d's counts will increase.
    # For each AP containing x, if the other two are in S_set, the AP becomes active.
    # This increases d_count[d] by 1.
    # If d_count[d] was 0, score increases by 1.
    
    # We don't want to modify d_count yet, just calculate delta.
    # But we need to know the current d_count[d].
    
    # Optimization: We can just check the conditions.
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            if d_count[d] == 0:
                delta += 1
            # else: d_count[d] > 0, so adding this AP doesn't change score
            # But it does change d_count[d], which might matter for future removals?
            # No, for score calculation, only 0->1 transition matters.
            # However, we MUST update d_count for the state to be consistent.
            # The function should return the changes to apply.
            # For now, let's just return delta and assume we will update state later.
            # Actually, we need to update d_count[d] += 1 for every such AP.
            
    # This simple delta calculation is correct for the score change.
    # But we need to update d_count correctly.
    # Let's create a list of (d, delta_count) to apply.
    # Actually, for SA, we just need the score delta to decide accept/reject.
    # If accepted, we apply the changes.
    
    return delta

def _delta_score_remove(S_set, d_count, score, x):
    """
    Calculate delta score if we remove element x from S_set.
    Assumes x in S_set.
    """
    delta = 0
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            # This AP is currently active.
            # Removing x will deactivate it.
            # d_count[d] will decrease by 1.
            if d_count[d] == 1:
                # It will become 0, so score decreases by 1
                delta -= 1
            # else: d_count[d] > 1, score unchanged
    return delta

def _apply_add(S_set, d_count, score, x):
    """Apply adding x to state. Returns new score."""
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            if d_count[d] == 0:
                score += 1
            d_count[d] += 1
    S_set.add(x)
    return score

def _apply_remove(S_set, d_count, score, x):
    """Apply removing x from state. Returns new score."""
    for d, e1, e2 in AP_BY_ELEM[x]:
        if e1 in S_set and e2 in S_set:
            d_count[d] -= 1
            if d_count[d] == 0:
                score -= 1
    S_set.remove(x)
    return score

def _greedy_set_cover_init(rng):
    """
    Initialize a set using a greedy set cover heuristic.
    """
    S = set()
    d_covered = np.zeros(256, dtype=bool)
    
    # Candidates: all elements 1..512
    # We want to pick elements that maximize new d's.
    # This is hard to do greedily by element.
    # Better: Greedy by AP.
    # Pick APs that cover new d's and use few new elements.
    
    # Let's stick to random init for simplicity, or structured.
    # The SA with delta scoring is the main improvement.
    
    return sorted(rng.sample(range(1, 513), 32))

def _simulated_annealing(S_init, time_budget, rng):
    S = list(S_init)
    S_set, d_count, current_score = _init_state(S)
    
    best_S = list(S)
    best_score = current_score
    
    start_time = time.time()
    T = 5.0
    cooling = 0.999
    
    # Pre-compute list of elements for fast random selection?
    # No, just use rng.
    
    while time.time() - start_time < time_budget:
        T *= cooling
        if T < 0.01:
            break
        
        i = rng.randint(0, 31)
        old_elem = S[i]
        
        # Generate new element
        # To avoid duplicates, we can sample from 1..512 and check.
        new_elem = rng.randint(1, 512)
        if new_elem == old_elem:
            continue
        if new_elem in S_set:
            continue
            
        # Calculate delta
        # Remove old, Add new
        # We must be careful: The delta for remove depends on S_set containing old_elem.
        # The delta for add depends on S_set NOT containing new_elem (but containing the other AP elements).
        # If we remove old_elem first, S_set changes.
        # So:
        # 1. Delta_remove = score change if we remove old_elem from current S_set.
        # 2. Delta_add = score change if we add new_elem to (S_set \ {old_elem}).
        
        # Let's compute remove delta first.
        delta_rem = _delta_score_remove(S_set, d_count, current_score, old_elem)
        
        # Now, to compute add delta, we need to consider S_set without old_elem.
        # But _delta_score_add checks if e1, e2 are in S_set.
        # If e1 or e2 is old_elem, the condition `e1 in S_set` will be True currently,
        # but False after removal.
        # So we need to handle this.
        
        # Optimization:
        # Just apply the move, calculate new score, and revert if needed?
        # That's 2 * O(deg) operations.
        # Or calculate delta carefully.
        
        # Let's just apply, score, revert. It's simpler and less error-prone.
        # O(deg) is ~500. 2*500 = 1000 ops per move.
        # 10^5 moves = 10^8 ops. Feasible in 10s? Maybe.
        
        # Apply Remove
        _apply_remove(S_set, d_count, current_score, old_elem)
        # Apply Add
        new_score = _apply_add(S_set, d_count, current_score, new_elem)
        
        delta = new_score - current_score
        
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_score = new_score
            S[i] = new_elem
            if current_score > best_score:
                best_score = current_score
                best_S = list(S)
        else:
            # Revert
            _apply_remove(S_set, d_count, current_score, new_elem)
            _apply_add(S_set, d_count, current_score, old_elem)
            # current_score is restored
    
    return best_S, best_score

def _greedy_polish(S_init, time_limit=5.0):
    """
    Greedy local search: try swapping each element with every other candidate.
    Use delta scoring for speed.
    """
    S = list(S_init)
    S_set, d_count, current_score = _init_state(S)
    
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(32):
            old_val = S[i]
            best_cand = None
            best_score = current_score
            
            # Try removing old_val
            _apply_remove(S_set, d_count, current_score, old_val)
            score_without_old = current_score
            
            # Try adding each candidate
            for cand in range(1, 513):
                if cand in S_set:
                    continue
                
                # Apply add
                new_score = _apply_add(S_set, d_count, score_without_old, cand)
                
                if new_score > best_score:
                    best_score = new_score
                    best_cand = cand
                    # Keep the state with best_cand applied?
                    # No, we are just exploring.
                    # We need to revert this add before trying next cand.
                    _apply_remove(S_set, d_count, new_score, cand)
                    # d_count and S_set are restored to state without old_val
                else:
                    # Revert add
                    _apply_remove(S_set, d_count, new_score, cand)
            
            # Restore old_val
            current_score = _apply_add(S_set, d_count, score_without_old, old_val)
            
            if best_cand is not None:
                S[i] = best_cand
                # State is currently with old_val added back.
                # We need to apply the best move: remove old_val, add best_cand.
                _apply_remove(S_set, d_count, current_score, old_val)
                current_score = _apply_add(S_set, d_count, current_score, best_cand)
                improved = True
    
    return S, current_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = random.Random(42)
    
    best_S = list(range(1, 33))
    best_score = _init_state(best_S)[2]
    
    # Try multiple restarts
    num_restarts = 10
    per_restart_time = 10.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            S_init = list(range(1, 33))
        else:
            # Random init
            S_init = sorted(rng.sample(range(1, 513), 32))
        
        remaining = min(per_restart_time, deadline - time.time() - 5.0)
        if remaining < 1.0:
            break
        
        S, score = _simulated_annealing(S_init, remaining, rng)
        
        # Greedy polish
        if time.time() < deadline - 3.0:
            S, score = _greedy_polish(S, time_limit=2.0)
        
        if score > best_score:
            best_score = score
            best_S = S
    
    # Final polish
    if time.time() < deadline - 1.0:
        best_S, best_score = _greedy_polish(best_S, time_limit=5.0)
    
    return sorted(best_S)

# EVOLVE-BLOCK-END
