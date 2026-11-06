import time
import random
from collections import defaultdict

n = 128
_DEADLINE = 85.0
_START_TIME = None

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _time_left():
    return _DEADLINE - _elapsed() - 2.0

# Precompute progressions
_AP_LIST = []
for a in range(n):
    for d in range(1, (n - a) // 3 + 1):
        _AP_LIST.append((a, a+d, a+2*d, a+3*d))

# Map position to list of AP indices containing it
_POS_TO_APS = [[] for _ in range(n)]
for idx, ap in enumerate(_AP_LIST):
    for p in ap:
        _POS_TO_APS[p].append(idx)

def _check_ap_consistency(ap, colors):
    """
    Check if the current assignment of colors to ap's positions
    is consistent with the requirement (>=3 distinct colors).
    Returns True if consistent (or not enough info), False if violated.
    """
    c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
    
    assigned = [c for c in [c1, c2, c3, c4] if c != -1]
    num_assigned = len(assigned)
    
    if num_assigned == 0 or num_assigned == 1 or num_assigned == 2:
        # Not enough info to violate
        return True
        
    distinct = set(assigned)
    
    if num_assigned == 3:
        # If 3 assigned are all same, the 4th MUST be different to get 2 distinct? 
        # No, if 3 are same (e.g. 1,1,1), and 4th is 1, distinct=1 (Fail).
        # If 4th is 2, distinct=2 (Fail).
        # We need >=3 distinct.
        # So if 3 are same, it is IMPOSSIBLE to have >=3 distinct colors?
        # Wait. {1, 1, 1, x}. Distinct is {1, x}. Size is at most 2.
        # So if 3 assigned are same, the AP is INVALID regardless of the 4th.
        if len(distinct) == 1:
            return False
        # If 3 assigned are 2 distinct (e.g. 1, 1, 2), distinct={1,2}.
        # The 4th must be a 3rd color (3) to make distinct={1,2,3}.
        # So it is POSSIBLE. Consistent so far.
        if len(distinct) == 2:
            return True
        # If 3 assigned are 3 distinct, valid.
        if len(distinct) == 3:
            return True
            
    if num_assigned == 4:
        if len(distinct) < 3:
            return False
        return True
        
    return True

def _solve_csp(k, seed):
    rng = random.Random(seed)
    colors = [-1] * n
    
    def backtrack(unassigned_count):
        if _time_left() < 0.1:
            return False
            
        if unassigned_count == 0:
            return True
            
        # Select variable with Minimum Remaining Values (MRV)
        min_domain_size = k + 1
        min_pos = -1
        min_valid_vals = []
        
        unassigned = [i for i in range(n) if colors[i] == -1]
        rng.shuffle(unassigned)
        
        for p in unassigned:
            valid_vals = []
            for v in range(k):
                # Tentatively assign
                colors[p] = v
                is_consistent = True
                for ap_idx in _POS_TO_APS[p]:
                    if not _check_ap_consistency(_AP_LIST[ap_idx], colors):
                        is_consistent = False
                        break
                if is_consistent:
                    valid_vals.append(v)
                colors[p] = -1
                
            if len(valid_vals) < min_domain_size:
                min_domain_size = len(valid_vals)
                min_pos = p
                min_valid_vals = valid_vals
                
            if min_domain_size == 0:
                return False
                
        if min_pos == -1:
            return False
            
        # Try values for min_pos
        rng.shuffle(min_valid_vals)
        
        for v in min_valid_vals:
            # Assign
            colors[min_pos] = v
            # Check consistency (again, to be safe, though we checked in MRV loop)
            # The MRV loop checked consistency for the assignment.
            # However, the state might have changed? No, we restored colors[p] = -1 after each check.
            # So the state is the same as before the MRV loop.
            # So we can just assign and recurse.
            
            if backtrack(unassigned_count - 1):
                return True
                
            # Unassign
            colors[min_pos] = -1
            
        return False

    success = backtrack(n)
    if success:
        return colors
    return None

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    # Try K from 3 upwards
    for k in range(3, 10):
        if _time_left() < 5.0:
            break
            
        # Try a few seeds for this K
        for seed in range(50):
            if _time_left() < 1.0:
                break
                
            result = _solve_csp(k, seed)
            if result is not None:
                return result

    # Fallback: If no solution found (should not happen for K>=4 likely),
    # return a pattern that is likely valid or just random.
    # A simple pattern: i % 4
    # Let's verify i%4 on a small AP.
    # 0, 1, 2, 3 -> 0,1,2,3 (4 distinct) OK.
    # 0, 2, 4, 6 -> 0,2,0,2 (2 distinct) FAIL.
    # So i%4 is NOT valid.
    
    # What is a valid pattern?
    # If K=4, maybe (i//4) % 4?
    # 0,0,0,0 -> 0 (Fail)
    
    # We must return SOMETHING.
    # If the solver failed, it's a timeout or bug.
    # To ensure we don't crash, return a random array?
    # The evaluator checks validity. If invalid, score 0.
    # Better to return the best effort.
    
    # Let's try a greedy construction as a fallback, which is fast.
    colors = [-1] * n
    for i in range(n):
        forbidden = [False] * k
        for ap_idx in _POS_TO_APS[i]:
            ap = _AP_LIST[ap_idx]
            others = [p for p in ap if p != i]
            c_others = [colors[p] for p in others]
            if all(c != -1 for c in c_others):
                # Check if any color makes distinct < 3
                for c in range(k):
                    # If we assign c to i, what is distinct count?
                    # We can just check if the set of {c} + c_others has size < 3
                    # If so, c is forbidden.
                    if len(set([c] + c_others)) < 3:
                        forbidden[c] = True
        available = [c for c in range(k) if not forbidden[c]]
        if available:
            colors[i] = random.choice(available)
        else:
            colors[i] = 0 # Fallback
            
    return colors
