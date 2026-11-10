import time
import random
import numpy as np
from fractions import Fraction

_N = 256
_KILL = 90.0
_MARGIN = 8.0
_RECIP = [0.0] + [1.0 / i for i in range(1, _N + 1)]

def _precompute_aps():
    """
    Precompute all 3-term arithmetic progressions in [1, N].
    Returns a list of tuples (a, b, c) where a < b < c and a+b+c = 3b.
    """
    aps = []
    for b in range(1, _N + 1):
        max_d = min(b - 1, _N - b)
        for d in range(1, max_d + 1):
            a = b - d
            c = b + d
            aps.append((a - 1, b - 1, c - 1)) # 0-indexed
    return aps

def _check_validity(mask):
    """
    Check if the bitmask represents a 3-AP free set.
    mask is a numpy array of bools or an integer bitmask.
    """
    if isinstance(mask, np.ndarray):
        # Efficient check using precomputed APs
        # For each AP (a,b,c), if mask[a] and mask[b] and mask[c] are all True, invalid
        # We can vectorize this slightly by checking sums
        # But for N=256, a simple loop over APs is fast enough if we don't do it too often
        # Let's use a set-based check for clarity and speed in Python for final verification
        S = set(i for i in range(_N) if mask[i])
        # Check all pairs
        # Optimization: iterate through sorted elements
        sorted_S = sorted(S)
        for i in range(len(sorted_S)):
            a = sorted_S[i]
            for j in range(i+1, len(sorted_S)):
                b = sorted_S[j]
                c = 2*b - a
                if c in S and c > b:
                    return False
                # Note: if c < a, it would have been caught in a previous iteration 
                # where c was 'a' and a was 'c'? No, we only check c > b.
                # Since a < b, c = 2b-a > b. So we only need to check c > b.
        return True
    else:
        # Integer bitmask
        S = set()
        m = mask
        i = 0
        while m:
            if m & 1:
                S.add(i)
            m >>= 1
            i += 1
        # Same check as above
        sorted_S = sorted(S)
        for i in range(len(sorted_S)):
            a = sorted_S[i]
            for j in range(i+1, len(sorted_S)):
                b = sorted_S[j]
                c = 2*b - a
                if c in S and c > b:
                    return False
        return True

def _count_violations(mask):
    """
    Count the number of 3-APs in the set represented by mask.
    mask is a numpy array of bools.
    """
    count = 0
    # Precomputed APs are 0-indexed
    for a, b, c in _APS:
        if mask[a] and mask[b] and mask[c]:
            count += 1
    return count

def _repair(mask):
    """
    Greedy repair: remove elements that are part of the most APs to make the set valid.
    """
    mask = mask.copy()
    valid = _check_validity(mask)
    while not valid:
        # Find all APs present
        active_aps = []
        for a, b, c in _APS:
            if mask[a] and mask[b] and mask[c]:
                active_aps.append((a, b, c))
        
        if not active_aps:
            break # Should not happen if valid check is consistent
            
        # Count occurrences of each element in active APs
        counts = np.zeros(_N, dtype=int)
        for a, b, c in active_aps:
            counts[a] += 1
            counts[b] += 1
            counts[c] += 1
        
        # Remove the element with the highest count (tie-break by smallest weight? or random?)
        # To maximize weight, we should remove the element with the smallest weight (largest index) among those with max count?
        # Or simply remove the one with highest count. If tie, remove largest index (smallest weight) to preserve high weights.
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        
        # Among candidates, pick the one with the smallest weight (largest index)
        # Weights are 1/i, so larger i means smaller weight.
        # We want to keep high weights (small i). So we remove the candidate with the largest i.
        idx_to_remove = np.max(candidates)
        
        mask[idx_to_remove] = False
        valid = _check_validity(mask)
        
    return mask

def _simulated_annealing():
    """
    Simulated Annealing to maximize reciprocal sum subject to no 3-APs.
    We use a penalty method: Objective = Sum(1/i * x_i) - Penalty * (#APs).
    """
    # Precompute APs globally
    global _APS
    if not hasattr(_simulated_annealing, '_aps_cache'):
        _APS = _precompute_aps()
        _simulated_annealing._aps_cache = True
    else:
        _APS = _simulated_annealing._aps_cache
        
    # Initial state: Random subset or Greedy
    # Let's start with a greedy set to have a good starting point
    # Greedy: Add elements in order of decreasing weight (increasing i? No, 1/1 > 1/2. So i=1 is highest weight).
    # Wait, 1/1 is largest. So we try to add 1, then 2, etc.
    mask = np.zeros(_N, dtype=bool)
    
    # Greedy init
    for i in range(1, _N + 1):
        idx = i - 1
        # Check if adding i creates an AP
        valid_add = True
        # Check against existing elements
        # For each existing j, check if (i, j, k) or (j, i, k) or (k, i, j) forms AP
        # Simpler: Just check if adding i to current mask creates any AP
        # We can do this by checking all APs involving i
        # But iterating all APs is O(N^2). For N=256, it's fine.
        # Let's just use the violation count delta
        old_violations = _count_violations(mask)
        mask[idx] = True
        new_violations = _count_violations(mask)
        if new_violations > old_violations:
            mask[idx] = False
        # Note: This greedy might not be optimal but provides a valid start.
        
    current_mask = mask.copy()
    current_weight = np.sum(_RECIP[1:] * current_mask)
    
    best_mask = current_mask.copy()
    best_weight = current_weight
    
    # Annealing parameters
    T_start = 1.0
    T_end = 0.01
    alpha = 0.999 # Cooling rate
    iterations = 50000 # Max steps
    
    start_time = time.time()
    deadline = start_time + _KILL - _MARGIN
    
    T = T_start
    
    # Precompute weights array for faster access
    weights = np.array(_RECIP[1:])
    
    while T > T_end:
        if time.time() > deadline:
            break
            
        # Random move: Flip a random bit
        idx = np.random.randint(0, _N)
        old_val = current_mask[idx]
        
        # Calculate change in weight
        if old_val:
            delta_weight = -weights[idx]
        else:
            delta_weight = weights[idx]
            
        # Calculate change in violations
        # Instead of recounting all, we can compute delta_violations
        # But _count_violations is O(N^2) ~ 256^2/2 = 32k ops. 
        # 50k steps * 32k ops = 1.6e9 ops. Too slow in Python.
        
        # Optimization: Precompute for each index, the list of APs it belongs to.
        # Let's restructure for speed.
        
        # Actually, let's use a faster violation check.
        # Precompute APs as a list of (a,b,c).
        # To check delta violations for flipping idx:
        # If turning ON: Count how many APs containing idx have the other two ON.
        # If turning OFF: Count how many APs containing idx have the other two ON.
        
        # Let's build an index of APs per element
        if not hasattr(_simulated_annealing, '_aps_by_elem'):
            aps_by_elem = [[] for _ in range(_N)]
            for ap in _APS:
                for idx in ap:
                    aps_by_elem[idx].append(ap)
            _simulated_annealing._aps_by_elem = aps_by_elem
        else:
            aps_by_elem = _simulated_annealing._aps_by_elem
            
        delta_violations = 0
        if old_val:
            # Turning OFF
            # Violations are removed where the other two are ON
            for ap in aps_by_elem[idx]:
                # Find the other two indices
                others = [x for x in ap if x != idx]
                if current_mask[others[0]] and current_mask[others[1]]:
                    delta_violations -= 1 # Violation count decreases
        else:
            # Turning ON
            # Violations are added where the other two are ON
            for ap in aps_by_elem[idx]:
                others = [x for x in ap if x != idx]
                if current_mask[others[0]] and current_mask[others[1]]:
                    delta_violations += 1 # Violation count increases
                    
        # Objective: Maximize Weight - Penalty * Violations
        # Penalty should be large enough to discourage violations but allow them during search.
        # Max possible weight sum is approx ln(256) ~ 5.5.
        # Max violations is ~N^2/12 ~ 5000.
        # Let's use Penalty = 0.01 initially? No, if we want to converge to valid sets, penalty should be high.
        # But for SA, we want to explore. 
        # Let's use a dynamic penalty or a fixed large penalty.
        # If Penalty is large, the search will mostly stick to valid sets.
        # If Penalty is small, it might get stuck in high-violation high-weight sets.
        # We want to end with a valid set. The repair step handles validity.
        # So we can use a moderate penalty to guide towards valid sets during SA.
        Penalty = 0.05
        
        delta_obj = delta_weight - Penalty * delta_violations
        
        # Acceptance
        if delta_obj > 0:
            # Always accept improvement
            current_mask[idx] = not old_val
        else:
            # Accept with probability exp(delta_obj / T)
            # delta_obj is negative
            p = np.exp(delta_obj / T)
            if np.random.random() < p:
                current_mask[idx] = not old_val
                
        # Update best
        if current_mask[idx] != old_val:
            # Recalculate weight? We have delta_weight, but it's safer to track current_weight
            current_weight += delta_weight
            if current_weight > best_weight:
                # Check if valid? 
                # SA might produce invalid sets. We only track "best_weight" among all states.
                # But we must return a VALID set.
                # So we should only update best_mask if it's valid, OR we keep the best valid set separately.
                # Checking validity every step is expensive.
                # Strategy: Keep track of the best VALID set.
                # To do this efficiently, we can check validity only when we find a new high weight state.
                # Or, we can just run SA to find a good state, then repair it.
                # Let's update best_mask only if it's valid.
                # To avoid frequent validity checks, we can check only if current_weight > best_valid_weight + epsilon
                pass # We will handle best tracking differently
        
        T *= alpha
        
    # The above loop is a bit messy with best tracking. Let's refine.
    # We'll restart the logic in a cleaner function if needed, but let's stick to this structure.
    # Actually, the `pass` above means we didn't update best_mask. 
    # Let's modify the loop to track best valid set.
    
    # Reset for a cleaner SA run with proper best tracking
    current_mask = mask.copy()
    current_weight = np.sum(weights * current_mask)
    
    # Ensure initial is valid (it is from greedy)
    best_mask = current_mask.copy()
    best_weight = current_weight
    
    T = T_start
    steps = 0
    max_steps = 100000
    
    while T > T_end and steps < max_steps:
        if time.time() > deadline:
            break
            
        idx = np.random.randint(0, _N)
        old_val = current_mask[idx]
        
        # Delta weight
        if old_val:
            delta_weight = -weights[idx]
        else:
            delta_weight = weights[idx]
            
        # Delta violations
        delta_violations = 0
        for ap in aps_by_elem[idx]:
            others = [x for x in ap if x != idx]
            if current_mask[others[0]] and current_mask[others[1]]:
                if old_val:
                    delta_violations -= 1
                else:
                    delta_violations += 1
                    
        Penalty = 0.05
        delta_obj = delta_weight - Penalty * delta_violations
        
        if delta_obj > 0 or np.random.random() < np.exp(delta_obj / T):
            current_mask[idx] = not old_val
            current_weight += delta_weight
            
            # Check if this new state is valid and better
            # Checking validity is O(N^2) in worst case, but usually fast if few violations.
            # However, doing it every step is too slow.
            # We can check validity only if current_weight > best_weight.
            if current_weight > best_weight:
                if _check_validity(current_mask):
                    best_mask = current_mask.copy()
                    best_weight = current_weight
                    # Optional: If we found a valid set, we can restart or continue
                    # Continuing is fine.
                    
        T *= alpha
        steps += 1
        
    return best_mask

def _local_search(mask):
    """
    1-swap local search on a valid mask.
    Try to swap one element for another to increase weight while maintaining validity.
    """
    mask = mask.copy()
    weights = np.array(_RECIP[1:])
    improved = True
    start_time = time.time()
    deadline = start_time + _KILL - _MARGIN
    
    # Precompute aps_by_elem for speed
    if not hasattr(_simulated_annealing, '_aps_by_elem'):
        aps_by_elem = [[] for _ in range(_N)]
        for ap in _APS:
            for idx in ap:
                aps_by_elem[idx].append(ap)
        _simulated_annealing._aps_by_elem = aps_by_elem
    else:
        aps_by_elem = _simulated_annealing._aps_by_elem
        
    while improved and time.time() < deadline:
        improved = False
        best_gain = 0
        best_swap = None
        
        # Try all 1-swaps: remove i, add j
        # To speed up, we can iterate over i in S and j not in S
        # N=256, so 256*256 = 65k swaps. Each swap check is O(deg(i)+deg(j)) ~ O(N).
        # Total O(N^3) ~ 16M ops per pass. Might be too slow in Python if many passes.
        # Let's limit the search or optimize.
        
        # Optimization: Only consider swaps where weight[j] > weight[i]
        # i.e., j < i (since 1/j > 1/i if j < i)
        
        S_indices = np.where(mask)[0]
        NotS_indices = np.where(~mask)[0]
        
        for i in S_indices:
            # If i is large (small weight), it's a good candidate to remove
            # Try to replace with smaller j (larger weight)
            for j in NotS_indices:
                if j >= i: # weight[j] <= weight[i], no gain
                    continue
                    
                # Check if S - {i} + {j} is valid
                # This is equivalent to checking if adding j to S\{i} creates no APs
                # And removing i doesn't break anything (it can't, since S was valid)
                
                # Efficient check:
                # 1. Temporarily remove i
                mask[i] = False
                # 2. Check if j can be added
                can_add = True
                for ap in aps_by_elem[j]:
                    others = [x for x in ap if x != j]
                    if mask[others[0]] and mask[others[1]]:
                        can_add = False
                        break
                # 3. Restore i
                mask[i] = True
                
                if can_add:
                    gain = weights[j] - weights[i]
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, j)
                        
        if best_swap:
            i, j = best_swap
            mask[i] = False
            mask[j] = True
            improved = True
            
    return mask

def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    
    # 1. Greedy Initialization
    mask = np.zeros(_N, dtype=bool)
    # Precompute APs for greedy
    if not hasattr(_simulated_annealing, '_aps_cache'):
        _APS = _precompute_aps()
        _simulated_annealing._aps_cache = True
    else:
        _APS = _simulated_annealing._aps_cache
        
    if not hasattr(_simulated_annealing, '_aps_by_elem'):
        aps_by_elem = [[] for _ in range(_N)]
        for ap in _APS:
            for idx in ap:
                aps_by_elem[idx].append(ap)
        _simulated_annealing._aps_by_elem = aps_by_elem
    else:
        aps_by_elem = _simulated_annealing._aps_by_elem
        
    weights = np.array(_RECIP[1:])
    
    # Greedy: Add in order of decreasing weight (increasing index? No, 1/1 is max. So index 0 is max weight).
    # So we try to add 0, 1, 2...
    for i in range(_N):
        # Check if adding i is valid
        can_add = True
        for ap in aps_by_elem[i]:
            others = [x for x in ap if x != i]
            if mask[others[0]] and mask[others[1]]:
                can_add = False
                break
        if can_add:
            mask[i] = True
            
    # 2. Simulated Annealing
    if time.time() < deadline:
        sa_mask = _simulated_annealing()
        # Repair SA result to ensure validity
        sa_mask = _repair(sa_mask)
        
        # Compare with greedy
        w_greedy = np.sum(weights * mask)
        w_sa = np.sum(weights * sa_mask)
        
        if w_sa > w_greedy:
            mask = sa_mask
            
    # 3. Local Search (1-swap)
    if time.time() < deadline:
        mask = _local_search(mask)
        
    # 4. Final Verification and Conversion
    if not _check_validity(mask):
        mask = _repair(mask)
        
    # Ensure non-empty
    if not np.any(mask):
        mask[0] = True
        
    # Convert to list of 1-based indices
    sol = [i + 1 for i in range(_N) if mask[i]]
    
    return sol

# Global cache initialization for _APS
_APS = _precompute_aps()
_simulated_annealing._aps_cache = True
_aps_by_elem = [[] for _ in range(_N)]
for ap in _APS:
    for idx in ap:
        _aps_by_elem[idx].append(ap)
_simulated_annealing._aps_by_elem = _aps_by_elem

# Expose for _local_search
# Note: _local_search uses _simulated_annealing._aps_by_elem
# We should ensure it's accessible.
# In _local_search, I used `if not hasattr(_simulated_annealing, '_aps_by_elem')`.
# Since we set it globally, it should be fine.

# One more thing: _count_violations in _simulated_annealing is not used in the final loop, 
# but _check_validity is.
# _check_validity uses a set-based approach which is slower than bitmask+precomputed APs.
# Let's optimize _check_validity to use precomputed APs for speed.

def _check_validity_fast(mask):
    """
    Fast validity check using precomputed APs.
    mask is a numpy array of bools.
    """
    # For each AP, if all 3 are set, return False
    # Vectorized: 
    # We can't easily vectorize the "all 3" check without building a huge matrix.
    # But N=256, num_aps ~ 5000.
    # Looping 5000 times in Python is fast (~ms).
    for a, b, c in _APS:
        if mask[a] and mask[b] and mask[c]:
            return False
    return True

# Replace _check_validity usage in _repair and _local_search with _check_validity_fast if mask is ndarray
# In _repair:
def _repair_fast(mask):
    mask = mask.copy()
    while not _check_validity_fast(mask):
        # Find active APs
        active_aps = []
        for a, b, c in _APS:
            if mask[a] and mask[b] and mask[c]:
                active_aps.append((a, b, c))
        
        if not active_aps:
            break
            
        counts = np.zeros(_N, dtype=int)
        for a, b, c in active_aps:
            counts[a] += 1
            counts[b] += 1
            counts[c] += 1
            
        max_count = np.max(counts)
        candidates = np.where(counts == max_count)[0]
        idx_to_remove = np.max(candidates) # Remove largest index (smallest weight)
        
        mask[idx_to_remove] = False
        
    return mask

# Update construct to use fast checks
def construct():
    start = time.time()
    deadline = start + _KILL - _MARGIN
    
    # 1. Greedy Initialization
    mask = np.zeros(_N, dtype=bool)
    weights = np.array(_RECIP[1:])
    
    for i in range(_N):
        can_add = True
        for ap in _aps_by_elem[i]:
            others = [x for x in ap if x != i]
            if mask[others[0]] and mask[others[1]]:
                can_add = False
                break
        if can_add:
            mask[i] = True
            
    # 2. Simulated Annealing
    if time.time() < deadline:
        # Use a wrapper that uses fast validity check
        # The SA function uses _check_validity which is slow.
        # Let's modify SA to use _check_validity_fast.
        
        # We'll just call a modified SA logic here or patch the function.
        # For simplicity, I'll reimplement the SA loop here with fast checks to avoid function overhead/attribute issues.
        
        current_mask = mask.copy()
        current_weight = np.sum(weights * current_mask)
        best_mask = current_mask.copy()
        best_weight = current_weight
        
        T = 1.0
        T_end = 0.01
        alpha = 0.999
        Penalty = 0.05
        
        steps = 0
        max_steps = 100000
        
        while T > T_end and steps < max_steps:
            if time.time() > deadline:
                break
                
            idx = np.random.randint(0, _N)
            old_val = current_mask[idx]
            
            if old_val:
                delta_weight = -weights[idx]
            else:
                delta_weight = weights[idx]
                
            delta_violations = 0
            for ap in _aps_by_elem[idx]:
                others = [x for x in ap if x != idx]
                if current_mask[others[0]] and current_mask[others[1]]:
                    if old_val:
                        delta_violations -= 1
                    else:
                        delta_violations += 1
                        
            delta_obj = delta_weight - Penalty * delta_violations
            
            if delta_obj > 0 or np.random.random() < np.exp(delta_obj / T):
                current_mask[idx] = not old_val
                current_weight += delta_weight
                
                if current_weight > best_weight:
                    if _check_validity_fast(current_mask):
                        best_mask = current_mask.copy()
                        best_weight = current_weight
                        
            T *= alpha
            steps += 1
            
        if best_weight > np.sum(weights * mask):
            mask = best_mask
            
    # 3. Local Search
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_gain = 0
            best_swap = None
            
            S_indices = np.where(mask)[0]
            NotS_indices = np.where(~mask)[0]
            
            for i in S_indices:
                for j in NotS_indices:
                    if j >= i:
                        continue
                        
                    mask[i] = False
                    can_add = True
                    for ap in _aps_by_elem[j]:
                        others = [x for x in ap if x != j]
                        if mask[others[0]] and mask[others[1]]:
                            can_add = False
                            break
                    mask[i] = True
                    
                    if can_add:
                        gain = weights[j] - weights[i]
                        if gain > best_gain:
                            best_gain = gain
                            best_swap = (i, j)
                            
            if best_swap:
                i, j = best_swap
                mask[i] = False
                mask[j] = True
                improved = True
                
    # 4. Final Check
    if not _check_validity_fast(mask):
        mask = _repair_fast(mask)
        
    if not np.any(mask):
        mask[0] = True
        
    sol = [i + 1 for i in range(_N) if mask[i]]
    return sol
