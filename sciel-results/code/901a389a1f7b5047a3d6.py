import time
import random
import sys
import math

_N = 128
_LO = 128
_HI = 256
_RANGE = _HI - _LO + 1

def _get_counts(A_set):
    """
    Returns a list of representation counts for sums in [_LO, _HI].
    Returns None if A is empty.
    """
    if not A_set:
        return None
    A_sorted = sorted(A_set)
    n = len(A_sorted)
    counts = [0] * _RANGE
    
    for i in range(n):
        a = A_sorted[i]
        for j in range(i, n):
            b = A_sorted[j]
            s = a + b
            if _LO <= s <= _HI:
                counts[s - _LO] += 1
    return counts

def _is_valid(counts):
    """Check if all sums in the range are covered."""
    if counts is None:
        return False
    return all(c >= 1 for c in counts)

def _get_peak(counts):
    """Get the maximum representation count."""
    if counts is None:
        return float('inf')
    return max(counts)

class SAState:
    def __init__(self, A_set):
        self.A = set(A_set)
        self.in_A = {x: True for x in self.A}
        self.counts = _get_counts(self.A)
        self.peak = _get_peak(self.counts)
        self.valid = _is_valid(self.counts)
        
    def apply_swap(self, x_out, x_in):
        """
        Swaps x_out (in A) with x_in (not in A).
        Returns the change in peak (new_peak - old_peak) or None if invalid.
        Modifies state in place.
        """
        if x_out not in self.A or x_in in self.A:
            return None
            
        # Snapshot old state for rollback
        old_A = self.A.copy()
        old_counts = self.counts.copy()
        old_peak = self.peak
        old_valid = self.valid
        
        # Perform swap
        self.A.discard(x_out)
        self.A.add(x_in)
        self.in_A[x_out] = False
        self.in_A[x_in] = True
        
        # Update counts incrementally
        # Pairs involving x_out are removed
        # Pairs involving x_in are added
        # Note: We must be careful with (x_out, x_out) and (x_in, x_in) if they existed, 
        # but x_out is removed so (x_out, x_out) is gone.
        # x_in is added so (x_in, x_in) is new.
        # Cross terms: (x_out, z) for z in A\{x_out} and (x_in, z) for z in A\{x_in}
        
        # It is safer and fast enough to just recompute if |A| is small, 
        # but let's try incremental for speed.
        # Actually, for |A| ~ 60, O(|A|^2) is ~3600 ops. 
        # Incremental is O(|A|) ~ 60 ops. Huge win.
        
        # Remove contributions of x_out
        for z in self.A: # z is in new A, so z != x_out
            s1 = x_out + z
            if _LO <= s1 <= _HI:
                self.counts[s1 - _LO] -= 1
        # Remove (x_out, x_out) if it was in range? 
        # x_out was in old A. 
        s_self = 2 * x_out
        if _LO <= s_self <= _HI:
            self.counts[s_self - _LO] -= 1
            
        # Add contributions of x_in
        for z in self.A: # z is in new A, so z != x_in
            s2 = x_in + z
            if _LO <= s2 <= _HI:
                self.counts[s2 - _LO] += 1
        # Add (x_in, x_in)
        s_self_in = 2 * x_in
        if _LO <= s_self_in <= _HI:
            self.counts[s_self_in - _LO] += 1
            
        # Check validity and peak
        new_valid = all(c >= 1 for c in self.counts)
        new_peak = max(self.counts)
        
        if not new_valid:
            # Rollback
            self.A = old_A
            self.in_A = {x: True for x in self.A}
            self.counts = old_counts
            self.peak = old_peak
            self.valid = old_valid
            return None
            
        self.peak = new_peak
        self.valid = new_valid
        
        return new_peak - old_peak

def _simulated_annealing(A_init, time_limit, rng):
    """
    Perform Simulated Annealing to minimize peak multiplicity.
    """
    state = SAState(A_init)
    if not state.valid:
        return None, float('inf')
        
    best_A = state.A.copy()
    best_peak = state.peak
    
    T = 5.0 # Initial temperature
    T_min = 0.01
    alpha = 0.995 # Cooling rate
    
    start_time = time.time()
    
    while time.time() - start_time < time_limit and T > T_min:
        # Generate a random swap
        if len(state.A) < 2:
            break
        in_list = list(state.A)
        not_in_list = [x for x in range(_N + 1) if x not in state.A]
        
        if not not_in_list:
            break
            
        x_out = rng.choice(in_list)
        x_in = rng.choice(not_in_list)
        
        delta = state.apply_swap(x_out, x_in)
        
        if delta is not None:
            # Valid move
            if delta <= 0:
                # Improvement or equal
                if state.peak < best_peak:
                    best_peak = state.peak
                    best_A = state.A.copy()
            else:
                # Worse move, accept with probability
                # Using delta_peak as energy difference
                prob = math.exp(-delta / T)
                if rng.random() < prob:
                    pass # Accepted
                else:
                    # Rollback
                    state.apply_swap(x_in, x_out) # Swap back
            
        T *= alpha
        
    return best_A, best_peak

def construct():
    start_time = time.time()
    DEADLINE = start_time + 85 # 5s margin
    
    rng = random.Random(42)
    
    best_A_global = None
    best_peak_global = float('inf')
    
    # 1. Initial Heuristic: Greedy Peel from Full Interval
    # This provides a good starting point with low peak.
    A_full = set(range(0, _N + 1))
    A_curr = set(A_full)
    counts = _get_counts(A_curr)
    
    # Simple greedy peel to reduce size/peak quickly
    improved = True
    while improved and time.time() < DEADLINE - 10:
        improved = False
        A_list = sorted(A_curr)
        for x in A_list:
            if len(A_curr) <= 2: break
            new_A = A_curr.copy()
            new_A.discard(x)
            new_counts = _get_counts(new_A)
            if _is_valid(new_counts):
                new_peak = _get_peak(new_counts)
                curr_peak = _get_peak(counts)
                if new_peak <= curr_peak:
                    A_curr = new_A
                    counts = new_counts
                    improved = True
                    break
        # If no single removal works, we are stuck for this phase.
        
    if _is_valid(counts):
        best_A_global = A_curr.copy()
        best_peak_global = _get_peak(counts)
    else:
        # Fallback to full set if peel broke it (shouldn't happen with check)
        best_A_global = set(A_full)
        best_peak_global = _get_peak(_get_counts(best_A_global))

    # 2. Simulated Annealing Refinement
    # Run SA starting from the best found so far.
    # We can run multiple restarts or one long run.
    # Given the time budget, one long SA run from a good start is usually effective.
    
    remaining_time = DEADLINE - time.time() - 2
    if remaining_time > 5:
        A_sa, peak_sa = _simulated_annealing(best_A_global, remaining_time, rng)
        if A_sa is not None and peak_sa < best_peak_global:
            best_peak_global = peak_sa
            best_A_global = A_sa

    # 3. If we have time left, try another SA restart from a slightly perturbed best
    if time.time() < DEADLINE - 5:
        A_perturbed = best_A_global.copy()
        # Perturb: swap 2-3 random elements
        for _ in range(3):
            in_list = list(A_perturbed)
            not_in_list = [x for x in range(_N + 1) if x not in A_perturbed]
            if not in_list or not not_in_list: break
            x_out = rng.choice(in_list)
            x_in = rng.choice(not_in_list)
            A_perturbed.discard(x_out)
            A_perturbed.add(x_in)
            
        # Check validity, if invalid, skip or repair
        counts_p = _get_counts(A_perturbed)
        if _is_valid(counts_p):
            A_sa2, peak_sa2 = _simulated_annealing(A_perturbed, DEADLINE - time.time() - 1, rng)
            if A_sa2 is not None and peak_sa2 < best_peak_global:
                best_peak_global = peak_sa2
                best_A_global = A_sa2

    if best_A_global is None:
        return list(range(64, 129))
        
    return sorted(best_A_global)
