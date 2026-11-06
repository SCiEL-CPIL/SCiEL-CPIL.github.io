"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    # Generate all 2^18 Boolean inputs as a (262144, 18) uint8 array
    all_inputs = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        all_inputs[:, i] = (np.arange(1 << N) >> i) & 1
    
    # State: each wire's value for each input
    # state[input_idx, wire_idx] = value of wire wire_idx for input input_idx
    state = all_inputs.copy()
    
    # Track which inputs are not yet sorted (wire 0 has min, wire N-1 has max, etc.)
    # An input is "sorted" if state[input] is non-decreasing across wires
    # We'll track violated pairs: for each input, how many adjacent wire pairs are out of order
    
    def count_violations(s):
        """Count total inversions across all inputs (adjacent wire pairs out of order)."""
        # For each input, count pairs (i, j) where i < j but s[input, i] > s[input, j]
        # For binary values, this means s[input, i] == 1 and s[input, j] == 0
        # We can vectorize: for each pair (i,j), count inputs where s[:,i]==1 and s[:,j]==0
        total = 0
        for i in range(N):
            for j in range(i+1, N):
                total += np.sum((s[:, i] == 1) & (s[:, j] == 0))
        return total
    
    def apply_comparator(s, i, j):
        """Apply compare-exchange on wires i and j. Returns new state."""
        new_s = s.copy()
        # min goes to wire i, max goes to wire j
        new_s[:, i] = np.minimum(s[:, i], s[:, j])
        new_s[:, j] = np.maximum(s[:, i], s[:, j])
        return new_s
    
    def is_sorted(s):
        """Check if all inputs are sorted (non-decreasing across wires)."""
        # For binary, sorted means all 0s come before all 1s on each input
        # i.e., no 1 appears before a 0
        for i in range(N - 1):
            if np.any((s[:, i] == 1) & (s[:, i+1] == 0)):
                return False
        return True
    
    # Greedy search: repeatedly pick the comparator that most reduces total inversions
    network = []
    current_state = state.copy()
    
    # Precompute all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Use a faster violation count: only check adjacent pairs initially for speed,
    # but we need full inversion count for proper greedy. Let's use a hybrid.
    
    # For efficiency, we'll compute the "disorder" as sum over all pairs (i,j) of 
    # number of inputs where wire i has 1 and wire j has 0 (i<j).
    # This is the total number of inversions across all inputs.
    
    while True:
        if time.time() > DEADLINE:
            break
        
        if is_sorted(current_state):
            break
        
        # Find the comparator that maximally reduces total inversions
        best_gain = -1
        best_comp = None
        
        # For each candidate comparator (i, j), compute how many inversions it fixes
        # A comparator (i,j) with i<j: 
        #   - Fixes inversions where input has wire_i=1, wire_j=0 (this pair becomes ordered)
        #   - But may create new inversions: for k < i, if wire_k=0 and wire_i becomes 0 (was 1), 
        #     no new inversion. If wire_i becomes 1 (was 0), and wire_k=0, still fine.
        #   Actually, let's think more carefully.
        #   
        # After compare-exchange on (i,j):
        #   new_wire_i = min(old_i, old_j), new_wire_j = max(old_i, old_j)
        #   
        # For binary: if old_i=0, old_j=0: no change
        #             if old_i=1, old_j=1: no change  
        #             if old_i=0, old_j=1: no change (already ordered)
        #             if old_i=1, old_j=0: swap -> new_i=0, new_j=1
        #
        # So the comparator only affects inputs where old_i=1 and old_j=0.
        # For those inputs:
        #   - The pair (i,j) inversion is fixed: gain +1 per such input
        #   - For k < i: old_k vs new_i=0. If old_k=1, that's an inversion (1,0). Before: old_k vs old_i=1. 
        #     If old_k=1, no inversion before (1,1). After: inversion (1,0). So we ADD inversions.
        #     If old_k=0, no inversion before or after (0,1) or (0,0).
        #   - For k > j: new_j=1 vs old_k. If old_k=0, inversion (1,0). Before: old_j=0 vs old_k=0, no inversion.
        #     So we ADD inversions. If old_k=1, no inversion before or after.
        #   - For i < k < j: new_i=0 vs old_k, and old_k vs new_j=1.
        #     Before: old_i=1 vs old_k, old_k vs old_j=0.
        #     After: 0 vs old_k, old_k vs 1.
        #     If old_k=0: before (1,0) inversion + (0,0) no inv = 1 inv. After (0,0) no inv + (0,1) no inv = 0 inv. Gain +1.
        #     If old_k=1: before (1,1) no inv + (1,0) inv = 1 inv. After (0,1) no inv + (1,1) no inv = 0 inv. Gain +1.
        #     So for i<k<j, always gain +1 per such input.
        #
        # Let me recompute total gain for inputs where old_i=1, old_j=0:
        #   Fixed: pair (i,j) itself: +1
        #   For k in (i,j): +1 each (as shown above)
        #   For k < i: if old_k=1, we create inversion (1,0) at (k,i): -1 each
        #   For k > j: if old_k=0, we create inversion (1,0) at (j,k): -1 each
        #
        # Net gain = 1 + (j-i-1) - (count of k<i with old_k=1) - (count of k>j with old_k=0)
        
        for (i, j) in all_comparators:
            # Only consider inputs where old_i=1 and old_j=0
            mask = (current_state[:, i] == 1) & (current_state[:, j] == 0)
            n_affected = np.sum(mask)
            if n_affected == 0:
                continue
            
            # Base gain: fix (i,j) pair + fix all intermediate pairs
            base_gain = 1 + (j - i - 1)  # per affected input
            
            # Subtract: for k < i, count of old_k=1 among affected inputs
            penalty_left = 0
            for k in range(i):
                penalty_left += np.sum(mask & (current_state[:, k] == 1))
            
            # Subtract: for k > j, count of old_k=0 among affected inputs
            penalty_right = 0
            for k in range(j+1, N):
                penalty_right += np.sum(mask & (current_state[:, k] == 0))
            
            total_gain = n_affected * base_gain - penalty_left - penalty_right
            
            if total_gain > best_gain:
                best_gain = total_gain
                best_comp = (i, j)
        
        if best_comp is None or best_gain <= 0:
            # No progress possible with greedy; break
            break
        
        # Apply the best comparator
        i, j = best_comp
        current_state = apply_comparator(current_state, i, j)
        network.append((i, j))
    
    # If we ran out of time or got stuck, return what we have
    # But we need a valid network! Let's check if current state is sorted.
    # If not, we should return the seed network to be safe.
    
    if not is_sorted(current_state):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return network
# EVOLVE-BLOCK-END
