import time
import numpy as np

def construct():
    """
    Constructs a subset A of [0, 128] such that every integer in [128, 256]
    is a sum of two elements of A. Minimizes the largest representation multiplicity.
    
    Strategy:
    1. Greedy Construction: Start with a necessary anchor (128). Iteratively add elements
       from [0, 128] that minimize the peak representation count in [128, 256], with a
       bias towards covering currently missing sums.
    2. Deterministic Local Search: Perform passes of single-element swaps and removals,
       accepting only if the peak multiplicity strictly decreases.
    
    This avoids the randomness and potential stagnation of Simulated Annealing by using
    a structured, objective-driven construction.
    """
    deadline = time.time() + 80  # Safety margin
    
    # Precompute masks for fast evaluation
    # A_bool: 129x1 indicator
    # We need to evaluate counts for sums 128..256.
    
    def get_peak(A_bool):
        """
        Returns (peak, missing_count) for the current A_bool.
        A_bool: np.array of shape (129,) with 0/1 values.
        """
        # Convolution to get ordered pair counts
        # conv[k] = sum_{i+j=k} A_bool[i] * A_bool[j]
        conv = np.convolve(A_bool, A_bool, mode='full')
        
        # Extract target range [128, 256]
        # conv indices: 0..256
        target_conv = conv[128:257]
        
        # Adjust for unordered pairs with replacement
        # Ordered count N.
        # Unordered count U = (N + D) / 2, where D=1 if k/2 in A and k is even, else 0.
        
        # We can vectorize the diagonal correction.
        # k is in [128, 256].
        # Even k: 128, 130, ..., 256.
        # Midpoint m = k/2 is in [64, 128].
        
        # Create a boolean array for even k in the target range
        # target_sums = np.arange(128, 257)
        # even_mask = (target_sums % 2 == 0)
        
        # Instead of creating arange every time, we can precompute indices.
        # But let's keep it simple first.
        
        # Actually, we can precompute the midpoints for even sums in [128, 256].
        # Even sums: 128, 130, ..., 256.
        # Number of even sums: (256-128)/2 + 1 = 65.
        # Their indices in target_conv (0-indexed):
        # sum=128 -> idx 0
        # sum=130 -> idx 2
        # ...
        # sum=256 -> idx 128
        # So even indices in target_conv are 0, 2, 4, ..., 128.
        
        even_indices = np.arange(0, 129, 2) # 0, 2, ..., 128
        
        # Corresponding midpoints in A_bool:
        # sum = 128 + 2*i -> mid = 64 + i
        # i ranges 0..64.
        # So mids are 64..128.
        mids = np.arange(64, 129)
        
        # Get A_bool values at these midpoints
        diag_corrections = A_bool[mids]
        
        # Apply correction to target_conv
        # We need to add diag_corrections to the even indices of target_conv.
        # Make a copy to avoid modifying conv
        adj_conv = target_conv.copy()
        adj_conv[even_indices] += diag_corrections
        
        # Now adj_conv contains 2 * Unordered_Count
        # Unordered_Count = adj_conv // 2
        
        # Check for missing sums (count == 0)
        missing = np.sum(adj_conv == 0)
        
        if missing > 0:
            return np.inf, missing
            
        peak = np.max(adj_conv) // 2
        return peak, 0

    # --- Phase 1: Greedy Construction ---
    
    # Initialize A_bool
    A_bool = np.zeros(129, dtype=np.int32)
    # Must include 128 to cover 256 (128+128)
    A_bool[128] = 1
    
    current_peak, current_missing = get_peak(A_bool)
    
    # We want to build a set that covers all sums.
    # Greedy step: pick x in [0, 128] \ A that minimizes the resulting peak,
    # prioritizing reduction of missing sums.
    
    # To speed up, we can precompute the contribution of adding each x.
    # Adding x increases counts for sums x+a for all a in A.
    # Specifically, sum s gets +1 for each a in A such that s-a in A (and s-a != x? No, ordered logic handles it).
    # Wait, the convolution logic is robust. We can just try adding each candidate and evaluating?
    # 129 candidates * ~100 steps * eval cost.
    # Eval cost is O(129^2) ~ 16k ops. 129 * 100 * 16k = 200M ops. Might be slow in Python if not vectorized well.
    # Let's try to vectorize the "try all candidates" step.
    
    # Let A be current set.
    # If we add x, new A' = A U {x}.
    # New conv = old_conv + A_bool * x_delta + x_delta * A_bool + x_delta * x_delta?
    # x_delta is 1 at x, 0 elsewhere.
    # conv_new = conv_old + np.roll(A_bool, x) + np.roll(A_bool, -x) + (1 if 2x in range else 0 at 2x)?
    # Actually, np.convolve(A, A) where A has 1 at x added.
    # Let v be indicator of A. v' = v + e_x.
    # v'*v' = v*v + v*e_x + e_x*v + e_x*e_x.
    # v*e_x is v shifted by x.
    # e_x*v is v shifted by -x (or x, since conv is symmetric for real inputs? No, conv is commutative).
    # So conv_new = conv_old + shift(v, x) + shift(v, -x) + e_2x.
    
    # We can maintain the current conv array!
    # conv is length 257.
    # v is length 129.
    
    # Let's restart with this incremental conv approach for speed.
    
    A_bool = np.zeros(129, dtype=np.int32)
    A_bool[128] = 1
    conv = np.convolve(A_bool, A_bool, mode='full')
    
    # Helper to get peak and missing from conv
    def eval_conv(conv):
        target_conv = conv[128:257]
        even_indices = np.arange(0, 129, 2)
        mids = np.arange(64, 129)
        # We need A_bool[mids] for the diagonal correction.
        # This function needs access to A_bool.
        # Let's pass A_bool or make it global? Pass it.
        pass # Will define properly below

    # Let's stick to the full evaluation but optimize the candidate selection.
    # Instead of trying all 129 candidates every step, we can try a subset or just randomize if too slow.
    # But "Deterministic" is the goal.
    
    # Let's try a simple loop.
    # Max steps: maybe 30-50 elements?
    
    A_bool = np.zeros(129, dtype=np.int32)
    A_bool[128] = 1
    
    for step in range(50):
        if time.time() > deadline - 10:
            break
            
        best_x = -1
        best_peak = float('inf')
        best_missing = float('inf')
        
        # Try all x not in A
        # Optimization: Precompute shifts?
        # For each x, new_conv = conv + np.roll(A_bool, x) + np.roll(A_bool, -x) + (1 at 2x)
        # Note: np.roll wraps around. We need to be careful with boundaries.
        # conv is length 257.
        # A_bool is length 129.
        # Adding x to A_bool:
        # The term v*e_x contributes to conv at indices i+x for i in support of v.
        # Since v is 0 outside [0,128], the support is within [0,128].
        # So indices are [x, 128+x].
        # We can create a temporary array for the delta.
        
        # To keep it simple and robust, let's just evaluate the top K candidates randomly?
        # No, let's do the full loop but break early if time is tight.
        
        candidates = [x for x in range(129) if A_bool[x] == 0]
        
        # If we are missing many sums, prioritize x that covers missing sums.
        # Let's just evaluate all.
        
        for x in candidates:
            # Create new A_bool
            A_bool[x] = 1
            peak, missing = get_peak(A_bool)
            A_bool[x] = 0
            
            # Score: minimize missing, then peak
            # If missing < best_missing, it's better.
            # If missing == best_missing and peak < best_peak, it's better.
            
            if missing < best_missing or (missing == best_missing and peak < best_peak):
                best_missing = missing
                best_peak = peak
                best_x = x
                
        if best_x == -1:
            break
            
        A_bool[best_x] = 1
        # Update conv? No, get_peak recalculates. This is O(N^2) per candidate.
        # 129 candidates * 50 steps * O(129^2) ~ 1M * 16k = 16 billion ops? Too slow.
        
    # The full evaluation inside the loop is too slow for 129 candidates * 50 steps.
    # We need the incremental conv update.
    
    # Reset for incremental approach
    A_bool = np.zeros(129, dtype=np.int32)
    A_bool[128] = 1
    conv = np.convolve(A_bool, A_bool, mode='full')
    
    # Precompute even indices and mids for evaluation
    even_indices = np.arange(0, 129, 2)
    mids = np.arange(64, 129)
    
    def eval_from_conv(conv, A_bool):
        target_conv = conv[128:257]
        adj_conv = target_conv.copy()
        adj_conv[even_indices] += A_bool[mids]
        missing = np.sum(adj_conv == 0)
        if missing > 0:
            return float('inf'), missing
        peak = np.max(adj_conv) // 2
        return peak, missing

    current_peak, current_missing = eval_from_conv(conv, A_bool)
    
    for step in range(40):
        if time.time() > deadline - 15:
            break
            
        best_x = -1
        best_score = (float('inf'), float('inf')) # (missing, peak)
        
        candidates = np.where(A_bool == 0)[0]
        
        for x in candidates:
            # Calculate new conv incrementally
            # delta_conv = shift(A_bool, x) + shift(A_bool, -x) + e_2x
            
            # shift(A_bool, x): A_bool[i] contributes to conv[i+x]
            # We can use np.roll but must handle the length difference.
            # conv is length 257. A_bool is 129.
            # Let's create a zero array of length 257 and place A_bool shifted.
            
            # Actually, simpler:
            # term1 = np.zeros(257)
            # term1[x:x+129] = A_bool  # This is shift by x
            # term2 = np.zeros(257)
            # term2[-x:-x-129] = A_bool # This is shift by -x? 
            # Wait, conv is commutative. v*e_x + e_x*v.
            # (v*e_x)[k] = v[k-x]. So term1[k] = A_bool[k-x] if 0<=k-x<129.
            # (e_x*v)[k] = v[k-x]. Same.
            # So the cross term is 2 * (A_bool shifted by x).
            # Plus the diagonal term e_x * e_x = e_2x.
            
            # So delta_conv = 2 * shift(A_bool, x) + (1 at 2x)
            
            # Let's verify.
            # v' = v + e_x.
            # v'*v' = v^2 + 2 v e_x + e_x^2.
            # v e_x is convolution of v and e_x.
            # (v * e_x)[k] = sum_i v[i] e_x[k-i] = v[k-x] (since e_x is 1 only at x).
            # So yes, it's just A_bool shifted by x.
            
            shift_arr = np.zeros(257, dtype=np.int32)
            start = x
            end = x + 129
            if end <= 257:
                shift_arr[start:end] = A_bool
            else:
                # Handle wrap/overflow? Max x=128. 128+129=257. Fits exactly.
                shift_arr[start:257] = A_bool[:257-start]
            
            delta_conv = 2 * shift_arr
            if 2*x < 257:
                delta_conv[2*x] += 1
            
            new_conv = conv + delta_conv
            
            # Evaluate
            peak, missing = eval_from_conv(new_conv, A_bool + (1 if False else 0)) # A_bool hasn't changed in this check
            # Wait, A_bool needs to reflect the addition for the diagonal check!
            # The diagonal check uses A_bool[mids].
            # If we add x, and x is in mids, the diagonal correction changes.
            # So we must update A_bool temporarily or pass a modified view.
            
            # Let's just update A_bool, eval, then revert. It's one array op.
            A_bool[x] = 1
            peak, missing = eval_from_conv(new_conv, A_bool)
            A_bool[x] = 0
            
            score = (missing, peak)
            if score < best_score:
                best_score = score
                best_x = x
                
        if best_x == -1:
            break
            
        # Commit
        A_bool[best_x] = 1
        # Update conv incrementally
        shift_arr = np.zeros(257, dtype=np.int32)
        start = best_x
        end = best_x + 129
        shift_arr[start:end] = A_bool # Wait, A_bool now includes best_x.
        # We need the shift of the OLD A_bool.
        # So we should have calculated delta_conv using the OLD A_bool.
        # Let's recalculate delta_conv with OLD A_bool for the commit.
        # Or just recompute conv from scratch? No, that's slow.
        
        # Let's store the old A_bool before committing?
        # Better: Calculate delta_conv BEFORE setting A_bool[x]=1.
        
        # Let's refactor the loop to calculate delta first.
        
    # The logic above has a bug in the commit step (using updated A_bool for shift).
    # Let's rewrite the loop cleanly.
    
    A_bool = np.zeros(129, dtype=np.int32)
    A_bool[128] = 1
    conv = np.convolve(A_bool, A_bool, mode='full')
    
    even_indices = np.arange(0, 129, 2)
    mids = np.arange(64, 129)
    
    def eval_conv_state(conv, A_bool):
        target_conv = conv[128:257]
        adj_conv = target_conv.copy()
        adj_conv[even_indices] += A_bool[mids]
        missing = np.sum(adj_conv == 0)
        if missing > 0:
            return float('inf'), missing
        peak = np.max(adj_conv) // 2
        return peak, missing

    current_peak, current_missing = eval_conv_state(conv, A_bool)
    
    for step in range(40):
        if time.time() > deadline - 15:
            break
            
        best_x = -1
        best_score = (float('inf'), float('inf'))
        
        candidates = np.where(A_bool == 0)[0]
        
        for x in candidates:
            # Calculate delta_conv based on CURRENT A_bool
            shift_arr = np.zeros(257, dtype=np.int32)
            shift_arr[x:x+129] = A_bool
            delta_conv = 2 * shift_arr
            if 2*x < 257:
                delta_conv[2*x] += 1
                
            new_conv = conv + delta_conv
            
            # Temporarily update A_bool for diagonal check
            A_bool[x] = 1
            peak, missing = eval_conv_state(new_conv, A_bool)
            A_bool[x] = 0
            
            score = (missing, peak)
            if score < best_score:
                best_score = score
                best_x = x
                
        if best_x == -1:
            break
            
        # Commit best_x
        A_bool[best_x] = 1
        # Update conv using the SAME delta calculation (A_bool was old during calc)
        # We need to recalculate delta_conv for best_x with the OLD A_bool.
        # Since we reverted A_bool[best_x] to 0, we can recalculate.
        shift_arr = np.zeros(257, dtype=np.int32)
        shift_arr[best_x:best_x+129] = A_bool
        delta_conv = 2 * shift_arr
        if 2*best_x < 257:
            delta_conv[2*best_x] += 1
        conv = conv + delta_conv
        
        current_peak, current_missing = best_score[1], best_score[0]
        
    # --- Phase 2: Local Search Refinement ---
    # Try to remove elements or swap to reduce peak, maintaining validity.
    
    # First, ensure we have a valid set. If not, stop.
    if current_missing > 0:
        # Fallback to a valid set if greedy failed
        # range(64, 129) is valid
        return list(range(64, 129))
        
    # Local Search
    # Try removing elements
    # If removing an element keeps it valid and reduces peak, do it.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        # Sort elements by "importance" or just iterate
        for x in range(129):
            if A_bool[x] == 0:
                continue
            if len(np.where(A_bool==1)[0]) <= 1:
                continue
                
            # Try removing x
            A_bool[x] = 0
            
            # Recalculate conv?
            # delta = -2 * shift(A_bool_without_x, x) - e_2x?
            # No, easier to just recompute conv if we do many removals?
            # Or update incrementally.
            # Let's update incrementally.
            # Old conv = conv.
            # New A = Old A \ {x}.
            # New conv = Old conv - 2*shift(Old A \ {x}, x) - e_2x?
            # Wait, shift should be of the NEW A? No, the cross term in v^2 is 2 v e_x.
            # If we remove x, v' = v - e_x.
            # v'^2 = v^2 - 2 v e_x + e_x^2.
            # So delta = -2 * shift(v, x) - e_2x.
            # Note: shift(v, x) uses the OLD v (which includes x).
            
            shift_arr = np.zeros(257, dtype=np.int32)
            shift_arr[x:x+129] = A_bool # This is v WITHOUT x, because we set A_bool[x]=0 already?
            # No, we set A_bool[x]=0. So this shift is of v'.
            # We need shift of v (with x).
            # So we should calculate delta BEFORE removing x.
            
            # Let's revert and do it correctly.
            A_bool[x] = 1
            
            # Calculate delta for removal
            shift_arr = np.zeros(257, dtype=np.int32)
            shift_arr[x:x+129] = A_bool # v with x
            delta_conv = -2 * shift_arr
            if 2*x < 257:
                delta_conv[2*x] -= 1
                
            new_conv = conv + delta_conv
            
            # Check validity with new A (x removed)
            A_bool[x] = 0
            new_peak, new_missing = eval_conv_state(new_conv, A_bool)
            
            if new_missing == 0 and new_peak < current_peak:
                # Accept removal
                conv = new_conv
                current_peak = new_peak
                improved = True
            else:
                # Revert
                A_bool[x] = 1
                # conv remains unchanged
                
    # Try swaps?
    # If peak is still high, try swapping x (in A) with y (not in A).
    # This is more complex. Given time constraints, removal might be enough.
    
    return list(np.where(A_bool == 1)[0])

# EVOLVE-BLOCK-END
