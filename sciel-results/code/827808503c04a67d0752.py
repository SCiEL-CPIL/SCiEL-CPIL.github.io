import time
import numpy as np

_START_TIME = time.time()
_DEADLINE = 85  # seconds, safety margin before 90s kill

def construct():
    """Find subset A of [0,128] covering [128,256] by pair sums, minimizing peak multiplicity."""
    n = 128
    target_lo, target_hi = 128, 256
    
    # Start with a valid baseline: A = [64, 65, ..., 128]
    # This covers [128, 256] since min sum = 64+64=128, max sum = 128+128=256
    A = np.zeros(n + 1, dtype=bool)
    A[64:129] = True
    
    def compute_sums(A_bool):
        """Compute pair sum multiplicities for unordered pairs including equal summands."""
        A_int = A_bool.astype(np.int32)
        ordered = np.convolve(A_int, A_int, mode='full')
        # ordered[s] = sum_{a+b=s} A[a]*A[b]
        # unordered[s] = (ordered[s] + self_count[s]) // 2
        self_count = np.zeros(len(ordered), dtype=np.int32)
        for s in range(target_lo, target_hi + 1):
            if s % 2 == 0 and s // 2 <= n and A_bool[s // 2]:
                self_count[s] = 1
        unordered = (ordered + self_count) // 2
        return unordered
    
    def get_cost(A_bool):
        """Return (peak_mult, sum_of_squares) for the target interval."""
        sums = compute_sums(A_bool)
        target_sums = sums[target_lo:target_hi + 1]
        # Check validity: all must be >= 1
        if np.any(target_sums < 1):
            return (float('inf'), float('inf'))
        peak = int(np.max(target_sums))
        sq = int(np.sum(target_sums ** 2))
        return (peak, sq)
    
    # Initial cost
    init_cost = get_cost(A)
    best_A = A.copy()
    best_cost = init_cost
    
    # Phase 1: Greedy Removal
    # Try to remove elements if it doesn't break coverage and improves cost
    while time.time() - _START_TIME < _DEADLINE:
        improved = False
        current_elements = np.where(A)[0]
        # Shuffle to avoid bias
        np.random.shuffle(current_elements)
        
        for val in current_elements:
            if time.time() - _START_TIME > _DEADLINE:
                break
            
            # Temporarily remove val
            A[val] = False
            new_cost = get_cost(A)
            
            # If valid and better or equal cost, keep removed
            if new_cost[0] < float('inf') and new_cost <= best_cost:
                best_cost = new_cost
                best_A = A.copy()
                improved = True
            else:
                # Revert
                A[val] = True
        
        if not improved:
            break
    
    # Phase 2: Local Search (Swaps)
    # Try to swap elements to further reduce peak/sq
    while time.time() - _START_TIME < _DEADLINE:
        improved = False
        current_elements = np.where(A)[0]
        if len(current_elements) == 0:
            break
        np.random.shuffle(current_elements)
        
        for rem_val in current_elements:
            if time.time() - _START_TIME > _DEADLINE:
                break
            
            candidates = np.where(~A)[0]
            if len(candidates) == 0:
                break
            np.random.shuffle(candidates)
            
            # Try a few random candidates for each removal to save time
            for add_val in candidates[:10]: 
                # Apply swap
                A[rem_val] = False
                A[add_val] = True
                
                new_cost = get_cost(A)
                
                if new_cost[0] < float('inf') and new_cost < best_cost:
                    best_cost = new_cost
                    best_A = A.copy()
                    improved = True
                    # Don't break out of inner loop immediately, might find better?
                    # But if we found an improvement, we should probably restart search or continue.
                    # For simplicity, we revert and let the outer loop handle it if we want strict improvement steps.
                    # Actually, if we accept, A is changed. We should break inner loop and restart outer?
                    # Let's just revert and mark improved, then the next iteration will pick up the new state.
                    A[add_val] = False
                    A[rem_val] = True
                    break # Break inner loop, restart outer with new best_A? No, A is current.
                    # Wait, if we accept, we should update A to the new state.
                    # Let's do this: If accepted, keep A changed, break inner, break outer?
                    # No, let's just keep A changed and continue checking other swaps?
                    # Risky. Let's revert, save best, and restart the whole swap phase?
                    # That's too slow.
                    
                    # Better: If accepted, keep A changed, break inner loop, continue outer loop.
                    A[add_val] = True # Re-apply to keep state consistent? No, we set it True above.
                    # We set A[rem]=False, A[add]=True.
                    # If accepted, we keep that state.
                    # We break inner loop to move to next rem_val? No, rem_val is no longer in A.
                    # So we should break outer loop and restart?
                    # Let's just break both and restart the while loop.
                    improved = True
                    # Reset A to best_A to ensure we are optimizing from best state
                    A = best_A.copy()
                    break # Break candidate loop
                
                # Revert if not accepted
                A[add_val] = False
                A[rem_val] = True
            
            if improved:
                break # Break rem_val loop, restart while loop
        
        if not improved:
            break
    
    # Return the best found as a list of Python ints
    return [int(x) for x in np.where(best_A)[0]]
