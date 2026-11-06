import numpy as np
import time
import itertools

N = 16

def verify_network(comparators, N=N):
    """
    Verify if the network sorts all 2^N binary inputs.
    Uses Zero-One Principle.
    """
    if not comparators:
        # Only valid if N <= 1
        return N <= 1
    
    # Generate all 2^N binary inputs
    # Shape: (2^N, N)
    inputs = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        # Fill column i with binary representation of 0..2^N-1
        # Bit i of the index
        inputs[:, i] = (np.arange(2**N, dtype=np.uint16) >> i) & 1
        
    # Apply comparators
    for i, j in comparators:
        min_val = np.minimum(inputs[:, i], inputs[:, j])
        max_val = np.maximum(inputs[:, i], inputs[:, j])
        inputs[:, i] = min_val
        inputs[:, j] = max_val
        
    # Check if sorted: each row should be 0...0 1...1
    # This means for each row, all 0s must come before 1s.
    # Equivalent to: inputs[:, k] <= inputs[:, k+1] for all k
    # Or simply: the sum of the row must match the number of 1s, 
    # and the first k elements must be 0 if sum < k? 
    # Faster check: 
    # A binary vector is sorted if and only if it is of the form 0^a 1^b.
    # This implies that once we see a 0 after a 1, it's invalid.
    # We can check: np.all(inputs[:, :-1] <= inputs[:, 1:])
    return np.all(inputs[:, :-1] <= inputs[:, 1:])

def generate_batcher_odd_merge(N):
    """
    Generate Batcher's Odd-Merge Sort network for N wires.
    Returns list of (i, j) comparators.
    """
    comparators = []
    
    def merge(a_start, a_len, b_start, b_len):
        """
        Merge two sorted sequences.
        a: [a_start, a_start+a_len)
        b: [b_start, b_start+b_len)
        Assumption: a_len == b_len for standard odd-merge steps in recursion,
        but the base case handles small sizes.
        Actually, standard Batcher Odd-Merge merges two equal size sorted lists.
        """
        if a_len == 0 or b_len == 0:
            return
        
        if a_len == 1 and b_len == 1:
            comparators.append((a_start, b_start))
            return
            
        # Split a and b into odd and even indexed elements
        # a_odd: a_start, a_start+2, ...
        # a_even: a_start+1, a_start+3, ...
        # b_odd: b_start, b_start+2, ...
        # b_even: b_start+1, b_start+3, ...
        
        # Recursively merge a_odd and b_odd
        # The result goes into a_odd positions? No, the standard algorithm
        # merges the two "halves" (odd/even indices) separately.
        
        # Let's follow the standard recursive definition:
        # 1. Merge the odd-indexed elements of A and B.
        # 2. Merge the even-indexed elements of A and B.
        # 3. Compare-exchange adjacent pairs to interleave.
        
        # Indices for odd positions in A: a_start, a_start+2, ...
        # Indices for even positions in A: a_start+1, a_start+3, ...
        
        # The recursive calls operate on the sub-arrays.
        # This implementation is tricky with flat indices.
        # It's easier to implement on a list of indices.
        
    # Alternative: Implement on a list of wire indices.
    def build_merge(indices_a, indices_b):
        """
        indices_a, indices_b: lists of wire indices.
        Returns list of comparators.
        """
        if not indices_a or not indices_b:
            return []
        if len(indices_a) == 1 and len(indices_b) == 1:
            return [(indices_a[0], indices_b[0])]
        
        # Split into odd/even positions
        a_odd = indices_a[0::2]
        a_even = indices_a[1::2]
        b_odd = indices_b[0::2]
        b_even = indices_b[1::2]
        
        comps = []
        # Recursively merge odd parts
        comps.extend(build_merge(a_odd, b_odd))
        # Recursively merge even parts
        comps.extend(build_merge(a_even, b_even))
        
        # Now interleave
        # The results of the recursive merges are in a_odd and b_odd?
        # No, the recursive calls just return comparators.
        # The wires in a_odd and b_odd are now sorted relative to each other?
        # Actually, after merging a_odd and b_odd, the combined set of wires 
        # (a_odd + b_odd) contains the smallest half of the final merged list?
        # No. 
        # Standard Odd-Merge:
        # 1. Merge A_odd and B_odd.
        # 2. Merge A_even and B_even.
        # 3. Compare A_odd[i] with B_odd[i]? No.
        
        # Let's use the explicit comparator list for the final step.
        # After recursive merges, the wires in a_odd and b_odd are such that
        # the sorted merged list of (a_odd + b_odd) is in those wires?
        # No, the recursive merge assumes the inputs are sorted.
        # The output of build_merge(a_odd, b_odd) is a set of comparators that
        # sorts the wires in a_odd + b_odd.
        # Similarly for even.
        
        # The final step of Odd-Merge:
        # Compare a_odd[i] with b_odd[i+1]? 
        # Actually, the standard formulation:
        # Let A' be the merged odd parts, B' be the merged even parts.
        # Then compare A'[i] with B'[i] for i=0..k/2-1?
        
        # Let's stick to the known property:
        # The final phase has len(indices_a) comparators.
        # Specifically, compare a_odd[i] with b_odd[i] ? No.
        
        # Correct Final Step for Odd-Merge:
        # After merging odd and even parts recursively:
        # For i in range(len(a_odd)):
        #    compare (a_odd[i], b_odd[i]) ? No.
        
        # Let's look at N=4 example.
        # A = [0, 1], B = [2, 3].
        # A_odd = [0], A_even = [1].
        # B_odd = [2], B_even = [3].
        # Merge(0, 2) -> (0,2).
        # Merge(1, 3) -> (1,3).
        # Final step: Compare A_odd[0] with B_odd[0]? No.
        # The wires are now:
        # Wire 0,2 contain sorted {0,2}. Wire 1,3 contain sorted {1,3}.
        # We need to interleave.
        # Compare 0 with 1? Compare 2 with 3?
        # Comparators: (0,1), (2,3).
        # Total: (0,2), (1,3), (0,1), (2,3).
        # Is this 4 comparators? Yes.
        
        # General Final Step:
        # Compare a_odd[i] with a_even[i]? No.
        # The wires a_odd and b_odd are interleaved in the final output?
        
        # Actually, the standard implementation is:
        # comps.extend(build_merge(a_odd, b_odd))
        # comps.extend(build_merge(a_even, b_even))
        # for i in range(len(a_odd)):
        #     comps.append((a_odd[i], b_odd[i])) # This is wrong.
        
        # Let's use a simpler, verified generator for small N or just hardcode the 80-comparator network if I can derive it.
        # Given the complexity, I will use the **NumPy Local Search** approach starting from a known valid network.
        # I can generate a valid network using **Batcher's Bitonic** (160 comps) or **Odd-Even Transposition** (120 comps) and prune.
        # Pruning 120 down to 60 is a big gap.
        # Pruning 80 (if I can generate it) is better.
        
        # I will generate the **Odd-Even Transposition** network (120 comps) which is easy to code.
        # Then I will use the powerful NumPy verifier to prune it.
        # With 90s, and ~10ms per check, I can do ~5000 checks.
        # Each check tries removing 1 comparator.
        # If I do 5000 removals, I might get from 120 to ~70-80.
        # To get to 60, I need smarter search.
        
        # I will add **Pair Removal** to the search.
        
        pass

    # Let's just generate Odd-Even Transposition for simplicity and reliability.
    # It's guaranteed to be correct initially.
    # The search will improve it.
    
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps

def local_search(initial_comps, N=N, time_limit=80):
    start_time = time.time()
    best_comps = list(initial_comps)
    best_len = len(best_comps)
    
    # Pre-generate inputs for verification
    inputs_template = np.zeros((2**N, N), dtype=np.uint8)
    for i in range(N):
        inputs_template[:, i] = (np.arange(2**N, dtype=np.uint16) >> i) & 1

    def is_valid(comps):
        if not comps:
            return N <= 1
        # Copy template to avoid modification
        # Using a view or copy? Copy is safer.
        # 65536 * 16 bytes = 1MB. Copy is fast.
        state = inputs_template.copy()
        for i, j in comps:
            min_val = np.minimum(state[:, i], state[:, j])
            max_val = np.maximum(state[:, i], state[:, j])
            state[:, i] = min_val
            state[:, j] = max_val
        return np.all(state[:, :-1] <= state[:, 1:])

    # Verify initial
    if not is_valid(best_comps):
        raise ValueError("Initial network invalid")

    # Local Search Loop
    while time.time() - start_time < time_limit:
        improved = False
        
        # 1. Try single removals
        for idx in range(len(best_comps)):
            if time.time() - start_time > time_limit:
                break
            trial = best_comps[:idx] + best_comps[idx+1:]
            if is_valid(trial):
                best_comps = trial
                best_len -= 1
                improved = True
                break # Restart search with new state
        
        if improved:
            continue
            
        # 2. Try pair removals
        found_pair = False
        for i in range(len(best_comps)):
            if time.time() - start_time > time_limit:
                break
            for j in range(i+1, len(best_comps)):
                if time.time() - start_time > time_limit:
                    break
                # Remove i and j
                trial = best_comps[:i] + best_comps[i+1:j] + best_comps[j+1:]
                if is_valid(trial):
                    best_comps = trial
                    best_len -= 2
                    found_pair = True
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
            
        # 3. If stuck, try to escape local minimum by random perturbation?
        # Or just stop. 
        # To escape, we could try swapping two comparators (doesn't change size).
        # Or removing 3 and adding 1?
        # Given time constraints, single/pair removal is the primary driver.
        
        # If we are stuck, we might try a different initial network?
        # For now, we assume the search is sufficient.
        
        # If no improvement, break to avoid infinite loop?
        # Or continue with triple removals?
        # Let's add triple removals if stuck.
        found_triple = False
        # Only do this if network is small enough to make triple combos feasible
        # C(80, 3) is ~ 80k. C(60, 3) is ~ 34k.
        # Each check is 10ms. 34k * 10ms = 340s. Too slow.
        # So we limit triple search to when len < 70.
        if len(best_comps) < 70:
            for i in range(len(best_comps)):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i+1, len(best_comps)):
                    if time.time() - start_time > time_limit:
                        break
                    for k in range(j+1, len(best_comps)):
                        if time.time() - start_time > time_limit:
                            break
                        trial = best_comps[:i] + best_comps[i+1:j] + best_comps[j+1:k] + best_comps[k+1:]
                        if is_valid(trial):
                            best_comps = trial
                            best_len -= 3
                            found_triple = True
                            improved = True
                            break
                    if improved: break
                if improved: break
            
        if not improved:
            # Stuck. Break.
            break
            
    return best_comps

def construct():
    # Start with Odd-Even Transposition (120 comparators)
    initial_comps = generate_batcher_odd_merge(N)
    
    # Run local search with a deadline
    # 90s limit, use 80s for search
    optimized_comps = local_search(initial_comps, N, time_limit=80)
    
    return optimized_comps
