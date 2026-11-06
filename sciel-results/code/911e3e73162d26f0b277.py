import time
import numpy as np

N = 17

def _batcher_oes(n):
    """Generates Batcher's Odd-Even Merge Sort network for n wires."""
    if n <= 1:
        return []
    mid = n // 2
    left = _batcher_oes(mid)
    right = _batcher_oes(n - mid)
    right = [(i + mid, j + mid) for i, j in right]
    merge = _batcher_merge(mid, n - mid)
    return left + right + merge

def _batcher_merge(n, m):
    """
    Generates the merge step for Batcher's OES.
    Merges two sorted lists of size n and m.
    The combined list has size n+m.
    Indices 0..n-1 are the first list, n..n+m-1 are the second.
    """
    if n == 0 or m == 0:
        return []
    if n == 1 and m == 1:
        return [(0, 1)]
    
    # Recursive structure of Odd-Even Merge
    # The merge of two sorted lists of size n and m is:
    # 1. Compare (0, 1) if n>=1 and m>=1? No.
    
    # Standard Recursive Definition:
    # Let L be the combined length.
    # Split the indices into "odd" and "even" positions.
    # Odd positions: 0, 2, 4, ...
    # Even positions: 1, 3, 5, ...
    
    # The merge is composed of:
    # 1. A recursive merge of the elements at odd positions?
    # 2. A recursive merge of the elements at even positions?
    # 3. A final phase of comparisons?
    
    # Actually, the standard algorithm is:
    # Merge(n, m):
    #   if n == 1 and m == 1: return [(0, 1)]
    #   if n == 1:
    #       # Insert 1 element into sorted list of m
    #       # This is handled by the general case if we are careful.
    #   
    # Let's use the iterative phase method for the merge, which is safer.
    # The merge of two sorted lists of total length L requires log2(L) phases.
    # However, L might not be a power of 2.
    
    # For simplicity and correctness, I will use a known hardcoded structure for small N
    # or a very robust recursive implementation.
    
    # Here is a recursive implementation that works for any n, m:
    # Reference: https://en.wikipedia.org/wiki/Odd%E2%80%93even_merge_sort
    # The merge step is:
    # 1. Compare (0, 1), (2, 3), ... ? No.
    
    # Let's use the property that Batcher's merge of n and m is:
    # Merge(n//2, m//2) on the first half of the combined array? No.
    
    # I will use a fallback: If I can't generate Batcher correctly, I use Odd-Even Transposition.
    # But I want Batcher.
    
    # Let's try this specific recursive merge:
    # def merge(n, m):
    #   if n == 1 and m == 1: return [(0, 1)]
    #   if n == 1:
    #      # Insert 0 into sorted list 1..m-1
    #      # This is equivalent to:
    #      # Compare (0, 1)
    #      # Then merge the result?
    #      # Actually, if n=1, the "Odd-Even" split results in:
    #      # Odd indices: 0
    #      # Even indices: 1, 2, ..., m-1
    #      # Recursively merge Odd (size 1) and Even (size m-1)?
    #      # No, the elements are not contiguous in the recursive call sense.
    
    # Given the risk, I will use **Odd-Even Transposition** as the seed, which is definitely correct,
    # and then apply **aggressive redundancy removal**.
    # Odd-Even Transposition for N=17 has 136 comparators.
    # Redundancy removal might get it down to ~100?
    # Batcher is ~75.
    # If I can get Batcher, it's much better.
    
    # I will attempt to use a hardcoded Batcher network for N=17 if I can construct it.
    # I will construct it using a simple recursive function that I am confident about.
    
    # Confidence check:
    # For N=2: [(0,1)]
    # For N=4:
    #   Left (0,1): [(0,1)]
    #   Right (2,3): [(2,3)]
    #   Merge(2,2):
    #     Compare (0,2), (1,3)
    #     Then (0,1), (2,3)
    #     Total: 4 comparators.
    #   Total N=4: 1+1+4 = 6. Correct.
    
    # For N=8:
    #   Left (0..3): 6 comps.
    #   Right (4..7): 6 comps.
    #   Merge(4,4):
    #     Phase 1: (0,4), (1,5), (2,6), (3,7). (4 comps)
    #     Phase 2: Merge(2,2) on indices (0,2,4,6) and (1,3,5,7)?
    #     The indices 0,2,4,6 form a list of size 4.
    #     The merge of two sorted lists of size 2 (from the previous phase's outputs) is needed.
    #     This recursive structure is tricky.
    
    # I will use the **Odd-Even Transposition** seed and optimize.
    # It is safer to have a valid, sub-optimal network than an invalid, optimal one.
    
    return [] # Placeholder, will use OET seed in construct

def _is_sorted_network(network):
    """
    Verifies if the network sorts all 2^N boolean inputs.
    Returns True if valid, False otherwise.
    """
    # Generate all 2^N inputs
    # N=17, so 131072 inputs.
    # Shape: (131072, 17)
    inputs = np.array([list(map(int, format(i, '017b'))) for i in range(2**N)])
    
    # Simulate
    state = inputs.copy()
    for i, j in network:
        # min to i, max to j
        min_vals = np.minimum(state[:, i], state[:, j])
        max_vals = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_vals
        state[:, j] = max_vals
    
    # Check if sorted
    # A list is sorted if state[:, k] <= state[:, k+1] for all k
    for k in range(N - 1):
        if not np.all(state[:, k] <= state[:, k + 1]):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 80.0  # 80s deadline to be safe
    
    # Seed: Odd-Even Transposition
    # Phases 0 to N-1
    # Phase p: compare (p%2, p%2+1), (p%2+2, p%2+3), ...
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    # Verify seed
    if not _is_sorted_network(network):
        # Should not happen, but if it does, return empty or seed
        return network
    
    best_network = network
    best_count = len(network)
    
    # Optimization: Redundancy Removal
    # Iterate through the network, try removing each comparator.
    # If valid, remove it.
    # Repeat until no more removals possible or time limit.
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        # Iterate in reverse to avoid index shifting issues during removal
        for idx in range(len(best_network) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
            
            # Try removing comparator at idx
            candidate = best_network[:idx] + best_network[idx+1:]
            
            # Verify candidate
            # To speed up verification, we can use a smaller set of inputs first?
            # No, the task requires certification over all inputs.
            # But for intermediate steps, we can use a random subset to filter out obvious failures.
            # However, for safety, let's use all inputs. It's fast enough.
            
            if _is_sorted_network(candidate):
                best_network = candidate
                best_count = len(best_network)
                improved = True
                # Restart the loop from the beginning?
                # Or continue?
                # If we remove one, the indices shift.
                # Since we are iterating in reverse, removing at idx means
                # we don't need to check idx again, but we should check earlier indices?
                # Actually, removing a comparator might make an earlier comparator redundant?
                # Unlikely, but possible.
                # To be thorough, we should restart the scan.
                # But that's expensive.
                # Let's just continue the scan.
    
    return best_network

# EVOLVE-BLOCK-START
def construct():
    # The function above is the implementation.
    # We need to wrap it or just define it here.
    # The prompt says "Keep the function names and signatures of the seed program."
    # The seed program has `def construct():` inside the evolve block.
    
    start_time = time.time()
    deadline = 80.0
    
    # Seed: Odd-Even Transposition
    network = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            network.append((i, i + 1))
    
    if not _is_sorted_network(network):
        return network
    
    best_network = network
    
    improved = True
    while improved and (time.time() - start_time) < deadline:
        improved = False
        for idx in range(len(best_network) - 1, -1, -1):
            if (time.time() - start_time) > deadline:
                break
            candidate = best_network[:idx] + best_network[idx+1:]
            if _is_sorted_network(candidate):
                best_network = candidate
                improved = True
    
    return best_network
# EVOLVE-BLOCK-END
