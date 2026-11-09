import time
import random
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.85 # Safety margin

# Pre-compute initial and expected wires for verification
_INITIAL_WIRES = None
_EXPECTED_WIRES = None

def _init_wires():
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES is not None:
        return
    
    _INITIAL_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    # This represents the input where the k-th mask has bit i set.
    # Actually, for the Zero-One principle, we verify on all 2^N inputs.
    # The "wire" representation uses a large integer where bit k of wire w
    # is 1 if the k-th input sequence has a 1 at position w.
    
    # Let's re-verify the standard bit-parallel verification setup.
    # Input: 2^N boolean vectors of length N.
    # We process them all at once.
    # Wire i holds a big int. Bit k of wire i is 1 iff the k-th input vector has a 1 at index i.
    # The k-th input vector is the binary representation of k.
    # So bit k of wire i is 1 iff (k >> i) & 1 is 1.
    
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
    # Expected wires: sorted output.
    # After sorting, the k-th input vector (which has popcount(k) ones) 
    # should have its ones in the last popcount(k) positions?
    # No, the sorted output of a vector with m ones is:
    # [0, 0, ..., 0, 1, 1, ..., 1] where there are N-m zeros and m ones.
    # So for the k-th input (with m = popcount(k) ones), 
    # the output has 1s at indices N-m, N-m+1, ..., N-1.
    # So bit k of wire i is 1 iff i >= N - popcount(k).
    
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= N - i:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

_init_wires()

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _build_batcher_odd_merge(n):
    """
    Batcher Odd-Merge sorting network.
    """
    if n <= 1:
        return []
    
    n1 = (n + 1) // 2
    n2 = n // 2
    
    net = []
    net.extend(_build_batcher_odd_merge(n1))
    
    if n2 > 1:
        sub_net = _build_batcher_odd_merge(n2)
        for i, j in sub_net:
            net.append((i + n1, j + n1))
            
    for i in range(n1):
        net.append((i, i + n2))
        
    if n1 > 1:
        sub_even = _build_batcher_odd_merge(n1)
        for i, j in sub_even:
            net.append((2*i, 2*j))
            
    if n2 > 1:
        sub_odd = _build_batcher_odd_merge(n2)
        for i, j in sub_odd:
            net.append((2*i+1, 2*j+1))
            
    return net

def _local_search(net, deadline):
    """
    Steepest descent local search.
    Tries to remove any comparator that is redundant.
    """
    current_net = net[:]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current_net) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            new_net = current_net[:idx] + current_net[idx+1:]
            
            if _verify_network(new_net):
                current_net = new_net
                improved = True
                break
                
    return current_net

def _hierarchical_search(deadline):
    """
    Strategy: Decompose the 17-wire network into:
    1. Sort first 9 wires (0-8)
    2. Sort last 8 wires (9-16)
    3. Merge the two sorted halves (17-wire merge network)
    
    We optimize each part independently where possible, or use specific structures.
    For now, we generate a Batcher Odd-Merge network, which naturally has this structure,
    but we will apply local search to the whole thing.
    
    To make this "hierarchical", we can try to fix the structure of the first two stages
    (the sorts) and only optimize the merge stage, or vice versa.
    However, the optimal network might not strictly follow the 9/8 split.
    
    Alternative Structural Change:
    Use a "Bitonic" structure? No, Batcher Odd-Merge is better.
    
    Let's try a different decomposition:
    Split 17 into 8 and 9? Or 7 and 10?
    The standard Batcher split is 9 and 8.
    
    Let's try to optimize the MERGE part specifically, assuming the inputs to the merge are sorted.
    This reduces the search space for the merge part because we only need to verify it against
    sorted inputs, not all 2^17 inputs. This allows for much faster verification of the merge network.
    
    Steps:
    1. Build a good 9-wire sort and 8-wire sort.
    2. Construct a candidate merge network.
    3. Verify the merge network only on inputs where the first 9 are sorted and last 8 are sorted.
    4. Optimize the merge network using this faster verifier.
    5. Combine and do final local search on the whole network.
    """
    
    # 1. Get base sorts
    # 9-wire sort
    sort_9 = _build_batcher_odd_merge(9)
    # 8-wire sort
    sort_8 = _build_batcher_odd_merge(8)
    
    # Map sort_8 to wires 9-16
    sort_8_mapped = [(i+9, j+9) for i, j in sort_8]
    
    # Base network: Sort 9, Sort 8, then a "cross" merge
    # The merge part in Batcher Odd-Merge for 17 wires is complex.
    # Let's just use the full Batcher network as the starting point for the hierarchical search.
    # But the KEY is to optimize the "tail" (the merge part) with a specialized verifier.
    
    # Identify the split point in the Batcher network for 17.
    # Batcher(17) = Batcher(9) + Batcher(8) + Merge(17)
    # The Merge part starts after the two sorts.
    
    base_net = _build_batcher_odd_merge(17)
    
    # How many comparators are in the sorts?
    len_sort_9 = len(sort_9)
    len_sort_8 = len(sort_8)
    
    # The first len_sort_9 + len_sort_8 comparators are the sorts.
    # The rest is the merge.
    split_idx = len_sort_9 + len_sort_8
    
    # Let's verify this assumption.
    # Batcher(17) structure:
    # 1. Sort(9) on wires 0-8
    # 2. Sort(8) on wires 9-16
    # 3. Merge(17)
    
    # So base_net[:split_idx] should be the sorts.
    # Let's extract the merge part.
    merge_part = base_net[split_idx:]
    
    # Now we want to optimize `merge_part`.
    # We need a verifier that checks if `merge_part` correctly merges two sorted lists.
    # Inputs to merge:
    # Wires 0-8 contain a sorted list of 9 elements.
    # Wires 9-16 contain a sorted list of 8 elements.
    # We need to check if the output of `merge_part` is fully sorted.
    
    # How many such inputs are there?
    # A sorted list of 9 elements from a 17-bit universe?
    # No, the Zero-One principle applies to the whole network.
    # For the merge network to be correct, it must work for ANY pair of sorted inputs.
    # The number of sorted inputs of length 9 is C(17, 9) + C(17, 8)? No.
    # The inputs are boolean vectors.
    # A "sorted" boolean vector of length 9 is of the form 0...01...1.
    # There are 10 such vectors (0 ones, 1 one, ..., 9 ones).
    # Similarly, 9 such vectors for length 8.
    # So there are 10 * 9 = 90 possible pairs of sorted inputs to the merge stage.
    # We can verify the merge network against these 90 cases very quickly.
    
    # Let's build a fast verifier for the merge part.
    # We represent the state of the 17 wires as a tuple of 17 integers (0 or 1).
    # Since we only have 90 cases, we can just simulate them directly.
    
    def _verify_merge(merge_net):
        # Generate all 90 sorted input pairs
        # For 9 wires: k ones at the end. k=0..9
        # For 8 wires: m ones at the end. m=0..8
        
        # We need to map these to the 17-wire state.
        # Wires 0-8 get the 9-sorted vector.
        # Wires 9-16 get the 8-sorted vector.
        
        # Pre-compute the 10 9-vectors and 9 8-vectors as bitmasks?
        # Or just lists of 0/1.
        
        sorted_9_vectors = []
        for k in range(10):
            vec = [0]*9
            for i in range(9-k, 9):
                vec[i] = 1
            sorted_9_vectors.append(tuple(vec))
            
        sorted_8_vectors = []
        for m in range(9):
            vec = [0]*8
            for i in range(8-m, 8):
                vec[i] = 1
            sorted_8_vectors.append(tuple(vec))
            
        # For each pair, simulate the merge network and check if output is sorted.
        # Output is sorted if it is of the form 0...01...1.
        
        for v9 in sorted_9_vectors:
            for v8 in sorted_8_vectors:
                # Construct initial 17-wire state
                wires = list(v9) + list(v8)
                
                # Apply merge_net
                for i, j in merge_net:
                    if wires[i] > wires[j]:
                        wires[i], wires[j] = wires[j], wires[i]
                        
                # Check if sorted
                # A boolean list is sorted if it is non-decreasing.
                for i in range(16):
                    if wires[i] > wires[i+1]:
                        return False
                        
        return True

    # Optimize the merge part using this fast verifier.
    # Start with the merge part from Batcher.
    current_merge = merge_part[:]
    
    # Local search on merge part
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(current_merge) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            new_merge = current_merge[:idx] + current_merge[idx+1:]
            if _verify_merge(new_merge):
                current_merge = new_merge
                improved = True
                break
                
    # Construct the full network
    full_net = base_net[:split_idx] + current_merge
    
    # Final local search on the full network to catch any cross-boundary redundancies
    # or to allow the sorts to be modified if it helps (though we fixed them, 
    # a full search might find a better sort+merge combo).
    # However, full search is slow. Let's do a few rounds.
    
    full_net = _local_search(full_net, deadline)
    
    return full_net

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    # Try hierarchical approach
    best_net = _hierarchical_search(deadline)
    best_size = len(best_net)
    
    # If time remains, try standard local search on the result
    # Or try a different decomposition?
    # For now, just return the best found.
    
    # Fallback if hierarchical fails or takes too long (shouldn't happen with deadline)
    if not _verify_network(best_net):
        best_net = _build_seed()
        best_net = _local_search(best_net, deadline)
        
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
