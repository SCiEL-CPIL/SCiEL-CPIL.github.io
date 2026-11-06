import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
# Set deadline well below 60s to ensure we return before kill
DEADLINE_SECONDS = 55

# Precompute the boolean input matrix for verification
# Shape: (2^N, N), dtype uint8
# wires[k, i] = bit i of k
_ALL_MASKS = np.arange(1 << N, dtype=np.int32)
_INPUT_MATRIX = np.zeros((1 << N, N), dtype=np.uint8)
for i in range(N):
    _INPUT_MATRIX[:, i] = (_ALL_MASKS >> i) & 1

# Precompute indices for faster comparator application
# We will store the network as a list of (i, j) tuples.

def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0

def _is_sorted_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle.
    """
    if not net:
        return True
    
    # Work on a copy of the input matrix
    # uint8 is sufficient for 0/1 values
    wires = _INPUT_MATRIX.copy()
    
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        
        # min to i, max to j
        # Since values are 0/1:
        # min = 0 if either is 0, else 1 -> wi & wj
        # max = 1 if either is 1, else 0 -> wi | wj
        # This is faster than np.minimum/maximum on integers? 
        # Bitwise ops on uint8 are very fast.
        wires[:, i] = wi & wj
        wires[:, j] = wi | wj
        
    # Check if sorted: wires[:, k] <= wires[:, k+1]
    # Equivalent to: no 1 followed by 0 in each row
    # i.e., wires[:, k] == 1 and wires[:, k+1] == 0 should never happen
    # i.e., (wires[:, :-1] == 1) & (wires[:, 1:] == 0) should be all False
    # Or simply: np.any(wires[:, :-1] > wires[:, 1:])
    
    # Vectorized check
    # This creates a boolean array of shape (2^N, N-1)
    # For N=18, this is 262144 * 17 ~ 4.4M booleans. Fast.
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _akl_sorter(n):
    """
    Generate the AKL (Akl, Krishnamurthy, Kung) sorting network.
    This is often smaller than odd-even transposition.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # AKL(n) = AKL(n/2) on even wires, AKL(n/2) on odd wires, then merge
    # Actually, the recursive definition is:
    # S(n) = S(n/2) on first half, S(n/2) on second half, then merge comparators
    
    # Let's implement the standard AKL construction.
    # For even n:
    # 1. Sort first n/2 and last n/2 in parallel (recursively)
    # 2. Merge: for k in 0..n/2-1, compare (k, k+n/2)
    #    Wait, that's just a merge network.
    
    # The AKL network is defined as:
    # A(n) = A(n/2) on indices 0..n/2-1, A(n/2) on indices n/2..n-1
    # followed by a merge network M(n/2, n/2)
    
    # Actually, a simpler recursive construction for sorting networks is:
    # If n is even:
    #   Sort first half and second half in parallel
    #   Then merge them
    # If n is odd:
    #   Sort first n-1, then insert the last element
    
    # Let's try a simple recursive merge sort network.
    # This might not be optimal but is a good starting point.
    
    def merge_sort_network(n, offset=0):
        if n <= 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        half = n // 2
        if n % 2 == 0:
            left = merge_sort_network(half, offset)
            right = merge_sort_network(half, offset + half)
            # Merge: compare (i, i+half) for i in 0..half-1
            # This is not a full merge, just a pairwise comparison.
            # A proper merge network is more complex.
            # Let's use a simple "butterfly" or just pairwise.
            # Actually, for sorting, pairwise comparison after sorting halves is NOT sufficient.
            # Example: [2, 1, 4, 3] -> [1, 2, 3, 4] needs more than just (0,2) and (1,3).
            # (0,2) compares 2,4 -> 2,4. (1,3) compares 1,3 -> 1,3. Result [2,1,4,3] -> [2,1,4,3]. Not sorted.
            
            # So simple merge sort network is not trivial.
            # Let's stick to Odd-Even Transposition or use a known good network.
            
            # For now, let's just return empty to avoid bad networks.
            # We will rely on optimization from OET.
            return []
        else:
            return []
            
    return []

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with odd-even transposition network
    net = _odd_even_transposition()
    best_net = net[:]
    best_count = len(net)
    
    # Initial check (should be true)
    # _is_sorted_network(net)
    
    # Phase 1: Greedy removal
    # We iterate until no more removals are possible or time runs out.
    # We try removing comparators from the end first, as they are often redundant.
    # Then we try from the beginning.
    
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return best_net
            
            # Create new network without idx
            new_net = net[:idx] + net[idx+1:]
            
            # Verify
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                changed = True
                break # Restart the loop with the new network
        
        # If no change from end-to-start, try start-to-end
        if not changed:
            for idx in range(len(net)):
                if time.time() > deadline:
                    return best_net
                
                new_net = net[:idx] + net[idx+1:]
                
                if _is_sorted_network(new_net):
                    net = new_net
                    best_net = net[:]
                    best_count = len(net)
                    changed = True
                    break
    
    # Phase 2: Random pair removal (if time remains)
    if time.time() < deadline - 5:
        # Try removing pairs of comparators
        # This is more expensive, so we limit attempts
        attempts = 0
        while time.time() < deadline - 2 and attempts < 500:
            if len(net) < 2:
                break
            idx1, idx2 = random.sample(range(len(net)), 2)
            # Ensure idx1 < idx2 for slicing logic simplicity, though set logic works
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
            
            new_net = net[:idx1] + net[idx1+1:idx2] + net[idx2+1:]
            
            if _is_sorted_network(new_net):
                net = new_net
                best_net = net[:]
                best_count = len(net)
                attempts = 0 # Reset attempts on success
            else:
                attempts += 1
                
    # Phase 3: Local Search - Swap adjacent comparators
    # This can sometimes allow further removals or reordering that leads to smaller networks.
    # Actually, swapping doesn't change count. But it might enable removals.
    # Let's try a simple hill climbing:
    # 1. Try swapping two random comparators.
    # 2. Then try greedy removal again.
    # If the resulting network is smaller, accept.
    
    if time.time() < deadline - 10:
        current_net = best_net[:]
        current_count = best_count
        no_improve_count = 0
        
        while time.time() < deadline - 5 and no_improve_count < 20:
            # Pick two random indices to swap
            if len(current_net) < 2:
                break
            idx1, idx2 = random.sample(range(len(current_net)), 2)
            
            new_net = current_net[:]
            new_net[idx1], new_net[idx2] = new_net[idx2], new_net[idx1]
            
            # Check validity
            if not _is_sorted_network(new_net):
                no_improve_count += 1
                continue
            
            # Try greedy removal on this new valid network
            # We do a limited greedy removal to save time
            temp_net = new_net[:]
            temp_changed = True
            while temp_changed and time.time() < deadline - 2:
                temp_changed = False
                # Try removing from end
                for idx in range(len(temp_net) - 1, -1, -1):
                    if time.time() > deadline - 2:
                        break
                    candidate = temp_net[:idx] + temp_net[idx+1:]
                    if _is_sorted_network(candidate):
                        temp_net = candidate
                        temp_changed = True
                        break
            
            if len(temp_net) < current_count:
                current_net = temp_net
                current_count = len(current_net)
                best_net = current_net[:]
                best_count = current_count
                no_improve_count = 0
            else:
                no_improve_count += 1
                
    return best_net

