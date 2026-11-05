"""Size-efficient sorting networks on 16 wires."""
import time
import numpy as np

N = 16
NUM_INPUTS = 1 << N  # 65536

def _apply_comparator_bitparallel(states, i, j):
    """Apply comparator (i,j) to all 65536 states at once.
    states: array of shape (65536, 16) with 0/1 values.
    """
    a = states[:, i]
    b = states[:, j]
    # min goes to i, max goes to j
    states[:, i] = np.minimum(a, b)
    states[:, j] = np.maximum(a, b)
    return states

def _verify_network(comparators):
    """Check if network sorts all 65536 Boolean inputs using bit-parallel method."""
    # Generate all 65536 inputs as a (65536, 16) array
    # Input k has bit i set if (k >> i) & 1
    indices = np.arange(NUM_INPUTS, dtype=np.uint8)
    states = np.zeros((NUM_INPUTS, N), dtype=np.uint8)
    for i in range(N):
        states[:, i] = (indices >> i) & 1
    
    for (i, j) in comparators:
        a = states[:, i]
        b = states[:, j]
        states[:, i] = np.minimum(a, b)
        states[:, j] = np.maximum(a, b)
    
    # Check sorted: for each row, states[:, 0] <= states[:, 1] <= ... <= states[:, 15]
    # A row is sorted iff it's of the form 0...01...1
    # Check: for each adjacent pair, states[:, k] <= states[:, k+1]
    for k in range(N - 1):
        if np.any(states[:, k] > states[:, k + 1]):
            return False
    return True

def _count_comparators(comparators):
    return len(comparators)

def _odd_even_transposition():
    """120-comparator baseline."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _batcher_odd_even_merge(n):
    """Batcher's odd-even merge sort network for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort two halves
    left = _batcher_odd_even_merge(half)
    right = [(i + half, j + half) for (i, j) in left]
    comps = left + right
    
    # Odd-even merge: merge two sorted halves of size half
    # For n=16: merge two sorted 8-element sequences
    # Odd-even merge: recursively merge
    # Phase 1: compare (0,1), (2,3), ... (odd-even pairs within each half? No)
    # Actually Batcher odd-even merge:
    # Compare elements at positions (i, i+half) for i in 0..half-1? No.
    
    # Let me use the standard Batcher odd-even merge:
    # To merge two sorted sequences of length m (total 2m):
    # 1. Compare (i, i+m) for i = 0, 1, ..., m-1  (this is the "even" phase offset)
    #    Actually: compare pairs (i, i+m) for i=0..m-1
    # 2. Recursively merge the even-indexed and odd-indexed subsequences
    
    # Standard Batcher odd-even merge for 2m elements from two sorted m-element lists:
    # Step 1: Compare (i, i+m) for i = 0..m-1
    # Step 2: Recursively odd-even merge the elements at even positions (0,2,4,...) 
    #         and odd positions (1,3,5,...)
    
    # Actually the standard formulation:
    # merge(2m): 
    #   compare (i, i+m) for i in 0..m-1
    #   recursively merge the even-indexed subsequence (positions 0,2,...,2m-2) 
    #   and the odd-indexed subsequence (positions 1,3,...,2m-1)
    #   But these subsequences are not contiguous in the original...
    
    # Let me use a cleaner recursive definition:
    # Batcher's odd-even merge sort:
    # sort(2m):
    #   sort(m) on first half, sort(m) on second half
    #   merge(m, m)
    # merge(2m) [merging two sorted m-element sequences]:
    #   compare (i, i+m) for i = 0..m-1
    #   merge(m/2, m/2) on even positions, merge(m/2, m/2) on odd positions
    #   where "even positions" means wires 0,2,4,...,2m-2 and "odd" means 1,3,...,2m-1
    
    # For the merge step, we need to map positions back to wire indices
    # This is getting complex. Let me just use a known good starting point.
    
    # Alternative: use a simple but effective construction
    # For now, return odd-even transposition as fallback
    return _odd_even_transposition()

def _known_60_comparator_network():
    """Try to construct or find a ~60 comparator network.
    This is a heuristic construction based on known results."""
    # Use a layered approach: build from smaller known networks
    # For 16 wires, a known optimal or near-optimal network has 60 comparators.
    # Let's construct one using a divide-and-conquer with known small networks.
    
    # Known: 4-wire sorting network needs 5 comparators (optimal)
    # 8-wire: 19 comparators (optimal per known results)
    # 16-wire: 60 comparators (reference)
    
    # Let me build a reasonable network and then optimize.
    # Start with a structured network:
    
    comps = []
    
    # Layer 1: sort 4-wire groups (4 groups of 4)
    # 4-wire optimal: (0,1),(2,3),(0,2),(1,3),(0,1),(2,3) - that's 6
    # Actually optimal 4-wire is 5: (0,1),(2,3),(0,2),(1,3),(0,1)? No.
    # Optimal 4-wire sorting network has 5 comparators:
    # (0,1), (2,3), (0,2), (1,3), (0,1) -- let me verify
    # Actually the known optimal 4-wire network is:
    # Phase 0: (0,1), (2,3)
    # Phase 1: (0,2), (1,3)  
    # Phase 2: (1,2)
    # That's 5 comparators. Let me verify: this is the odd-even transposition for 4.
    # OET for 4: phase0: (0,1),(2,3); phase1: (1,2); phase2: (0,1),(2,3) = 5 comps
    
    def sort4(offset):
        return [(offset+0, offset+1), (offset+2, offset+3),
                (offset+1, offset+2),
                (offset+0, offset+1), (offset+2, offset+3)]
    
    for g in range(4):
        comps.extend(sort4(g * 4))
    
    # Now we have 4 sorted 4-element groups
    # Merge them pairwise: merge(4,4) to get 8, then merge(8,8) to get 16
    # Batcher odd-even merge for 4+4:
    # Compare (i, i+4) for i=0..3
    # Then sort the even positions (0,2,4,6) and odd positions (1,3,5,7) as 4-wire networks
    def merge8(offset):
        """Merge two sorted 4-element sequences starting at offset."""
        m = []
        # Compare (i, i+4) for i=0..3
        for i in range(4):
            m.append((offset + i, offset + i + 4))
        # Sort even positions: wires offset+0, offset+2, offset+4, offset+6
        # Map to 4-wire sort on these positions
        even_pos = [offset + 0, offset + 2, offset + 4, offset + 6]
        odd_pos = [offset + 1, offset + 3, offset + 5, offset + 7]
        
        # 4-wire sort on even positions
        m.append((even_pos[0], even_pos[1]))
        m.append((even_pos[2], even_pos[3]))
        m.append((even_pos[1], even_pos[2]))
        m.append((even_pos[0], even_pos[1]))
        m.append((even_pos[2], even_pos[3]))
        
        # 4-wire sort on odd positions
        m.append((odd_pos[0], odd_pos[1]))
        m.append((odd_pos[2], odd_pos[3]))
        m.append((odd_pos[1], odd_pos[2]))
        m.append((odd_pos[0], odd_pos[1]))
        m.append((odd_pos[2], odd_pos[3]))
        
        return m
    
    # Merge groups 0&1 (wires 0-7) and groups 2&3 (wires 8-15)
    comps.extend(merge8(0))
    comps.extend(merge8(8))
    
    # Now merge the two 8-element sorted sequences (wires 0-7 and 8-15)
    def merge16(offset):
        """Merge two sorted 8-element sequences starting at offset."""
        m = []
        # Compare (i, i+8) for i=0..7
        for i in range(8):
            m.append((offset + i, offset + i + 8))
        # Sort even positions (0,2,4,...,14) as 8-wire, odd (1,3,...,15) as 8-wire
        # For 8-wire sort, we need a good 8-wire network
        # Use a simple 8-wire: OET has 28 comparators, but we can do better
        # For now use OET for 8 wires
        even_pos = [offset + 2*i for i in range(8)]
        odd_pos = [offset + 2*i + 1 for i in range(8)]
        
        # 8-wire OET
        for phase in range(8):
            for i in range(phase % 2, 7, 2):
                m.append((even_pos[i], even_pos[i+1]))
                m.append((odd_pos[i], odd_pos[i+1]))
        
        return m
    
    comps.extend(merge16(0))
    
    return comps

def _greedy_remove(comparators, deadline):
    """Try to remove redundant comparators one at a time."""
    comps = list(comparators)
    start = time.time()
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for idx in range(len(comps) - 1, -1, -1):
            if time.time() >= deadline:
                break
            # Try removing comparator at idx
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network(candidate):
                comps = candidate
                changed = True
                break  # Restart scan from end
    
    return comps

def _local_search(comparators, deadline):
    """Local search: try moving comparators, swapping, etc."""
    comps = list(comparators)
    start = time.time()
    best = len(comps)
    
    while time.time() < deadline:
        improved = False
        # Try removing
        for idx in range(len(comps)):
            if time.time() >= deadline:
                break
            candidate = comps[:idx] + comps[idx+1:]
            if _verify_network(candidate):
                comps = candidate
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping adjacent comparators
        for idx in range(len(comps) - 1):
            if time.time() >= deadline:
                break
            candidate = comps[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if _verify_network(candidate):
                # Only accept if it doesn't increase count (it doesn't) 
                # but might help future removals
                # For now, accept swaps that maintain validity
                # Actually swaps don't change count, so only do if it enables removal
                # Skip swaps for now to keep it simple
                pass
        
        if not improved and time.time() > deadline - 5:
            break
    
    return comps

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80  # 80s safety margin
    
    # Start with a structured network
    # Use odd-even transposition as initial (120 comps)
    comps = _odd_even_transposition()
    
    # Greedy removal
    comps = _greedy_remove(comps, deadline - 10)
    
    # Local search for further improvements
    if time.time() < deadline - 5:
        comps = _local_search(comps, deadline - 5)
    
    return comps
# EVOLVE-BLOCK-END
