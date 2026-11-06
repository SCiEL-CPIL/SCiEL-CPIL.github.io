"""Size-efficient sorting network search for N=16 using a SAT/CP-style constructive approach."""
import time
import random
from collections import deque

N = 16
DEADLINE = 85.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _simulate(network, mask):
    """Simulate network on a bitmask (16-bit). Returns sorted bitmask if sorted."""
    wires = [0] * N
    # Load bits
    for i in range(N):
        wires[i] = (mask >> i) & 1
    for (i, j) in network:
        if wires[i] > wires[j]:
            wires[i], wires[j] = wires[j], wires[i]
    # Check sorted: all 1s before 0s? Actually sorted means non-decreasing.
    # For binary, sorted means all 0s then all 1s.
    for k in range(N - 1):
        if wires[k] > wires[k + 1]:
            return None  # not sorted
    # Return canonical sorted form
    ones = sum(wires)
    return (1 << ones) - 1  # bottom 'ones' bits are 1

def _verify_network(network):
    """Verify network sorts all 2^16 inputs using zero-one principle."""
    # Optimization: use the fact that we only need to check all 2^16 masks.
    # But we can prune: if the network is symmetric or we use incremental checking.
    # For 16 wires, 65536 checks is fast enough in Python if we optimize the simulation.
    
    # Pre-compute for speed: represent wires as a single int? No, list is fine.
    # Let's use a faster simulation: track only the positions of 1s.
    # Actually, for binary inputs, the state is just which wires have 1s.
    # A compare-exchange (i,j) on binary: if wire i has 1 and wire j has 0, swap.
    # So we can track a bitmask of which wires are 1.
    
    for mask in range(1 << N):
        # Simulate
        state = mask
        for (i, j) in network:
            # If i has 1 and j has 0, swap them
            if (state >> i) & 1 and not ((state >> j) & 1):
                state ^= (1 << i) | (1 << j)
        # Check if sorted: all 1s are in the lower positions (or upper? depends on convention)
        # min goes to i, max to j. So smaller values go to lower indices? 
        # Wait: "writes min to wire i and max to j". 
        # If i < j, and we put min at i, max at j, then after sorting, 
        # wire 0 has the global min, wire N-1 has the global max.
        # For binary, min=0, max=1. So sorted means 0s at low indices, 1s at high indices.
        # So the bitmask of 1s should be of the form: 00...011...1
        # i.e., the lowest k bits are 0, the highest (N-k) bits are 1.
        # If there are 'ones' ones, the mask should be ((1 << N) - 1) ^ ((1 << (N - ones)) - 1)
        # Let's check:
        ones = bin(state).count('1')
        expected = ((1 << N) - 1) ^ ((1 << (N - ones)) - 1) if N - ones > 0 else (1 << N) - 1
        # Actually, if ones=0, expected=0. If ones=N, expected=(1<<N)-1.
        # Let's re-derive:
        # Sorted ascending: wire 0 <= wire 1 <= ... <= wire N-1.
        # Binary: 0,0,...,0,1,1,...,1
        # So the 1s are in the highest positions.
        # If there are 'ones' ones, they occupy positions N-ones, ..., N-1.
        # Mask for these positions: sum(1 << p for p in range(N-ones, N))
        # = (1 << N) - (1 << (N - ones))
        if N - ones >= 0:
            expected = (1 << N) - (1 << (N - ones))
        else:
            expected = (1 << N) - 1
        if state != expected:
            return False
    return True

def _count_comparators(network):
    return len(network)

def _build_odd_even_transposition():
    """Standard odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _build_bitonic_merge(i, j, k):
    """
    Bitonic merge on wires [i, i+k) and [i+k, i+2k).
    Assumes first half is ascending, second is descending.
    Merges into ascending order on [i, i+2k).
    """
    network = []
    for d in range(k, 0, -1):
        for l in range(0, 2 * k, d):
            for m in range(l, l + d):
                if m < i + k and m + d < i + 2 * k:
                    network.append((i + m, i + m + d))
    return network

def _build_bitonic_sort():
    """Standard bitonic sort network for N=16."""
    network = []
    # Bitonic sort: for k=2,4,8,16:
    #   for j=k/2, k/4, ..., 1:
    #     for i=0, k, 2k, ...:
    #       merge(i, j)
    # But we need to create bitonic sequences first.
    
    # Standard bitonic sort algorithm:
    # for k = 2, 4, 8, 16:
    #   for j = k/2, k/4, ..., 1:
    #     for i = 0, k, 2k, ..., N-k:
    #       # compare-exchange (i, i+j), (i+j, i+2j), etc.
    #       for m = 0; m < j; m += 1:
    #         compare-exchange (i + m, i + m + j)
    
    # Wait, the standard bitonic sort uses a different structure.
    # Let's use the recursive definition:
    # BitonicSort(k):
    #   if k == 1: return []
    #   return BitonicSort(k/2) + BitonicSort(k/2) + Merge(k)
    # But this doesn't work directly because the inputs need to be bitonic.
    
    # Let's use the iterative version:
    # for k = 2, 4, 8, 16:
    #   for j = k/2, k/4, ..., 1:
    #     for i = 0, k, 2k, ..., N-k:
    #       for m = 0 to j-1:
    #         compare-exchange (i + m, i + m + j)
    
    # This creates a bitonic sequence and then sorts it.
    # Let's implement this.
    
    for k in [2, 4, 8, 16]:
        for j in range(k // 2, 0, -1):
            for i in range(0, N, k):
                for m in range(j):
                    network.append((i + m, i + m + j))
    
    return network

def _local_search(network, max_iter=5000):
    """
    Local search: try to remove comparators or swap them to reduce count while maintaining correctness.
    """
    current = list(network)
    best = current
    best_count = len(current)
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        improved = False
        
        # Try removing each comparator
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > DEADLINE - 5:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network(candidate):
                current = candidate
                best = current
                best_count = len(current)
                improved = True
                break  # restart after improvement
        
        if not improved:
            # Try swapping adjacent comparators
            swapped = False
            for idx in range(len(current) - 1):
                if time.time() > DEADLINE - 5:
                    break
                candidate = current[:]
                candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
                if _verify_network(candidate):
                    current = candidate
                    best = current
                    improved = True
                    swapped = True
                    break
            if not swapped:
                break  # stuck
    
    return best

def _random_search(max_iter=2000):
    """
    Random search: start from a valid network and try random modifications.
    """
    # Start from bitonic sort
    network = _build_bitonic_sort()
    best = network
    best_count = len(network)
    
    for _ in range(max_iter):
        if time.time() > DEADLINE - 5:
            break
        candidate = list(best)
        
        # Random modification
        op = random.choice(['remove', 'add', 'swap'])
        if op == 'remove' and len(candidate) > 0:
            idx = random.randint(0, len(candidate) - 1)
            candidate.pop(idx)
        elif op == 'add':
            i = random.randint(0, N - 1)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(candidate))
            candidate.insert(pos, (i, j))
        elif op == 'swap' and len(candidate) > 1:
            i = random.randint(0, len(candidate) - 1)
            j = random.randint(0, len(candidate) - 1)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        
        if _verify_network(candidate):
            if len(candidate) < best_count:
                best = candidate
                best_count = len(candidate)
    
    return best

def construct():
    """
    Construct a sorting network for N=16 wires.
    Returns a list of (i, j) pairs.
    """
    start_time = time.time()
    
    # Start with bitonic sort (known correct, ~50-60 comparators for N=16)
    network = _build_bitonic_sort()
    best = network
    best_count = len(network)
    
    # Verify bitonic sort
    if not _verify_network(best):
        # Fallback to odd-even transposition
        best = _build_odd_even_transposition()
        best_count = len(best)
    
    # Local search to improve
    if time.time() < DEADLINE - 10:
        improved = _local_search(best, max_iter=3000)
        if len(improved) < best_count:
            best = improved
            best_count = len(best)
    
    # Random search if time permits
    if time.time() < DEADLINE - 5:
        rand_result = _random_search(max_iter=1000)
        if len(rand_result) < best_count:
            best = rand_result
            best_count = len(best)
    
    # Final verification
    if not _verify_network(best):
        # Fallback
        best = _build_odd_even_transposition()
    
    return best

# EVOLVE-BLOCK-END
