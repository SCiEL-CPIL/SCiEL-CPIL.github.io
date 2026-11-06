"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

N = 40

# EVOLVE-BLOCK-START
def construct():
    # Start time to enforce deadline
    start_time = time.time()
    deadline = start_time + 80  # 80s out of 90s limit
    
    # Base cycle edges: (0,1), (1,2), ..., (39,0)
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Chords: (0, k) for k in 2..20
    # These generate cycles of lengths k+1 and 41-k
    # k=2 -> 3, 39
    # k=3 -> 4, 38
    # ...
    # k=20 -> 21, 21
    for k in range(2, 21):
        edges.add((0, k))
        
    # Convert to list for output
    edge_list = list(edges)
    
    # Construct cycles
    # We need one vertex-list cycle for every length 3..40
    cycles = []
    
    # Helper to get the path from u to v on the base cycle in the "short" or specific direction
    # Since we have chords (0, k), the cycles are:
    # 1. 0 -> 1 -> ... -> k -> 0 (length k+1)
    # 2. 0 -> k -> k+1 -> ... -> 39 -> 0 (length 41-k)
    
    for length in range(3, N + 1):
        if length == N:
            # The base cycle
            cycles.append(list(range(N)))
        else:
            # Find k such that k+1 == length or 41-k == length
            # Case 1: k+1 == length => k = length - 1
            # Case 2: 41-k == length => k = 41 - length
            
            k1 = length - 1
            k2 = 41 - length
            
            valid_k = None
            if 2 <= k1 <= 20:
                valid_k = k1
                # Cycle: 0, 1, ..., k, 0
                cycle = list(range(k + 1)) # 0 to k
                # This is a path 0..k. Adding edge (k,0) closes it.
                # The list of vertices is 0, 1, ..., k.
                # Length is k+1.
                cycles.append(cycle)
            elif 2 <= k2 <= 20:
                valid_k = k2
                # Cycle: 0, k, k+1, ..., 39, 0
                # Vertices: 0, k, k+1, ..., 39
                cycle = [0] + list(range(k, N))
                # This path goes 0 -> k (chord) -> k+1 ... -> 39 -> 0 (base)
                # Length is 1 + (39 - k + 1) = 41 - k.
                cycles.append(cycle)
            else:
                # This should not happen for the given construction
                # Fallback: use base cycle if length is 40, else error
                # But our construction covers all 3..39
                pass

    return {'edges': edge_list, 'cycles': cycles}
# EVOLVE-BLOCK-END
