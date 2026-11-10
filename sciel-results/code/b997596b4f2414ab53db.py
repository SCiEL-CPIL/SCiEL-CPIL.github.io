import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin for 60s limit
    
    N = 256
    # 1. Build the conflict graph
    # adj[i] is a bitset (Python int) where bit j is 1 if i and j are connected (conflict)
    adj = [0] * (N + 1)
    
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Maximum Independent Set via Branch and Bound
    # We use a bitset to represent the set of available vertices.
    # We maintain a global best_size and best_set.
    
    best_size = 0
    best_set = 0
    
    # Helper to count bits
    def popcount(x):
        return x.bit_count()
    
    # We need an order of vertices to branch on. 
    # A static order based on initial degree is often good, but dynamic is better.
    # For simplicity and speed in Python, we pick the vertex with the highest degree in the current candidate set.
    # To make this fast, we can just iterate through bits.
    
    def solve(candidates, current_size, current_set):
        nonlocal best_size, best_set
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        if not candidates:
            if current_size > best_size:
                best_size = current_size
                best_set = current_set
            return
            
        # Upper bound: current_size + popcount(candidates)
        # If this doesn't beat best_size, prune
        if current_size + popcount(candidates) <= best_size:
            return
            
        # Select a vertex to branch on
        # Heuristic: Pick vertex with highest degree in the induced subgraph of candidates
        # To avoid O(N^2) per step, we can just pick the first one or use a precomputed order.
        # Given N=256, finding max degree is fast enough.
        
        # Find vertex with max degree in candidates
        max_deg = -1
        v = -1
        c = candidates
        while c:
            # Get lowest set bit
            lsb = c & -c
            idx = lsb.bit_length() - 1
            # Degree of idx in candidates
            deg = popcount(adj[idx] & candidates)
            if deg > max_deg:
                max_deg = deg
                v = idx
            c ^= lsb
            
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates = candidates & ~adj[v] & ~(1<<v)
        new_candidates = candidates & ~adj[v] & ~(1 << v)
        solve(new_candidates, current_size + 1, current_set | (1 << v))
        
        # Branch 2: Exclude v
        # New candidates = candidates & ~(1<<v)
        new_candidates = candidates & ~(1 << v)
        solve(new_candidates, current_size, current_set)

    # Initial candidates: all vertices 1..256
    initial_candidates = 0
    for i in range(1, N + 1):
        initial_candidates |= (1 << i)
        
    solve(initial_candidates, 0, 0)
    
    # If best_size is 0, something went wrong, return a safe fallback (e.g., all odds)
    if best_size == 0:
        return list(range(1, 257, 2))
        
    result = []
    for i in range(1, N + 1):
        if best_set & (1 << i):
            result.append(i)
            
    return result

# For local testing / sanity check if run directly
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            a, b = res[i], res[j]
            if (a*b) % (a+b) == 0:
                print(f"Conflict found: {a}, {b}")
                sys.exit(1)
    print("Valid solution")
