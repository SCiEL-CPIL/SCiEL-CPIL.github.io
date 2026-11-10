"""Direct construction for dense odd-order graphs avoiding equal-degree path endpoints."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from collections import Counter

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    def is_valid(edge_set, n):
        deg = [0] * n
        adj = [set() for _ in range(n)]
        for (u, v) in edge_set:
            deg[u] += 1
            deg[v] += 1
            adj[u].add(v)
            adj[v].add(u)
        
        for (u, v) in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # Strategy:
    # 1. Start with a graph where Part A (size 7) has distinct degrees.
    # 2. Part B (size 8) can have edges among themselves.
    # 3. No edges within A.
    # This structure allows for higher edge counts if valid.
    
    A = list(range(7))
    B = list(range(7, 15))
    
    # Target degrees for A: 2,3,4,5,6,7,8 (distinct)
    # Sum = 35. Edges A-B = 35.
    # We can add edges within B.
    
    best_edges = set()
    best_count = 0
    
    # Baseline: K_{7,8}
    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))
    if is_valid(base_edges, n):
        best_edges = base_edges
        best_count = len(base_edges)
        
    # Construct a specific candidate:
    # A degrees: 2,3,4,5,6,7,8
    # Assign degrees to A vertices
    target_deg_A = [2, 3, 4, 5, 6, 7, 8]
    
    # We need to select edges A-B such that deg(a_i) = target_deg_A[i]
    # This is a bipartite degree sequence realization.
    # Since max degree is 8 (size of B) and min is 2, and sum is 35, it should be realizable.
    
    # Simple greedy assignment for A-B edges
    edges_AB = set()
    # Sort A by target degree descending
    # Sort B by index (all capacity 7 initially)
    
    # To ensure distinct degrees in A, we can just pick random subsets of B for each A
    # and check if degrees are distinct.
    
    random.seed(42)
    
    for restart in range(100):
        if time.time() > deadline:
            break
        
        # Generate a random bipartite graph A-B with distinct degrees for A
        # Degrees must be between 0 and 8.
        # Let's pick a random permutation of a subset of {0..8} of size 7?
        # No, we want high degrees.
        # Pick 7 distinct values from {0..8}.
        possible_degrees = list(range(0, 9))
        chosen_degrees = random.sample(possible_degrees, 7)
        
        # Try to realize this degree sequence
        # Sort chosen degrees descending
        chosen_degrees.sort(reverse=True)
        
        # Greedy: For each a in A, connect to 'deg' random b's
        # This might not work perfectly, but let's try.
        # Actually, we can just use a simple construction:
        # a_i connected to first d_i vertices in B?
        # If we do that, degrees of B will be structured.
        
        current_AB = set()
        valid_realization = True
        # Shuffle B to randomize
        B_shuffled = B[:]
        random.shuffle(B_shuffled)
        
        # To make it more likely to be valid, we should probably ensure
        # that the degrees of B are also somewhat distinct or high.
        
        # Let's just try to build it
        for i, a in enumerate(A):
            d = chosen_degrees[i]
            # Connect to d random vertices in B
            # Ensure no duplicates
            neighbors = random.sample(B_shuffled, d)
            for b in neighbors:
                current_AB.add((a, b))
                
        # Check if degrees of A are as desired
        deg_check = [0]*n
        for u,v in current_AB:
            deg_check[u]+=1
            deg_check[v]+=1
            
        if sorted([deg_check[a] for a in A]) != sorted(chosen_degrees):
            continue
            
        # Now add edges within B
        # Start with empty B-B
        current_BB = set()
        
        # Try to add all possible B-B edges
        all_BB = list(combinations(B, 2))
        random.shuffle(all_BB)
        
        current_edges = set(current_AB)
        
        for e in all_BB:
            if time.time() > deadline:
                break
            trial = current_edges | {e}
            if is_valid(trial, n):
                current_edges = trial
        
        if len(current_edges) > best_count:
            best_edges = current_edges
            best_count = len(current_edges)
            
    # Also try the reverse: Distinct degrees in B, edges in A?
    # We proved that's likely worse (max 49), but let's be safe.
    # Actually, K_{7,8} is 56.
    
    # Final verification
    if not is_valid(best_edges, n):
        # Fallback to K_{7,8}
        best_edges = set()
        for a in A:
            for b in B:
                best_edges.add((a, b))
        best_count = len(best_edges)
        
    result = [list(e) for e in sorted(best_edges)]
    final = []
    for e in result:
        final.append([int(e[0]), int(e[1])])
    return final

# EVOLVE-BLOCK-END
