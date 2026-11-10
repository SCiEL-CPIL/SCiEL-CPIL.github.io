import time
from math import gcd
from collections import defaultdict
import random

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 128
    
    # Precompute LCM table
    lcm_table = [[0]*(n+1) for _ in range(n+1)]
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Find all bad triples
    bad_triples = []
    triple_set = set()
    for a in range(1, n+1):
        for b in range(a+1, n+1):
            lab = lcm_table[a][b]
            for c in range(b+1, n+1):
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    bad_triples.append((a, b, c))
                    triple_set.add((a,b,c))
    
    # For each element, list of pairs (u, w) such that (elem, u, w) is a bad triple
    # This allows fast check: if adding elem, are any of its conflict pairs fully present?
    # We store pairs as frozenset or just tuple (u,w) with u < w for consistency?
    # Actually, we just need to check if both u and w are in the current set.
    conflicts = defaultdict(list)
    for a, b, c in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))
    
    # Function to check if adding x to current set S is valid
    # S is a set of integers
    def can_add(S, x):
        for u, w in conflicts[x]:
            if u in S and w in S:
                return False
        return True

    # Function to check if removing x from S keeps it valid (always true)
    # Function to get size
    def size(S):
        return len(S)

    best_size = 0
    best_set = set()
    
    # Run multiple randomized greedy initializations
    num_restarts = 20
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        random.seed(12345 + restart)
        
        # Greedy Set Cover to find minimum vertex cover (to remove)
        # We want to remove minimum vertices to hit all bad triples.
        # This is equivalent to finding a maximum independent set in the hypergraph.
        # Greedy: repeatedly pick the vertex that covers the most *remaining* bad triples.
        
        remaining_triples = set(bad_triples)
        selected = set(range(1, n+1)) # Start with all, remove as we go? 
        # Or start with empty and add? 
        # The previous code started with all and removed. 
        # Let's stick to that: Start with all, remove vertices to break triples.
        
        current_set = set(range(1, n+1))
        
        # Precompute for each vertex, which triples it hits
        vertex_to_triples = defaultdict(list)
        for t in bad_triples:
            for v in t:
                vertex_to_triples[v].append(t)
        
        active_triples = set(bad_triples)
        
        # Greedy removal
        while active_triples:
            if time.time() > deadline - 25:
                break
            best_v = None
            best_count = 0
            # Randomize tie-breaking
            candidates = list(vertex_to_triples.keys())
            random.shuffle(candidates)
            
            for v in candidates:
                if v not in current_set:
                    continue
                count = 0
                for t in vertex_to_triples[v]:
                    if t in active_triples:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            
            if best_v is None or best_count == 0:
                break
                
            current_set.remove(best_v)
            # Update active triples
            to_remove = set()
            for t in vertex_to_triples[best_v]:
                if t in active_triples:
                    to_remove.add(t)
            active_triples -= to_remove
        
        # Now current_set is a valid solution.
        # Try to improve it by adding back vertices.
        
        # Local Search: Try to add vertices
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for x in range(1, n+1):
                if x in current_set:
                    continue
                if can_add(current_set, x):
                    current_set.add(x)
                    improved = True
        
        # Local Search: Swaps (1 out, 1 in)
        # Try to swap out a vertex y and in a vertex x to maintain or increase size?
        # Actually, if we can just add, size increases.
        # If we can't add, maybe swapping allows future additions?
        # Let's try simple swaps that maintain size but might unblock additions.
        # Or just try to add after removing one vertex?
        
        # Phase 2: 1-out, 1-in swaps to potentially enable further additions
        # This is complex. Let's try a simpler "Kick and Fill"
        # Pick a random vertex y in current_set, remove it, then try to add as many as possible.
        # If the size increases, keep it.
        
        for _ in range(50):
            if time.time() > deadline - 10:
                break
            if not current_set:
                break
            y = random.choice(list(current_set))
            old_size = len(current_set)
            
            # Remove y
            current_set.remove(y)
            
            # Try to add all possible
            added = []
            for x in range(1, n+1):
                if x in current_set:
                    continue
                if can_add(current_set, x):
                    current_set.add(x)
                    added.append(x)
            
            new_size = len(current_set)
            if new_size <= old_size:
                # Revert
                for x in added:
                    current_set.remove(x)
                current_set.add(y)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Final polish on best_set
    # Try to add more
    for x in range(1, n+1):
        if x not in best_set and can_add(best_set, x):
            best_set.add(x)
            
    result = list(best_set)
    if not result:
        result = [1]
        
    return result
