"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    
    N = 180
    # Precompute conflict triples: (a, b, c) distinct, 1/a = 1/b + 1/c
    # For each pair (b, c) with b < c, compute a = bc/(b+c). If a is integer, 1 <= a <= N, a != b, a != c, then (min, mid, max) is a conflict.
    triples = []
    triple_set = set()
    for b in range(1, N + 1):
        for c in range(b + 1, N + 1):
            num = b * c
            den = b + c
            if num % den == 0:
                a = num // den
                if 1 <= a <= N and a != b and a != c:
                    t = tuple(sorted((a, b, c)))
                    if t not in triple_set:
                        triple_set.add(t)
                        triples.append(t)
    
    # For each element, list of triples it participates in
    elem_to_triples = [[] for _ in range(N + 1)]
    for t in triples:
        for x in t:
            elem_to_triples[x].append(t)
    
    # Conflict graph: two elements conflict if they appear together in some triple
    # Actually, we need to check if adding an element creates a triple with two already-present elements.
    # But for the vertex cover approach, we can build a graph where edges are conflicts? No, it's hyperedges.
    # Let's use a different approach: Local Search.
    
    # Start with the seed solution: [91..180]
    S = set(range(91, N + 1))
    
    def count_violations(S):
        # Count triples (a,b,c) in S where 1/a = 1/b + 1/c
        # This is O(|triples|)
        v = 0
        for t in triples:
            if t[0] in S and t[1] in S and t[2] in S:
                v += 1
        return v
    
    # Check if seed is valid
    violations = count_violations(S)
    if violations == 0:
        best_size = len(S)
    else:
        # Fix seed by removing elements
        # Simple greedy: while violations > 0, remove element with highest degree in violation graph
        while violations > 0 and time.time() < deadline:
            # Find a violating triple
            # We can just pick any violating triple
            v = 0
            for t in triples:
                if t[0] in S and t[1] in S and t[2] in S:
                    # Remove one of them. Which one?
                    # Heuristic: remove the one that appears in the most triples
                    # Count appearances in ALL triples (not just violating) as a proxy
                    best_x = None
                    best_count = -1
                    for x in t:
                        count = len(elem_to_triples[x])
                        if count > best_count:
                            best_count = count
                            best_x = x
                    S.remove(best_x)
                    # Recount
                    violations = count_violations(S)
                    break
    
    best_size = len(S)
    best_S = set(S)
    
    # Local search to maximize size
    # We can try to add elements back, or swap elements
    # Let's try a simple hill climbing: try to add each element not in S, if it doesn't create violations, add it.
    # Repeat until no improvement.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x not in S:
                S.add(x)
                if count_violations(S) == 0:
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = set(S)
                    improved = True
                else:
                    S.remove(x)
    
    # Now try to remove one element and add two others? Or just try removing one element to allow adding more?
    # This is getting complex. Let's try a simpler approach: 
    # Start with the set of all numbers, and remove the minimum vertex cover of the hypergraph.
    # Since the hypergraph has small arity (3), we can use a greedy approach:
    # While there is a violating triple, remove the element with the highest degree.
    # But we want to MAXIMIZE the set, so we want to MINIMIZE the removals.
    # This is exactly the minimum vertex cover problem for hypergraphs.
    # Since the hypergraph is 3-uniform, we can use approximation algorithms.
    # But let's stick to local search.
    
    # Try a different initial solution: start with all numbers, and remove elements to fix violations.
    S2 = set(range(1, N + 1))
    violations2 = count_violations(S2)
    while violations2 > 0 and time.time() < deadline:
        # Find a violating triple
        found = False
        for t in triples:
            if t[0] in S2 and t[1] in S2 and t[2] in S2:
                # Remove the element with the highest degree
                best_x = None
                best_count = -1
                for x in t:
                    count = len(elem_to_triples[x])
                    if count > best_count:
                        best_count = count
                        best_x = x
                S2.remove(best_x)
                violations2 = count_violations(S2)
                found = True
                break
        if not found:
            break
    
    if len(S2) > best_size:
        best_size = len(S2)
        best_S = set(S2)
    
    # Try to add elements back to S2
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x not in S2:
                S2.add(x)
                if count_violations(S2) == 0:
                    if len(S2) > best_size:
                        best_size = len(S2)
                        best_S = set(S2)
                    improved = True
                else:
                    S2.remove(x)
    
    # Try swapping: remove one element, add two others?
    # Or remove two elements, add three others?
    # Let's try: for each element x in S, try to remove it and add any two elements not in S, if the new set is valid.
    # This is O(|S| * N^2 * |triples|) which might be too slow.
    
    # Let's try a simpler heuristic: 
    # Iterate over all pairs (x, y) not in S, and see if we can add both.
    # Or iterate over all triples of elements not in S, and see if we can add all three.
    
    # Actually, let's just return the best set found.
    return sorted(best_S)

# EVOLVE-BLOCK-END
