"""Improved program for K16 edge-colouring: minimise colours subject to every K4 using >=5 colours."""
import itertools
import random

def construct():
    N = 16
    # Precompute all K4s
    vertices = list(range(N))
    k4s = list(itertools.combinations(vertices, 4))
    
    # Each edge is indexed by (i, j) with i < j
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_to_idx = {(i, j): idx for idx, (i, j) in enumerate(edges)}
    
    # Map each K4 to the list of edge indices in it
    k4_edges = []
    for k4 in k4s:
        idxs = []
        for a in range(4):
            for b in range(a + 1, 4):
                idxs.append(edge_to_idx[(k4[a], k4[b])])
        k4_edges.append(idxs)
    
    # We'll use a greedy approach with backtracking to find a valid colouring
    # with as few colours as possible.
    
    # First, let's try to construct with 15 colours using a systematic approach.
    
    # Strategy: Use a probabilistic method / local search.
    # Start with a random assignment of 15 colours, then iteratively fix violations.
    
    def count_violations(colours, k4_edges):
        """Count K4s that have fewer than 5 distinct colours."""
        v = 0
        for edge_idxs in k4_edges:
            cs = set()
            for idx in edge_idxs:
                cs.add(colours[idx])
            if len(cs) < 5:
                v += 1
        return v
    
    def is_valid(colours, k4_edges):
        for edge_idxs in k4_edges:
            cs = set()
            for idx in edge_idxs:
                cs.add(colours[idx])
            if len(cs) < 5:
                return False
        return True
    
    # Try different numbers of colours from 13 upwards
    # But first let's see if 15 is achievable quickly.
    
    # Use simulated annealing / local search
    best_k = 15  # Start with 15 colours
    
    for k in [15, 14, 13]:
        # Try to find a valid colouring with k colours
        best = None
        best_violations = float('inf')
        
        # Multiple random restarts
        for restart in range(500):
            # Random initial assignment
            colours = [random.randint(0, k - 1) for _ in edges]
            
            # Local search: repeatedly fix random violations
            for iteration in range(2000):
                # Count violations
                violations = []
                for ki, edge_idxs in enumerate(k4_edges):
                    cs = set()
                    for idx in edge_idxs:
                        cs.add(colours[idx])
                    if len(cs) < 5:
                        violations.append(ki)
                
                if not violations:
                    best = colours[:]
                    best_violations = 0
                    break
                
                # Pick a random violated K4
                ki = random.choice(violations)
                edge_idxs = k4_edges[ki]
                
                # Get current colours in this K4
                cs = set()
                for idx in edge_idxs:
                    cs.add(colours[idx])
                
                # Need to change some edges to introduce new colours
                # Try changing a random edge to a new colour
                for _ in range(10):
                    idx = random.choice(edge_idxs)
                    new_colour = random.randint(0, k - 1)
                    if new_colour not in cs:
                        colours[idx] = new_colour
                        break
                else:
                    # If no new colour found, just change randomly
                    idx = random.choice(edge_idxs)
                    colours[idx] = random.randint(0, k - 1)
            
            v = count_violations(colours, k4_edges)
            if v < best_violations:
                best_violations = v
                best = colours[:]
            
            if best_violations == 0:
                break
        
        if best is not None and best_violations == 0:
            return best
    
    # Fallback: use the sum-mod-17 construction which is known valid
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
