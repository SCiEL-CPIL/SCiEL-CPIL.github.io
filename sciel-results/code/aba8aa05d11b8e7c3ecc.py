import time
import random

N = 40

def construct():
    # Base Hamiltonian cycle
    edges = set()
    for i in range(N):
        edges.add((i, (i + 1) % N))
    
    # Add chords of distances 2 to 19 from vertex 0
    # This covers lengths 3..20 and 22..39
    for d in range(2, 20):
        edges.add((0, d))
        
    # Current edge count: 40 + 18 = 58
    # Missing length: 21
    # Check if a 21-cycle exists
    # If not, add a chord to ensure it
    
    # For efficiency, we assume the star topology might miss 21.
    # We add a chord (10, 31) which has distance 21 (or 19).
    # This adds 1 edge, total 59.
    # But we want to try 58 first.
    
    # Let's just return the 59 edge solution to be safe and fast, 
    # as finding a 58 edge pancyclic graph is non-trivial.
    # However, the inspiration says 59 is 1.947.
    # If I can get 58, score is 2.05.
    
    # Let's try to verify 21-cycle in the 58-edge graph.
    # Graph: 0 connected to 1..19, 39. Path 1-2-...-39.
    # 21-cycle must pass through 0.
    # Needs path of length 19 between two neighbors of 0.
    # Neighbors: 1..19, 39.
    # Distances in path 1-...-39:
    # |u - v| = 19.
    # Pairs: (1, 20) - 20 not neighbor.
    # (2, 21) - 21 not neighbor.
    # ...
    # (20, 39) - 20 not neighbor.
    # So no 21-cycle.
    
    # So 58 edges with this topology fails.
    # We need a different topology or 59 edges.
    
    # Let's use 59 edges with a better distribution to be safe?
    # Or just add the missing chord.
    
    edges.add((0, 20)) # Distance 20. Covers 21, 21.
    
    # Total edges: 59.
    # This is the same as the inspiration.
    # To beat it, we need 58.
    
    # Let's try a different 58-edge construction.
    # Chords: (0,2)..(0,18) (17 chords) and (19, 39) (1 chord, dist 20).
    # Total 18 chords.
    # Covers:
    # d=2..18 from 0: lengths 3..19 and 23..39.
    # d=20 from (19,39): lengths 21, 21.
    # Missing: 20, 22.
    # Can we form 20 and 22?
    # Likely yes with mixed chords.
    
    # Given the risk, I will stick to the 59 edge solution which is known to work, 
    # but optimize the cycle finding to be fast.
    
    # Actually, if I return 59, score is 1.947.
    # If I return 58 (if valid), score is 2.05.
    
    # I will attempt the 58 edge construction with (0,2)..(0,19) and (10, 31) is NOT added.
    # I will add (10, 29) (dist 19).
    # This might help form 21.
    
    # For now, I will return the 59 edge solution to guarantee validity and beat 1.0.
    # The task asks to "rewrite ... with a stronger algorithm".
    # 59 is stronger than 63.
    
    edge_list = list(edges)
    
    # Generate cycles
    cycles = []
    # Length 3..40
    # We can construct them explicitly using the chords.
    # For length k, if k <= 20, use chord (0, k-1).
    # Cycle: 0, 1, ..., k-1, 0.
    for k in range(3, 21):
        cycles.append(list(range(k)))
        
    # For length k > 20, use chord (0, 41-k).
    # Distance d = 41-k.
    # Cycle: 0, d, d-1, ..., 1, 0? No.
    # Chord (0, d). Path 0->1->...->d->0 is length d+1.
    # Other path: 0->39->38->...->d->0 is length 40-d+1 = 41-d = k.
    # Vertices: 0, 39, 38, ..., d.
    for k in range(22, 41):
        d = 41 - k
        # Cycle: 0, 39, 38, ..., d
        # List: [0, 39, 38, ..., d]
        cycle_vertices = [0] + list(range(39, d - 1, -1))
        cycles.append(cycle_vertices)
        
    # Length 21
    # Chord (0, 20). Distance 20.
    # Cycle: 0, 1, ..., 20, 0.
    cycles.append(list(range(21)))
    
    # Reorder to 3..40
    # My loop added 3..20, then 22..40, then 21.
    # Need to sort by length.
    # The task requires "one vertex-list cycle of every length 3..40 in that order".
    
    final_cycles = [None] * 38
    for k in range(3, 41):
        if k <= 20:
            final_cycles[k-3] = list(range(k))
        elif k == 21:
            final_cycles[k-3] = list(range(21))
        else:
            d = 41 - k
            final_cycles[k-3] = [0] + list(range(39, d - 1, -1))
            
    return {'edges': edge_list, 'cycles': final_cycles}
