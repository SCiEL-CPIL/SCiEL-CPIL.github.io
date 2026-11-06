"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    # Start with the known 32-vertex construction: 8 disjoint K4's
    n_base = 32
    edges_base = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges_base.append([base + i, base + j])
    
    best_n = n_base
    best_edges = edges_base[:]
    
    # Try to find a 33-vertex graph by modifying the base construction
    # Strategy: try adding a 33rd vertex with careful edge connections
    # and see if we can maintain C5-free and alpha < 9
    
    # First, let's try a different approach: use a complete multipartite graph
    # or other C5-free constructions that might give better n/alpha ratio
    
    # Actually, let's think about this differently.
    # The key insight is that we need C5-free and alpha <= 8.
    # K4 gives n=4, alpha=1, ratio 4.
    # Can we do better?
    
    # Consider K5: n=5, alpha=1, ratio 5. But K5 contains C5! So no.
    # Consider K4 minus an edge: n=4, alpha=2 (the two non-adjacent vertices). Ratio 2.
    # Consider K4: ratio 4.
    
    # What about C4? n=4, alpha=2. Ratio 2.
    # What about K3? n=3, alpha=1. Ratio 3.
    
    # So K4 is optimal for the ratio among small C5-free graphs.
    
    # Let's try to construct a 33-vertex graph by taking 8 K4's and adding one more vertex
    # that connects to some existing vertices to reduce the independence number.
    
    # Actually, let's try a completely different construction.
    # What about a complete bipartite graph? K_{m,n} is C5-free if min(m,n) <= 2 or max(m,n) <= 2.
    # K_{2,n}: n+2 vertices, alpha = max(2, n). For alpha <= 8, n <= 8. So n+2 <= 10. Not good.
    # K_{1,n}: n+1 vertices, alpha = n. For alpha <= 8, n <= 8. So n+1 <= 9. Not good.
    
    # What about a complete tripartite graph? K_{a,b,c} is C5-free?
    # A C5 in a complete tripartite graph would need to alternate between parts.
    # With 3 parts, a 5-cycle would need to visit parts in a pattern that repeats after 5 steps,
    # which is impossible with only 3 parts (since 5 and 3 are coprime, you'd need all parts to have
    # appropriate sizes, but the cycle would force a vertex to be in two parts).
    # Actually, complete multipartite graphs are C5-free if and only if they have at most 2 parts
    # or specific conditions. Let me reconsider.
    
    # Actually, K_{a,b,c} contains a C5 if and only if a,b,c are all at least 2? No, that's not right either.
    # Let me just stick with the K4 construction which is known to work.
    
    # Let's try a randomized local search to see if we can find a 33-vertex graph.
    # Start from the 32-vertex graph and try to add a vertex with various edge sets.
    
    for attempt in range(100):
        if time.time() > deadline:
            break
        
        # Try adding a 33rd vertex (vertex 32)
        # The new vertex can connect to any subset of the existing 32 vertices.
        # We need to ensure:
        # 1. No C5 is created
        # 2. alpha remains <= 8
        
        # For alpha: if the new vertex is isolated or connects to few vertices,
        # it might be part of an independent set of size 9.
        # To keep alpha <= 8, the new vertex must connect to enough vertices
        # that no independent set of size 9 can include it.
        
        # Actually, the simplest approach: make the new vertex connect to all vertices
        # in one K4 block. Then any independent set can include at most one vertex from
        # that K4 (the new vertex or one of the K4 vertices, but not both if they're connected).
        # Wait, if the new vertex connects to all 4 vertices of K4 block 0, then in any independent set,
        # we can include either the new vertex or at most one vertex from K4 block 0, but not both.
        # So the independence number would be 1 (from the new vertex or K4 block 0) + 7 (from other K4's) = 8.
        # That works!
        
        # But we need to check for C5. If the new vertex connects to all 4 vertices of K4 block 0,
        # then within K4 block 0, we have K4 plus a new vertex connected to all 4.
        # This is K5 minus... wait, no. The new vertex is connected to all 4 vertices of K4,
        # and the 4 vertices form a K4. So we have a K5! And K5 contains C5.
        # So that doesn't work.
        
        # Let's try connecting the new vertex to only 2 vertices of a K4 block.
        # Say vertices 0 and 1 of block 0.
        # Then the new vertex (32) is connected to 0 and 1.
        # Within block 0, we have K4 on {0,1,2,3} plus vertex 32 connected to 0 and 1.
        # Is there a C5? Let's see: 32-0-2-3-1-32. That's a C5! So that doesn't work.
        
        # What if we connect the new vertex to 3 vertices of a K4 block?
        # Say 0, 1, 2 of block 0.
        # Then 32-0-3-1-2-32? 32-0-3-1-2-32: 32 connected to 0, 0 connected to 3, 3 connected to 1, 1 connected to 2, 2 connected to 32. Yes, that's a C5.
        
        # It seems hard to add a vertex without creating a C5.
        
        # Let's try a different strategy: modify the existing graph by adding edges between blocks.
        # But adding edges between blocks might create C5's.
        
        # Given the difficulty, let's just return the 32-vertex construction.
        pass
    
    return {'n': best_n, 'edges': best_edges}
# EVOLVE-BLOCK-END
