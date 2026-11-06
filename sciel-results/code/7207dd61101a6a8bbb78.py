import time
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    # We'll try to find the largest n with a valid coloring
    # Start with known constructions and try to extend
    
    # First, return a known valid coloring as fallback
    # Paley graph on 17 vertices
    def paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        colors = [0 if (j - i) % 17 in squares else 1 
                  for i in range(17) for j in range(i + 1, 17)]
        return {'n': 17, 'colors': colors}
    
    best_result = paley_17()
    
    # Try to find larger colorings using backtracking
    # For n vertices, there are n*(n-1)//2 edges
    # We'll use a bitset-based approach
    
    def check_no_k222(n, colors):
        """Check if neither color class contains K_{2,2,2}."""
        # Build adjacency bitsets for each color
        adj = [[0] * n for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[i][j] = 1 << c
                adj[j][i] = 1 << c
                idx += 1
        
        # For each color, check if K_{2,2,2} exists
        # K_{2,2,2} has 6 vertices partitioned into 3 pairs {a,b}, {c,d}, {e,f}
        # where all 12 cross-edges are present in that color
        # Equivalently: there exist 6 vertices with a specific structure
        
        # More efficient: for each color, build the graph and check
        for color in [0, 1]:
            # Build adjacency for this color
            g = [0] * n  # bitset representation
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == color:
                        g[i] |= (1 << j)
                        g[j] |= (1 << i)
                    idx += 1
            
            # Check for K_{2,2,2}: need 6 vertices a,b,c,d,e,f
            # such that all edges between {a,b}, {c,d}, {e,f} are in color
            # i.e., a-c, a-d, a-e, a-f, b-c, b-d, b-e, b-f, c-e, c-f, d-e, d-f all present
            # And no requirement on a-b, c-d, e-f
            
            # For small n, we can check all 6-tuples
            if n <= 31:
                for a in range(n):
                    for b in range(a + 1, n):
                        # a and b are one pair
                        # We need to find c,d,e,f from remaining vertices
                        # such that c,d form a pair connected to both a and b
                        # and e,f form a pair connected to both a and b
                        # and c,e,c,f,d,e,d,f all connected
                        
                        # Common neighbors of a and b in this color
                        common_ab = g[a] & g[b]
                        # Remove a and b
                        common_ab &= ~(1 << a)
                        common_ab &= ~(1 << b)
                        
                        # We need to pick c,d from common_ab such that c-d edge exists
                        # and e,f from common_ab \ {c,d} such that e-f edge exists
                        # and c-e, c-f, d-e, d-f all exist
                        
                        # Try all pairs (c,d) from common_ab
                        bits = list(range(n))
                        for c in range(n):
                            if not (common_ab >> c) & 1:
                                continue
                            for d in range(c + 1, n):
                                if not (common_ab >> d) & 1:
                                    continue
                                if not (g[c] >> d) & 1:
                                    continue
                                # c and d are a valid pair
                                # Now find e,f from common_ab \ {a,b,c,d}
                                remaining = common_ab & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                                # Need e,f in remaining with e-f edge, and
                                # c-e, c-f, d-e, d-f all present
                                for e in range(n):
                                    if not (remaining >> e) & 1:
                                        continue
                                    if not (g[c] >> e) & 1:
                                        continue
                                    if not (g[d] >> e) & 1:
                                        continue
                                    for f in range(e + 1, n):
                                        if not (remaining >> f) & 1:
                                            continue
                                        if not (g[c] >> f) & 1:
                                            continue
                                        if not (g[d] >> f) & 1:
                                            continue
                                        if not (g[e] >> f) & 1:
                                            continue
                                        # Found K_{2,2,2}
                                        return False
        return True
    
    # Backtracking solver
    def solve(n, max_edges=None):
        """Try to find a valid 2-coloring of K_n avoiding monochromatic K_{2,2,2}."""
        if max_edges is None:
            max_edges = n * (n - 1) // 2
        
        colors = [None] * max_edges
        adj0 = [0] * n  # bitsets for color 0
        adj1 = [0] * n  # bitsets for color 1
        
        def edge_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
        def would_create_k222(color, i, j):
            """Check if adding edge (i,j) in given color creates K_{2,2,2}."""
            if color == 0:
                adj = adj0
            else:
                adj = adj1
            
            # Adding edge (i,j) to this color's graph
            # Check if this creates a K_{2,2,2}
            # K_{2,2,2} containing edge (i,j): 
            # Either (i,j) is within a pair, or (i,j) is a cross-edge
            
            # Case 1: (i,j) is a cross-edge in K_{2,2,2}
            # Then there exist two other vertices a,b in one part and c,d in another
            # such that i and j are in different parts
            # This is complex; for simplicity, do a full check after adding
            
            # Simpler: just add the edge and check locally
            # For efficiency, check if there's a K_{2,2,2} using the new edge
            
            # New edge (i,j) in this color. For K_{2,2,2} using this edge:
            # The edge (i,j) must be a cross-edge between two of the three pairs.
            # So i is in one pair {i, x} and j is in another pair {j, y},
            # and the third pair is {u, v}.
            # All cross-edges must be present: i-j (new), i-u, i-v, j-u, j-v, x-u, x-v, x-y? No...
            # Actually in K_{2,2,2}, pairs are {a,b}, {c,d}, {e,f}.
            # Cross-edges are all edges between different pairs.
            # If (i,j) is a cross-edge, say i in pair 1, j in pair 2.
            # Then pair 1 = {i, i2}, pair 2 = {j, j2}, pair 3 = {u, v}.
            # Required edges: i-j (this one), i-u, i-v, i2-j, i2-u, i2-v, j-u, j-v, j2-u, j2-v, u-v? No, u-v is within pair 3, not required.
            # Wait, in K_{2,2,2}, edges are ALL between different pairs. So u-v is NOT an edge.
            # Required: i-j, i-u, i-v, i2-j, i2-u, i2-v, j-u, j-v, j2-u, j2-v.
            # That's 10 other edges plus i-j = 11 edges. But K_{2,2,2} has 12 edges.
            # I missed: the edge between i2 and j2! Yes, i2 is in pair 1, j2 in pair 2, so i2-j2 is a cross-edge.
            # So required: i-j, i-u, i-v, i2-j, i2-u, i2-v, j-u, j-v, j2-u, j2-v, i2-j2.
            # That's 11 edges total including i-j. But K_{2,2,2} has C(6,2) - 3 = 15 - 3 = 12 edges.
            # I'm missing one. Let me recount: pairs {i,i2}, {j,j2}, {u,v}.
            # Cross-edges: i-j, i-j2, i-u, i-v, i2-j, i2-j2, i2-u, i2-v, j-u, j-v, j2-u, j2-v = 12.
            # So we need all 12. The new edge is i-j. We need the other 11.
            
            # For each possible i2 (partner of i in its pair):
            for i2 in range(n):
                if i2 == i:
                    continue
                # i2 must be connected to j, u, v, j2 in this color
                # But we don't know j2, u, v yet.
                # This is getting complex. Let me just do a full check.
                pass
            
            # For correctness and simplicity, do a local check:
            # Temporarily add the edge and check for K_{2,2,2} in this color
            # that uses this edge.
            
            # Alternative: check all 6-sets containing i and j
            # For n <= 31, this is feasible but slow.
            
            # Let me use a different approach: precompute and check
            # Actually, for the backtracking to be efficient, I need a fast check.
            
            # Let me just return True (no pruning) for now and rely on the final check.
            # This will be slow but correct.
            return False
        
        # Given the complexity, let me use a simpler approach:
        # Try known constructions and small perturbations
        
        # Actually, let me try a randomized local search approach
        # Start from a valid coloring and try to extend
        
        return None
    
    # Try known larger constructions
    # The record is 29. Let me try to find one.
    
    # For now, let me try a simple backtracking for small n to verify,
    # then try to construct larger ones.
    
    # Let me try a different approach: use the fact that we can verify
    # colorings quickly, and use a smart search.
    
    # Try n from 18 upward, using backtracking with good pruning
    # But this is very expensive. Instead, let me try known constructions.
    
    # A known approach: use quadratic residues or other algebraic constructions.
    
    # For the purpose of this task, let me try to find a valid coloring
    # for n up to 29 using a combination of known methods.
    
    # Try the Paley graph on prime p where p ≡ 1 mod 4
    # Paley graph on p vertices: edge (i,j) if (j-i) is a quadratic residue mod p
    # The Paley graph on 17 works. Let me try other primes.
    
    def try_paley(p):
        """Try Paley construction on p vertices."""
        if p < 2:
            return None
        squares = {x * x % p for x in range(1, p)}
        colors = [0 if (j - i) % p in squares else 1 
                  for i in range(p) for j in range(i + 1, p)]
        if check_no_k222(p, colors):
            return {'n': p, 'colors': colors}
        return None
    
    # Try various primes
    for p in [17, 19, 23, 29, 31]:
        if time.time() > deadline:
            break
        result = try_paley(p)
        if result and result['n'] > best_result['n']:
            best_result = result
    
    # Try to extend the best result found
    # For now, return the best we have
    return best_result
# EVOLVE-BLOCK-END
