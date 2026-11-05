"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    # Paley colouring of K29 as baseline.
    p = 29
    squares = {x * x % p for x in range(1, p)}
    
    # Build adjacency bitmasks for color 0 (edges) in the Paley graph on 29 vertices
    # adj0[i] is a bitmask where bit j is set if edge (i,j) has color 0
    adj0 = [0] * p
    adj1 = [0] * p  # complement: color 1 edges
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
    
    # Check for K_{2,2,2} in a color class using bitmasks
    # K_{2,2,2} is a complete tripartite graph with parts of size 2
    # It exists if there are vertices a,b,c,d,e,f such that all cross-edges exist
    # More efficiently: K_{2,2,2} exists iff there exist two vertices u,v 
    # whose common neighborhood (in the given color) contains a K_{2,2}
    # K_{2,2} in a set S exists iff there are x,y in S such that 
    # N(x) ∩ N(y) ∩ S has size >= 2
    
    def has_k222(adj, n):
        """Check if the graph defined by adjacency bitmasks 'adj' on n vertices 
        contains K_{2,2,2} as a subgraph."""
        # For each pair (u,v), compute common neighbors
        # Then check if those common neighbors contain a K_{2,2}
        # K_{2,2} in subset S: exists x,y in S with |N(x)∩N(y)∩S| >= 2
        
        for u in range(n):
            for v in range(u + 1, n):
                # Common neighbors of u and v
                common = adj[u] & adj[v]
                # Remove u and v themselves
                common &= ~(1 << u)
                common &= ~(1 << v)
                if common == 0:
                    continue
                
                # Check if common neighbors contain K_{2,2}
                # K_{2,2} means: exist x,y in common such that 
                # adj[x] & adj[y] & common has at least 2 bits set
                # But we need x,y to be in common, and their common neighbors 
                # within common to form the other part of K_{2,2}
                
                # Iterate over pairs in common
                bits = []
                temp = common
                while temp:
                    b = temp & (-temp)
                    bits.append(b.bit_length() - 1)
                    temp &= temp - 1
                
                if len(bits) < 4:
                    continue
                    
                # Check for K_{2,2} in the induced subgraph on 'bits'
                # K_{2,2}: partition into two sets of 2 where all cross-edges exist
                # Equivalently: exists x,y such that adj[x] & adj[y] & common_mask 
                # contains at least 2 vertices from common (which would be the other part)
                # Actually for K_{2,2}: we need x,y,z,w all in common such that
                # x-z, x-w, y-z, y-w all exist. This means adj[x] & adj[y] & common
                # should contain z and w.
                
                common_mask = common
                for i_idx in range(len(bits)):
                    x = bits[i_idx]
                    for j_idx in range(i_idx + 1, len(bits)):
                        y = bits[j_idx]
                        # Common neighbors of x and y within common
                        cn = adj[x] & adj[y] & common_mask
                        # Remove x and y
                        cn &= ~(1 << x)
                        cn &= ~(1 << y)
                        if cn != 0:
                            # At least one common neighbor, but we need at least 2
                            # for K_{2,2} (the other part has 2 vertices)
                            # Wait: K_{2,2} has parts {x,y} and {z,w}
                            # So we need |adj[x] & adj[y] & common| >= 2 (excluding x,y)
                            # But actually z and w must both be in common and both
                            # adjacent to x and y
                            if (cn & (cn - 1)) != 0:  # at least 2 bits set
                                return True
        return False
    
    # Verify Paley(29) is valid
    if has_k222(adj0, p) or has_k222(adj1, p):
        # Should not happen, but safety check
        pass
    
    # Try to extend to n=30
    deadline = time.time() + 140  # Leave 10s safety margin
    
    # Strategy: Try to add vertex 29 (index 29) with appropriate colors
    # For each possible color assignment of vertex 29 to vertices 0..28,
    # check if the resulting graph is K_{2,2,2}-free in both colors
    
    # We'll use local search: start with a random assignment and flip colors
    # to reduce violations
    
    best_n = p
    best_colors = None
    
    # Generate baseline colors in lexicographic order
    base_colors = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                base_colors.append(0)
            else:
                base_colors.append(1)
    
    # Try multiple random initializations for vertex 29's connections
    random.seed(42)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Random initial assignment for vertex 29
        new_edges = [random.randint(0, 1) for _ in range(p)]
        
        # Build full adjacency for n=30
        n = 30
        adj0_full = [0] * n
        adj1_full = [0] * n
        
        for i in range(p):
            for j in range(i + 1, p):
                diff = (j - i) % p
                if diff in squares:
                    adj0_full[i] |= (1 << j)
                    adj0_full[j] |= (1 << i)
                else:
                    adj1_full[i] |= (1 << j)
                    adj1_full[j] |= (1 << i)
        
        for i in range(p):
            if new_edges[i] == 0:
                adj0_full[i] |= (1 << 29)
                adj0_full[29] |= (1 << i)
            else:
                adj1_full[i] |= (1 << 29)
                adj1_full[29] |= (1 << i)
        
        # Local search: try to flip edges to eliminate K_{2,2,2}
        # We'll do a simple hill-climbing with restarts
        
        for iteration in range(500):
            if time.time() > deadline:
                break
            
            # Check for violations
            v0 = has_k222(adj0_full, n)
            v1 = has_k222(adj1_full, n)
            
            if not v0 and not v1:
                # Found a valid coloring!
                best_n = n
                # Build colors in lexicographic order
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if adj0_full[i] & (1 << j):
                            colors.append(0)
                        else:
                            colors.append(1)
                best_colors = colors
                break
            
            # If we have violations, try flipping a random edge involving vertex 29
            # or between existing vertices
            if random.random() < 0.7:
                # Flip edge involving vertex 29
                i = random.randint(0, p - 1)
                current = 0 if (adj0_full[i] & (1 << 29)) else 1
                new_color = 1 - current
                
                # Update
                if current == 0:
                    adj0_full[i] &= ~(1 << 29)
                    adj0_full[29] &= ~(1 << i)
                    adj1_full[i] |= (1 << 29)
                    adj1_full[29] |= (1 << i)
                else:
                    adj1_full[i] &= ~(1 << 29)
                    adj1_full[29] &= ~(1 << i)
                    adj0_full[i] |= (1 << 29)
                    adj0_full[29] |= (1 << i)
            else:
                # Flip an edge between existing vertices
                i = random.randint(0, p - 1)
                j = random.randint(0, p - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                
                current = 0 if (adj0_full[i] & (1 << j)) else 1
                new_color = 1 - current
                
                if current == 0:
                    adj0_full[i] &= ~(1 << j)
                    adj0_full[j] &= ~(1 << i)
                    adj1_full[i] |= (1 << j)
                    adj1_full[j] |= (1 << i)
                else:
                    adj1_full[i] &= ~(1 << j)
                    adj1_full[j] &= ~(1 << i)
                    adj0_full[i] |= (1 << j)
                    adj0_full[j] |= (1 << i)
        
        if best_colors is not None:
            break
    
    if best_colors is not None:
        return {'n': best_n, 'colors': best_colors}
    
    # Return the Paley(29) baseline
    return {'n': p, 'colors': base_colors}
# EVOLVE-BLOCK-END
