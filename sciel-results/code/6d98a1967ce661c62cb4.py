import time

N = 16
NUM_EDGES = N * (N - 1) // 2

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def construct():
    # Proper edge coloring of K_16 using 15 colors (1-factorization)
    # For K_{2m}, we can use: c(i,j) = (i+j) % (2m-1) if i+j < 2m-1, else 2m-1
    # Wait, for K_16 (2m=16, m=8), we need 15 colors
    # Standard 1-factorization: label vertices 0..14 and ∞
    # Edge (i,∞) gets color i
    # Edge (i,j) with i,j in 0..14 gets color (i+j)/2 mod 15... 
    
    # Actually, simpler: for K_{2m}, vertices 0..2m-1
    # c(i,j) = (i+j) % (2m-1) if i+j < 2m-1, else 2m-2... 
    # Let me use the standard construction:
    # For even n, c(i,j) = (i + j) % (n-1) if i != j and {i,j} != {0, n-1}... 
    
    # Simpler approach: 
    # c(i,j) = (i + j) % 15 for i != j, and handle the case where i+j >= 15
    # Actually for K_{2m}, the 1-factorization is:
    # Fix vertex 2m-1. For k in 0..2m-2, the k-th perfect matching contains:
    #   (2m-1, k) and for each r in 0..m-1, ((k+r)%(2m-1), (k-r+1)%(2m-1))
    # but this is getting complex. Let me just use:
    
    colors = [0] * NUM_EDGES
    
    # Use c(i,j) = (i + j) % 15, but when i+j >= 15, use 14 (the last color)
    # Wait, that won't give a proper coloring.
    
    # Let me just use the min(i,j) base which is known to work, and try to add more colors
    # by being more clever about which edges to modify.
    
    for i in range(N):
        for j in range(i + 1, N):
            colors[edge_idx(i, j)] = i
    
    # The min(i,j) coloring uses colors 0..14 (15 colors)
    # We can add color 15 on edge (13,15) as before
    # Can we add more? Let's try adding colors on edges incident to vertex 15
    # Edge (i,15) currently has color i for i in 0..14
    # If we change edge (14,15) to color 15, we need to check all 4-cycles containing this edge
    
    # Actually, let me try a completely different base: 
    # c(i,j) = (i * j) % 16, but ensure it's in 0..119 and check validity
    
    # Let me try: c(i,j) = min(i,j) * 8 + max(i,j) % 8
    # This gives up to 14*8+7 = 119 colors, but likely creates rainbow cycles
    
    # Better: stick with min(i,j) base and try to add more colors strategically
    # The key constraint: for any 4 vertices a<b<c<d, the three 4-cycles must not be rainbow
    
    # In min(i,j) coloring:
    # Cycle a-b-c-d-a: colors a, b, c, d -> all distinct if a<b<c<d! RAINBOW!
    # Wait, that can't be right. Let me recheck.
    # Edge (a,b): min(a,b) = a
    # Edge (b,c): min(b,c) = b  
    # Edge (c,d): min(c,d) = c
    # Edge (d,a): min(d,a) = a
    # So colors are a, b, c, a -> only 3 distinct! OK, not rainbow.
    
    # Cycle a-b-d-c-a: 
    # (a,b): a, (b,d): b, (d,c): c, (c,a): a -> colors a,b,c,a -> 3 distinct
    # Cycle a-c-b-d-a:
    # (a,c): a, (c,b): b, (b,d): b, (d,a): a -> colors a,b,b,a -> 2 distinct
    
    # So min(i,j) is valid! Now, which edges can we recolor?
    # Edge (i,j) with i<j has color i. If we change it to color c != i,
    # we need to check all 4-cycles containing edge (i,j).
    
    # Let's try to add as many new colors as possible.
    # Start with min(i,j) coloring (15 colors: 0-14)
    # Try to recolor edges to use new colors 15, 16, ...
    
    # For each edge, try assigning a new color and check validity
    # This is expensive, so let's be strategic.
    
    # Edge (13,15) can be recolored to 15 (as in current solution)
    # Can we recolor more edges?
    
    # Let me try: recolor (12,15) to 15, (13,15) to 16, (14,15) to 17
    # But need to verify.
    
    # Actually, let me just try to greedily add colors.
    # For each edge in some order, try to assign the smallest unused color
    # that keeps the coloring valid.
    
    # This is complex. Let me just go with the known 16-color solution
    # and see if I can squeeze out one more.
    
    colors[edge_idx(13, 15)] = 15
    colors[edge_idx(14, 15)] = 16
    
    # Try to also recolor (12, 15) to 17
    # Check if this creates any rainbow cycles
    # Edge (12,15) is in 4-cycles with any other 2 vertices from 0..14 (excluding 12,15)
    # That's C(14,2) = 91 4-cycles to check, each with 3 cycles = 273 checks
    
    # Let me just try it and verify
    colors[edge_idx(12, 15)] = 17
    
    # Verify the entire coloring
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    c1 = [colors[edge_idx(a,b)], colors[edge_idx(b,c)], 
                          colors[edge_idx(c,d)], colors[edge_idx(d,a)]]
                    if len(set(c1)) == 4:
                        # Revert
                        colors[edge_idx(12, 15)] = 12
                        break
                    c2 = [colors[edge_idx(a,b)], colors[edge_idx(b,d)], 
                          colors[edge_idx(d,c)], colors[edge_idx(c,a)]]
                    if len(set(c2)) == 4:
                        colors[edge_idx(12, 15)] = 12
                        break
                    c3 = [colors[edge_idx(a,c)], colors[edge_idx(c,b)], 
                          colors[edge_idx(b,d)], colors[edge_idx(d,a)]]
                    if len(set(c3)) == 4:
                        colors[edge_idx(12, 15)] = 12
                        break
                else:
                    continue
                break
            else:
                continue
            break
        else:
            continue
        break
    
    return colors
