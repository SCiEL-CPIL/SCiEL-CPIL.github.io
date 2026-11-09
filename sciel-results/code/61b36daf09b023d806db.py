import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # Try Paley graph for n=29
    n = 29
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    # Build color array
    # color[i][j] is 0 if (j-i)%n in squares, else 1
    # We need to return a flat list in lexicographic order (i,j) with i<j
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
    
    # Verify
    if is_valid_coloring(n, colors):
        return {'n': n, 'colors': colors}
    
    # If Paley 29 fails, try local search
    # We need a representation for fast checking
    # Let's use a 2D numpy array for adjacency
    # A[i][j] = color(i,j)
    
    A = np.zeros((n, n), dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            A[i][j] = c
            A[j][i] = c
            idx += 1
            
    best_n = n
    best_colors = colors
    
    # Local Search
    # We will try to reduce the number of mono K2,2,2
    # Count violations
    def count_violations(A):
        count = 0
        # Precompute partitions of 6 indices into 3 pairs
        # Indices 0..5
        # Partitions:
        # (0,1)(2,3)(4,5)
        # (0,1)(2,4)(3,5)
        # (0,1)(2,5)(3,4)
        # (0,2)(1,3)(4,5)
        # (0,2)(1,4)(3,5)
        # (0,2)(1,5)(3,4)
        # (0,3)(1,2)(4,5)
        # (0,3)(1,4)(2,5)
        # (0,3)(1,5)(2,4)
        # (0,4)(1,2)(3,5)
        # (0,4)(1,3)(2,5)
        # (0,4)(1,5)(2,3)
        # (0,5)(1,2)(3,4)
        # (0,5)(1,3)(2,4)
        # (0,5)(1,4)(2,3)
        
        partitions = [
            ((0,1),(2,3),(4,5)),
            ((0,1),(2,4),(3,5)),
            ((0,1),(2,5),(3,4)),
            ((0,2),(1,3),(4,5)),
            ((0,2),(1,4),(3,5)),
            ((0,2),(1,5),(3,4)),
            ((0,3),(1,2),(4,5)),
            ((0,3),(1,4),(2,5)),
            ((0,3),(1,5),(2,4)),
            ((0,4),(1,2),(3,5)),
            ((0,4),(1,3),(2,5)),
            ((0,4),(1,5),(2,3)),
            ((0,5),(1,2),(3,4)),
            ((0,5),(1,3),(2,4)),
            ((0,5),(1,4),(2,3))
        ]
        
        for verts in combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = verts
            # Check color 0
            for p in partitions:
                (a,b), (c,d), (e,f) = p
                # Map local indices to verts
                va, vb = verts[a], verts[b]
                vc, vd = verts[c], verts[d]
                ve, vf = verts[e], verts[f]
                
                # Check all cross edges are 0
                if (A[va][vc] == 0 and A[va][vd] == 0 and 
                    A[vb][vc] == 0 and A[vb][vd] == 0 and
                    A[va][ve] == 0 and A[va][vf] == 0 and
                    A[vb][ve] == 0 and A[vb][vf] == 0 and
                    A[vc][ve] == 0 and A[vc][vf] == 0 and
                    A[vd][ve] == 0 and A[vd][vf] == 0):
                    count += 1
                    break # Count each K2,2,2 once? Or count all? 
                    # If we count all, it's more robust for optimization.
                    # But for validity, 1 is enough.
                    # Let's count all for better gradient.
                    # Actually, if we break, we miss multiple partitions for the same set?
                    # A K2,2,2 has a unique partition into 3 parts?
                    # No, the vertices are just a set. The partition defines the parts.
                    # If a set of 6 vertices forms a K2,2,2, it might do so for multiple partitions?
                    # If it's a complete graph K6, it works for all 15 partitions.
                    # So we should NOT break. We should count all valid partitions.
            # Check color 1
            for p in partitions:
                (a,b), (c,d), (e,f) = p
                va, vb = verts[a], verts[b]
                vc, vd = verts[c], verts[d]
                ve, vf = verts[e], verts[f]
                
                if (A[va][vc] == 1 and A[va][vd] == 1 and 
                    A[vb][vc] == 1 and A[vb][vd] == 1 and
                    A[va][ve] == 1 and A[va][vf] == 1 and
                    A[vb][ve] == 1 and A[vb][vf] == 1 and
                    A[vc][ve] == 1 and A[vc][vf] == 1 and
                    A[vd][ve] == 1 and A[vd][vf] == 1):
                    count += 1
                    
        return count

    # This function is too slow for repeated calls in local search.
    # We need a faster check.
    
    # Alternative: Just return the Paley 29 if it's valid.
    # If not, we might not have time to fix it properly.
    # But let's check validity first.
    
    def is_valid_coloring(n, colors):
        # Reconstruct A
        A = np.zeros((n, n), dtype=int)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                A[i][j] = c
                A[j][i] = c
                idx += 1
        
        # Check for mono K2,2,2
        # Use the same logic as above but return False immediately upon finding one
        partitions = [
            ((0,1),(2,3),(4,5)),
            ((0,1),(2,4),(3,5)),
            ((0,1),(2,5),(3,4)),
            ((0,2),(1,3),(4,5)),
            ((0,2),(1,4),(3,5)),
            ((0,2),(1,5),(3,4)),
            ((0,3),(1,2),(4,5)),
            ((0,3),(1,4),(2,5)),
            ((0,3),(1,5),(2,4)),
            ((0,4),(1,2),(3,5)),
            ((0,4),(1,3),(2,5)),
            ((0,4),(1,5),(2,3)),
            ((0,5),(1,2),(3,4)),
            ((0,5),(1,3),(2,4)),
            ((0,5),(1,4),(2,3))
        ]
        
        for verts in combinations(range(n), 6):
            v0, v1, v2, v3, v4, v5 = verts
            # Check color 0
            for p in partitions:
                (a,b), (c,d), (e,f) = p
                va, vb = verts[a], verts[b]
                vc, vd = verts[c], verts[d]
                ve, vf = verts[e], verts[f]
                
                if (A[va][vc] == 0 and A[va][vd] == 0 and 
                    A[vb][vc] == 0 and A[vb][vd] == 0 and
                    A[va][ve] == 0 and A[va][vf] == 0 and
                    A[vb][ve] == 0 and A[vb][vf] == 0 and
                    A[vc][ve] == 0 and A[vc][vf] == 0 and
                    A[vd][ve] == 0 and A[vd][vf] == 0):
                    return False
            # Check color 1
            for p in partitions:
                (a,b), (c,d), (e,f) = p
                va, vb = verts[a], verts[b]
                vc, vd = verts[c], verts[d]
                ve, vf = verts[e], verts[f]
                
                if (A[va][vc] == 1 and A[va][vd] == 1 and 
                    A[vb][vc] == 1 and A[vb][vd] == 1 and
                    A[va][ve] == 1 and A[va][vf] == 1 and
                    A[vb][ve] == 1 and A[vb][vf] == 1 and
                    A[vc][ve] == 1 and A[vc][vf] == 1 and
                    A[vd][ve] == 1 and A[vd][vf] == 1):
                    return False
        return True

    if is_valid_coloring(n, colors):
        return {'n': n, 'colors': colors}
        
    # If invalid, try n=28? Or try to fix n=29?
    # Given time constraints, if 29 fails, try 28.
    n = 28
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    if is_valid_coloring(n, colors):
        return {'n': n, 'colors': colors}
        
    # Fallback to n=17 if 28 also fails (unlikely)
    n = 17
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n in squares:
                colors.append(0)
            else:
                colors.append(1)
                
    return {'n': n, 'colors': colors}

# EVOLVE-BLOCK-END
