"""Spectral graph theory approach for 4-colouring K_n without C6 in any colour class."""
import numpy as np
from collections import deque

def _has_c6(adj, n):
    """Check if adjacency matrix (list of sets or boolean matrix) contains a C6 (not necessarily induced).
    Returns True if any 6-cycle exists."""
    # For efficiency with n up to ~20, we can use a bitset-based approach or simple enumeration.
    # Since n is small, let's enumerate all 6-cycles.
    # A 6-cycle is a sequence v0-v1-v2-v3-v4-v5-v0 where all edges exist and all vertices are distinct.
    # To avoid duplicates, we fix v0 = 0 and v1 > v0, and require the cycle to be "minimal" in some sense.
    # Actually, let's just check all combinations of 6 vertices and see if they form a C6.
    # Number of 6-subsets from n vertices: C(n,6). For n=20, C(20,6)=38760. For each, check if they form a C6.
    # A set of 6 vertices forms a C6 if there exists a cyclic ordering where consecutive vertices are adjacent.
    # This is equivalent to the induced subgraph on those 6 vertices containing a Hamiltonian cycle.
    
    # Better approach: for each pair of vertices at distance 3 in the potential cycle...
    # Let's use a direct approach: for each 6-tuple (v0,v1,v2,v3,v4,v5) with v0 < v1 < ... no, that's not right.
    # 
    # Let's use the standard method: a C6 exists iff there exist vertices v0,...,v5 such that
    # v0~v1, v1~v2, v2~v3, v3~v4, v4~v5, v5~v0, all distinct.
    # To avoid checking too many, we can fix v0=0, v1 in neighbors of 0, v5 in neighbors of 0, v1 != v5,
    # then look for paths of length 3 from v1 to v5 that don't use 0.
    
    # Given small n, let's do a BFS-based check for each pair.
    # Actually, let me just enumerate all 6-cycles directly.
    
    # adj is a list where adj[i] is a set of neighbors
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v5 in adj[v0]:
                if v5 == v1 or v5 <= v0:
                    continue
                # Now find paths of length 3 from v1 to v5 avoiding v0
                # v1 - v2 - v3 - v4 - v5
                for v2 in adj[v1]:
                    if v2 == v0 or v2 == v5:
                        continue
                    for v3 in adj[v2]:
                        if v3 == v0 or v3 == v1 or v3 == v5:
                            continue
                        for v4 in adj[v3]:
                            if v4 == v0 or v4 == v1 or v4 == v2 or v4 == v5:
                                continue
                            if v5 in adj[v4]:
                                return True
    return False

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    np.random.seed(42)
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    # Try to find colorings for larger n using spectral approach
    for n in range(10, 18):
        # Generate a random regular graph (or use a structured one)
        # We'll try multiple random graphs
        found = False
        for trial in range(50):
            # Create a random 4-regular graph on n vertices (if possible)
            # For simplicity, use a random permutation-based construction
            perm = np.random.permutation(n)
            
            # Build adjacency: connect each vertex to next 4 in a circular ordering
            # This gives a 4-regular circulant-like graph
            adj = [set() for _ in range(n)]
            for i in range(n):
                for k in range(1, 5):
                    j = (i + k) % n
                    if i != j:
                        adj[i].add(j)
                        adj[j].add(i)
            
            # Compute Laplacian
            degree = np.array([len(adj[i]) for i in range(n)], dtype=float)
            D = np.diag(degree)
            A = np.zeros((n, n))
            for i in range(n):
                for j in adj[i]:
                    A[i, j] = 1.0
            L = D - A
            
            # Get top 4 eigenvectors (smallest non-zero eigenvalues)
            eigvals, eigvecs = np.linalg.eigh(L)
            # eigvals[0] should be 0, take eigvecs[1:5]
            vecs = eigvecs[:, 1:5]
            
            # Assign colors based on quadrant signs of the 4 eigenvector coordinates
            colors = []
            for i in range(n):
                signs = np.sign(vecs[i, :])
                # Map signs to color: each sign pattern is a color
                # 2^4 = 16 possible sign patterns, but we only have 4 colors
                # Use a hash or projection to map to 4 colors
                # Better: use the two most significant components
                # Actually, let's use a different approach: project onto 2D and use quadrants
                # Or: use the first two eigenvectors for x,y and next two for refinement
                
                # Let's use: color = (sign of vec1, sign of vec2) mapped to 4 colors
                s0 = 1 if vecs[i, 0] >= 0 else 0
                s1 = 1 if vecs[i, 1] >= 0 else 0
                color = s0 * 2 + s1
                colors.append(color)
            
            # Check if this coloring is valid (no C6 in any color class)
            # Build color class adjacency
            valid = True
            for c in range(4):
                adj_c = [set() for _ in range(n)]
                for i in range(n):
                    for j in range(i + 1, n):
                        if colors[i] == c and colors[j] == c:
                            adj_c[i].add(j)
                            adj_c[j].add(i)
                if _has_c6(adj_c, n):
                    valid = False
                    break
            
            if valid:
                # Convert to edge color list in lexicographic order
                edge_colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        # Color of edge (i,j): use the color of vertex i? No, that's vertex coloring.
                        # We need edge coloring. Let me reconsider.
                        pass
                # The above approach gives vertex coloring, not edge coloring.
                # We need to color edges. Let me redo.
                found = True
                break
        
        if found:
            break
    
    # The spectral approach above is flawed because it gives vertex colors, not edge colors.
    # Let me implement a proper edge-coloring approach.
    
    # New approach: Use spectral embedding to assign edge colors.
    # For each edge (i,j), compute a feature vector and assign to one of 4 colors.
    
    best_n = 9
    best_colors = [min(j - i, 9 - j + i) - 1 for i in range(9) for j in range(i + 1, 9)]
    
    for n in range(10, 18):
        found_valid = False
        for trial in range(30):
            np.random.seed(trial * n + 42)
            
            # Generate random symmetric matrix for spectral embedding
            M = np.random.randn(n, n)
            M = (M + M.T) / 2
            
            # Get top 4 eigenvectors
            eigvals, eigvecs = np.linalg.eigh(M)
            # Take the 4 largest eigenvalues' eigenvectors
            idx = np.argsort(eigvals)[::-1][:4]
            vecs = eigvecs[:, idx]
            
            # For each edge (i,j), compute a feature based on the eigenvector difference
            # and assign a color
            edge_colors = []
            for i in range(n):
                for j in range(i + 1, n):
                    # Feature: difference in eigenvector coordinates
                    diff = vecs[j, :] - vecs[i, :]
                    # Assign color based on which component has largest magnitude
                    max_idx = np.argmax(np.abs(diff))
                    # Use sign to subdivide
                    sign = 1 if diff[max_idx] >= 0 else 0
                    color = max_idx * 2 + sign
                    # But we only have 4 colors, and max_idx is 0..3, so color is 0..7
                    # Need to map to 4 colors
                    color = max_idx % 4
                    edge_colors.append(color)
            
            # Check validity
            valid = True
            for c in range(4):
                adj_c = [set() for _ in range(n)]
                idx = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if edge_colors[idx] == c:
                            adj_c[i].add(j)
                            adj_c[j].add(i)
                        idx += 1
                if _has_c6(adj_c, n):
                    valid = False
                    break
            
            if valid:
                best_n = n
                best_colors = edge_colors
                found_valid = True
                break
        
        if found_valid:
            continue
        else:
            # If we can't find a valid coloring for this n, stop
            break
    
    return {'n': best_n, 'colors': best_colors}
