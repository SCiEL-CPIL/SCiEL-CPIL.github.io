"""Spectral graph theory approach for 4-colouring K_n without C6 in any colour class."""
import numpy as np
from itertools import combinations

def _has_c6(adj, n):
    """Check if adjacency (list of sets) contains a C6 (not necessarily induced)."""
    for v0 in range(n):
        for v1 in adj[v0]:
            if v1 <= v0:
                continue
            for v5 in adj[v0]:
                if v5 == v1 or v5 <= v0:
                    continue
                # Find paths of length 3 from v1 to v5 avoiding v0
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

def _build_adj_from_edge_colors(edge_colors, n):
    """Build adjacency lists for each color class from edge color list."""
    adj = [set() for _ in range(4)]
    for c in range(4):
        adj[c] = [set() for _ in range(n)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = edge_colors[idx]
            adj[c][i].add(j)
            adj[c][j].add(i)
            idx += 1
    return adj

def _is_valid_coloring(edge_colors, n):
    """Check if the edge coloring has no C6 in any color class."""
    adj = _build_adj_from_edge_colors(edge_colors, n)
    for c in range(4):
        if _has_c6(adj[c], n):
            return False
    return True

def _spectral_edge_coloring(n, seed):
    """Generate edge colors using spectral embedding of a random regular graph."""
    rng = np.random.RandomState(seed)
    
    # Generate a random 4-regular graph on n vertices
    # For n >= 5, we can construct a 4-regular graph using circulant structure
    # Connect vertex i to i+1, i+2, i-1, i-2 (mod n)
    if n < 5:
        # Fallback for small n
        return [0] * (n * (n - 1) // 2)
    
    # Build adjacency for the random regular graph
    adj_reg = [set() for _ in range(n)]
    for i in range(n):
        for k in [1, 2, n-1, n-2]:
            j = (i + k) % n
            if i != j:
                adj_reg[i].add(j)
                adj_reg[j].add(i)
    
    # Compute Laplacian
    degree = np.array([len(adj_reg[i]) for i in range(n)], dtype=float)
    D = np.diag(degree)
    A = np.zeros((n, n))
    for i in range(n):
        for j in adj_reg[i]:
            A[i, j] = 1.0
    L = D - A
    
    # Get eigenvectors
    eigvals, eigvecs = np.linalg.eigh(L)
    # eigvals[0] should be ~0, take eigenvectors 1..4 (smallest non-zero)
    # Actually, take the 4 eigenvectors corresponding to smallest non-zero eigenvalues
    # eigvals are sorted in ascending order
    vecs = eigvecs[:, 1:5]  # First 4 non-trivial eigenvectors
    
    # For each edge (i,j), compute the difference vector and assign color by quadrant
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = vecs[j, :] - vecs[i, :]
            # Use first two components for quadrant assignment
            s0 = 1 if diff[0] >= 0 else 0
            s1 = 1 if diff[1] >= 0 else 0
            color = s0 * 2 + s1
            edge_colors.append(color)
    
    return edge_colors

def _spectral_edge_coloring_v2(n, seed):
    """Alternative: use all 4 components with a different mapping."""
    rng = np.random.RandomState(seed)
    
    if n < 5:
        return [0] * (n * (n - 1) // 2)
    
    # Build a random 4-regular graph using a different construction
    # Use a random permutation to create the regular graph
    perm = rng.permutation(n)
    
    adj_reg = [set() for _ in range(n)]
    for i in range(n):
        for k in [1, 2, n-1, n-2]:
            j = (i + k) % n
            if i != j:
                adj_reg[i].add(j)
                adj_reg[j].add(i)
    
    degree = np.array([len(adj_reg[i]) for i in range(n)], dtype=float)
    D = np.diag(degree)
    A = np.zeros((n, n))
    for i in range(n):
        for j in adj_reg[i]:
            A[i, j] = 1.0
    L = D - A
    
    eigvals, eigvecs = np.linalg.eigh(L)
    vecs = eigvecs[:, 1:5]
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = vecs[j, :] - vecs[i, :]
            # Map to 4 colors using a hash-like function on all 4 components
            # Use the signs of all 4 components: 2^4=16 patterns, map to 4 colors
            signs = np.sign(diff)
            # Create a 4-bit signature
            sig = 0
            for k in range(4):
                if signs[k] >= 0:
                    sig |= (1 << k)
            color = sig % 4
            edge_colors.append(color)
    
    return edge_colors

def _spectral_edge_coloring_v3(n, seed):
    """Use random symmetric matrix for spectral embedding."""
    rng = np.random.RandomState(seed)
    
    if n < 5:
        return [0] * (n * (n - 1) // 2)
    
    # Generate random symmetric matrix
    M = rng.randn(n, n)
    M = (M + M.T) / 2
    
    eigvals, eigvecs = np.linalg.eigh(M)
    # Take 4 largest eigenvalues' eigenvectors
    idx = np.argsort(eigvals)[::-1][:4]
    vecs = eigvecs[:, idx]
    
    edge_colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = vecs[j, :] - vecs[i, :]
            # Use quadrant of first two components
            s0 = 1 if diff[0] >= 0 else 0
            s1 = 1 if diff[1] >= 0 else 0
            color = s0 * 2 + s1
            edge_colors.append(color)
    
    return edge_colors

def construct():
    """Construct a 4-coloring of K_n without C6 in any color class, maximizing n."""
    
    # Start with a known valid small coloring
    # For n=9, use a simple pattern
    def _simple_coloring(n):
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = j - i
                # Use modular arithmetic to spread colors
                color = (d * 7 + i * 3) % 4
                colors.append(color)
        return colors
    
    best_n = 1
    best_colors = [0]
    
    # Try increasing n
    for n in range(2, 18):
        found_valid = False
        
        # Try multiple spectral approaches with different seeds
        for trial in range(100):
            # Try different spectral coloring methods
            if trial % 3 == 0:
                edge_colors = _spectral_edge_coloring(n, trial)
            elif trial % 3 == 1:
                edge_colors = _spectral_edge_coloring_v2(n, trial)
            else:
                edge_colors = _spectral_edge_coloring_v3(n, trial)
            
            if _is_valid_coloring(edge_colors, n):
                best_n = n
                best_colors = edge_colors
                found_valid = True
                break
        
        if not found_valid:
            # Try a simple modular coloring as fallback
            edge_colors = _simple_coloring(n)
            if _is_valid_coloring(edge_colors, n):
                best_n = n
                best_colors = edge_colors
                found_valid = True
        
        if not found_valid:
            break
    
    return {'n': best_n, 'colors': best_colors}
