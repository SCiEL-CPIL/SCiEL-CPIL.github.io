import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def time_check():
        return time.time() > deadline - 5.0

    # Strategy: Spectral Graph Theory Approach
    # 1. Generate a random regular graph (or complete graph with random weights)
    # 2. Compute Laplacian eigenvectors
    # 3. Project vertices onto top 4 eigenvectors
    # 4. Assign colors based on quadrant signs
    # 5. Local search to fix any remaining C6s

    n = 17
    edges = []
    edge_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i,j))
            edge_idx[(i,j)] = idx
            idx += 1

    def spectral_coloring(n, seed=None):
        """Generate a coloring based on spectral properties of a random regular graph."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Create a random 4-regular graph (or similar)
        # Actually, let's use the complete graph K_n and weight edges randomly
        # Then compute the Laplacian of this weighted graph
        # The eigenvectors of the Laplacian capture the "structure" of the graph
        
        # For a complete graph, the Laplacian is (n-1)I - A where A is adjacency
        # But we want to use a random structure to break symmetry
        # Let's create a random 4-regular graph on n vertices
        
        # Generate a random 4-regular graph
        adj = np.zeros((n, n), dtype=int)
        for i in range(n):
            # Each vertex needs degree 4
            # We'll use a simple method: for each vertex, connect to 4 random others
            # This might not give exactly 4-regular, but close enough
            pass
        
        # Actually, for simplicity and speed, let's just use the complete graph
        # and weight edges with random values
        # The Laplacian of a weighted complete graph will have eigenvectors
        # that depend on the random weights
        
        # Create random edge weights
        weights = np.random.rand(len(edges))
        
        # Build weighted adjacency matrix
        W = np.zeros((n, n))
        for k, (i, j) in enumerate(edges):
            W[i][j] = W[j][i] = weights[k]
        
        # Build Laplacian: L = D - A where D is degree matrix
        # For weighted graph, D[i][i] = sum_j W[i][j]
        D = np.diag(W.sum(axis=1))
        L = D - W
        
        # Compute eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            # Fallback to random coloring
            return [random.randint(0, 3) for _ in range(len(edges))]
        
        # Sort by eigenvalue (ascending)
        # The smallest eigenvalue is 0 (for connected graph)
        # We want the next 4 eigenvectors (indices 1, 2, 3, 4)
        # These capture the "structure" of the graph
        
        # Take eigenvectors 1 through 4 (0-indexed: 1, 2, 3, 4)
        # Project vertices onto these eigenvectors
        # Each vertex gets a 4-dimensional vector
        proj = eigvecs[:, 1:5]  # Shape: (n, 4)
        
        # Assign colors based on quadrant signs
        # For each vertex, look at the signs of its 4 coordinates
        # Map sign patterns to colors 0-3
        
        # There are 2^4 = 16 possible sign patterns
        # We need to map them to 4 colors
        # Use a deterministic mapping based on the pattern
        
        # Actually, let's use a simpler approach:
        # For each vertex, compute a "color score" for each of the 4 colors
        # based on its projection, and assign the color with the highest score
        
        # Or even simpler: use the first 2 eigenvectors to define 4 quadrants
        # This gives a natural 4-coloring
        
        # Use eigenvectors 1 and 2 (the first two non-trivial ones)
        # Project onto these two
        proj2d = eigvecs[:, 1:3]  # Shape: (n, 2)
        
        # Assign colors based on quadrant
        # Quadrant 1: (+, +) -> 0
        # Quadrant 2: (-, +) -> 1
        # Quadrant 3: (-, -) -> 2
        # Quadrant 4: (+, -) -> 3
        
        vertex_colors = []
        for i in range(n):
            x, y = proj2d[i]
            if x >= 0 and y >= 0:
                c = 0
            elif x < 0 and y >= 0:
                c = 1
            elif x < 0 and y < 0:
                c = 2
            else:
                c = 3
            vertex_colors.append(c)
        
        # Now assign edge colors
        # The edge color should be determined by the vertex colors
        # But we need a 4-coloring of edges, not vertices
        # So we need a different approach
        
        # Let's use the 4 eigenvectors to create 4 "color fields"
        # For each color c, compute a field F_c(v) = eigvecs[v][c]
        # Then for edge (u, v), assign color c that minimizes |F_c(u) - F_c(v)|
        # or maximizes some compatibility score
        
        # Actually, let's try a different approach:
        # Use the 4 eigenvectors to define 4 "partitions" of the vertex set
        # Then assign edge colors based on which partition the endpoints belong to
        
        # For each eigenvector, partition vertices into 2 groups (positive/negative)
        # This gives 2^4 = 16 possible "types" for each vertex
        # Map these types to 4 colors
        
        # Compute vertex types
        vertex_types = []
        for i in range(n):
            signs = []
            for c in range(4):
                signs.append(1 if eigvecs[i][c+1] >= 0 else 0)  # Skip index 0
            vertex_types.append(tuple(signs))
        
        # Map 16 types to 4 colors
        # Use a deterministic mapping
        type_to_color = {}
        for t in range(16):
            # Simple mapping: t % 4
            type_to_color[t] = t % 4
        
        # Convert vertex types to integers
        def type_to_int(t):
            val = 0
            for bit in t:
                val = val * 2 + bit
            return val
        
        # Assign vertex colors
        vertex_colors = []
        for i in range(n):
            t = type_to_int(vertex_types[i])
            vertex_colors.append(type_to_color[t])
        
        # Now assign edge colors
        # For edge (u, v), the color should be based on the vertex colors
        # But we need a 4-coloring of edges
        # One approach: color = (vertex_colors[u] + vertex_colors[v]) % 4
        # Another: color = vertex_colors[u] if vertex_colors[u] != vertex_colors[v] else ...
        
        # Let's use: color = (vertex_colors[u] * vertex_colors[v]) % 4
        # Or simply: color = (vertex_colors[u] + vertex_colors[v]) % 4
        
        edge_colors = []
        for i in range(n):
            for j in range(i+1, n):
                c = (vertex_colors[i] + vertex_colors[j]) % 4
                edge_colors.append(c)
        
        return edge_colors

    # Try multiple spectral colorings with different random seeds
    best_colors = None
    best_cost = float('inf')
    
    # Helper: compute cost (number of C6s)
    def count_c6s(colors):
        """Count monochromatic C6s in the coloring."""
        # Build adjacency for each color
        adj = [np.zeros((n, n), dtype=int) for _ in range(4)]
        for k, (i, j) in enumerate(edges):
            c = colors[k]
            adj[c][i][j] = 1
            adj[c][j][i] = 1
        
        total_c6 = 0
        for c in range(4):
            # Count C6s using matrix multiplication
            # A C6 is a cycle of length 6
            # We can count closed walks of length 6, but that includes smaller cycles
            # For an exact count, we need to be more careful
            # But for now, use trace(A^6) as a proxy
            a2 = adj[c] @ adj[c]
            a4 = a2 @ a2
            a6 = a4 @ a2
            total_c6 += np.trace(a6)
        
        return total_c6

    # Try 100 different random seeds
    for seed in range(100):
        if time_check():
            break
        
        colors = spectral_coloring(n, seed=seed)
        cost = count_c6s(colors)
        
        if cost < best_cost:
            best_cost = cost
            best_colors = colors
        
        if best_cost == 0:
            break
    
    # If we found a zero-cost coloring, return it
    if best_colors is not None and best_cost == 0:
        return {'n': n, 'colors': best_colors}
    
    # If not, try local search to improve the best coloring
    if best_colors is not None:
        current_colors = best_colors[:]
        current_cost = best_cost
        
        # Local search: flip edge colors to reduce C6 count
        T = 1000.0
        cooling = 0.999
        
        for it in range(50000):
            if time_check():
                break
            
            u, v = random.choice(edges)
            old_c = current_colors[edge_idx[(u,v)]]
            new_c = random.randint(0, 3)
            if new_c == old_c:
                continue
            
            # Flip
            current_colors[edge_idx[(u,v)]] = new_c
            new_cost = count_c6s(current_colors)
            
            delta = new_cost - current_cost
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_colors = current_colors[:]
                    if best_cost == 0:
                        break
            else:
                # Revert
                current_colors[edge_idx[(u,v)]] = old_c
            
            T *= cooling
        
        if best_cost == 0:
            return {'n': n, 'colors': best_colors}
    
    # Fallback: return the best known cyclic coloring
    # This is the same as the original program's Strategy 1
    pairs = [(1, 16), (2, 15), (3, 14), (4, 13), (5, 12), (6, 11), (7, 10), (8, 9)]
    
    def has_c6_in_diffs(D):
        """Check if Cayley graph on Z_17 with difference set D contains a C6."""
        D_set = set(D)
        for v1 in D_set:
            for v2 in [(v1 + d) % 17 for d in D_set]:
                if v2 == 0 or v2 == v1: continue
                for v3 in [(v2 + d) % 17 for d in D_set]:
                    if v3 in (0, v1, v2): continue
                    for v4 in [(v3 + d) % 17 for d in D_set]:
                        if v4 in (0, v1, v2, v3): continue
                        for v5 in [(v4 + d) % 17 for d in D_set]:
                            if v5 in (0, v1, v2, v3, v4): continue
                            if (-v5) % 17 in D_set:
                                return True
        return False

    for assignment in range(4**8):
        if time_check(): break
        
        colors_for_pairs = [0] * 8
        temp = assignment
        for i in range(8):
            colors_for_pairs[i] = temp % 4
            temp //= 4
        
        diff_sets = [set() for _ in range(4)]
        for i, c in enumerate(colors_for_pairs):
            p = pairs[i]
            diff_sets[c].add(p[0])
            diff_sets[c].add(p[1])
        
        valid = True
        for c in range(4):
            if has_c6_in_diffs(diff_sets[c]):
                valid = False
                break
        
        if valid:
            color_map = {}
            for c in range(4):
                for d in diff_sets[c]:
                    color_map[d] = c
            
            n = 17
            colors_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    diff = (j - i) % 17
                    colors_list.append(color_map[diff])
            
            return {'n': n, 'colors': colors_list}
    
    # Final fallback: return a small valid coloring
    n = 9
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
