import random
import time

def is_3_colorable_mask(adj):
    n = 16
    # Order vertices by degree descending
    degrees = [bin(adj[i]).count('1') for i in range(n)]
    order = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    color = [-1] * n
    # Precompute neighbor lists for faster forbidden computation
    neighbors = [0] * n
    for i in range(n):
        neighbors[i] = adj[i]
    
    def backtrack(idx):
        if idx == n:
            return True
        u = order[idx]
        # Compute forbidden colors from colored neighbors
        forbidden = 0
        mask = neighbors[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if color[v] >= 0:
                forbidden |= (1 << color[v])
            mask ^= lsb
        
        # If all 3 colors forbidden, fail fast
        if forbidden == 0b111:
            return False
        
        # Try colors in order of least constraining (fewest options for neighbors)
        # Simple heuristic: try color 0,1,2 but skip forbidden
        for c in range(3):
            if not (forbidden >> c) & 1:
                color[u] = c
                if backtrack(idx + 1):
                    return True
                color[u] = -1
        return False

    return backtrack(0)

def build_groetzsch_edges():
    """Build the Grötzsch graph (11 vertices, 20 edges, triangle-free, 4-chromatic)"""
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],  # C5
        [5, 4], [5, 1],  # w0 connected to neighbors of v0 (4,1)
        [6, 0], [6, 2],  # w1 connected to neighbors of v1 (0,2)
        [7, 1], [7, 3],  # w2 connected to neighbors of v2 (1,3)
        [8, 2], [8, 4],  # w3 connected to neighbors of v3 (2,4)
        [9, 3], [9, 0],  # w4 connected to neighbors of v4 (3,0)
        [10, 5], [10, 6], [10, 7], [10, 8], [10, 9]  # apex connected to all w
    ]
    return edges

def generate_blowup_distributions():
    """Generate various blow-up distributions of the Grötzsch core onto 16 vertices"""
    # Grötzsch has 11 vertices: 5 (C5), 5 (Mycielski copies), 1 (apex)
    # We need to map 11 vertices to 16 vertices
    # Strategy: blow up some vertices to multiple copies
    
    distributions = []
    
    # Base: each of 11 vertices maps to 1 vertex, leaving 5 isolated
    # Then we can blow up: give some vertices 2 copies
    
    # Try different blow-up patterns
    # Pattern: which of the 11 Grötzsch vertices get blown up to 2 copies
    # We need sum of sizes = 16, so we need 5 extra copies total
    
    # Generate all ways to distribute 5 extra copies among 11 vertices
    # This is stars and bars: C(5+11-1, 11-1) = C(15,10) = 3003, too many
    # Instead, try targeted patterns
    
    # Pattern 1: Blow up 5 different vertices by 1 each
    for i in range(11):
        for j in range(i+1, 11):
            for k in range(j+1, 11):
                for l in range(k+1, 11):
                    for m in range(l+1, 11):
                        sizes = [1]*11
                        sizes[i] = 2
                        sizes[j] = 2
                        sizes[k] = 2
                        sizes[l] = 2
                        sizes[m] = 2
                        distributions.append(sizes)
    
    # Pattern 2: Blow up 2 vertices by 2, 1 vertex by 1
    for i in range(11):
        for j in range(i+1, 11):
            for k in range(11):
                if k != i and k != j:
                    sizes = [1]*11
                    sizes[i] = 3
                    sizes[j] = 3
                    sizes[k] = 2
                    distributions.append(sizes)
    
    # Pattern 3: Blow up 1 vertex by 5
    for i in range(11):
        sizes = [1]*11
        sizes[i] = 6
        distributions.append(sizes)
    
    # Pattern 4: Blow up 1 vertex by 4, 1 by 1
    for i in range(11):
        for j in range(11):
            if i != j:
                sizes = [1]*11
                sizes[i] = 5
                sizes[j] = 2
                distributions.append(sizes)
    
    # Pattern 5: Blow up 2 vertices by 2, 1 by 1 (different from pattern 2)
    # Already covered above
    
    # Remove duplicates
    seen = set()
    unique_distributions = []
    for d in distributions:
        t = tuple(d)
        if t not in seen:
            seen.add(t)
            unique_distributions.append(d)
    
    return unique_distributions

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    best_edges = None
    best_count = -1
    
    # Build Grötzsch core
    groetzsch_edges = build_groetzsch_edges()
    
    # Generate blow-up distributions
    distributions = generate_blowup_distributions()
    random.shuffle(distributions)
    
    # Also try the base case (no blow-up, 5 isolated vertices)
    base_distribution = [1]*11 + [0]*5  # 11 vertices used, 5 isolated
    
    all_distributions = [base_distribution] + distributions
    
    for dist in all_distributions:
        if time.time() > deadline:
            break
        
        # Build adjacency from blow-up
        adj = [0] * 16
        
        # Map Grötzsch vertices to actual vertex indices
        vertex_map = []  # vertex_map[i] = list of actual vertices for Grötzsch vertex i
        current_idx = 0
        for i in range(11):
            count = dist[i] if i < len(dist) else 1
            if count == 0:
                vertex_map.append([])
            else:
                vertex_map.append(list(range(current_idx, current_idx + count)))
                current_idx += count
        
        # Add edges based on Grötzsch structure
        for u_g, v_g in groetzsch_edges:
            for u in vertex_map[u_g]:
                for v in vertex_map[v_g]:
                    if u != v:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
        
        # Check if this blow-up is triangle-free and not 3-colorable
        # First check triangle-free
        triangle_free = True
        for u in range(16):
            for v in range(u+1, 16):
                if (adj[u] >> v) & 1:
                    if (adj[u] & adj[v]) != 0:
                        triangle_free = False
                        break
            if not triangle_free:
                break
        
        if not triangle_free:
            continue
        
        # Check not 3-colorable
        if is_3_colorable_mask(adj):
            continue
        
        current_edge_count = sum(bin(adj[i]).count('1') for i in range(16)) // 2
        
        # Now do saturation: add all triangle-safe non-edges that preserve non-3-colorability
        all_possible = [(u, v) for u in range(16) for v in range(u+1, 16)]
        random.shuffle(all_possible)
        
        for u, v in all_possible:
            if time.time() > deadline:
                break
            
            # Check if edge exists
            if (adj[u] >> v) & 1:
                continue
                
            # Check triangle free
            if (adj[u] & adj[v]) != 0:
                continue
                
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            # Check if graph is still NOT 3-colorable
            if is_3_colorable_mask(adj):
                # Remove
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
            else:
                # Keep
                current_edge_count += 1
                
        if current_edge_count > best_count:
            best_count = current_edge_count
            # Reconstruct edge list
            best_edges = []
            for u in range(16):
                for v in range(u+1, 16):
                    if (adj[u] >> v) & 1:
                        best_edges.append([u, v])
    
    # If we still have time, do more random restarts with the best distribution
    if best_edges is None:
        # Fallback: just return the Grötzsch graph with isolated vertices
        best_edges = groetzsch_edges[:]
        return best_edges
    
    # If we have significant time left, try to improve further
    remaining_time = deadline - time.time()
    if remaining_time > 30:
        # Do additional random restarts with random edge additions
        for restart in range(500):
            if time.time() > deadline:
                break
            
            adj = [0] * 16
            for u, v in best_edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
            
            current_edge_count = len(best_edges)
            
            all_possible = [(u, v) for u in range(16) for v in range(u+1, 16)]
            random.shuffle(all_possible)
            
            for u, v in all_possible:
                if time.time() > deadline:
                    break
                
                if (adj[u] >> v) & 1:
                    continue
                    
                if (adj[u] & adj[v]) != 0:
                    continue
                    
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_3_colorable_mask(adj):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                else:
                    current_edge_count += 1
                    
            if current_edge_count > best_count:
                best_count = current_edge_count
                best_edges = []
                for u in range(16):
                    for v in range(u+1, 16):
                        if (adj[u] >> v) & 1:
                            best_edges.append([u, v])
    
    return best_edges
