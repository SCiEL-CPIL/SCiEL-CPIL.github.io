# EVOLVE-BLOCK-START
import time
import random

def construct():
    """
    Construct a triangle-free graph with no independent set of 10 vertices.
    Uses bitset optimization for speed.
    """
    max_time = 115
    start_time = time.time()
    best_result = None
    
    def is_triangle_free(adj_bits, n):
        # adj_bits[i] is a bitmask of neighbors of i
        for i in range(n):
            # For each neighbor j of i
            # We need to ensure no common neighbor between i and j
            # Actually, we can just check during construction, but for verification:
            for j in range(i+1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    if common:
                        return False
        return True

    def get_mis_size_and_set(adj_bits, n):
        """
        Returns (size, set_of_vertices) of a maximum independent set.
        Uses branch and bound with bitsets.
        """
        best_size = 0
        best_set = []
        
        def branch(current_size, current_set, candidates_mask):
            nonlocal best_size, best_set
            if time.time() - start_time > max_time:
                return
            if current_size + bin(candidates_mask).count('1') <= best_size:
                return
            if candidates_mask == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                return
            
            # Heuristic: pick vertex with max degree in candidates
            # To speed up, just pick the first one or one with fewest neighbors in candidates?
            # Greedy: pick vertex with max degree in induced subgraph?
            # For simplicity and speed in Python, picking a random vertex or first is okay if B&B is tight.
            # Let's pick the vertex with the highest degree in the candidate subgraph.
            
            # Extract candidates
            candidates = []
            m = candidates_mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                candidates.append(v)
                m ^= lsb
            
            if not candidates:
                return

            # Find vertex with max degree in candidates
            max_deg = -1
            v_idx = 0
            for i, v in enumerate(candidates):
                # Count neighbors of v within candidates
                neighbors = adj_bits[v] & candidates_mask
                deg = bin(neighbors).count('1')
                if deg > max_deg:
                    max_deg = deg
                    v_idx = i
            
            v = candidates[v_idx]
            
            # Branch 1: Include v
            # New candidates: candidates that are not neighbors of v and not v
            new_mask = candidates_mask & ~adj_bits[v] & ~(1 << v)
            current_set.append(v)
            branch(current_size + 1, current_set, new_mask)
            current_set.pop()
            
            # Branch 2: Exclude v
            branch(current_size, current_set, candidates_mask & ~(1 << v))
            
        branch(0, [], (1 << n) - 1)
        return best_size, best_set

    def try_one(seed):
        rng = random.Random(seed)
        n = 39
        # Initialize with a random triangle-free graph?
        # Better: Start with empty graph and add edges?
        # Or start with a random graph and remove triangles?
        # Removing triangles from random graph on 39 vertices usually leaves a very sparse graph, 
        # leading to large alpha.
        # Let's try starting with a specific structure or just random and fixing.
        
        # Let's try a "Random Greedy" triangle-free graph generation.
        # Add edges one by one if they don't form a triangle.
        adj_bits = [0] * n
        edges_to_try = [(i, j) for i in range(n) for j in range(i+1, n)]
        rng.shuffle(edges_to_try)
        
        for u, v in edges_to_try:
            # Check if adding (u,v) creates a triangle
            if (adj_bits[u] & adj_bits[v]) == 0:
                adj_bits[u] |= (1 << v)
                adj_bits[v] |= (1 << u)
                
        # Now we have a maximal triangle-free graph.
        # Check alpha
        for _ in range(200): # Iterations to reduce alpha
            if time.time() - start_time > max_time:
                return None
            
            alpha, mis = get_mis_size_and_set(adj_bits, n)
            if alpha <= 9:
                break
            
            # Try to add an edge within MIS to break it
            found = False
            # Shuffle MIS to randomize
            mis_shuffled = mis[:]
            rng.shuffle(mis_shuffled)
            
            for i in range(len(mis_shuffled)):
                for j in range(i+1, len(mis_shuffled)):
                    u = mis_shuffled[i]
                    v = mis_shuffled[j]
                    if not ((adj_bits[u] >> v) & 1):
                        # Check triangle
                        if (adj_bits[u] & adj_bits[v]) == 0:
                            adj_bits[u] |= (1 << v)
                            adj_bits[v] |= (1 << u)
                            found = True
                            break
                if found:
                    break
            
            if not found:
                # If no edge can be added, we are stuck.
                # We might need to remove a vertex? 
                # But we want to maximize n. 
                # If we can't reduce alpha, this seed failed.
                # Try removing a vertex from MIS?
                # This reduces n, which is bad for the score if we can find 39.
                # But better than 0.
                # However, for n=39, we should be able to find a valid graph.
                # If stuck, return None for this seed.
                pass

        # Final verification
        if alpha <= 9:
            # Verify triangle-free
            if is_triangle_free(adj_bits, n):
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj_bits[i] >> j) & 1:
                            edges.append([i, j])
                return {'n': n, 'edges': edges}
        return None

    # Try multiple seeds
    for seed in range(50):
        if time.time() - start_time > max_time:
            break
        result = try_one(seed)
        if result:
            if best_result is None or result['n'] > best_result['n']:
                best_result = result
                if best_result['n'] >= 39:
                    break

    if best_result:
        return best_result
    
    # Fallback: Try n=38
    # ... (Similar logic for n=38)
    
    # Hardcoded fallback if all else fails (should not happen)
    # Return a small valid graph
    # Clebsch graph n=16, alpha=5
    # Blow up by 2? n=32, alpha=10 (Too big)
    # Blow up by 1? n=16.
    # Let's just return a valid graph on 16 vertices if needed.
    n = 16
    edges = []
    # Clebsch graph edges
    # ...
    # For now, return None if not found, but the task requires a valid graph.
    # The prompt says "invalid result ... scores 0".
    # So we must return something valid.
    # The seed program returns a fallback. I will return a small valid graph.
    
    # Construct a valid graph on 10 vertices?
    # C5 x C5? n=25.
    # Let's use the Clebsch graph (16 vertices, alpha=5).
    # It is triangle-free.
    # So n=16 is valid.
    
    # Clebsch graph edges:
    # Vertices 0..15.
    # ...
    # I will generate a simple valid graph:
    # Complete bipartite K_5,5 is triangle free, alpha=5. n=10.
    # n=10 is valid.
    n = 10
    edges = []
    for i in range(5):
        for j in range(5, 10):
            edges.append([i, j])
    return {'n': n, 'edges': edges}
# EVOLVE-BLOCK-END
