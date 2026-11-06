import time

def construct():
    n = 35
    start_time = time.time()
    
    # Helper to check if a circulant graph defined by mask is valid
    def is_valid(adj_bits):
        # 1. Check for K4
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_bits[i] >> j) & 1:
                    common = adj_bits[i] & adj_bits[j]
                    # Check if there is any edge within 'common'
                    # If yes, {i, j, k, l} forms a K4
                    c = common
                    while c:
                        lsb = c & -c
                        k = lsb.bit_length() - 1
                        c ^= lsb
                        # If k is connected to any other vertex in common, we have an edge
                        # Note: adj_bits[k] contains neighbors of k.
                        # If (adj_bits[k] & common) is not 0, there is an edge.
                        # We must ensure we don't pick k itself (not in adj) or i/j (not in common? i,j are not in common unless triangle? No, common is N(i) \cap N(j). i is not in N(i). So i,j not in common).
                        if (adj_bits[k] & common) != 0:
                            return False
                            
        # 2. Check for Independent Set of 6 (K6 in complement)
        # Build complement bitmasks
        full_mask = (1 << n) - 1
        comp_bits = []
        for i in range(n):
            # Complement of adj_bits[i], excluding self
            comp_bits.append((full_mask ^ adj_bits[i]) & ~(1 << i))
            
        # Backtracking search for K6 in comp_bits
        # Use a list of candidate vertices
        # To speed up, order candidates by degree in complement (descending)
        # But for simplicity and speed in Python, we'll just use a list and prune.
        
        def has_clique(bits, k, candidates):
            if k == 0:
                return True
            if len(candidates) < k:
                return False
            if not candidates:
                return False
            
            # Pick the first candidate
            v = candidates[0]
            # New candidates: those connected to v in bits
            new_cands = []
            for u in candidates[1:]:
                if (bits[v] >> u) & 1:
                    new_cands.append(u)
            
            if has_clique(bits, k - 1, new_cands):
                return True
            
            # Exclude v
            if has_clique(bits, k, candidates[1:]):
                return True
            
            return False
        
        if has_clique(comp_bits, 6, list(range(n))):
            return False
            
        return True

    # Search for a valid circulant graph
    # Vertices 0..n-1
    # Edges (i, i+s) % n for s in S
    # S is a subset of {1, ..., n//2}
    
    best_n = 17
    best_edges = []
    
    # Fallback seed (17 vertices, known valid for R(4,4) but we need R(4,6)? 
    # Wait, the 17 vertex seed in the original code was for something else.
    # Let's just use a simple valid small graph if 35 fails.
    # A random graph on 20 vertices might work?
    # Or just return the 17-vertex one if 35 fails.
    
    squares = {x * x % 17 for x in range(1, 17)}
    seed_edges = set((i, j) for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares)
    best_edges = [[u, v] for u, v in seed_edges]
    
    # Try to find 35-vertex circulant graph
    for mask in range(1 << (n // 2)):
        if time.time() - start_time > 115:
            break
            
        adj_bits = [0] * n
        # Build adjacency for this mask
        for s in range(1, n // 2 + 1):
            if (mask >> (s - 1)) & 1:
                for i in range(n):
                    j = (i + s) % n
                    if i < j:
                        adj_bits[i] |= (1 << j)
                        adj_bits[j] |= (1 << i)
                    
        # Optional: Quick degree check
        # Avg degree should be around n/2
        # deg(i) = sum(1 for s in S if s in S or n-s in S)
        # Actually, for circulant, all degrees are same.
        # deg = 2 * popcount(mask)
        # If deg > 20 or deg < 10, skip? 
        # R(4,6) graphs are usually around p=0.5, so deg ~ 17.
        # popcount ~ 8 or 9.
        
        if is_valid(adj_bits):
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_bits[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
    # If not found, return best known (17)
    return {'n': best_n, 'edges': best_edges}
