"""Combined approach: Paley(37) seed + structured vertex addition with
efficient clique/independent-set checking via bitsets."""

import time
import random
import sys

# Deadline guard: evaluator kills at 330s, use 290s internal for safety
_DEADLINE = 290.0
_start_time = time.time()

def _elapsed():
    return time.time() - _start_time

def _time_left():
    return _DEADLINE - _elapsed()

def _paley_37():
    """Return adjacency matrix (list of lists of int) and edge list for Paley(37)."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [[0] * q for _ in range(q)]
    edges = []
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append((i, j))
    return q, adj, edges

def _to_bitsets(adj, n):
    """Convert adjacency matrix to list of Python ints (bitsets)."""
    bits = [0] * n
    for i in range(n):
        b = 0
        for j in range(n):
            if adj[i][j]:
                b |= (1 << j)
        bits[i] = b
    return bits

def _has_k_clique(bits, n, k):
    """Check if graph (bitsets) has a clique of size k. Returns True/False."""
    if n < k:
        return False
    
    # Precompute global degrees for ordering
    global_deg = [bits[v].bit_count() for v in range(n)]
    
    def backtrack(candidates_mask, size_needed):
        if size_needed == 0:
            return True
        if size_needed == 1:
            return candidates_mask != 0
        if candidates_mask.bit_count() < size_needed:
            return False
        # Collect candidate vertices and sort by degree within candidates (descending)
        cm = candidates_mask
        verts = []
        while cm:
            lb = cm & (-cm)
            v = lb.bit_length() - 1
            cm ^= lb
            deg = (candidates_mask & bits[v]).bit_count()
            verts.append((deg, v))
        verts.sort(reverse=True)
        
        for deg, v in verts:
            if deg < size_needed - 1:
                break  # No more vertices can work
            new_cands = candidates_mask & bits[v]
            if new_cands.bit_count() < size_needed - 1:
                continue
            if backtrack(new_cands, size_needed - 1):
                return True
        return False
    
    all_mask = (1 << n) - 1
    # Try starting from each vertex, ordered by degree
    verts_by_deg = list(enumerate(global_deg))
    verts_by_deg.sort(key=lambda x: x[1], reverse=True)
    
    for v, d in verts_by_deg:
        if d < k - 1:
            break
        cands = all_mask & bits[v]
        if cands.bit_count() < k - 1:
            continue
        if backtrack(cands, k - 1):
            return True
    return False

def _has_k_independent(bits, n, k):
    """Check if graph (bitsets) has an independent set of size k."""
    if n < k:
        return False
    # Independent set in G = clique in complement
    comp_bits = [0] * n
    all_mask = (1 << n) - 1
    for i in range(n):
        comp_bits[i] = all_mask & ~bits[i]
        comp_bits[i] &= ~(1 << i)  # remove self
    return _has_k_clique(comp_bits, n, k)

def _is_valid(bits, n):
    """Check (5,5)-free: no K5, no independent 5."""
    if _has_k_clique(bits, n, 5):
        return False
    if _has_k_independent(bits, n, 5):
        return False
    return True

def _has_k_clique_in_subgraph(bits, n, k, subset_mask):
    """Check if there's a k-clique within the given subset of vertices."""
    if subset_mask.bit_count() < k:
        return False
    # Use the same backtracking but restricted to subset
    def backtrack(candidates_mask, size_needed):
        if size_needed == 0:
            return True
        if size_needed == 1:
            return candidates_mask != 0
        if candidates_mask.bit_count() < size_needed:
            return False
        cm = candidates_mask
        verts = []
        while cm:
            lb = cm & (-cm)
            v = lb.bit_length() - 1
            cm ^= lb
            deg = (candidates_mask & bits[v]).bit_count()
            verts.append((deg, v))
        verts.sort(reverse=True)
        for deg, v in verts:
            if deg < size_needed - 1:
                break
            new_cands = candidates_mask & bits[v]
            if new_cands.bit_count() < size_needed - 1:
                continue
            if backtrack(new_cands, size_needed - 1):
                return True
        return False
    
    return backtrack(subset_mask, k)

def _is_valid_add_vertex(bits, n, new_vertex_mask):
    """Check if adding a vertex with adjacency new_vertex_mask (bits 0..n-1)
    keeps the graph (5,5)-free. The new vertex is at index n."""
    # New vertex is at position n
    # Check for K5 including new vertex: need K4 in neighbors of new vertex
    neigh_mask = new_vertex_mask  # bits 0..n-1 where new vertex connects
    if _has_k_clique_in_subgraph(bits, n, 4, neigh_mask):
        return False
    
    # Check for independent 5-set including new vertex:
    # Need independent 4-set in non-neighbors of new vertex
    non_neigh_mask = ((1 << n) - 1) & ~new_vertex_mask
    # Independent set in G = clique in complement
    # Build complement bitsets restricted to non_neigh_mask
    comp_bits = [0] * n
    for i in range(n):
        comp_bits[i] = non_neigh_mask & ~bits[i] & ~(1 << i)
    if _has_k_clique_in_subgraph(comp_bits, n, 4, non_neigh_mask):
        return False
    
    # Also need to verify no existing K5 or independent 5 (shouldn't change, but safe)
    # Actually, adding a vertex can't create a K5 or independent 5 that doesn't include it,
    # since the old graph was already (5,5)-free.
    return True

def _try_add_vertex(adj, n, bits, density, rng):
    """Try to add one vertex with given edge density to existing vertices.
    Returns (new_n, new_adj, new_bits, added_edges) or None if invalid."""
    new_n = n + 1
    new_row = [0] * new_n
    added_edges = []
    new_vertex_mask = 0
    
    for i in range(n):
        if rng.random() < density:
            new_row[i] = 1
            added_edges.append((i, n))
            new_vertex_mask |= (1 << i)
    
    # Fast incremental check
    if not _is_valid_add_vertex(bits, n, new_vertex_mask):
        return None
    
    # Build new adjacency
    new_adj = [row[:] for row in adj]
    for i in range(n):
        new_adj[i].append(new_row[i])
    new_adj.append(new_row[:])
    
    # Build new bitsets
    new_bits = bits[:] + [0]
    new_bits[n] = 0
    for i in range(n):
        if new_row[i]:
            new_bits[i] |= (1 << n)
            new_bits[n] |= (1 << i)
    
    # Full verification (should pass if incremental check passed)
    if _is_valid(new_bits, new_n):
        return new_n, new_adj, new_bits, added_edges
    return None

def _try_remove_edge_and_add(adj, n, bits, edge_idx, density, rng):
    """Remove edge edge_idx from graph, then try to add a vertex.
    Returns (new_n, new_adj, new_bits, new_edges_list) or None."""
    i, j = edge_idx
    # Remove edge
    new_adj = [row[:] for row in adj]
    new_adj[i][j] = 0
    new_adj[j][i] = 0
    
    new_bits = bits[:]
    new_bits[i] &= ~(1 << j)
    new_bits[j] &= ~(1 << i)
    
    # Rebuild edge list
    edges = []
    for a in range(n):
        for b in range(a + 1, n):
            if new_adj[a][b]:
                edges.append((a, b))
    
    result = _try_add_vertex(new_adj, n, new_bits, density, rng)
    if result is not None:
        new_n, new_adj2, new_bits2, added = result
        edges.extend(added)
        return new_n, new_adj2, new_bits2, edges
    return None

def _structured_add_vertex(adj, n, bits, rng):
    """Try to add a vertex using structured patterns based on existing vertex neighborhoods.
    Returns (new_n, new_adj, new_bits, added_edges) or None if invalid."""
    new_n = n + 1
    full_mask = (1 << n) - 1
    
    # Generate diverse strategies
    strategies = []
    
    # Strategy 1: Copy a random vertex's neighborhood
    for _ in range(4):
        v = rng.randrange(n)
        strategies.append(('copy', v))
        strategies.append(('complement', v))
    
    # Strategy 2: Intersection of two random vertices' neighborhoods
    for _ in range(3):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        if v1 != v2:
            strategies.append(('intersection', v1, v2))
    
    # Strategy 3: Union of two random vertices' neighborhoods  
    for _ in range(3):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        if v1 != v2:
            strategies.append(('union', v1, v2))
    
    # Strategy 4: XOR of two random vertices' neighborhoods
    for _ in range(3):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        if v1 != v2:
            strategies.append(('xor', v1, v2))
    
    # Strategy 5: Majority vote of 3 random vertices
    for _ in range(3):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        v3 = rng.randrange(n)
        if len({v1, v2, v3}) >= 2:
            strategies.append(('majority', v1, v2, v3))
    
    # Strategy 6: Random with target density
    for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
        strategies.append(('random', d))
    
    # Strategy 7: Random flip from a base vertex
    for _ in range(3):
        v = rng.randrange(n)
        strategies.append(('flip', v, 0.2))
    
    # NEW: Strategy 8 - Symmetric difference with controlled density
    for _ in range(2):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        if v1 != v2:
            strategies.append(('symdiff', v1, v2))
    
    # NEW: Strategy 9 - Weighted combination of 3 vertices
    for _ in range(2):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        v3 = rng.randrange(n)
        if len({v1, v2, v3}) >= 2:
            strategies.append(('weighted3', v1, v2, v3))
    
    # NEW: Strategy 10 - Neighborhood of a non-neighbor pair
    for _ in range(2):
        v1 = rng.randrange(n)
        # Find a non-neighbor of v1
        non_neigh = full_mask & ~bits[v1] & ~(1 << v1)
        if non_neigh:
            cm = non_neigh
            lb = cm & (-cm)
            v2 = lb.bit_length() - 1
            strategies.append(('nonneigh_pair', v1, v2))
    
    rng.shuffle(strategies)
    
    for strat in strategies:
        if _time_left() < 5:
            break
        new_vertex_mask = 0
        added_edges = []
        
        if strat[0] == 'copy':
            v = strat[1]
            new_vertex_mask = bits[v] & full_mask & ~(1 << v)
        elif strat[0] == 'complement':
            v = strat[1]
            new_vertex_mask = full_mask & ~bits[v] & ~(1 << v)
        elif strat[0] == 'intersection':
            v1, v2 = strat[1], strat[2]
            new_vertex_mask = bits[v1] & bits[v2] & full_mask
        elif strat[0] == 'union':
            v1, v2 = strat[1], strat[2]
            new_vertex_mask = (bits[v1] | bits[v2]) & full_mask
        elif strat[0] == 'xor':
            v1, v2 = strat[1], strat[2]
            new_vertex_mask = (bits[v1] ^ bits[v2]) & full_mask
        elif strat[0] == 'majority':
            v1, v2, v3 = strat[1], strat[2], strat[3]
            # For each vertex, count how many of v1,v2,v3 connect to it
            b1, b2, b3 = bits[v1], bits[v2], bits[v3]
            # Count bits set in each position
            common = (b1 & b2) | (b1 & b3) | (b2 & b3)
            # Majority: at least 2 of 3
            # Positions where exactly 2 or 3 are set
            # b1&b2 includes positions where all 3 are also set
            # We want: (b1&b2) | (b1&b3) | (b2&b3)
            new_vertex_mask = common & full_mask
        elif strat[0] == 'random':
            d = strat[1]
            for i in range(n):
                if rng.random() < d:
                    new_vertex_mask |= (1 << i)
        elif strat[0] == 'flip':
            v = strat[1]
            p_flip = strat[2]
            base = bits[v] & full_mask & ~(1 << v)
            new_vertex_mask = base
            for i in range(n):
                if rng.random() < p_flip:
                    new_vertex_mask ^= (1 << i)
        elif strat[0] == 'symdiff':
            v1, v2 = strat[1], strat[2]
            # Symmetric difference: connect to exactly one of v1, v2
            b1, b2 = bits[v1], bits[v2]
            new_vertex_mask = (b1 ^ b2) & full_mask
        elif strat[0] == 'weighted3':
            v1, v2, v3 = strat[1], strat[2], strat[3]
            b1, b2, b3 = bits[v1], bits[v2], bits[v3]
            # Connect if at least 2 of 3 connect (majority) OR if exactly 1 connects with high probability
            majority = (b1 & b2) | (b1 & b3) | (b2 & b3)
            # Also include some vertices where only 1 connects, with random selection
            only_one = ((b1 & ~b2 & ~b3) | (~b1 & b2 & ~b3) | (~b1 & ~b2 & b3)) & full_mask
            # Randomly select from only_one
            selected = 0
            cm = only_one
            while cm:
                lb = cm & (-cm)
                if rng.random() < 0.3:
                    selected |= lb
                cm ^= lb
            new_vertex_mask = (majority | selected) & full_mask
        elif strat[0] == 'nonneigh_pair':
            v1, v2 = strat[1], strat[2]
            # Connect to common non-neighbors of v1 and v2
            # This creates a vertex that is non-adjacent to both v1 and v2
            common_non_neigh = full_mask & ~bits[v1] & ~bits[v2]
            new_vertex_mask = common_non_neigh
        else:
            continue
        
        # Remove self if present (shouldn't be, but safe)
        new_vertex_mask &= full_mask
        
        # Fast incremental check
        if not _is_valid_add_vertex(bits, n, new_vertex_mask):
            continue
        
        # Build new row
        new_row = [0] * new_n
        for i in range(n):
            if (new_vertex_mask >> i) & 1:
                new_row[i] = 1
                added_edges.append((i, n))
        
        # Build new adjacency
        new_adj = [row[:] for row in adj]
        for i in range(n):
            new_adj[i].append(new_row[i])
        new_adj.append(new_row[:])
        
        # Build new bitsets
        new_bits = bits[:] + [0]
        new_bits[n] = new_vertex_mask
        for i in range(n):
            if new_row[i]:
                new_bits[i] |= (1 << n)
        
        # Full verification
        if _is_valid(new_bits, new_n):
            return new_n, new_adj, new_bits, added_edges
    
    return None

def construct():
    global _start_time
    _start_time = time.time()
    
    # Start with Paley(37)
    n, adj, edges = _paley_37()
    bits = _to_bitsets(adj, n)
    
    # Verify seed
    if not _is_valid(bits, n):
        # Shouldn't happen, but return as-is
        return n, edges
    
    best_n = n
    best_adj = [row[:] for row in adj]
    best_edges = edges[:]
    best_bits = bits[:]
    
    rng = random.Random(42)
    
    # Phase 1: Try structured vertex addition with increased attempts
    added_count = 0
    max_rounds = 200
    for round_num in range(max_rounds):
        if _time_left() < 30:
            break
        
        found = False
        # Try structured addition first (more attempts with different rng)
        for attempt in range(5):
            if _time_left() < 30:
                break
            if attempt == 0:
                r = rng
            else:
                r = random.Random(rng.randint(0, 2**31))
            result = _structured_add_vertex(adj, n, bits, r)
            if result is not None:
                n, adj, bits, added_edges = result
                edges.extend(added_edges)
                best_n = n
                best_adj = [row[:] for row in adj]
                best_edges = edges[:]
                best_bits = bits[:]
                added_count += 1
                found = True
                break
        
        if not found:
            # Try random addition with various densities
            densities = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
            for d in densities:
                if _time_left() < 30:
                    break
                for trial in range(20):
                    if _time_left() < 30:
                        break
                    result = _try_add_vertex(adj, n, bits, d, rng)
                    if result is not None:
                        n, adj, bits, added_edges = result
                        edges.extend(added_edges)
                        best_n = n
                        best_adj = [row[:] for row in adj]
                        best_edges = edges[:]
                        best_bits = bits[:]
                        added_count += 1
                        found = True
                        break
                if found:
                    break
        
        if not found:
            # Phase 2: Try removing an edge to create room, then add
            if _time_left() < 60:
                break
            
            removed = False
            for _ in range(100):
                if _time_left() < 60:
                    break
                if not edges:
                    break
                ei = rng.randrange(len(edges))
                d = rng.choice([0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75])
                result = _try_remove_edge_and_add(adj, n, bits, edges[ei], d, rng)
                if result is not None:
                    n, adj, bits, edges = result
                    best_n = n
                    best_adj = [row[:] for row in adj]
                    best_edges = edges[:]
                    best_bits = bits[:]
                    removed = True
                    break
            
            if not removed:
                # Try a few more random addition attempts
                for _ in range(70):
                    if _time_left() < 60:
                        break
                    d = rng.choice([0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75])
                    result = _try_add_vertex(adj, n, bits, d, rng)
                    if result is not None:
                        n, adj, bits, added_edges = result
                        edges.extend(added_edges)
                        best_n = n
                        best_adj = [row[:] for row in adj]
                        best_edges = edges[:]
                        best_bits = bits[:]
                        removed = True
                        break
                
                if not removed:
                    # Try structured addition one more time with fresh rng
                    rng2 = random.Random(rng.randint(0, 2**31))
                    result = _structured_add_vertex(adj, n, bits, rng2)
                    if result is not None:
                        n, adj, bits, added_edges = result
                        edges.extend(added_edges)
                        best_n = n
                        best_adj = [row[:] for row in adj]
                        best_edges = edges[:]
                        best_bits = bits[:]
                    else:
                        break  # Give up adding more
    
    # Return best result
    return best_n, best_edges

# Keep the required entry point name
# The function is already named construct()
