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
    
    # Precompute degrees within the graph for better ordering
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
    verts_by_deg = []
    for v in range(n):
        d = bits[v].bit_count()
        verts_by_deg.append((d, v))
    verts_by_deg.sort(reverse=True)
    
    for d, v in verts_by_deg:
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
    
    strategies = []
    
    # Strategy 1: Copy a random vertex's neighborhood
    for _ in range(8):
        v = rng.randrange(n)
        strategies.append(('copy', v))
        strategies.append(('complement', v))
    
    # Strategy 2: Intersection of two random vertices' neighborhoods
    for _ in range(5):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        if v1 != v2:
            strategies.append(('intersection', v1, v2))
    
    # Strategy 3: Union of two random vertices' neighborhoods  
    for _ in range(5):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        if v1 != v2:
            strategies.append(('union', v1, v2))
    
    # Strategy 4: XOR of two random vertices' neighborhoods
    for _ in range(4):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        if v1 != v2:
            strategies.append(('xor', v1, v2))
    
    # Strategy 5: Majority vote of 3 random vertices
    for _ in range(4):
        v1 = rng.randrange(n)
        v2 = rng.randrange(n)
        v3 = rng.randrange(n)
        if len({v1, v2, v3}) == 3:
            strategies.append(('majority3', v1, v2, v3))
    
    # Strategy 6: Random with target density
    for d in [0.3, 0.4, 0.5, 0.6, 0.7]:
        strategies.append(('random', d))
    
    # Strategy 7: Perturb a random vertex's neighborhood (flip 1-3 bits)
    for _ in range(6):
        v = rng.randrange(n)
        num_flips = rng.randint(1, 3)
        strategies.append(('perturb', v, num_flips))
    
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
        elif strat[0] == 'majority3':
            v1, v2, v3 = strat[1], strat[2], strat[3]
            # For each vertex, check if it's connected to at least 2 of the 3
            new_vertex_mask = 0
            for i in range(n):
                count = 0
                if (bits[v1] >> i) & 1:
                    count += 1
                if (bits[v2] >> i) & 1:
                    count += 1
                if (bits[v3] >> i) & 1:
                    count += 1
                if count >= 2:
                    new_vertex_mask |= (1 << i)
        elif strat[0] == 'random':
            d = strat[1]
            for i in range(n):
                if rng.random() < d:
                    new_vertex_mask |= (1 << i)
        elif strat[0] == 'perturb':
            v, num_flips = strat[1], strat[2]
            new_vertex_mask = bits[v] & full_mask & ~(1 << v)
            # Flip num_flips random bits
            candidates = [i for i in range(n) if i != v]
            rng.shuffle(candidates)
            for i in range(min(num_flips, len(candidates))):
                bit_pos = candidates[i]
                new_vertex_mask ^= (1 << bit_pos)
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
    
    # Try multiple random seeds to explore different search paths
    seed_list = [42, 123, 456, 789, 1000, 2024, 31415, 27182]
    
    for seed in seed_list:
        if _time_left() < 45:
            break
        
        rng = random.Random(seed)
        
        # Work on a copy of the current best
        cur_n = best_n
        cur_adj = [row[:] for row in best_adj]
        cur_bits = best_bits[:]
        cur_edges = best_edges[:]
        
        # Try structured vertex addition multiple times
        max_rounds = 50
        for round_num in range(max_rounds):
            if _time_left() < 45:
                break
            
            found = False
            
            # Try structured addition first
            result = _structured_add_vertex(cur_adj, cur_n, cur_bits, rng)
            if result is not None:
                cur_n, cur_adj, cur_bits, added_edges = result
                cur_edges.extend(added_edges)
                
                # Update best if improved
                if cur_n > best_n:
                    best_n = cur_n
                    best_adj = [row[:] for row in cur_adj]
                    best_edges = cur_edges[:]
                    best_bits = cur_bits[:]
                
                found = True
            
            if not found:
                # Try random addition with various densities
                densities = [0.3, 0.4, 0.5, 0.6, 0.7]
                for d in densities:
                    if _time_left() < 45:
                        break
                    for trial in range(8):
                        if _time_left() < 45:
                            break
                        result = _try_add_vertex(cur_adj, cur_n, cur_bits, d, rng)
                        if result is not None:
                            cur_n, cur_adj, cur_bits, added_edges = result
                            cur_edges.extend(added_edges)
                            
                            if cur_n > best_n:
                                best_n = cur_n
                                best_adj = [row[:] for row in cur_adj]
                                best_edges = cur_edges[:]
                                best_bits = cur_bits[:]
                            
                            found = True
                            break
                    if found:
                        break
            
            if not found:
                # Try removing an edge to create room, then add
                if _time_left() < 60:
                    break
                
                removed = False
                for _ in range(30):
                    if _time_left() < 60:
                        break
                    if not cur_edges:
                        break
                    ei = rng.randrange(len(cur_edges))
                    d = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
                    result = _try_remove_edge_and_add(cur_adj, cur_n, cur_bits, cur_edges[ei], d, rng)
                    if result is not None:
                        cur_n, cur_adj, cur_bits, cur_edges = result
                        
                        if cur_n > best_n:
                            best_n = cur_n
                            best_adj = [row[:] for row in cur_adj]
                            best_edges = cur_edges[:]
                            best_bits = cur_bits[:]
                        
                        removed = True
                        break
                
                if not removed:
                    # Try a few more random addition attempts
                    for _ in range(20):
                        if _time_left() < 60:
                            break
                        d = rng.choice([0.3, 0.4, 0.5, 0.6, 0.7])
                        result = _try_add_vertex(cur_adj, cur_n, cur_bits, d, rng)
                        if result is not None:
                            cur_n, cur_adj, cur_bits, added_edges = result
                            cur_edges.extend(added_edges)
                            
                            if cur_n > best_n:
                                best_n = cur_n
                                best_adj = [row[:] for row in cur_adj]
                                best_edges = cur_edges[:]
                                best_bits = cur_bits[:]
                            
                            removed = True
                            break
                    
                    if not removed:
                        # If we can't add more vertices, break out of this seed's loop
                        break
    
    # Return best result
    return best_n, best_edges

# Keep the required entry point name
# The function is already named construct()
