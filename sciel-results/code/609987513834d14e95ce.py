# EVOLVE-BLOCK-START
import time
import sys

_DEADLINE = time.time() + 140.0  # Safety margin

def _time_up():
    return time.time() > _DEADLINE

def _has_triangle(adj, n):
    """Exact triangle check using adjacency bitmasks."""
    for a in range(n):
        # Only check edges (a, b) where b > a
        # We can iterate b by popping bits from adj[a] that are > a
        mask = adj[a]
        # Clear bits <= a
        mask &= ~((1 << (a + 1)) - 1)
        while mask:
            if _time_up():
                return True # Assume timeout means fail, but triangle check is fast
            lsb = mask & -mask
            b = lsb.bit_length() - 1
            mask ^= lsb
            # Check if a and b share a common neighbor c > b
            common = adj[a] & adj[b]
            if common >> (b + 1):
                return True
    return False

def _check_alpha_less_than_10(adj, n):
    """
    Returns True if the graph has NO independent set of size 10.
    Equivalently, the maximum independent set size < 10.
    Uses iterative Branch and Bound with bitmasks for speed.
    """
    if _time_up():
        return False # Timeout safety: assume invalid to avoid returning bad graph
    
    # We want to find if there is an independent set of size 10.
    # Independent set in G = Clique in Complement.
    # But it's easier to work with G directly for "non-adjacent" logic?
    # Actually, standard MIS B&B works on G:
    # State: (candidates_mask, current_size, current_set_mask)
    # Branch: pick v from candidates.
    #   Include v: new_candidates = candidates & ~adj[v] (remove neighbors of v)
    #   Exclude v: new_candidates = candidates without v
    
    # Optimization: Order vertices by degree to prune faster?
    # For a fixed graph, we can precompute an ordering.
    # However, dynamic re-ordering is complex. 
    # Simple heuristic: Pick vertex with max degree in induced subgraph? 
    # Or just fixed order 0..n-1.
    # Given n=39, fixed order is usually fine if the graph is "dense enough" or structured.
    
    # Let's use a standard DFS with bitmasks.
    # To speed up, we can use a greedy lower bound for the remaining candidates.
    # But we only need to know if size >= 10 exists.
    
    # Precompute "neighbors in G" as bitmasks (already have adj)
    
    # We will search for a clique of size 10 in the complement graph.
    # Complement adjacency: comp_adj[i] = ~adj[i] & ((1<<n)-1) & ~(1<<i)
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask ^ adj[i] ^ (1 << i)
        
    # Reorder vertices by degree in complement (descending) for better pruning
    # High degree in complement = High non-degree in G = Good candidate for independent set
    order = list(range(n))
    order.sort(key=lambda v: comp_adj[v].bit_count(), reverse=True)
    
    # Remap adjacency to new order
    # new_index -> old_index
    remap_idx = [0] * n
    for new_i, old_v in enumerate(order):
        remap_idx[new_i] = old_v
        
    new_comp_adj = [0] * n
    for new_i in range(n):
        old_v = order[new_i]
        mask = 0
        c = comp_adj[old_v]
        while c:
            lsb = c & -c
            old_u = lsb.bit_length() - 1
            c ^= lsb
            # Find new index of old_u
            # This is O(n) per bit, which is slow. 
            # Better: create a map old->new
            pass
        
    # Let's just use a lookup table for old->new
    old_to_new = [0] * n
    for new_i, old_v in enumerate(order):
        old_to_new[old_v] = new_i
        
    for new_i in range(n):
        old_v = order[new_i]
        mask = 0
        c = comp_adj[old_v]
        while c:
            lsb = c & -c
            old_u = lsb.bit_length() - 1
            c ^= lsb
            new_u = old_to_new[old_u]
            mask |= (1 << new_u)
        new_comp_adj[new_i] = mask
        
    # Now run B&B on new_comp_adj to find clique of size 10
    # If found, return False (graph IS invalid because it HAS independent set 10)
    # If not found, return True (graph is valid)
    
    # Iterative DFS to avoid recursion limits and overhead
    # Stack items: (candidates_mask, current_size)
    # We want to know if any path reaches size 10.
    
    # Heuristic bound: if current_size + popcount(candidates) < 10, prune.
    # Actually, we want to find a clique of size 10.
    # Upper bound on clique size in induced subgraph of candidates is popcount(candidates).
    
    stack = [(full_mask, 0)]
    
    while stack:
        if _time_up():
            return False # Timeout
        
        candidates, size = stack.pop()
        
        if size >= 10:
            return False # Found independent set of size 10
        
        if candidates == 0:
            continue
            
        # Prune if even taking all remaining candidates won't reach 10
        if size + candidates.bit_count() < 10:
            continue
            
        # Pick a vertex from candidates
        # Heuristic: Pick vertex with highest degree in induced subgraph (complement)
        # to maximize chance of finding clique? 
        # Or lowest degree to prune faster?
        # For clique finding, picking high degree is often good for finding, 
        # but for proving non-existence, bounds are key.
        # Let's pick the first available vertex (lowest index in new order)
        # Since we sorted by degree desc, the first bit is the highest degree vertex.
        
        lsb = candidates & -candidates
        v = lsb.bit_length() - 1
        candidates_without_v = candidates ^ lsb
        
        # Branch 1: Include v
        # New candidates must be neighbors of v in complement
        new_candidates = candidates_without_v & new_comp_adj[v]
        stack.append((new_candidates, size + 1))
        
        # Branch 2: Exclude v
        stack.append((candidates_without_v, size))
        
    return True # No independent set of size 10 found

def _validate(adj, n):
    if _has_triangle(adj, n):
        return False
    if not _check_alpha_less_than_10(adj, n):
        return False
    return True

def _paley_graph_41():
    """
    Constructs the Paley graph on 41 vertices.
    Vertices 0..40.
    Edge (i, j) exists if (j-i)^2 is a quadratic residue mod 41.
    Quadratic residues mod 41:
    1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25, 6^2=36, 7^2=49=8, 8^2=64=23, 
    9^2=81=39, 10^2=100=18, 11^2=121=40, 12^2=144=21, 13^2=169=5, 14^2=196=32,
    15^2=225=20, 16^2=256=10, 17^2=289=2, 18^2=324=37, 19^2=361=33
    Residues: {1, 2, 4, 5, 8, 9, 10, 16, 18, 20, 21, 23, 25, 32, 33, 36, 37, 39, 40}
    Note: 0 is not a residue.
    The set of residues R has size (41-1)/2 = 20.
    """
    residues = {1, 2, 4, 5, 8, 9, 10, 16, 18, 20, 21, 23, 25, 32, 33, 36, 37, 39, 40}
    # Wait, I listed 19 above. Let's re-verify.
    # 1, 4, 9, 16, 25, 36, 8, 23, 39, 18, 40, 21, 5, 32, 20, 10, 2, 37, 33
    # Sorted: 1, 2, 4, 5, 8, 9, 10, 16, 18, 20, 21, 23, 25, 32, 33, 36, 37, 39, 40
    # Count: 19? No, (p-1)/2 = 20.
    # Let's check 14^2 = 196 = 4*41 + 32. Correct.
    # 15^2 = 225 = 5*41 + 20. Correct.
    # 16^2 = 256 = 6*41 + 10. Correct.
    # 17^2 = 289 = 7*41 + 2. Correct.
    # 18^2 = 324 = 7*41 + 37. Correct.
    # 19^2 = 361 = 8*41 + 33. Correct.
    # Missing one?
    # 0 is not in R.
    # Let's list 1..20 squares:
    # 1: 1
    # 2: 4
    # 3: 9
    # 4: 16
    # 5: 25
    # 6: 36
    # 7: 49-41=8
    # 8: 64-41=23
    # 9: 81-82=-1=40
    # 10: 100-82=18
    # 11: 121-123=-2=39
    # 12: 144-123=21
    # 13: 169-164=5
    # 14: 196-164=32
    # 15: 225-205=20
    # 16: 256-246=10
    # 17: 289-287=2
    # 18: 324-287=37
    # 19: 361-328=33
    # 20: 400-369=31
    # Ah, 20^2 = 400. 400 = 9*41 + 31. So 31 is a residue.
    # My previous list missed 31.
    residues = {1, 2, 4, 5, 8, 9, 10, 16, 18, 20, 21, 23, 25, 31, 32, 33, 36, 37, 39, 40}
    
    adj = [0] * 41
    for i in range(41):
        for j in range(i + 1, 41):
            diff = (j - i) % 41
            if diff in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return adj

def construct():
    best_result = None
    best_n = 0
    
    # 1. Try the known 39-vertex construction from Paley(41)
    # Delete vertices 0 and 1 from Paley(41).
    # This graph is triangle-free (subgraph of triangle-free Paley).
    # Does it have alpha < 10?
    # Paley(41) has alpha = 9? 
    # If Paley(41) has alpha 9, then any subgraph has alpha <= 9.
    # Actually, we need to be sure Paley(41) doesn't have alpha 10.
    # R(3,10) > 39 implies existence. Paley(41) is a strong candidate.
    
    adj_41 = _paley_graph_41()
    
    # Map 41 vertices to 39 vertices by removing 0 and 1.
    # New vertices 0..38 correspond to old 2..40.
    # Old index v -> New index v-2 (if v>=2)
    
    n = 39
    adj_39 = [0] * n
    
    for new_i in range(n):
        old_i = new_i + 2
        mask = 0
        for new_j in range(n):
            if new_i == new_j:
                continue
            old_j = new_j + 2
            if (adj_41[old_i] >> old_j) & 1:
                mask |= (1 << new_j)
        adj_39[new_i] = mask
        
    # Validate
    if _validate(adj_39, n):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj_39[i] >> j) & 1:
                    edges.append([i, j])
        best_result = {'n': n, 'edges': edges}
        best_n = n
        # If we found a valid 39, we can return immediately as it's the max possible (R(3,10)<=41 -> max 40, but 39 is known).
        # Actually, R(3,10) is 40 or 41. If it's 40, max n is 39. If 41, max n is 40.
        # The task says "survey records 39 as largest known".
        # If we can find 40, that's better. But 39 is safe.
        # Let's check if we have time to try 40.
        if not _time_up():
            # Try to see if a 40-vertex graph exists?
            # Unlikely to be found easily. 
            # However, if R(3,10)=40, no 40-vertex graph exists.
            # If R(3,10)=41, a 40-vertex graph exists.
            # Given the difficulty, returning 39 is the robust play.
            pass
        
    return best_result

# EVOLVE-BLOCK-END
