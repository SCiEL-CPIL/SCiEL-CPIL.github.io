import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s buffer
    
    q = 5
    n = 31
    
    # 1. Generate 31 points of PG(2,5)
    points = []
    points.append((1, 0, 0))
    points.append((0, 1, 0))
    points.append((0, 0, 1))
    for y in range(1, q):
        points.append((1, y, 0))
    for z in range(1, q):
        points.append((1, 0, z))
    for z in range(1, q):
        points.append((0, 1, z))
    for y in range(1, q):
        for z in range(1, q):
            points.append((1, y, z))
            
    # 2. Construct Polarity Graph
    # Adjacency as list of sets for fast intersection
    adj = [set() for _ in range(n)]
    base_edges = []
    
    for i in range(n):
        for j in range(i + 1, n):
            p_i = points[i]
            p_j = points[j]
            if (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                base_edges.append((i, j))
                
    target_edges = 91
    edges_to_add = target_edges - len(base_edges) # 26
    
    # 3. Precompute C4 count for all 4-sets based on edge mask
    # There are 2^6 = 64 possible edge configurations for a 4-set.
    # We need to map the 6 edges to a bitmask.
    # For a set {a,b,c,d}, the edges are (a,b), (a,c), (a,d), (b,c), (b,d), (c,d).
    # We precompute the number of C4s for each mask.
    
    def get_c4_count_for_mask(mask):
        # Edges in the 4-set
        # 0: a-b, 1: a-c, 2: a-d, 3: b-c, 4: b-d, 5: c-d
        # A C4 is a cycle of length 4.
        # Possible C4s in K4:
        # 1-2-3-4-1 (edges 0,3,5,2) -> mask bits 0,2,3,5
        # 1-2-4-3-1 (edges 0,4,5,1) -> mask bits 0,1,4,5
        # 1-3-2-4-1 (edges 1,3,4,2) -> mask bits 1,2,3,4
        
        # Check if the 4 edges of a specific C4 are present.
        # C4 1: 0, 2, 3, 5
        # C4 2: 0, 1, 4, 5
        # C4 3: 1, 2, 3, 4
        
        count = 0
        if (mask & (1<<0)) and (mask & (1<<2)) and (mask & (1<<3)) and (mask & (1<<5)):
            count += 1
        if (mask & (1<<0)) and (mask & (1<<1)) and (mask & (1<<4)) and (mask & (1<<5)):
            count += 1
        if (mask & (1<<1)) and (mask & (1<<2)) and (mask & (1<<3)) and (mask & (1<<4)):
            count += 1
            
        return count

    c4_table = [0] * 64
    for m in range(64):
        c4_table[m] = get_c4_count_for_mask(m)
        
    # Helper to calculate total C4s given a set of added edges
    # We maintain the full adjacency list
    # To speed up, we can just recalculate from scratch using bitmasks if needed,
    # but iterating 4-sets is straightforward.
    
    def count_total_c4s(adj_sets, added_edges_set):
        # adj_sets includes base edges + added edges
        # We iterate over all 4-combinations
        total = 0
        # To speed this up, we can precompute the 4-combinations list
        # But we need to check edge presence.
        # Edge presence: base_edges_set or added_edges_set
        
        # Optimization: Precompute list of 4-sets
        # This is done once outside? No, inside is fine if cached.
        # Let's just loop.
        for combo in range(31465): # Placeholder, need actual list
            pass
            
        # Better: Precompute all 4-combinations and their edge bit positions
        # There are 31465 combos.
        # For each combo (a,b,c,d), the edges are:
        # e0: a-b, e1: a-c, e2: a-d, e3: b-c, e4: b-d, e5: c-d
        
        # We will precompute this list of (a,b,c,d, e0_idx, e1_idx, ...) 
        # Actually, we just need to check if the pair is in the adjacency.
        
        # For efficiency in SA, we will maintain a global edge set.
        # And just iterate.
        
        # Since 31465 is small, we can just do it.
        # But Python loops are slow.
        # Let's use a list of 4-tuples.
        
        # This function will be called many times in SA.
        # We need it to be fast.
        
        # Alternative: Use the formula with common neighbors?
        # Sum over all pairs (u,v) of C(u,v) ... 
        # Let's stick to 4-set enumeration but optimize the inner loop.
        
        total = 0
        # Use the precomputed combos list
        for a, b, c, d in combos_4:
            mask = 0
            if b in adj_sets[a]: mask |= 1
            if c in adj_sets[a]: mask |= 2
            if d in adj_sets[a]: mask |= 4
            if c in adj_sets[b]: mask |= 8
            if d in adj_sets[b]: mask |= 16
            if d in adj_sets[c]: mask |= 32
            total += c4_table[mask]
        return total

    # Precompute combos_4
    combos_4 = []
    import itertools
    for c in itertools.combinations(range(n), 4):
        combos_4.append(c)
        
    # Initialize with Greedy
    added_edges = []
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j not in adj[i]:
                non_edges.append((i, j))
                
    # Greedy selection
    current_adj = [set(s) for s in adj] # Copy
    
    # Helper for greedy
    def get_common_neighbors(u, v, adj_list):
        return len(adj_list[u] & adj_list[v])
        
    while len(added_edges) < edges_to_add:
        best_edge = None
        min_c = float('inf')
        for e in non_edges:
            if e in added_edges: continue
            u, v = e
            c = get_common_neighbors(u, v, current_adj)
            if c < min_c:
                min_c = c
                best_edge = e
        if best_edge is None: break
        added_edges.append(best_edge)
        u, v = best_edge
        current_adj[u].add(v)
        current_adj[v].add(u)
        
    # Now we have a starting solution.
    # Start Simulated Annealing
    
    current_added_set = set(added_edges)
    current_adj = [set(s) for s in adj]
    for u, v in current_added_set:
        current_adj[u].add(v)
        current_adj[v].add(u)
        
    current_c4s = count_total_c4s(current_adj, current_added_set)
    best_c4s = current_c4s
    best_added_set = set(current_added_set)
    
    T = 50.0
    T_min = 1.0
    alpha = 0.99
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick random edge to remove
        if not current_added_set:
            break
        e_out = random.choice(list(current_added_set))
        
        # Pick random edge to add
        # Non-edges not in current_added_set
        # We can just pick a random non-edge and check if it's in added_set
        # Or maintain a list of available non-edges
        # For simplicity, pick random from non_edges and check
        
        e_in = None
        while e_in is None or e_in in current_added_set:
            e_in = random.choice(non_edges)
            
        # Calculate new C4s
        # Remove e_out, Add e_in
        u_out, v_out = e_out
        u_in, v_in = e_in
        
        # Temporarily modify
        current_adj[u_out].remove(v_out)
        current_adj[v_out].remove(u_out)
        current_added_set.remove(e_out)
        
        current_adj[u_in].add(v_in)
        current_adj[v_in].add(u_in)
        current_added_set.add(e_in)
        
        new_c4s = count_total_c4s(current_adj, current_added_set)
        
        delta = new_c4s - current_c4s
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_c4s = new_c4s
            if new_c4s < best_c4s:
                best_c4s = new_c4s
                best_added_set = set(current_added_set)
        else:
            # Revert
            current_adj[u_in].remove(v_in)
            current_adj[v_in].remove(u_in)
            current_added_set.remove(e_in)
            
            current_adj[u_out].add(v_out)
            current_adj[v_out].add(u_out)
            current_added_set.add(e_out)
            
        T *= alpha
        
    # Return best solution
    final_edges = []
    for u, v in base_edges:
        final_edges.append([u, v])
    for u, v in best_added_set:
        final_edges.append([u, v])
        
    return final_edges
