import random
import itertools
import time

N = 16
ALL_SUBSETS_8 = list(itertools.combinations(range(N), 8))

# Precompute Q3 properties
Q3_ADJ = [0]*8
for i in range(8):
    for j in range(8):
        if i != j and bin(i^j).count('1') == 1:
            Q3_ADJ[i] |= (1 << j)

def contains_q3_fast(subset, adj_bits):
    """
    Fast check if induced subgraph on 8 vertices contains Q3.
    """
    # Build local adjacency bitmasks
    local_adj = [0]*8
    for i in range(8):
        vi = subset[i]
        ai = adj_bits[vi]
        mask = 0
        for j in range(8):
            if i != j and (ai >> subset[j]) & 1:
                mask |= (1 << j)
        local_adj[i] = mask
    
    # Degree check: all must have degree >= 3
    for i in range(8):
        if bin(local_adj[i]).count('1') < 3:
            return False
    
    # Edge count check
    total_deg = sum(bin(local_adj[i]).count('1') for i in range(8))
    if total_deg < 24:
        return False
    
    # Backtracking
    mapping = [-1]*8
    
    def backtrack(q3_v):
        if q3_v == 8:
            return True
        used_mask = 0
        for u in range(q3_v):
            used_mask |= (1 << mapping[u])
        
        for s_idx in range(8):
            if (used_mask >> s_idx) & 1:
                continue
            valid = True
            for u in range(q3_v):
                if (Q3_ADJ[u] >> q3_v) & 1:
                    if not ((local_adj[mapping[u]] >> s_idx) & 1):
                        valid = False
                        break
            if valid:
                mapping[q3_v] = s_idx
                if backtrack(q3_v + 1):
                    return True
                mapping[q3_v] = -1
        return False
    
    for first in range(8):
        mapping[0] = first
        if backtrack(1):
            return True
        mapping[0] = -1
    
    return False

def check_add_edge(u, v, adj_bits):
    """
    Check if adding edge (u,v) creates a Q3.
    Only checks subsets containing u and v.
    """
    other_vertices = [x for x in range(N) if x != u and x != v]
    for rest in itertools.combinations(other_vertices, 6):
        subset = tuple(rest) + (u, v)
        if contains_q3_fast(subset, adj_bits):
            return False
    return True

def find_q3_subset(adj_bits):
    """Find any 8-vertex subset containing Q3. Returns subset or None."""
    for subset in ALL_SUBSETS_8:
        if contains_q3_fast(subset, adj_bits):
            return subset
    return None

def construct():
    n = N
    adj_bits = [0]*n
    
    # Phase 1: Initialize with a dense random graph
    # p=0.8 is dense, likely has many Q3s
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.8:
                adj_bits[i] |= (1 << j)
                adj_bits[j] |= (1 << i)
    
    start_time = time.time()
    time_limit = 115
    
    # Phase 2: Repair - Remove edges to destroy all Q3s
    while time.time() - start_time < time_limit - 30:
        q3_sub = find_q3_subset(adj_bits)
        if q3_sub is None:
            break
        
        # Remove a random edge from this Q3
        # Find edges within the subset
        edges_in_q3 = []
        for i in range(8):
            for j in range(i+1, 8):
                if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                    edges_in_q3.append((q3_sub[i], q3_sub[j]))
        
        if not edges_in_q3:
            break
            
        u, v = random.choice(edges_in_q3)
        adj_bits[u] &= ~(1 << v)
        adj_bits[v] &= ~(1 << u)
        
    # Ensure we are valid
    # If time ran out during repair, we might still have Q3s.
    # Force clean if needed.
    if time.time() - start_time < time_limit - 5:
        while True:
            q3_sub = find_q3_subset(adj_bits)
            if q3_sub is None:
                break
            edges_in_q3 = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                        edges_in_q3.append((q3_sub[i], q3_sub[j]))
            if not edges_in_q3: break
            u, v = random.choice(edges_in_q3)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
    # Phase 3: Local Search to maximize edges
    # Strategy: Try to add edges. If blocked, try swaps.
    
    current_edges = sum(bin(adj_bits[i]).count('1') for i in range(n)) // 2
    best_edges = current_edges
    best_adj_bits = adj_bits[:]
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() - start_time < time_limit - 2:
        # Get missing edges
        missing = [e for e in all_edges if not ((adj_bits[e[0]] >> e[1]) & 1)]
        if not missing:
            break
            
        random.shuffle(missing)
        
        added_any = False
        for u, v in missing:
            if time.time() - start_time > time_limit - 2:
                break
                
            # Try to add (u, v)
            adj_bits[u] |= (1 << v)
            adj_bits[v] |= (1 << u)
            
            if check_add_edge(u, v, adj_bits):
                current_edges += 1
                added_any = True
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj_bits = adj_bits[:]
            else:
                # Revert
                adj_bits[u] &= ~(1 << v)
                adj_bits[v] &= ~(1 << u)
                
        if not added_any:
            # Try 2-swaps: Remove one edge, add one edge
            existing = [e for e in all_edges if (adj_bits[e[0]] >> e[1]) & 1]
            if not existing:
                break
            random.shuffle(existing)
            
            swapped = False
            for rem_u, rem_v in existing:
                if time.time() - start_time > time_limit - 2:
                    break
                for add_u, add_v in missing:
                    if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                        continue
                        # Skip if same edge (shouldn't happen as missing vs existing)
                    
                    # Perform swap
                    adj_bits[rem_u] &= ~(1 << rem_v)
                    adj_bits[rem_v] &= ~(1 << rem_u)
                    adj_bits[add_u] |= (1 << add_v)
                    adj_bits[add_v] |= (1 << add_u)
                    
                    # Check validity of the added edge
                    if check_add_edge(add_u, add_v, adj_bits):
                        # Valid swap
                        swapped = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj_bits = adj_bits[:]
                        break
                    else:
                        # Revert
                        adj_bits[add_u] &= ~(1 << add_v)
                        adj_bits[add_v] &= ~(1 << add_u)
                        adj_bits[rem_u] |= (1 << rem_v)
                        adj_bits[rem_v] |= (1 << rem_u)
                        
                if swapped:
                    break
            if not swapped:
                # If we can't add or swap, we might be stuck in a local optimum.
                # Try a 3-swap? Or just break?
                # For now, break to save time, or try random restart?
                # Let's try a few random 3-swaps
                break
                
    # Use best solution
    adj_bits = best_adj_bits
    
    # Final cleanup: ensure no Q3s
    # (Should be clean, but just in case)
    if time.time() - start_time < time_limit - 0.5:
        while True:
            q3_sub = find_q3_subset(adj_bits)
            if q3_sub is None:
                break
            edges_in_q3 = []
            for i in range(8):
                for j in range(i+1, 8):
                    if (adj_bits[q3_sub[i]] >> q3_sub[j]) & 1:
                        edges_in_q3.append((q3_sub[i], q3_sub[j]))
            if not edges_in_q3: break
            u, v = random.choice(edges_in_q3)
            adj_bits[u] &= ~(1 << v)
            adj_bits[v] &= ~(1 << u)
            
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj_bits[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
