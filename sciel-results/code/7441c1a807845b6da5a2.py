"""Combined program: Fast bitmask C6 check + Greedy + 2-Swap Local Search."""
# EVOLVE-BLOCK-START
import time
import random
import sys

DEADLINE = 145.0  # Safety margin before 150s kill

def _check_c6_through_edge(adj, u, v, n=40):
    """
    Check if adding edge (u,v) creates a C6.
    A C6 containing edge (u,v) has the form u-v-a-b-c-d-u.
    This is equivalent to finding a path of length 4 from u to v in the graph 
    G - {u, v} (where u and v are removed, but we keep their neighbors for the path endpoints).
    Actually, simpler: 
    Path u - x1 - x2 - x3 - x4 - v.
    x1 in N(v) \ {u}
    x4 in N(u) \ {v, x1}
    x2 in N(x1) \ {v, x4, u}
    x3 in N(x4) \ {u, x2, v, x1}
    Edge (x2, x3) must exist.
    """
    # x1 in N(v) \ {u}
    x1_mask = adj[v] & ~(1 << u)
    while x1_mask:
        x1 = (x1_mask & (-x1_mask)).bit_length() - 1
        x1_mask &= x1_mask - 1
        
        # x4 in N(u) \ {v, x1}
        x4_mask = adj[u] & ~(1 << v) & ~(1 << x1)
        while x4_mask:
            x4 = (x4_mask & (-x4_mask)).bit_length() - 1
            x4_mask &= x4_mask - 1
            
            # x2 in N(x1) \ {v, x4, u}
            # Note: x2 cannot be u, v, x1, x4.
            # adj[x1] contains u? No, x1 is neighbor of v, not necessarily u. 
            # But if x1 is neighbor of u, then x2 could be u? 
            # Path is u-x1-x2... so x2 is neighbor of x1.
            # x2 cannot be u (would be u-x1-u... cycle of length 2? No, simple graph).
            # x2 cannot be v (would be u-x1-v... but v is endpoint).
            # x2 cannot be x4 (would be u-x1-x4-x4... no).
            x2_mask = adj[x1] & ~(1 << v) & ~(1 << x4) & ~(1 << u)
            while x2_mask:
                x2 = (x2_mask & (-x2_mask)).bit_length() - 1
                x2_mask &= x2_mask - 1
                
                # x3 in N(x4) \ {u, x2, v, x1}
                x3_mask = adj[x4] & ~(1 << u) & ~(1 << x2) & ~(1 << v) & ~(1 << x1)
                while x3_mask:
                    x3 = (x3_mask & (-x3_mask)).bit_length() - 1
                    x3_mask &= x3_mask - 1
                    
                    # Check if x2-x3 is an edge
                    if adj[x2] & (1 << x3):
                        return False  # C6 found
    return True  # No C6 through this edge

def _is_c6_free(adj_bits, n=40):
    """Full check: verify no C6 exists in the graph."""
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                if not _check_c6_through_edge(adj_bits, i, j, n):
                    return False
    return True

def _count_edges(adj_bits, n=40):
    count = 0
    for i in range(n):
        count += bin(adj_bits[i]).count('1')
    return count // 2

def _get_edges(adj_bits, n=40):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj_bits[i] & (1 << j):
                edges.append([i, j])
    return edges

def _random_greedy(n=40, seed=None, max_time=2.0):
    """Randomized greedy: try edges in random order, add if C6-free."""
    if seed is not None:
        random.seed(seed)
    start_time = time.time()
    adj = [0] * n
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    for u, v in all_edges:
        if time.time() - start_time > max_time:
            break
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        if not _check_c6_through_edge(adj, u, v, n):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
    return adj, _count_edges(adj, n)

def _local_search_2swap(adj, n=40, max_time=10.0):
    """
    Local search using 1-add and 2-swap moves.
    1-add: Add a non-edge if it doesn't create a C6.
    2-swap: Remove 2 edges, add 2 non-edges, if the result is C6-free.
    This helps escape local optima where no single edge can be added.
    """
    start_time = time.time()
    current = _count_edges(adj, n)
    best_adj = adj[:]
    best_count = current
    
    # Precompute edge list and non-edge list periodically? 
    # For n=40, 780 edges. Recomputing is fast.
    
    while time.time() - start_time < max_time:
        improved = False
        
        # 1. Try to add edges (1-add)
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() - start_time > max_time:
                return best_adj, best_count
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if _check_c6_through_edge(adj, u, v, n):
                current += 1
                improved = True
                if current > best_count:
                    best_count = current
                    best_adj = adj[:]
            else:
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        if improved:
            # If we added edges, loop back to try more adds
            continue
            
        # 2. Try 2-swaps if no edges were added
        # Pick 2 edges to remove and 2 non-edges to add
        cur_edges = [(i, j) for i in range(n) for j in range(i+1, n) if adj[i] & (1 << j)]
        non_edges = [(i, j) for i in range(n) for j in range(i+1, n) if not (adj[i] & (1 << j))]
        
        if len(cur_edges) < 2 or len(non_edges) < 2:
            break
            
        # Try a few random 2-swaps
        swaps_tried = 0
        max_swaps_to_try = 1000
        
        while swaps_tried < max_swaps_to_try and time.time() - start_time < max_time:
            swaps_tried += 1
            # Pick 2 distinct edges to remove
            e1, e2 = random.sample(cur_edges, 2)
            # Ensure they are distinct pairs
            if e1[0] == e2[0] and e1[1] == e2[1]:
                continue
            if e1[0] == e2[1] and e1[1] == e2[0]:
                continue
                
            # Pick 2 distinct non-edges to add
            a1, a2 = random.sample(non_edges, 2)
            if a1[0] == a2[0] and a1[1] == a2[1]:
                continue
            if a1[0] == a2[1] and a1[1] == a2[0]:
                continue
            
            # Check if the add-edges are already in the remove set (they shouldn't be, as one is edge, one is non-edge)
            # But check if a1 or a2 are in e1/e2? No, non_edges are not in adj.
            
            # Apply removals
            adj[e1[0]] &= ~(1 << e1[1])
            adj[e1[1]] &= ~(1 << e1[0])
            adj[e2[0]] &= ~(1 << e2[1])
            adj[e2[1]] &= ~(1 << e2[0])
            
            # Apply additions
            adj[a1[0]] |= (1 << a1[1])
            adj[a1[1]] |= (1 << a1[0])
            adj[a2[0]] |= (1 << a2[1])
            adj[a2[1]] |= (1 << a2[0])
            
            # Check if C6-free
            # We only need to check edges involved in the change? 
            # No, removing edges cannot create C6. Adding edges can.
            # So we just need to check if a1 or a2 creates a C6.
            # But adding a1 might create a C6 that involves a2? 
            # The check `_check_c6_through_edge` for a1 assumes the rest of the graph is fixed.
            # If we check a1 in the context of the graph with a2 added, it's correct.
            
            is_valid = True
            if not _check_c6_through_edge(adj, a1[0], a1[1], n):
                is_valid = False
            elif not _check_c6_through_edge(adj, a2[0], a2[1], n):
                is_valid = False
            
            if is_valid:
                # Accept the swap (edge count remains same, structure changes)
                # We don't update best_count because it's the same.
                # But we update the current state.
                # We should probably limit how many consecutive swaps we do without an add
                # to avoid infinite loops, but randomization helps.
                # Update cur_edges and non_edges lists for next iteration?
                # They are stale now. We break to recompute.
                improved = True
                break
            else:
                # Revert
                adj[a2[0]] &= ~(1 << a2[1])
                adj[a2[1]] &= ~(1 << a2[0])
                adj[a1[0]] &= ~(1 << a1[1])
                adj[a1[1]] &= ~(1 << a1[0])
                adj[e2[0]] |= (1 << e2[1])
                adj[e2[1]] |= (1 << e2[0])
                adj[e1[0]] |= (1 << e1[1])
                adj[e1[1]] |= (1 << e1[0])
        
        if not improved:
            # No valid 2-swap found in this batch, try again or break?
            # Let's try again, maybe with different random seed or just loop.
            # To prevent infinite loop if stuck, we can break after a few failed batches.
            # But for now, just continue to next while iteration which re-shuffles.
            pass

    return best_adj, best_count

def _known_good(n=40):
    """8 disjoint K5s = 100 edges, C6-free."""
    adj = [0] * n
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a+1, 5):
                adj[base+a] |= (1 << (base+b))
                adj[base+b] |= (1 << (base+a))
    return adj

def construct():
    start_time = time.time()
    n = 40
    best_adj = None
    best_edges = 0
    
    # Phase 1: Start from 8xK5
    adj = _known_good(n)
    best_adj = adj[:]
    best_edges = _count_edges(adj, n)
    
    # Phase 2: Local search with 2-swaps from 8xK5
    if time.time() - start_time < DEADLINE - 10:
        adj_ls, count_ls = _local_search_2swap(adj, n, max_time=min(10.0, DEADLINE - (time.time() - start_time) - 5))
        if count_ls > best_edges:
            best_edges = count_ls
            best_adj = adj_ls[:]
    
    # Phase 3: Multiple random greedy restarts + 2-swap
    for attempt in range(20):
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining < 2:
            break
        
        # Short greedy
        mt_greedy = min(1.0, remaining / 4)
        adj_g, count_g = _random_greedy(n, seed=500 + attempt, max_time=mt_greedy)
        
        if count_g > best_edges:
            best_edges = count_g
            best_adj = adj_g[:]
            
        # Apply 2-swap local search on the greedy result
        # Allocate remaining time split between this attempt and future ones
        # Simple heuristic: give it 50% of remaining time if it's promising
        if count_g >= best_edges - 5:
            mt_ls = min(5.0, (remaining / 2))
            adj_l, count_l = _local_search_2swap(adj_g, n, max_time=mt_ls)
            if count_l > best_edges:
                best_edges = count_l
                best_adj = adj_l[:]
    
    # Final verification
    if best_adj is None:
        best_adj = _known_good(n)
    
    if not _is_c6_free(best_adj, n):
        best_adj = _known_good(n)
    
    return _get_edges(best_adj, n)
# EVOLVE-BLOCK-END
