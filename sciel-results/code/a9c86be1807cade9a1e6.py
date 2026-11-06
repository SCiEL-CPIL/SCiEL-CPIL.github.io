"""
K2,3-free graph construction on 20 vertices.
Uses a combination of:
1. Bitmask-based fast common-neighbor tracking
2. Greedy construction with multiple orderings
3. Local search via edge swaps and removals
4. Polarity graph / finite geometry inspired structure
"""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n = 20
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute bit masks
    bit = [1 << i for i in range(n)]
    
    def init_adj():
        return [0] * n
    
    def count_edges(adj):
        c = 0
        for i in range(n):
            c += bin(adj[i]).count('1')
        return c // 2
    
    def common_count(adj, u, v):
        """Count common neighbors of u and v."""
        cn = adj[u] & adj[v]
        return bin(cn).count('1')
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u,v) maintains K2,3-freeness.
        """
        # Check 1: u and v can't have >2 common neighbors
        if common_count(adj, u, v) > 2:
            return False
        
        # Check 2: For each w in N(v), |N(u) ∩ N(w)| <= 1 after adding (u,v)
        # After adding, u becomes a neighbor of v, so for w in N(v):
        # N(u) ∩ N(w) gains v if v in N(w) (always true since w in N(v) and graph is simple)
        # Actually, we need to check that adding (u,v) doesn't create a violation
        # For each w in N(v) (w != u): after adding, N(u) includes v
        # So N(u) ∩ N(w) = (old N(u) ∩ N(w)) ∪ ({v} if v in N(w))
        # Since w is neighbor of v, v is in N(w), so we add v to the intersection
        # But v is not in old N(u) (since we're checking before adding), so:
        # new |N(u) ∩ N(w)| = old |N(u) ∩ N(w)| + 1 (because v is in N(w) but not in old N(u))
        # Wait, that's not right. Let me think again.
        
        # Before adding: N(u) and N(v) are as they are.
        # After adding (u,v): u ∈ N(v) and v ∈ N(u).
        # For a pair (a,b) to have 3 common neighbors after adding:
        # Case 1: a=u, b=v: common neighbors = old N(u) ∩ N(v) (u,v don't count as their own neighbors)
        #   This is already checked above.
        # Case 2: a=u, b=w where w ∈ N(v), w ≠ u:
        #   New common neighbors of u and w = (old N(u) ∩ N(w)) ∪ {v} (since v is now in N(u) and v is in N(w))
        #   But wait, is v in old N(u)? No, we're adding (u,v) so v was not in N(u) before.
        #   Is v in N(w)? Yes, because w ∈ N(v) implies v ∈ N(w).
        #   So new count = old_count + 1. Need old_count <= 1.
        # Case 3: a=v, b=w where w ∈ N(u), w ≠ v:
        #   New common neighbors of v and w = (old N(v) ∩ N(w)) ∪ {u}
        #   Similarly, need old_count <= 1.
        # Case 4: a,w both in N(v) or both in N(u) - adding (u,v) doesn't change their common neighbors
        #   unless one of them is u or v, which is covered above.
        
        # So we need:
        # For each w in N(v), w != u: common_count(adj, u, w) <= 1
        # For each w in N(u), w != v: common_count(adj, v, w) <= 1
        
        nv = adj[v]
        w = 0
        while nv:
            low = nv & (-nv)
            w = low.bit_length() - 1
            if w != u:
                if common_count(adj, u, w) > 1:
                    return False
            nv ^= low
        
        nu = adj[u]
        w = 0
        while nu:
            low = nu & (-nu)
            w = low.bit_length() - 1
            if w != v:
                if common_count(adj, v, w) > 1:
                    return False
            nu ^= low
        
        return True
    
    def add_edge(adj, u, v):
        adj[u] |= bit[v]
        adj[v] |= bit[u]
    
    def remove_edge(adj, u, v):
        adj[u] &= ~bit[v]
        adj[v] &= ~bit[u]
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges
    
    def greedy_fill(adj, order):
        """Add all valid edges from the given order."""
        for u, v in order:
            if not ((adj[u] >> v) & 1):
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
    
    def local_search(adj, max_iters=500, deadline=deadline):
        """
        Local search via edge swaps:
        - Remove a random edge
        - Try to add back plus other edges via greedy fill
        - If we gain edges, keep; else revert
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        current_count = best_count
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Try a swap: remove one edge, then greedy fill
            cur_edges = get_edges(current_adj)
            
            if not cur_edges:
                break
            
            idx = random.randint(0, len(cur_edges) - 1)
            u_rem, v_rem = cur_edges[idx]
            
            backup = current_adj[:]
            remove_edge(current_adj, u_rem, v_rem)
            
            # Greedy fill
            missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if not ((current_adj[i] >> j) & 1)]
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(current_adj, u, v):
                    add_edge(current_adj, u, v)
            
            new_count = count_edges(current_adj)
            
            if new_count > best_count:
                best_adj = current_adj[:]
                best_count = new_count
                current_adj = best_adj[:]
                no_improve = 0
            else:
                # Revert
                current_adj = backup[:]
                no_improve += 1
                if no_improve > 40:
                    break
        
        return best_adj, best_count
    
    def edge_swap_local_search(adj, max_iters=300, deadline=deadline):
        """
        More aggressive local search: try swapping one edge for another.
        """
        best_adj = adj[:]
        best_count = count_edges(best_adj)
        current_adj = adj[:]
        no_improve = 0
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            cur_edges = get_edges(current_adj)
            if not cur_edges:
                break
            
            # Pick an edge to remove
            u_rem, v_rem = cur_edges[random.randint(0, len(cur_edges) - 1)]
            
            # Pick a missing edge to add
            missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                       if not ((current_adj[i] >> j) & 1)]
            if not missing:
                break
            
            u_add, v_add = missing[random.randint(0, len(missing) - 1)]
            
            backup = current_adj[:]
            remove_edge(current_adj, u_rem, v_rem)
            
            if is_valid_add(current_adj, u_add, v_add):
                add_edge(current_adj, u_add, v_add)
                
                # Now try to add more edges greedily
                still_missing = [(i, j) for i in range(n) for j in range(i+1, n) 
                                if not ((current_adj[i] >> j) & 1)]
                random.shuffle(still_missing)
                for u, v in still_missing:
                    if is_valid_add(current_adj, u, v):
                        add_edge(current_adj, u, v)
                
                new_count = count_edges(current_adj)
                if new_count > best_count:
                    best_adj = current_adj[:]
                    best_count = new_count
                    no_improve = 0
                else:
                    current_adj = backup[:]
                    no_improve += 1
            else:
                current_adj = backup[:]
                no_improve += 1
            
            if no_improve > 50:
                break
        
        return best_adj, best_count
    
    def try_structured_init(steps_list, deadline=deadline):
        """Try various structured initializations."""
        results = []
        
        for steps in steps_list:
            if time.time() > deadline - 10:
                break
            
            adj = init_adj()
            # Circulant-like: add edges based on distance
            for i in range(n):
                for j in range(i+1, n):
                    diff = (j - i) % n
                    if diff in steps or (n - diff) in steps:
                        if is_valid_add(adj, i, j):
                            add_edge(adj, i, j)
            
            cnt = count_edges(adj)
            results.append((adj, cnt))
        
        return results
    
    # Main strategy:
    # 1. Run multiple random restarts with greedy + local search
    # 2. Also try some structured initializations
    # 3. Use edge swap local search for refinement
    
    best_adj_overall = None
    best_count_overall = 0
    
    # Strategy 1: Random restarts with varied greedy orderings
    max_restarts = 30
    for r in range(max_restarts):
        if time.time() > deadline - 15:
            break
        
        adj = init_adj()
        
        # Different ordering strategies
        strategy = r % 5
        order = all_pairs[:]
        
        if strategy == 0:
            random.shuffle(order)
        elif strategy == 1:
            # Sort by sum of indices
            order.sort(key=lambda x: x[0] + x[1])
            random.shuffle(order)
        elif strategy == 2:
            # Sort by difference
            order.sort(key=lambda x: abs(x[1] - x[0]))
            random.shuffle(order)
        elif strategy == 3:
            # Random with bias toward smaller indices
            order.sort(key=lambda x: x[0])
            random.shuffle(order)
        else:
            # Fully random
            random.shuffle(order)
        
        # Greedy construct
        greedy_fill(adj, order)
        
        # Local search
        adj, cnt = local_search(adj, max_iters=200)
        
        if cnt > best_count_overall:
            best_count_overall = cnt
            best_adj_overall = adj[:]
        
        # Edge swap refinement
        adj2, cnt2 = edge_swap_local_search(adj, max_iters=100)
        if cnt2 > best_count_overall:
            best_count_overall = cnt2
            best_adj_overall = adj2[:]
        
        # Early exit if we hit a high count
        if best_count_overall >= 65:
            break
    
    # Strategy 2: Try structured initialization with circulant graphs
    if time.time() < deadline - 15:
        step_sets = [
            [1], [2], [3], [4], [5], [6], [7],
            [1,2], [1,3], [1,4], [2,3], [2,4], [3,4],
            [1,2,3], [1,2,4], [2,3,4],
        ]
        
        for steps in step_sets:
            if time.time() > deadline - 10:
                break
            
            adj = init_adj()
            for i in range(n):
                for j in range(i+1, n):
                    diff = (j - i) % n
                    if diff in steps or (n - diff) in steps:
                        if is_valid_add(adj, i, j):
                            add_edge(adj, i, j)
            
            adj, cnt = local_search(adj, max_iters=150)
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
            
            adj2, cnt2 = edge_swap_local_search(adj, max_iters=80)
            if cnt2 > best_count_overall:
                best_count_overall = cnt2
                best_adj_overall = adj2[:]
    
    # Strategy 3: Try complete bipartite-like structures
    if time.time() < deadline - 12:
        for split in range(5, 15):
            if time.time() > deadline - 8:
                break
            
            adj = init_adj()
            # First try adding all cross edges
            for i in range(split):
                for j in range(split, n):
                    if is_valid_add(adj, i, j):
                        add_edge(adj, i, j)
            
            # Then try adding intra-part edges
            for part in [range(split), range(split, n)]:
                for i in part:
                    for j in part:
                        if i < j:
                            if is_valid_add(adj, i, j):
                                add_edge(adj, i, j)
            
            adj, cnt = local_search(adj, max_iters=150)
            if cnt > best_count_overall:
                best_count_overall = cnt
                best_adj_overall = adj[:]
            
            adj2, cnt2 = edge_swap_local_search(adj, max_iters=80)
            if cnt2 > best_count_overall:
                best_count_overall = cnt2
                best_adj_overall = adj2[:]
    
    # Strategy 4: Final intensive local search on best result
    if best_adj_overall is not None and time.time() < deadline - 10:
        adj_final, cnt_final = local_search(best_adj_overall, max_iters=500)
        if cnt_final > best_count_overall:
            best_count_overall = cnt_final
            best_adj_overall = adj_final[:]
        
        adj_final2, cnt_final2 = edge_swap_local_search(best_adj_overall, max_iters=300)
        if cnt_final2 > best_count_overall:
            best_count_overall = cnt_final2
            best_adj_overall = adj_final2[:]
    
    # Convert best_adj to edge list
    if best_adj_overall is None:
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj_overall[i] >> j) & 1:
                edges.append([i, j])
    
    return edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
