import random
import time
import sys
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s limit
    
    # Vertices 0..19
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) keeps the graph K2,3-free.
        A K2,3 is formed if u and v have > 2 common neighbors.
        Also, for any neighbor w of u, if (w, v) is added to the set of common neighbors,
        we must ensure that w and v don't already have 2 common neighbors that are also neighbors of u?
        Actually, the condition is simply: 
        For any pair of vertices (a, b), |N(a) ∩ N(b)| <= 2.
        
        If we add (u, v):
        1. The common neighbors of u and v become N(u) ∩ N(v). This size must be <= 2.
           Note: N(u) and N(v) do not include each other yet in the check for common neighbors 
           if we consider the graph state before addition. 
           Wait, if we add (u, v), then u is in N(v) and v is in N(u).
           The common neighbors of u and v in the NEW graph are (N(u) ∩ N(v)).
           u is not in N(u), v is not in N(v). So N(u) ∩ N(v) is the set of w such that w~u and w~v.
           So we just need |N(u) ∩ N(v)| <= 2.
        
        2. For any neighbor w of u (w != v), the pair (w, v) now has a new common neighbor u.
           Previously, common neighbors of (w, v) were N(w) ∩ N(v).
           Now it is (N(w) ∩ N(v)) ∪ {u} (if u is not already in N(w) ∩ N(v), which it isn't unless w~u and w~v, 
           but w is neighbor of u. If w~v, then w is in N(u) ∩ N(v).)
           
           Let's refine:
           When adding (u, v):
           - For the pair (u, v): new common neighbors are N(u) ∩ N(v). Must be <= 2.
           - For any w in N(u): The pair (w, v) gains u as a common neighbor IF u is not already a common neighbor of w and v.
             u is a neighbor of w. Is u a neighbor of v? No (we are adding it). 
             So u becomes a NEW common neighbor of (w, v) only if w is NOT a neighbor of v? 
             No. Common neighbors of (w, v) are vertices x such that x~w and x~v.
             If we add (u, v), u is now a neighbor of v. 
             Is u a neighbor of w? Yes, w is in N(u) implies u in N(w).
             So u becomes a common neighbor of (w, v) IF u was not already one.
             u was a common neighbor of (w, v) if u~w and u~v. But u~v is false before addition.
             So u is a NEW common neighbor for ALL w in N(u).
             Thus, for every w in N(u), the number of common neighbors of (w, v) increases by 1.
             We must ensure that for all w in N(u), old_common(w, v) + 1 <= 2 => old_common(w, v) <= 1.
             
           - Similarly, for any w in N(v), the pair (u, w) gains v as a new common neighbor.
             We must ensure old_common(u, w) <= 1 for all w in N(v).
             
           So the conditions to add (u, v) are:
           1. |N(u) ∩ N(v)| <= 2
           2. For all w in N(u), |N(w) ∩ N(v)| <= 1  (Note: w is in N(u), so w is a neighbor of u. 
               The common neighbors of w and v are counted. If w is also in N(v), then w is in N(u)∩N(v).
               This condition covers all w in N(u).)
           3. For all w in N(v), |N(u) ∩ N(w)| <= 1
           
           Let's verify condition 2 and 3.
           If w in N(u) and w in N(v), then w is a common neighbor of u and v.
           Condition 1 says there are at most 2 such w's.
           For a specific w in N(u), we check common neighbors of w and v.
           This is O(deg(u) * deg(v)) or similar if done naively.
           With n=20, this is very fast.
        """
        # 1. Check common neighbors of u and v
        common_uv = np.sum(adj[u] & adj[v])
        if common_uv > 2:
            return False
        
        # 2. Check neighbors of u
        neighbors_u = np.where(adj[u] == 1)[0]
        for w in neighbors_u:
            # w is neighbor of u. Check common neighbors of w and v.
            # If w == v, skip (v is not in N(u) before addition)
            if w == v:
                continue
            common_wv = np.sum(adj[w] & adj[v])
            if common_wv > 1:
                return False
                
        # 3. Check neighbors of v
        neighbors_v = np.where(adj[v] == 1)[0]
        for w in neighbors_v:
            if w == u:
                continue
            common_uw = np.sum(adj[u] & adj[w])
            if common_uw > 1:
                return False
                
        return True

    def verify(edges):
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            if u == v or not (0 <= u < n and 0 <= v < n):
                return None
            if adj[u, v]:
                return None
            adj[u, v] = 1
            adj[v, u] = 1
        for u in range(n):
            for v in range(u + 1, n):
                if np.sum(adj[u] & adj[v]) > 2:
                    return None
        return edges

    def greedy_build(seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.uint8)
        pairs = all_pairs[:]
        rng.shuffle(pairs)
        
        for u, v in pairs:
            if is_valid_addition(adj, u, v):
                adj[u, v] = 1
                adj[v, u] = 1
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u, v]:
                    edges.append([u, v])
        return edges

    def local_search(edges, seed):
        rng = random.Random(seed)
        adj = np.zeros((n, n), dtype=np.uint8)
        for u, v in edges:
            adj[u, v] = 1
            adj[v, u] = 1
            
        current_count = len(edges)
        best_count = current_count
        best_adj = adj.copy()
        
        improved = True
        iters = 0
        max_iters = 50000 # Higher limit, time controlled
        
        while improved and iters < max_iters and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            
            # 1. Try to add an edge (try all non-edges, pick first valid)
            non_edges = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
            rng.shuffle(non_edges)
            
            added = False
            for u, v in non_edges:
                if is_valid_addition(adj, u, v):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    current_count += 1
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj.copy()
                    improved = True
                    added = True
                    break
            
            if added:
                continue
                
            # 2. Try 1-2 swap (remove 1, add 2) - more aggressive
            edge_list = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v]]
            if not edge_list:
                break
            
            # Try more random removals
            swaps_tried = 0
            max_swaps = 30
            while swaps_tried < max_swaps and time.time() < deadline - 5.0:
                swaps_tried += 1
                u_rem, v_rem = rng.choice(edge_list)
                
                # Save state
                adj_save = adj.copy()
                count_save = current_count
                
                # Remove edge
                adj[u_rem, v_rem] = 0
                adj[v_rem, u_rem] = 0
                current_count -= 1
                
                # Try to add 2 edges - try more candidates
                added_count = 0
                non_edges_swap = [(u, v) for u in range(n) for v in range(u + 1, n) if not adj[u, v]]
                rng.shuffle(non_edges_swap)
                
                # Try first 50 non-edges for addition
                for u_add, v_add in non_edges_swap[:50]:
                    if added_count == 2:
                        break
                    if is_valid_addition(adj, u_add, v_add):
                        adj[u_add, v_add] = 1
                        adj[v_add, u_add] = 1
                        current_count += 1
                        added_count += 1
                        
                if current_count > count_save:
                    if current_count > best_count:
                        best_count = current_count
                        best_adj = adj.copy()
                    improved = True
                    break
                else:
                    # Revert
                    adj = adj_save
                    current_count = count_save
        
        # Final polish: systematic check for any addable edge
        if time.time() < deadline - 2.0:
            non_edges_final = [(u, v) for u in range(n) for v in range(u + 1, n) if not best_adj[u, v]]
            for u, v in non_edges_final:
                if is_valid_addition(best_adj, u, v):
                    best_adj[u, v] = 1
                    best_adj[v, u] = 1
                    best_count += 1
            
            # Systematic 1-2 swap polish
            if time.time() < deadline - 1.0:
                edge_list_final = [(u, v) for u in range(n) for v in range(u + 1, n) if best_adj[u, v]]
                for u_rem, v_rem in edge_list_final:
                    if time.time() > deadline - 1.0:
                        break
                    adj_save = best_adj.copy()
                    count_save = best_count
                    best_adj[u_rem, v_rem] = 0
                    best_adj[v_rem, u_rem] = 0
                    best_count -= 1
                    
                    added_count = 0
                    non_edges_swap = [(u, v) for u in range(n) for v in range(u + 1, n) if not best_adj[u, v]]
                    for u_add, v_add in non_edges_swap:
                        if added_count == 2:
                            break
                        if is_valid_addition(best_adj, u_add, v_add):
                            best_adj[u_add, v_add] = 1
                            best_adj[v_add, u_add] = 1
                            best_count += 1
                            added_count += 1
                    
                    if best_count <= count_save:
                        best_adj = adj_save
                        best_count = count_save
        
        # Return best found
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if best_adj[u, v]:
                    edges.append([u, v])
        return edges

    best_edges = None
    best_count = 0
    
    # Run multiple restarts with adaptive stopping
    num_restarts = 500
    stagnant_count = 0
    last_improvement_iter = 0
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Greedy construction
        edges = greedy_build(i * 12345 + 6789)
        
        # Local search
        improved_edges = local_search(edges, i * 98765 + 4321)
        
        if len(improved_edges) > best_count:
            best_count = len(improved_edges)
            best_edges = improved_edges
            stagnant_count = 0
            last_improvement_iter = i
        else:
            stagnant_count += 1
        
        # Stop if no improvement for 50 consecutive restarts
        if stagnant_count >= 50:
            break
            
        # Early stop if we reach a very high number
        if best_count > 110:
             pass

    if best_edges is None:
        return []
        
    # Final verification and cleanup
    final_edges = verify(best_edges)
    if final_edges is None:
        # Fallback: rebuild from best_adj if verification failed
        adj_check = np.zeros((n, n), dtype=np.uint8)
        for u, v in best_edges:
            adj_check[u, v] = 1
            adj_check[v, u] = 1
        result = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj_check[u, v]:
                    result.append([u, v])
        return result
    
    return final_edges

if __name__ == "__main__":
    result = construct()
    print(f"Edges: {len(result)}")
