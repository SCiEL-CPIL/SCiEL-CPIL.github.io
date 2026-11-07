import time
import sys
from itertools import combinations

def construct():
    n = 15
    start_time = time.time()
    deadline = 55.0  # Safety margin before 60s kill
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_pairs)
    
    # Map pair to index for bitmask
    pair_index = {}
    for idx, (u, v) in enumerate(all_pairs):
        pair_index[(u, v)] = idx
        
    # Precompute neighbors for each vertex in the complete graph
    # neighbors[i] is a list of (neighbor_vertex, edge_index)
    neighbors = [[] for _ in range(n)]
    for idx, (u, v) in enumerate(all_pairs):
        neighbors[u].append((v, idx))
        neighbors[v].append((u, idx))

    # State: 
    # adj_mask: bitmask of present edges
    # deg: list of degrees
    # forbidden: bitmask of edges that CANNOT be added without violating constraints immediately?
    # Actually, the constraint is global. 
    # Constraint: For every path s-u-v-t, deg(s) != deg(t).
    # This is hard to check incrementally. 
    # However, we can check validity of the *current* graph at each step.
    # To speed up backtracking, we can use a "conflict" check.
    
    def is_valid_full(adj_mask, deg):
        # Check all paths s-u-v-t
        # For each edge (u,v) in the graph:
        #   For each s in N(u) \ {v}:
        #     For each t in N(v) \ {u, s}:
        #       if deg[s] == deg[t]: return False
        # This is O(E * D^2) roughly. With E~50, D~10, it's ~5000 ops. Fast enough.
        
        # Iterate over edges in the graph
        # We need to iterate over set bits in adj_mask
        mask = adj_mask
        while mask:
            bit = mask & -mask
            idx = bit.bit_length() - 1
            u, v = all_pairs[idx]
            
            # Get neighbors of u excluding v
            # We can use the precomputed neighbors list and check adj_mask
            # But iterating lists is slower than bit ops if we had bitmasks for neighbors.
            # Let's use the list and check mask.
            
            # Optimization: Precompute neighbor bitmasks for each vertex?
            # No, adj_mask changes. 
            # Let's just iterate over the small degree lists.
            
            for su, idx_su in neighbors[u]:
                if su == v: continue
                if not (adj_mask >> idx_su) & 1: continue
                
                for tv, idx_tv in neighbors[v]:
                    if tv == u or tv == su: continue
                    if not (adj_mask >> idx_tv) & 1: continue
                    
                    if deg[su] == deg[tv]:
                        return False
        return True

    # Backtracking search
    # We want to maximize edges. 
    # Strategy: Try to include edges. If conflict, exclude.
    # Order of edges: High degree vertices first? Or random?
    # Let's try a recursive DFS with pruning.
    
    # To make it efficient, we can use a "forward checking" style.
    # If adding edge (u,v) creates a path s-u-v-t where deg(s)==deg(t), 
    # we cannot add it *unless* we also change degrees of s or t (by removing/adding other edges).
    # Since we are building from empty, degrees only increase.
    # If we add (u,v), deg(u)++, deg(v)++.
    # New paths appear involving (u,v).
    # Existing paths remain valid IF the degree changes don't break them.
    # Wait, if deg(s) changes, it might become equal to deg(t) for an existing path s'-u'-v'-t' where s=s' or t=t'.
    # So adding an edge can invalidate *existing* valid paths.
    # This makes simple incremental "add if valid" tricky because validity is not monotonic.
    
    # However, if we start from EMPTY, every step we add an edge, we must re-check ALL paths.
    # Given n=15, max edges ~50. 
    # Number of paths is sum over edges (u,v) of (deg(u)-1)*(deg(v)-1).
    # In a dense graph, this is large. But 15 nodes is small.
    
    # Let's use a randomized backtracking with a time limit.
    # We will try to build a graph with as many edges as possible.
    
    best_edges = []
    best_count = 0
    current_adj = 0
    current_deg = [0] * n
    
    # Heuristic: Sort edges by some priority? 
    # Let's try edges in a fixed order but use backtracking.
    # To avoid deep recursion limits and overhead, we can use an iterative deepening or just a standard DFS.
    # Given 105 edges, depth 105 is too deep for pure recursion in Python if we explore much.
    # But we prune heavily.
    
    # Alternative: Local search is better for this size if done well, but the prompt asks for a DIFFERENT strategy.
    # Let's try a "Greedy with Randomized Restart and Conflict Resolution".
    # 1. Start with empty graph.
    # 2. Pick a random edge. Add it.
    # 3. If invalid, we must remove some edges to fix it.
    #    Which edges? The ones causing the conflict.
    #    Conflict: path s-u-v-t with deg(s)==deg(t).
    #    To fix, we can remove an edge incident to s or t to change their degree.
    #    Or remove (u,v) itself.
    
    # This is essentially a SAT solver approach.
    
    # Let's implement a simple backtracking solver that tries to include edges in a specific order.
    # Order: Randomize the edge order for each restart.
    
    def backtrack(edge_idx, adj_mask, deg, count):
        nonlocal best_count, best_edges
        nonlocal start_time, deadline
        
        if time.time() > deadline:
            return
        
        if count > best_count:
            best_count = count
            # Extract edges
            best_edges = [list(all_pairs[i]) for i in range(m) if (adj_mask >> i) & 1]
            
        if edge_idx == m:
            return
            
        # Try to include edge edge_idx
        u, v = all_pairs[edge_idx]
        
        # Tentative add
        new_adj = adj_mask | (1 << edge_idx)
        new_deg = deg[:]
        new_deg[u] += 1
        new_deg[v] += 1
        
        if is_valid_full(new_adj, new_deg):
            backtrack(edge_idx + 1, new_adj, new_deg, count + 1)
            
        # Try to exclude edge edge_idx
        backtrack(edge_idx + 1, adj_mask, deg, count)

    # The above is too slow for 105 edges without strong pruning.
    # is_valid_full is O(E*D^2). 105 edges * 50 paths ~ 5000 checks per node.
    # Branching factor 2. 2^10 is 1000. 2^20 is 1M. 
    # We need to prune earlier.
    
    # Better Pruning: 
    # When adding (u,v), check if it creates a *new* conflict.
    # A new conflict exists if there is a path s-u-v-t such that deg_new[s] == deg_new[t].
    # Note: deg_new[s] = deg[s] + (1 if s==u or s==v else 0).
    # So we only need to check paths involving u or v as endpoints?
    # No, the path is s-u-v-t. s is neighbor of u, t is neighbor of v.
    # s and t are NOT u or v.
    # So deg_new[s] = deg[s] (since s != u, s != v).
    # deg_new[t] = deg[t] (since t != u, t != v).
    # Wait, if s == u? No, s is a neighbor of u, s != v. s can be any neighbor.
    # If s == u, that's a loop, not allowed.
    # So s and t are distinct from u and v.
    # Therefore, deg_new[s] == deg[s] and deg_new[t] == deg[t].
    # So the condition is deg[s] == deg[t] in the OLD graph?
    # No! The path s-u-v-t uses the NEW edge (u,v).
    # The degrees of s and t in the NEW graph are the same as in the OLD graph (unless s or t is u or v, which they are not).
    # So if we add (u,v), the only new paths are those using (u,v).
    # For these new paths, the endpoint degrees are unchanged from the previous state.
    # So if deg[s] == deg[t] in the current state, adding (u,v) creates a conflict.
    # BUT, adding (u,v) might also break existing valid paths?
    # Existing paths do not use (u,v). Their endpoint degrees might change if the endpoints are u or v.
    # So we must check:
    # 1. New paths using (u,v) are valid: For all s in N(u)\{v}, t in N(v)\{u,s}, deg[s] != deg[t].
    # 2. Existing paths remain valid: For all existing paths s'-u'-v'-t' not using (u,v), deg_new[s'] != deg_new[t'].
    #    deg_new[x] = deg[x] + (1 if x in {u,v} else 0).
    #    So if s' or t' is u or v, their degree changes.
    
    # This incremental check is much faster than full check.
    
    def is_valid_incremental(adj_mask, deg, added_edge_idx):
        u, v = all_pairs[added_edge_idx]
        
        # 1. Check new paths using (u,v)
        # s in N(u) \ {v}, t in N(v) \ {u, s}
        # Note: N(u) in the NEW graph includes v. But s != v.
        # So s is in N_old(u).
        for su, idx_su in neighbors[u]:
            if su == v: continue
            if not (adj_mask >> idx_su) & 1: continue # Must be in OLD graph (and thus NEW)
            
            for tv, idx_tv in neighbors[v]:
                if tv == u or tv == su: continue
                if not (adj_mask >> idx_tv) & 1: continue # Must be in OLD graph
                
                # Check degrees in NEW graph
                # deg_new[su] = deg[su] + (1 if su==u or su==v else 0) -> su is not u,v
                # deg_new[tv] = deg[tv] + (1 if tv==u or tv==v else 0) -> tv is not u,v
                if deg[su] == deg[tv]:
                    return False
        
        # 2. Check existing paths that might be broken by degree change of u or v
        # A path s'-u'-v'-t' is broken if deg_new[s'] == deg_new[t'] but deg_old[s'] != deg_old[t'].
        # This can only happen if s' or t' is u or v.
        # Case A: s' == u. Path u-u'-v'-t'.
        #   deg_new[u] = deg[u] + 1.
        #   deg_new[t'] = deg[t'] (t' not u,v).
        #   Conflict if deg[u] + 1 == deg[t'].
        # Case B: t' == u. Path s'-u'-v'-u.
        #   deg_new[s'] = deg[s'].
        #   deg_new[u] = deg[u] + 1.
        #   Conflict if deg[s'] == deg[u] + 1.
        # Case C: s' == v. Path v-u'-v'-t'.
        #   deg_new[v] = deg[v] + 1.
        #   deg_new[t'] = deg[t'].
        #   Conflict if deg[v] + 1 == deg[t'].
        # Case D: t' == v. Path s'-u'-v'-v.
        #   deg_new[s'] = deg[s'].
        #   deg_new[v] = deg[v] + 1.
        #   Conflict if deg[s'] == deg[v] + 1.
        
        # We need to iterate over all existing edges (u', v') != (u,v) and check if they form a path with u or v as endpoint.
        # Actually, simpler: Just re-run full check? 
        # With n=15, full check is very fast. 
        # 105 edges. 
        # Let's stick to full check for correctness and simplicity, but optimize the loop.
        
        return True # Placeholder, we will use full check in the main loop for safety
        
    # Let's optimize the full check using bitmasks for neighbor lists?
    # No, let's just use the optimized full check.
    
    def is_valid_fast(adj_mask, deg):
        mask = adj_mask
        while mask:
            bit = mask & -mask
            idx = bit.bit_length() - 1
            u, v = all_pairs[idx]
            
            # Get neighbors of u in the graph
            # We can iterate neighbors[u] and check mask
            # To speed up, we can precompute a bitmask for neighbors of u? 
            # But mask changes.
            
            # Let's just iterate.
            for su, idx_su in neighbors[u]:
                if su == v: continue
                if not (adj_mask >> idx_su) & 1: continue
                
                for tv, idx_tv in neighbors[v]:
                    if tv == u or tv == su: continue
                    if not (adj_mask >> idx_tv) & 1: continue
                    
                    if deg[su] == deg[tv]:
                        return False
        return True

    # Restart strategy
    import random
    random.seed(42)
    
    # We will try multiple random orderings of edges.
    # For each ordering, we do a DFS that prefers "Include".
    # If "Include" fails, we "Exclude".
    # This finds a maximal valid graph for that ordering.
    
    def solve_ordering(order):
        nonlocal best_count, best_edges
        nonlocal start_time, deadline
        
        # Recursive helper
        def dfs(idx, adj_mask, deg, count):
            nonlocal best_count, best_edges
            nonlocal start_time, deadline
            
            if time.time() > deadline:
                return False # Timeout
            
            if count > best_count:
                best_count = count
                best_edges = [list(all_pairs[i]) for i in range(m) if (adj_mask >> i) & 1]
                
            if idx == len(order):
                return True
                
            e_idx = order[idx]
            u, v = all_pairs[e_idx]
            
            # Try Include
            new_adj = adj_mask | (1 << e_idx)
            new_deg = deg[:]
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_fast(new_adj, new_deg):
                if dfs(idx + 1, new_adj, new_deg, count + 1):
                    return True # Found a good solution, propagate success? 
                    # Actually, we want to find the BEST. 
                    # If we return True, we stop exploring this branch? 
                    # No, we should explore both to find max.
                    # But if we have a time limit, we can't explore everything.
                    # Let's just explore "Include" first. If it leads to a solution, we keep it.
                    # We don't need to backtrack if we found a valid completion?
                    # No, we want MAX edges. 
                    # So we must explore both.
                    
            # Try Exclude
            dfs(idx + 1, adj_mask, deg, count)
            
            return True # Continue searching

        dfs(0, 0, [0]*n, 0)

    # Run multiple trials with different random orders
    # To avoid deep recursion issues, we limit the depth or use iterative?
    # Python recursion limit is usually 1000. Depth 105 is fine.
    
    # However, DFS explores 2^105 nodes in worst case. 
    # Pruning is key.
    # If is_valid_fast is fast, we can prune many branches.
    
    # Let's add a heuristic: If we have already found a solution with count > current count + remaining_edges, we can prune.
    # But we don't know remaining valid edges.
    
    # Let's just run a few restarts.
    
    for trial in range(20):
        if time.time() > deadline - 5:
            break
            
        order = list(range(m))
        random.shuffle(order)
        
        # We will use a modified DFS that keeps the best found so far.
        # To speed up, we can use a "Greedy with Backtracking" approach:
        # Try to include. If conflict, exclude.
        # This is exactly what the DFS above does if we prioritize Include.
        
        # Let's implement a manual stack to avoid recursion overhead and limit depth?
        # No, recursion is fine.
        
        # To make it faster, we can use a "Lookahead" or just rely on the fast validity check.
        
        # One optimization: If the current count + (m - idx) <= best_count, prune.
        def dfs_opt(idx, adj_mask, deg, count):
            nonlocal best_count, best_edges
            nonlocal start_time, deadline
            
            if time.time() > deadline:
                return
            
            if count + (m - idx) <= best_count:
                return
                
            if count > best_count:
                best_count = count
                best_edges = [list(all_pairs[i]) for i in range(m) if (adj_mask >> i) & 1]
                
            if idx == m:
                return
                
            e_idx = order[idx]
            u, v = all_pairs[e_idx]
            
            # Try Include
            new_adj = adj_mask | (1 << e_idx)
            new_deg = deg[:]
            new_deg[u] += 1
            new_deg[v] += 1
            
            if is_valid_fast(new_adj, new_deg):
                dfs_opt(idx + 1, new_adj, new_deg, count + 1)
                
            # Try Exclude
            dfs_opt(idx + 1, adj_mask, deg, count)

        dfs_opt(0, 0, [0]*n, 0)

    # Final conversion
    # Ensure types are Python int
    final_edges = [[int(e[0]), int(e[1])] for e in best_edges]
    
    return final_edges

# Test locally? No, just return.
# The function construct is the entry point.
