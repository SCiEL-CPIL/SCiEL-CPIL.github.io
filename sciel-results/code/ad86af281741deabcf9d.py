import random
import time

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s, we target 110s to be safe
    deadline = start_time + 110
    
    n = 20
    range_n = range(n)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_addition(adj, u, v):
        """
        Check if adding edge (u, v) to graph with adjacency bitmasks 'adj' 
        creates a K3,3 subgraph.
        Returns True if valid (no K3,3 created), False otherwise.
        """
        # u and v are NOT currently connected.
        # We check if u in X, v in Y or u in Y, v in X for some K3,3 partition.
        
        # Case 1: u in X, v in Y
        # X = {u, x1, x2}, Y = {v, y1, y2}
        # Edges required:
        # u-v (being added)
        # u-y1, u-y2 (u connected to y1, y2) -> y1, y2 in adj[u]
        # v-x1, v-x2 (v connected to x1, x2) -> x1, x2 in adj[v]
        # x1-y1, x1-y2, x2-y1, x2-y2 (x1, x2 connected to y1, y2)
        
        # Let C_y = adj[u] & ~(1 << v)  (candidates for y1, y2)
        # Let C_x = adj[v] & ~(1 << u)  (candidates for x1, x2)
        
        # We need to find y1, y2 in C_y and x1, x2 in C_x
        # such that x1, x2 are in adj[y1] AND adj[y2].
        
        # Iterate over pairs in C_y
        # To speed up, we can iterate over subsets or just pairs.
        # Max size of C_y is 18. Pairs ~ 153.
        
        c_y_mask = adj[u] & ~(1 << v)
        c_x_mask = adj[v] & ~(1 << u)
        
        # Extract list of vertices in c_y_mask
        y_candidates = [i for i in range_n if (c_y_mask >> i) & 1]
        
        # If fewer than 2 candidates for Y, no K3,3 of this type
        if len(y_candidates) < 2:
            return True
            
        for i in range(len(y_candidates)):
            y1 = y_candidates[i]
            for j in range(i + 1, len(y_candidates)):
                y2 = y_candidates[j]
                
                # Common neighbors of y1 and y2
                common_y1_y2 = adj[y1] & adj[y2]
                
                # We need at least 2 common neighbors that are in C_x
                # common_y1_y2 & c_x_mask must have at least 2 bits set
                potential_x = common_y1_y2 & c_x_mask
                if potential_x:
                    # Check if at least 2 bits are set
                    p = potential_x
                    # Count bits or check if p & (p-1) is non-zero
                    if p & (p - 1):
                        return False
        return True

    def count_edges(adj):
        count = 0
        for i in range_n:
            count += bin(adj[i]).count('1')
        return count // 2

    best_edges = []
    best_count = 0
    best_adj = [0] * n

    # Helper to convert adj bitmask list to edge list
    def adj_to_edges(adj):
        edges = []
        for i in range_n:
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    # Main search loop
    while time.time() < deadline:
        # 1. Greedy Construction
        random.shuffle(all_edges)
        adj = [0] * n
        current_edges = []
        
        for u, v in all_edges:
            # Tentatively add
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
            if is_valid_addition(adj, u, v):
                current_edges.append((u, v))
            else:
                # Revert
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        
        # 2. Local Search / Hill Climbing
        # We will try to improve the current solution by removing one edge and adding others.
        # To make this efficient, we maintain the list of edges.
        # However, re-calculating validity from scratch is safer and still fast enough for n=20.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each edge? Or random edge?
            # Random edge is faster to explore.
            if not current_edges:
                break
                
            # Pick a random edge to remove
            edge_idx = random.randint(0, len(current_edges) - 1)
            u_rem, v_rem = current_edges[edge_idx]
            
            # Remove it
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            current_edges.pop(edge_idx)
            
            # Try to add any missing edges
            # We need to identify missing edges. 
            # Instead of checking all 190 edges, we can check neighbors of u_rem and v_rem first?
            # Actually, removing an edge can unblock many edges.
            # Let's iterate through all non-edges.
            added_any = False
            # We need a list of all potential edges not in current_edges
            # For simplicity, we iterate all possible edges and check if they are in current_edges (O(N^2) set lookup)
            # To optimize, keep a set of current edges
            edge_set = set(current_edges)
            # Note: (u,v) and (v,u) should be normalized. Our all_edges has u<v.
            
            # To avoid O(N^2) set lookups in inner loop, we can just check bitmask
            # But we need to know which edges to try.
            # Let's just try all edges in all_edges that are not in current_adj
            # Wait, we just removed one. Maybe we can just try all edges?
            # There are 190 edges. 190 * check_cost.
            # check_cost is ~150 ops. 190 * 150 = 28k ops.
            # This is very fast.
            
            for u, v in all_edges:
                if (adj[u] >> v) & 1:
                    continue # Already connected
                
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
                if is_valid_addition(adj, u, v):
                    current_edges.append((u, v))
                    added_any = True
                else:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if added_any:
                improved = True
            else:
                # Revert the removal if no improvement
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
                current_edges.append((u_rem, v_rem))

        # Update best
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_adj = adj.copy()
            best_edges = adj_to_edges(best_adj)
            
        # If we found a very high quality graph, we might want to try more local search
        # But the loop structure above already does local search per restart.
        
    return best_edges
