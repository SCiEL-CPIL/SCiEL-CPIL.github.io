import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    total_edges = n * (n - 1) // 2  # 120
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def check_not_3_colorable(adj_mask):
        """Check if graph with adjacency masks is NOT 3-colorable."""
        order = sorted(range(n), key=lambda x: bin(adj_mask[x]).count('1'), reverse=True)
        
        colors = [-1] * n
        
        def backtrack(idx):
            if idx == n:
                return True
            
            v = order[idx]
            
            forbidden = 0
            # Iterate neighbors
            mask = adj_mask[v]
            while mask:
                bit = mask & (-mask)
                nb = bit.bit_length() - 1
                c = colors[nb]
                if c != -1:
                    forbidden |= (1 << c)
                mask ^= bit
            
            available = 7 & ~forbidden
            
            if available == 0:
                return False
            
            while available:
                bit = available & (-available)
                available ^= bit
                c = bit.bit_length() - 1
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
            
            return False
        
        return not backtrack(0)

    def get_clebsch_adj_mask():
        adj_mask = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                diff = u ^ v
                dist = bin(diff).count('1')
                if dist == 1 or dist == 4:
                    adj_mask[u] |= (1 << v)
                    adj_mask[v] |= (1 << u)
        return adj_mask

    adj_mask = get_clebsch_adj_mask()
    
    if not check_not_3_colorable(adj_mask):
        return []
        
    current_count = sum(bin(adj_mask[i]).count('1') for i in range(n)) // 2
    best_count = current_count
    best_adj_mask = adj_mask[:]
    
    def can_add_edge(u, v, adj_mask):
        if (adj_mask[u] >> v) & 1:
            return False
        common = adj_mask[u] & adj_mask[v]
        if common != 0:
            return False
        return True

    random.seed(42)
    
    # State: adj_mask is known to be non-3-colorable
    # We maintain the invariant that adj_mask is non-3-colorable.
    
    while time.time() < deadline - 2:
        # Phase 1: Greedy Add
        # Since graph is non-3-colorable, any triangle-free addition preserves this.
        # So we just add all possible triangle-free edges.
        added_any = False
        for (u, v) in all_edges:
            if (adj_mask[u] >> v) & 1:
                continue
            if can_add_edge(u, v, adj_mask):
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                current_count += 1
                added_any = True
        
        if current_count > best_count:
            best_count = current_count
            best_adj_mask = adj_mask[:]
            
        if not added_any:
            # Phase 2: Local Search / Removal
            # We are at a locally maximal triangle-free graph.
            # Try removing one edge to potentially allow more additions later.
            # We need to ensure the graph remains non-3-colorable after removal.
            
            current_edge_list = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj_mask[i] >> j) & 1:
                        current_edge_list.append((i, j))
                        
            if not current_edge_list:
                break
                
            random.shuffle(current_edge_list)
            
            improved = False
            for (ru, rv) in current_edge_list:
                if time.time() > deadline:
                    break
                
                # Remove edge
                adj_mask[ru] &= ~(1 << rv)
                adj_mask[rv] &= ~(1 << ru)
                current_count -= 1
                
                # Check if still non-3-colorable
                if check_not_3_colorable(adj_mask):
                    # It is still valid.
                    # Now try to add edges again (Phase 1)
                    # We can just break and let the outer loop handle the add phase.
                    # But we want to be aggressive.
                    # Let's do a quick greedy add to see if we gain.
                    # Actually, just breaking to the outer loop is cleaner and reuses the logic.
                    # However, if we just break, we might not add the edge back immediately.
                    # The outer loop will try to add all possible edges.
                    # This is effectively a "kick".
                    improved = True
                    break
                else:
                    # Revert removal
                    adj_mask[ru] |= (1 << rv)
                    adj_mask[rv] |= (1 << ru)
                    current_count += 1
                    
            if not improved:
                # No single removal led to a valid state?
                # This means the graph is "critically" non-3-colorable?
                # Or we just didn't find a good removal.
                # Try 2-edge removals? Or just restart?
                # For now, if no improvement, we might be stuck.
                # We can try to restart from best_adj_mask with a different seed?
                # Or just break to save time?
                # Let's try to force a change by removing 2 edges?
                # For simplicity, if we can't improve, we stop or shuffle?
                # Since we are maximizing, if we can't find a better path, we stay.
                # But we might be in a local optimum of the "add" phase.
                # If no removal works, we are likely at the global optimum for this seed.
                break

    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (best_adj_mask[u] >> v) & 1:
                final_edges.append([u, v])
                
    return final_edges
