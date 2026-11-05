import random
import time
import numpy as np

def construct():
    n = 20
    start_time = time.time()
    max_time = 115.0  # Safety margin
    
    best_edges = []
    best_count = 0
    
    # Precompute all pairs
    all_pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_valid_add(u, v, adj_mask, common, common_mask):
        """Check if edge (u,v) can be added without violating K2,3-free constraint.
        Uses bit-level checks: if common[u][w] >= 2, then bit w is set in common_mask[u].
        """
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 2:
            return False
        
        # Adding (u,v) makes u and v share each other's other neighbors.
        # For each w in N(v)\{u}: common[u][w] increases by 1.
        #   Valid iff common[u][w] < 2 for all such w, i.e., no bit w set in common_mask[u].
        mask_v = adj_mask[v] & ~(1 << u)
        if mask_v & common_mask[u]:
            return False
        
        # Similarly for v's side
        mask_u = adj_mask[u] & ~(1 << v)
        if mask_u & common_mask[v]:
            return False
        
        return True

    def add_edge(u, v, adj_mask, common, common_mask, edges, edge_set):
        """Add edge (u,v) to the graph state."""
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append((u, v))
        edge_set.add((u, v))
        edge_set.add((v, u))
        
        # Update common neighbor counts and masks
        # For each w in N(v)\{u}, common[u][w] += 1
        mask_v = adj_mask[v] & ~(1 << u)
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            if common[u][w] >= 2:
                common_mask[u] |= (1 << w)
                common_mask[w] |= (1 << u)
            m -= lsb
        
        # For each w in N(u)\{v}, common[v][w] += 1
        mask_u = adj_mask[u] & ~(1 << v)
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            if common[v][w] >= 2:
                common_mask[v] |= (1 << w)
                common_mask[w] |= (1 << v)
            m -= lsb

    def remove_edge(u, v, adj_mask, common, common_mask, edges, edge_set):
        """Remove edge (u,v) from the graph state."""
        # Capture neighbor sets BEFORE clearing the edge
        mask_v = adj_mask[v] & ~(1 << u)
        mask_u = adj_mask[u] & ~(1 << v)
        
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove((u, v))
        edge_set.discard((u, v))
        edge_set.discard((v, u))
        
        # For each w in N(v)\{u}, common[u][w] -= 1
        m = mask_v
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            if common[u][w] < 2:
                common_mask[u] &= ~(1 << w)
                common_mask[w] &= ~(1 << u)
            m -= lsb
        
        # For each w in N(u)\{v}, common[v][w] -= 1
        m = mask_u
        while m:
            lsb = m & -m
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            if common[v][w] < 2:
                common_mask[v] &= ~(1 << w)
                common_mask[w] &= ~(1 << v)
            m -= lsb

    def greedy_fill(adj_mask, common, common_mask, edges, edge_set):
        """Greedy fill: add edges in random order if valid."""
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if is_valid_add(u, v, adj_mask, common, common_mask):
                add_edge(u, v, adj_mask, common, common_mask, edges, edge_set)

    def local_search(initial_edges, max_iters=500, deadline=None):
        """Local search with kicks to escape local optima."""
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        common_mask = [0] * n
        edges = []
        edge_set = set()
        
        for u, v in initial_edges:
            add_edge(u, v, adj_mask, common, common_mask, edges, edge_set)
            
        best_local_edges = list(edges)
        best_local_count = len(edges)
        
        for it in range(max_iters):
            if deadline is not None and time.time() > deadline - 5:
                break
            
            # Try to add all valid edges in one pass
            added_any = False
            pairs = all_pairs[:]
            random.shuffle(pairs)
            for u, v in pairs:
                if is_valid_add(u, v, adj_mask, common, common_mask):
                    add_edge(u, v, adj_mask, common, common_mask, edges, edge_set)
                    added_any = True
                    if len(edges) > best_local_count:
                        best_local_count = len(edges)
                        best_local_edges = list(edges)
            
            if not added_any:
                # Local optimum - apply a kick
                if len(edges) == 0:
                    break
                
                # Smarter kick: prefer edges that block many potential additions
                # Identify "blocking" edges: edges (u,v) where removing them would
                # allow more edges to be added
                kick_size = random.choice([1, 1, 2, 2, 3, 4, 5])
                
                # Score each edge by how many pairs it blocks
                # An edge (u,v) blocks pair (a,b) if adding (a,b) is invalid due to
                # common neighbor constraints involving u or v
                # Simplified: just pick random edges, but with bias toward high-degree vertices
                if len(edges) > kick_size:
                    # Weighted sampling: edges incident to high-degree vertices are more "blocking"
                    weights = []
                    for u, v in edges:
                        deg_u = bin(adj_mask[u]).count('1')
                        deg_v = bin(adj_mask[v]).count('1')
                        weights.append(deg_u * deg_v + 1)
                    
                    # Use weighted random sampling
                    to_remove = set()
                    total_w = sum(weights)
                    for _ in range(min(kick_size, len(edges))):
                        r = random.random() * total_w
                        cum = 0
                        for i, (u, v) in enumerate(edges):
                            cum += weights[i]
                            if r <= cum:
                                to_remove.add((u, v))
                                break
                    to_remove = list(to_remove)
                else:
                    to_remove = random.sample(edges, min(kick_size, len(edges)))
                
                for u, v in to_remove:
                    remove_edge(u, v, adj_mask, common, common_mask, edges, edge_set)
                
                # After kick, try to fill again with multiple passes
                for _ in range(80):
                    if deadline is not None and time.time() > deadline - 5:
                        break
                    pairs = all_pairs[:]
                    random.shuffle(pairs)
                    added = False
                    for u, v in pairs:
                        if is_valid_add(u, v, adj_mask, common, common_mask):
                            add_edge(u, v, adj_mask, common, common_mask, edges, edge_set)
                            added = True
                            if len(edges) > best_local_count:
                                best_local_count = len(edges)
                                best_local_edges = list(edges)
                    if not added:
                        break
        
        return best_local_edges

    def verify_edges(edge_list):
        """Verify that the edge list forms a K2,3-free graph."""
        adj = [0] * n
        for u, v in edge_list:
            if u == v:
                return False
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if bin(common).count('1') > 2:
                    return False
        return True

    # Main search loop
    deadline = start_time + max_time
    while time.time() < deadline:
        # Greedy initialization with random ordering
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        common_mask = [0] * n
        edges = []
        edge_set = set()
        greedy_fill(adj_mask, common, common_mask, edges, edge_set)
        
        # Local search with kicks
        result_edges = local_search(edges, deadline=deadline)
        
        # Verify and update best
        if verify_edges(result_edges) and len(result_edges) > best_count:
            best_count = len(result_edges)
            best_edges = [[u, v] for u, v in result_edges]
        
        # Check time
        if time.time() > deadline - 5:
            break
    
    # Final polish: try to add any remaining valid edges to the best solution
    if best_edges:
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        common_mask = [0] * n
        polish_edges = []
        edge_set = set()
        for u, v in best_edges:
            add_edge(u, v, adj_mask, common, common_mask, polish_edges, edge_set)
        
        # Try to add more edges - multiple passes for better convergence
        for _ in range(10):
            pairs = all_pairs[:]
            random.shuffle(pairs)
            added_any = False
            for u, v in pairs:
                if is_valid_add(u, v, adj_mask, common, common_mask):
                    add_edge(u, v, adj_mask, common, common_mask, polish_edges, edge_set)
                    added_any = True
            if not added_any:
                break
        
        if verify_edges(polish_edges) and len(polish_edges) > best_count:
            best_count = len(polish_edges)
            best_edges = [[u, v] for u, v in polish_edges]

    # Fallback if no edges found
    if not best_edges:
        # Try a simple construction: 5 disjoint K4s: each K4 has 6 edges, total 30 edges
        # K4 is K2,3-free since max common neighbors in K4 is 2
        best_edges = []
        for i in range(5):
            base = i * 4
            for u in range(4):
                for v in range(u + 1, 4):
                    best_edges.append([base + u, base + v])
    
    return best_edges

if __name__ == "__main__":
    random.seed(42)
    edges = construct()
    print(f"Found {len(edges)} edges")
