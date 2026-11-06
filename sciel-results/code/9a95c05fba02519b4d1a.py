"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a K2,3-free graph on 20 vertices maximizing edges.
    
    Approach:
    1. Start with a good initial graph (greedy or structured).
    2. Use local search / simulated annealing to improve.
    3. Use branch-and-bound for final optimization if time permits.
    
    A K2,3-free graph has no two vertices with 3+ common neighbors.
    Equivalently, for any pair of vertices {u,v}, |N(u) ∩ N(v)| ≤ 2.
    """
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline to be safe (150s limit)
    
    n = 20
    # All possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    total_possible = len(all_edges)  # 190
    
    # Represent graph as a set of edges
    # For efficiency, use a boolean matrix or set
    
    def is_k23_free(edge_set):
        """Check if a set of edges forms a K2,3-free graph."""
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        # Check all pairs of vertices
        for u in range(n):
            for v in range(u + 1, n):
                common = adj[u] & adj[v]
                if len(common) >= 3:
                    return False
        return True
    
    def count_common_neighbors(adj, u, v):
        """Count common neighbors of u and v."""
        return len(adj[u] & adj[v])
    
    def can_add_edge(adj, edge_set, u, v):
        """Check if adding edge (u,v) keeps the graph K2,3-free."""
        if (u, v) in edge_set or (v, u) in edge_set:
            return True  # already there
        
        # Adding edge (u,v) means u and v become neighbors.
        # This affects common neighbor counts for pairs involving u or v.
        # Specifically, for any w, the pair (u,w) now has v as a potential common neighbor
        # if w is adjacent to v. Similarly for (v,w) and u.
        
        # New common neighbors:
        # For pair (u, w): common = (N(u) ∪ {v}) ∩ (N(w) ∪ {u if w==u? no})
        # Actually, N(u) gains v, N(v) gains u.
        
        # Let's check: for any w != u, v:
        #   New common(u, w) = (N(u) ∪ {v}) ∩ (N(w)) 
        #   = (N(u) ∩ N(w)) ∪ ({v} ∩ N(w))
        #   So if v ∈ N(w), then v becomes a common neighbor of u and w.
        #   We need |new common(u,w)| ≤ 2.
        #   Currently |N(u) ∩ N(w)| ≤ 2. If v ∈ N(w), then new size = |N(u) ∩ N(w)| + 1 (since v ∉ N(u) before).
        #   So we need |N(u) ∩ N(w)| ≤ 1 for all w ∈ N(v).
        
        # Similarly, for w ∈ N(u), we need |N(v) ∩ N(w)| ≤ 1.
        
        # Also check pair (u, v) itself: common = N(u) ∩ N(v). Need ≤ 2.
        
        # Check pair (u, v):
        if count_common_neighbors(adj, u, v) >= 3:
            return False
        
        # Check for all w in N(v): need |N(u) ∩ N(w)| ≤ 1
        for w in adj[v]:
            if w == u:
                continue
            if count_common_neighbors(adj, u, w) >= 2:
                return False
        
        # Check for all w in N(u): need |N(v) ∩ N(w)| ≤ 1
        for w in adj[u]:
            if w == v:
                continue
            if count_common_neighbors(adj, v, w) >= 2:
                return False
        
        return True
    
    def add_edge(adj, edge_set, u, v):
        """Add edge (u,v) to the graph."""
        edge_set.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
    
    def remove_edge(adj, edge_set, u, v):
        """Remove edge (u,v) from the graph."""
        edge_set.discard((u, v))
        adj[u].discard(v)
        adj[v].discard(u)
    
    # Initialize with a greedy construction
    adj = [set() for _ in range(n)]
    edge_set = set()
    
    # Greedy: try to add edges in some order
    random.seed(42)
    edge_order = all_edges[:]
    random.shuffle(edge_order)
    
    for u, v in edge_order:
        if can_add_edge(adj, edge_set, u, v):
            add_edge(adj, edge_set, u, v)
    
    best_edge_set = set(edge_set)
    best_adj = [set(s) for s in adj]
    best_count = len(edge_set)
    
    # Local search: try to add edges, or swap edges
    def evaluate(edge_set):
        return len(edge_set)
    
    # Simulated annealing style local search
    current_adj = [set(s) for s in adj]
    current_edge_set = set(edge_set)
    current_count = len(edge_set)
    
    temperature = 10.0
    cooling = 0.999
    min_temperature = 0.01
    
    while time.time() < deadline - 10:
        if temperature < min_temperature:
            break
        
        # Try a random move: add an edge, remove an edge, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random edge
            u, v = random.choice(all_edges)
            if (u, v) not in current_edge_set and (v, u) not in current_edge_set:
                if can_add_edge(current_adj, current_edge_set, u, v):
                    add_edge(current_adj, current_edge_set, u, v)
                    new_count = len(current_edge_set)
                    if new_count > current_count:
                        current_count = new_count
                        if current_count > best_count:
                            best_count = current_count
                            best_edge_set = set(current_edge_set)
                            best_adj = [set(s) for s in current_adj]
                    # Always accept improvements, sometimes accept worse
                else:
                    # Can't add, try removing a random edge instead
                    pass
        elif move_type < 0.6:
            # Remove a random edge
            if current_edge_set:
                u, v = random.choice(list(current_edge_set))
                remove_edge(current_adj, current_edge_set, u, v)
                new_count = len(current_edge_set)
                # Accept with some probability
                if new_count >= current_count or random.random() < 0.1:
                    current_count = new_count
                else:
                    # Reject, add back
                    add_edge(current_adj, current_edge_set, u, v)
                    current_count = len(current_edge_set)
        else:
            # Swap: remove one edge, add another
            if current_edge_set:
                u1, v1 = random.choice(list(current_edge_set))
                remove_edge(current_adj, current_edge_set, u1, v1)
                
                # Try to add a random edge
                improved = False
                for _ in range(10):
                    u2, v2 = random.choice(all_edges)
                    if (u2, v2) not in current_edge_set and (v2, u2) not in current_edge_set:
                        if can_add_edge(current_adj, current_edge_set, u2, v2):
                            add_edge(current_adj, current_edge_set, u2, v2)
                            new_count = len(current_edge_set)
                            if new_count > current_count:
                                current_count = new_count
                                improved = True
                                if current_count > best_count:
                                    best_count = current_count
                                    best_edge_set = set(current_edge_set)
                                    best_adj = [set(s) for s in current_adj]
                            break
                
                if not improved:
                    # Try to put back the removed edge if possible
                    if can_add_edge(current_adj, current_edge_set, u1, v1):
                        add_edge(current_adj, current_edge_set, u1, v1)
                    else:
                        # Just accept the state
                        current_count = len(current_edge_set)
        
        temperature *= cooling
    
    # Final greedy pass: try to add any remaining edges
    for u, v in all_edges:
        if (u, v) not in best_edge_set and (v, u) not in best_edge_set:
            # Check if we can add it to the best graph
            temp_adj = [set(s) for s in best_adj]
            temp_set = set(best_edge_set)
            if can_add_edge(temp_adj, temp_set, u, v):
                add_edge(temp_adj, temp_set, u, v)
                best_edge_set = temp_set
                best_adj = temp_adj
                best_count = len(best_edge_set)
    
    # Convert to list of edges
    result = [[u, v] for u, v in best_edge_set]
    result.sort()
    
    # Verify the result is valid
    if not is_k23_free(set(best_edge_set)):
        # Fallback to a known valid construction
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges
    
    return result

# EVOLVE-BLOCK-END
