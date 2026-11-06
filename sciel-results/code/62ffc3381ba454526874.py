import random
import time
import math

def construct():
    n = 40
    # Adjacency list representation
    adj = [set() for _ in range(n)]
    
    def creates_c6(u, v):
        # Check if adding edge (u, v) creates a C6.
        # Path: u - a - b - c - d - v
        # We need a path of length 5 between u and v in G - {(u,v)}.
        # u - a - b - c - d - v
        # a in N(u), b in N(a), c in N(b), d in N(c) intersect N(v)
        # d not in {u, a, b, c, v}
        # c not in {u, a, b, v}
        # b not in {u, a, v}
        # a not in {u, v}
        
        for a in adj[u]:
            if a == v: continue
            for b in adj[a]:
                if b in (u, v): continue
                for c in adj[b]:
                    if c in (u, a, v): continue
                    # d must be in N(c) and N(v)
                    # d must not be u, a, b, c, v
                    candidates = adj[c] & adj[v]
                    if candidates - {u, a, b, c, v}:
                        return True
        return False

    def try_add_edge(u, v):
        if v in adj[u]:
            return False
        if creates_c6(u, v):
            return False
        adj[u].add(v)
        adj[v].add(u)
        return True

    def remove_edge(u, v):
        if v in adj[u]:
            adj[u].remove(v)
            adj[v].remove(u)
            return True
        return False

    def count_edges():
        return sum(len(s) for s in adj) // 2

    def get_missing_edges():
        return [(u, v) for u in range(n) for v in range(u+1, n) if v not in adj[u]]

    def get_existing_edges():
        return [(u, v) for u in range(n) for v in adj[u] if u < v]

    start_time = time.time()
    limit = 115.0
    
    best_edges = 0
    best_adj_copy = None
    best_edges_list = []

    # Simulated Annealing Parameters
    T_start = 5.0
    T_end = 0.01
    cooling_rate = 0.995
    max_iterations = 500000 # Safety cap
    
    # Restart SA multiple times if time permits
    restarts = 0
    max_restarts = 10
    
    while restarts < max_restarts and time.time() - start_time < limit:
        restarts += 1
        
        # Reset graph
        adj = [set() for _ in range(n)]
        
        # Initialization: Random graph with average degree ~5
        # Try to add edges randomly until we hit a high density or no more valid edges
        # We start with a sparse graph and add edges greedily/randomly
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_pairs)
        for u, v in all_pairs:
            if try_add_edge(u, v):
                pass # Just add if possible
        
        current_edges = count_edges()
        if current_edges > best_edges:
            best_edges = current_edges
            best_adj_copy = [set(s) for s in adj]
            best_edges_list = [[u, v] for u in range(n) for v in adj[u] if u < v]

        T = T_start
        iter_count = 0
        
        while T > T_end and time.time() - start_time < limit - 1:
            iter_count += 1
            if iter_count > max_iterations:
                break
            
            # Generate a move
            # 50% chance: Add an edge
            # 50% chance: Remove an edge
            # This simple 1-move neighborhood is slow. 
            # Let's use a "Swap" move: Remove one, try to add two.
            # Or simply: Pick a random edge to remove, then try to add random missing edges.
            
            existing = get_existing_edges()
            missing = get_missing_edges()
            
            if not existing or not missing:
                break
            
            # Move: Remove a random edge (u1, v1)
            idx = random.randint(0, len(existing) - 1)
            u1, v1 = existing[idx]
            remove_edge(u1, v1)
            
            # Try to add 1 or 2 edges from missing
            # We want to maintain or increase edge count.
            # If we remove 1, we should try to add 2 to net +1, or 1 to net 0.
            # Let's try to add 2.
            added = 0
            added_edges = []
            
            # Sample a few missing edges to try
            sample_size = min(5, len(missing))
            candidates = random.sample(missing, sample_size)
            
            for u2, v2 in candidates:
                if try_add_edge(u2, v2):
                    added += 1
                    added_edges.append((u2, v2))
                    if added == 2:
                        break
            
            delta_edges = added - 1 # Removed 1, added 'added'
            
            # Acceptance Criterion
            if delta_edges >= 0:
                # Always accept if it improves or maintains edge count
                pass
            else:
                # delta_edges is -1 (if added 0)
                # Accept with probability exp(-delta / T) ? 
                # Energy E = -edges. Delta E = -delta_edges.
                # If delta_edges = -1, Delta E = 1.
                # Prob = exp(-1 / T)
                prob = math.exp(-1.0 / T)
                if random.random() > prob:
                    # Reject: Revert move
                    # Re-add removed edge
                    adj[u1].add(v1)
                    adj[v1].add(u1)
                    # Remove added edges
                    for u2, v2 in added_edges:
                        remove_edge(u2, v2)
            
            # Update best
            current_edges = count_edges()
            if current_edges > best_edges:
                best_edges = current_edges
                best_adj_copy = [set(s) for s in adj]
                best_edges_list = [[u, v] for u in range(n) for v in adj[u] if u < v]
                
            # Cool down
            T *= cooling_rate

    return best_edges_list
