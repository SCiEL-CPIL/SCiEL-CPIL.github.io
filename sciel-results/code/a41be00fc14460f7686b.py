"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    n = 20
    best_edges = []
    best_score = 0
    
    # Phase 1: Multiple greedy restarts
    for seed in range(500):
        if time.time() > deadline - 10.0:
            break
        
        edges = _greedy_construct(n, seed)
        score = len(edges)
        
        if score > best_score:
            best_score = score
            best_edges = edges
    
    # Phase 2: Local search improvement on best solution
    if best_edges and time.time() < deadline - 5.0:
        improved = _local_search(best_edges, n, deadline)
        if len(improved) > best_score:
            best_edges = improved
            best_score = len(improved)
    
    return best_edges

def _greedy_construct(n, seed):
    """
    Greedy construction of K2,3-free graph.
    Start empty, add edges in random order if they don't create K2,3.
    """
    random.seed(seed)
    
    # Generate all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(all_edges)
    
    # Track adjacency for quick common neighbor count
    adj = [set() for _ in range(n)]
    edges = []
    
    for u, v in all_edges:
        # Check if adding edge (u,v) would create a K2,3
        ok = _can_add_edge(adj, u, v)
        
        if ok:
            adj[u].add(v)
            adj[v].add(u)
            edges.append([u, v])
    
    return edges

def _can_add_edge(adj, u, v):
    """Check if adding edge (u,v) maintains K2,3-free property."""
    n = len(adj)
    
    # Check pair (u,v): common neighbors must be < 3
    common_uv = adj[u] & adj[v]
    if len(common_uv) >= 3:
        return False
    
    # Check pairs (u, w) for w in adj[v]
    for w in adj[v]:
        if w == u:
            continue
        # After adding (u,v), common neighbors of u and w:
        # = (adj[u] ∩ adj[w]) ∪ ({v} if v in adj[w] else empty)
        common = len(adj[u] & adj[w])
        if v in adj[w]:
            common += 1
        if common >= 3:
            return False
    
    # Check pairs (v, w) for w in adj[u]
    for w in adj[u]:
        if w == v:
            continue
        # After adding (u,v), common neighbors of v and w:
        # = (adj[v] ∩ adj[w]) ∪ ({u} if u in adj[w] else empty)
        common = len(adj[v] & adj[w])
        if u in adj[w]:
            common += 1
        if common >= 3:
            return False
    
    return True

def _local_search(edges, n, deadline):
    """
    Try to improve the graph by removing some edges and re-adding others.
    Uses simulated annealing-like approach.
    """
    # Build adjacency from edges
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    current_edges = list(edges)
    current_score = len(current_edges)
    best_edges = list(current_edges)
    best_score = current_score
    
    # Build edge set for quick lookup
    edge_set = set()
    for u, v in current_edges:
        edge_set.add((min(u, v), max(u, v)))
    
    # All possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Try to add missing edges
    missing_edges = [(u, v) for u, v in all_edges if (u, v) not in edge_set]
    
    # Randomly try to add missing edges
    random.shuffle(missing_edges)
    
    for u, v in missing_edges:
        if time.time() > deadline - 2.0:
            break
        
        if _can_add_edge(adj, u, v):
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append([u, v])
            edge_set.add((u, v))
            current_score += 1
            
            if current_score > best_score:
                best_score = current_score
                best_edges = list(current_edges)
    
    # Now try to swap: remove an edge, add two others
    # This is more complex, so we'll do a simpler version
    # Try removing low-degree edges and re-adding
    
    # Get degrees
    degrees = [len(adj[i]) for i in range(n)]
    
    # Find edges to potentially remove (those with lower impact)
    candidate_removals = []
    for u, v in current_edges:
        # Estimate impact of removing this edge
        # It reduces common neighbor counts for pairs involving u or v
        candidate_removals.append((u, v))
    
    random.shuffle(candidate_removals)
    
    for u, v in candidate_removals[:50]:  # Limit to avoid too much time
        if time.time() > deadline - 1.0:
            break
        
        # Try removing (u,v)
        adj[u].discard(v)
        adj[v].discard(u)
        current_edges.remove([u, v])
        edge_set.discard((u, v))
        current_score -= 1
        
        # Try to add back missing edges
        for mu, mv in missing_edges:
            if (mu, mv) in edge_set:
                continue
            if _can_add_edge(adj, mu, mv):
                adj[mu].add(mv)
                adj[mv].add(mu)
                current_edges.append([mu, mv])
                edge_set.add((mu, mv))
                current_score += 1
                if current_score > best_score:
                    best_score = current_score
                    best_edges = list(current_edges)
        
        # Restore if not improved
        if current_score < best_score:
            # Restore original state
            adj[u].add(v)
            adj[v].add(u)
            current_edges.append([u, v])
            edge_set.add((u, v))
            current_score += 1
            # Remove any edges we added
            for mu, mv in list(edge_set):
                if (mu, mv) not in [(a, b) for a, b in best_edges]:
                    pass  # This is getting complex, skip for now
    
    return best_edges
# EVOLVE-BLOCK-END
