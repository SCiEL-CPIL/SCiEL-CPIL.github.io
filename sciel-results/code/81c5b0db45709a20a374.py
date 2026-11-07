"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

N = 16
DEADLINE = 80  # safety margin before 90s kill

def _build_adj(edges):
    """Build adjacency list from edge list."""
    n = N
    adj = [set() for _ in range(n)]
    for e in edges:
        u, v = e[0], e[1]
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _has_diameter2(adj):
    """Check if graph has diameter exactly 2 (all pairs dist <= 2, at least one pair dist 2)."""
    n = N
    has_dist2 = False
    for a in range(n):
        for b in range(a + 1, n):
            if b in adj[a]:
                continue
            found = False
            for c in adj[a]:
                if c != b and b in adj[c]:
                    found = True
                    break
            if not found:
                return False
            has_dist2 = True
    return has_dist2

def _all_edges_essential(adj):
    """Check if every edge is essential (removing it increases diameter above 2 or disconnects)."""
    n = N
    eset = set()
    for u in range(n):
        for v in adj[u]:
            if v > u:
                eset.add((u, v))
    
    for (u, v) in eset:
        # Remove edge (u, v)
        adj2 = [set(s) for s in adj]
        adj2[u].discard(v)
        adj2[v].discard(u)
        
        # Check if diameter is still <= 2
        for a in range(n):
            for b in range(a + 1, n):
                if b in adj2[a]:
                    continue
                found = False
                for c in adj2[a]:
                    if b in adj2[c]:
                        found = True
                        break
                if not found:
                    # This edge was essential
                    break
            else:
                continue
            break
        else:
            # All pairs still have distance <= 2, so this edge was NOT essential
            return False
    return True

def _validate_full(adj):
    """Full validation: diameter exactly 2 and all edges essential."""
    if not _has_diameter2(adj):
        return False
    if not _all_edges_essential(adj):
        return False
    return True

def _edges_from_adj(adj):
    """Convert adjacency list to edge list."""
    n = N
    edges = []
    for u in range(n):
        for v in adj[u]:
            if v > u:
                edges.append([u, v])
    return edges

def construct():
    start = time.time()
    best_edges = None
    best_count = -1
    
    def try_expand(adj):
        """Try to add edges while maintaining all-essential property."""
        n = N
        adj = [set(s) for s in adj]
        
        # Verify current state is valid
        if not _validate_full(adj):
            return None
        
        pairs = list(combinations(range(n), 2))
        random.shuffle(pairs)
        improved = True
        while improved:
            improved = False
            for (u, v) in pairs:
                if time.time() > DEADLINE:
                    return adj
                if v in adj[u]:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                if _validate_full(adj):
                    improved = True
                else:
                    adj[u].discard(v)
                    adj[v].discard(u)
        return adj
    
    # Strategy: Start from K_{8,8} (the current best) and try to expand
    # Also try random seeds for diversity
    
    # First, try expanding from K_{8,8}
    adj_k88 = [set() for _ in range(N)]
    for u in range(8):
        for v in range(8, 16):
            adj_k88[u].add(v)
            adj_k88[v].add(u)
    
    adj_exp = try_expand(adj_k88)
    if adj_exp is not None:
        edges = _edges_from_adj(adj_exp)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
    
    # Try random diameter-2 seeds
    attempts = 0
    while time.time() < DEADLINE - 5 and attempts < 50:
        attempts += 1
        adj_seed = [set() for _ in range(N)]
        
        # Create a random spanning tree
        order = list(range(N))
        random.shuffle(order)
        for i in range(1, N):
            u = order[i]
            v = random.choice(order[:i])
            adj_seed[u].add(v)
            adj_seed[v].add(u)
        
        # Add random edges to get diameter 2
        pairs = list(combinations(range(N), 2))
        random.shuffle(pairs)
        for (u, v) in pairs:
            if v not in adj_seed[u] and random.random() < 0.4:
                adj_seed[u].add(v)
                adj_seed[v].add(u)
        
        # Ensure diameter 2
        while not _has_diameter2(adj_seed):
            missing = [(u, v) for (u, v) in combinations(range(N), 2) if v not in adj_seed[u]]
            if not missing:
                break
            u, v = random.choice(missing)
            adj_seed[u].add(v)
            adj_seed[v].add(u)
        
        # Expand
        adj_exp = try_expand(adj_seed)
        if adj_exp is not None:
            edges = _edges_from_adj(adj_exp)
            if len(edges) > best_count:
                best_count = len(edges)
                best_edges = edges
    
    # Fallback: return K_{8,8} if nothing better found
    if best_edges is None:
        best_edges = _edges_from_adj(adj_k88)
    
    # Ensure types are correct
    result = []
    for e in best_edges:
        result.append([int(e[0]), int(e[1])])
    return result
# EVOLVE-BLOCK-END
