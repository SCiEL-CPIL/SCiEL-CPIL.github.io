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
    """
    Returns the edges of K_{8,8}, which is the maximum edge count (64) for a 
    triangle-free graph on 16 vertices. Since any triangle makes its edges 
    non-essential, the graph must be triangle-free. K_{8,8} has diameter 2 
    and is triangle-free, so all its edges are essential.
    """
    edges = []
    for u in range(8):
        for v in range(8, 16):
            edges.append([u, v])
    return edges
# EVOLVE-BLOCK-END
