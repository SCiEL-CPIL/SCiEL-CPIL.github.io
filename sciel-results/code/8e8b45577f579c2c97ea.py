"""Search for a dense edge-critical diameter-2 graph on 16 vertices."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _bit(u, n=16):
    return 1 << u

def _diameter_le_2(adj, n=16):
    for u in range(n):
        reachable = adj[u]
        for w in range(n):
            if w == u:
                continue
            if (reachable & bit(w)) and (adj[u] & bit(w)):
                continue  # direct edge, distance 1
            if reachable & bit(w):
                continue  # distance 2 via some neighbor of u
            # check if there exists v with v~u and v~w
            # reachable already includes all v such that v~u
            # We need to check if any such v is also adjacent to w
            # adj[v] & bit(w) for some v in neighbors of u
            # Let's check: for each v in neighbors of u, is v adjacent to w?
            # That's: OR over v in N(u) of (adj[v] & bit(w))
            # Which is: (sum over v in N(u) of adj[v]) & bit(w)
            # We can precompute or just iterate
            found = False
            v = adj[u]
            while v:
                lsb = v & (-v)
                idx = lsb.bit_length() - 1
                if adj[idx] & bit(w):
                    found = True
                    break
                v ^= lsb
            if not found:
                return False
    return True

def _edge_critical(adj, edge, n=16):
    u, v = edge
    # Remove edge (u,v)
    new_adj = adj[:]
    new_adj[u] &= ~bit(v)
    new_adj[v] &= ~bit(u)
    # Check diameter <= 2
    for s in range(n):
        if s == u or s == v:
            # These are the most likely to be affected
            pass
        for t in range(n):
            if s == t:
                continue
            # BFS from s to t in the new graph
            # Check distance 1
            if new_adj[s] & bit(t):
                continue
            # Check distance 2
            found = False
            nb = new_adj[s]
            while nb:
                lsb = nb & (-nb)
                idx = lsb.bit_length() - 1
                if new_adj[idx] & bit(t):
                    found = True
                    break
                nb ^= lsb
            if not found:
                return False
    return True

def _all_edges_critical(adj, edges, n=16):
    for edge in edges:
        if not _edge_critical(adj, edge, n):
            return False
    return True

def _score_and_check(adj, edges, n=16):
    if not _diameter_le_2(adj, n):
        return 0
    if not _all_edges_critical(adj, edges, n):
        return 0
    return len(edges)

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin
    
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Start with K_{8,8} as baseline: 64 edges
    best_edges = [(u, v) for u in range(8) for v in range(8, 16)]
    best_adj = [0] * n
    for u, v in best_edges:
        best_adj[u] |= bit(v)
        best_adj[v] |= bit(u)
    best_score = _score_and_check(best_adj, best_edges, n)
    
    if best_score == 0:
        # K_{8,8} is not valid (shouldn't happen, but safety)
        # This means every edge must be critical, so K_{8,8} fails
        pass
    
    # Try to improve: start from dense random graphs and remove non-critical edges
    random.seed(42)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        
        # Random dense graph
        p = 0.8
        edges = []
        adj = [0] * n
        for u, v in all_edges:
            if random.random() < p:
                edges.append((u, v))
                adj[u] |= bit(v)
                adj[v] |= bit(u)
        
        # Iterate: remove non-critical edges until all are critical or graph disconnected
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Check diameter first (cheap-ish)
            if not _diameter_le_2(adj, n):
                break
            
            # Find a non-critical edge
            non_critical = -1
            for i, (u, v) in enumerate(edges):
                # Quick check: if u and v have a common neighbor, edge is non-critical
                # because removing it still allows u-v path via common neighbor
                # Actually we need to check if diameter stays <= 2
                # Common neighbor implies distance 2 path exists, so edge is NOT critical
                common = adj[u] & adj[v]
                # Remove u and v themselves from common if present
                common &= ~bit(u)
                common &= ~bit(v)
                if common:
                    # There's a common neighbor, so edge (u,v) is not critical
                    # (removing it still gives distance 2 via common neighbor)
                    non_critical = i
                    break
            
            if non_critical == -1:
                # All edges are critical
                break
            
            # Remove this edge
            u, v = edges[non_critical]
            edges.pop(non_critical)
            adj[u] &= ~bit(v)
            adj[v] &= ~bit(u)
        
        if time.time() > deadline:
            break
        
        sc = _score_and_check(adj, edges, n)
        if sc > best_score:
            best_score = sc
            best_edges = edges[:]
            best_adj = adj[:]
            print(f"New best: {sc} edges at trial {trial}")
    
    return [list(e) for e in best_edges]
# EVOLVE-BLOCK-END
