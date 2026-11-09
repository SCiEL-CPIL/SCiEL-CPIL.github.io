"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START

N = 20
DEADLINE = 115.0  # seconds, safety margin before 120s kill

def _can_add_edge(adj, u, v):
    """
    Check if adding edge (u,v) to a K3,3-free graph keeps it K3,3-free.
    
    A K3,3 forms if there exist sets A={a1,a2,a3} and B={b1,b2,b3}
    such that all 9 edges between A and B exist.
    
    When we add edge (u,v), the only new K3,3 must use this edge.
    So either u is in A and v in B, or u in B and v in A.
    
    Case 1: u in A, v in B.
    We need 2 more vertices a2,a3 in A (neighbors of v, not u)
    and 2 more vertices b2,b3 in B (neighbors of u, not v)
    such that all cross-edges exist: a2-a' in B, a3-a' in B, b2-b' in A, b3-b' in A.
    More precisely: for the K3,3 with A={u,a2,a3}, B={v,b2,b3}:
      - u-v exists (we're adding it)
      - u-b2, u-b3 must exist (u adjacent to b2,b3)
      - a2-v, a3-v must exist (a2,a3 adjacent to v)
      - a2-b2, a2-b3, a3-b2, a3-b3 must exist
    
    So: pick b2,b3 from N(u) \ {v}, pick a2,a3 from N(v) \ {u},
    check all 4 cross edges a_i - b_j exist.
    
    Case 2: symmetric (u in B, v in A) - same check with roles swapped,
    but since we check both (u in A, v in B) and (u in B, v in A),
    and the edge is undirected, we just need to check both orientations.
    Actually for undirected, case 1 with (u,v) and case 2 with (u,v) are the same
    as case 1 with (v,u). So we only need one check.
    """
    # Get neighbors of u and v (excluding each other since edge not yet added)
    nu = adj[u]
    nv = adj[v]
    
    # Candidates for B-side (neighbors of u, excluding v)
    b_cands = [w for w in nu if w != v]
    # Candidates for A-side (neighbors of v, excluding u)
    a_cands = [w for w in nv if w != u]
    
    if len(b_cands) < 2 or len(a_cands) < 2:
        return True
    
    # Check all pairs
    for i in range(len(a_cands)):
        a = a_cands[i]
        for j in range(len(b_cands)):
            b = b_cands[j]
            if j <= i:
                continue
            # Check cross edges: a-b2, a-b3, a2-b, a3-b
            # Here a is a2, b is b2. We need a adjacent to b (and to other b's),
            # and b adjacent to a (and to other a's).
            # Actually we need: a-b exists (this is one cross edge)
            # But we also need a to be adjacent to all b's in B, and b adjacent to all a's in A.
            # For the specific K3,3: A={u,a,a'}, B={v,b,b'}
            # Cross edges needed: a-b, a-b', a'-b, a'-b'
            # We're checking a specific pair (a from a_cands, b from b_cands)
            # but we need TWO a's and TWO b's.
            pass
    
    # More efficient: for each pair of a_cands and pair of b_cands, check all 4 cross edges
    la = len(a_cands)
    lb = len(b_cands)
    for i1 in range(la):
        a1 = a_cands[i1]
        for i2 in range(i1+1, la):
            a2 = a_cands[i2]
            for j1 in range(lb):
                b1 = b_cands[j1]
                for j2 in range(j1+1, lb):
                    b2 = b_cands[j2]
                    # Check 4 cross edges
                    if (adj[a1][b1] and adj[a1][b2] and 
                        adj[a2][b1] and adj[a2][b2]):
                        return False
    return True


def _can_add_edge_fast(adj, u, v):
    """Faster version using bitmasks."""
    # adj is a list of int bitmasks
    nu = adj[u]
    nv = adj[v]
    
    # b_cands: neighbors of u, not v
    b_mask = nu & ~(1 << v)
    # a_cands: neighbors of v, not u
    a_mask = nv & ~(1 << u)
    
    # Need at least 2 in each
    if bin(b_mask).count('1') < 2 or bin(a_mask).count('1') < 2:
        return True
    
    # For each pair of a's, check if they share 2+ common neighbors in b_mask
    # that are also... wait, let me think again.
    
    # We need: exist a1,a2 in a_cands, b1,b2 in b_cands
    # such that a1-b1, a1-b2, a2-b1, a2-b2 all exist.
    
    # For a fixed pair (a1,a2), the valid b's must be in b_mask AND in N(a1) AND in N(a2).
    # So count = popcount(b_mask & adj[a1] & adj[a2]). If >= 2, then K3,3 exists.
    
    a_list = [w for w in range(N) if (a_mask >> w) & 1]
    la = len(a_list)
    for i in range(la):
        a1 = a_list[i]
        for j in range(i+1, la):
            a2 = a_list[j]
            common = b_mask & adj[a1] & adj[a2]
            if bin(common).count('1') >= 2:
                return False
    return True


def _greedy_build(start_edges, time_limit):
    """
    Start from a set of edges, greedily add edges that don't create K3,3.
    adj is a list of int bitmasks.
    Returns (adj, edge_count).
    """
    adj = [0] * N
    for u, v in start_edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Build list of candidate edges not yet present
    all_edges = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    # Shuffle for diversity
    random.shuffle(all_edges)
    
    start_time = time.time()
    added = 0
    for u, v in all_edges:
        if time.time() - start_time > time_limit:
            break
        if (adj[u] >> v) & 1:
            continue
        if _can_add_edge_fast(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            added += 1
    
    return adj, added


def _local_search(adj, time_limit):
    """
    Local search with Add and 1-2 Swap (Remove 1, Add 2) moves.
    """
    start_time = time.time()
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        
        # Phase 1: Try adding all missing edges
        for u in range(N):
            for v in range(u+1, N):
                if time.time() - start_time > time_limit:
                    return adj
                if (adj[u] >> v) & 1:
                    continue
                if _can_add_edge_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        
        # Phase 2: Try 1-2 Swaps (Remove edge e, try to add 2 new edges)
        # We iterate over existing edges
        for u in range(N):
            for v in range(u+1, N):
                if time.time() - start_time > time_limit:
                    return adj
                if not ((adj[u] >> v) & 1):
                    continue
                
                # Save state
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                
                added_count = 0
                new_edges = []
                
                # Try to add all possible missing edges
                for u2 in range(N):
                    for v2 in range(u2+1, N):
                        if time.time() - start_time > time_limit:
                            break
                        if (adj[u2] >> v2) & 1:
                            continue
                        if _can_add_edge_fast(adj, u2, v2):
                            adj[u2] |= (1 << v2)
                            adj[v2] |= (1 << u2)
                            added_count += 1
                            new_edges.append((u2, v2))
                
                if time.time() - start_time > time_limit:
                    break
                    
                # If we added more than 1 edge, it's a net gain
                if added_count > 1:
                    improved = True
                    # Keep the new state (don't restore u-v, keep new edges)
                    # Note: We don't restore u-v because we want to keep the gain
                    # But wait, if we break out of loops, we need to be careful.
                    # Let's just mark improved and continue.
                else:
                    # Restore original state
                    # Remove newly added edges
                    for eu, ev in new_edges:
                        adj[eu] &= ~(1 << ev)
                        adj[ev] &= ~(1 << eu)
                    # Restore u-v
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    
    return adj


def _random_start(density):
    """Generate a random graph with given edge probability."""
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if random.random() < density:
                edges.append((u, v))
    return edges


def _k218_start():
    """K2,18: vertices 0,1 connected to 2..19."""
    edges = [(0, v) for v in range(2, N)] + [(1, v) for v in range(2, N)]
    return edges


def _valid_bipartite_start(n1, n2):
    """
    Generates a bipartite graph K(n1, n2) that is K3,3-free.
    This is only possible if n1 < 3 or n2 < 3.
    If not, we return a sparse valid graph instead.
    """
    if n1 < 3 or n2 < 3:
        return [(u, v) for u in range(n1) for v in range(n1, n1+n2)]
    else:
        # Fallback: Just a random sparse graph or empty
        return []


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    best_adj = [0] * N
    best_count = 0
    
    # Strategy 1: K2,18 base + greedy
    edges = _k218_start()
    adj, added = _greedy_build(edges, 10.0)
    count = sum(bin(adj[u]).count('1') for u in range(N)) // 2
    if count > best_count:
        best_adj = adj[:]
        best_count = count
    
    if time.time() > deadline - 10:
        return _adj_to_edges(best_adj)
    
    # Strategy 2: Random starts with various densities
    for density in [0.3, 0.5, 0.7]:
        if time.time() > deadline - 15:
            break
        edges = _random_start(density)
        # Filter: only keep edges that don't create K3,3 when added in order
        adj = [0] * N
        valid_edges = []
        for u, v in edges:
            if (adj[u] >> v) & 1:
                continue
            if _can_add_edge_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                valid_edges.append((u, v))
        
        # Now greedy fill
        adj, added = _greedy_build(valid_edges, 5.0)
        count = sum(bin(adj[u]).count('1') for u in range(N)) // 2
        if count > best_count:
            best_adj = adj[:]
            best_count = count
    
    if time.time() > deadline - 10:
        return _adj_to_edges(best_adj)
    
    # Strategy 3: Valid sparse bipartite starts (n1 < 3 or n2 < 3)
    # K(2, 18) is already covered by Strategy 1.
    # Let's try K(1, 19) or K(2, 17) with different partitions?
    # Actually, random starts are better. Let's add more random restarts.
    for _ in range(5):
        if time.time() > deadline - 15:
            break
        edges = _random_start(0.4)
        adj = [0] * N
        valid_edges = []
        for u, v in edges:
            if (adj[u] >> v) & 1:
                continue
            if _can_add_edge_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                valid_edges.append((u, v))
        
        adj, added = _greedy_build(valid_edges, 5.0)
        count = sum(bin(adj[u]).count('1') for u in range(N)) // 2
        if count > best_count:
            best_adj = adj[:]
            best_count = count
    
    # Strategy 4: Local search on best
    if time.time() < deadline - 5:
        adj = best_adj[:]
        remaining = deadline - time.time() - 2
        adj = _local_search(adj, remaining)
        count = sum(bin(adj[u]).count('1') for u in range(N)) // 2
        if count > best_count:
            best_adj = adj[:]
            best_count = count
    
    return _adj_to_edges(best_adj)


def _adj_to_edges(adj):
    """Convert adjacency bitmask list to list of [u,v] edges."""
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (adj[u] >> v) & 1:
                edges.append([u, v])
    return edges

# EVOLVE-BLOCK-END
