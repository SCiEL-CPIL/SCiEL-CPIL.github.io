"""Q3-free graph on 16 vertices - maximize edges via bipartite local search."""
import time
import random
from itertools import combinations

DEADLINE = 140.0  # safety margin before 150s kill

def _has_q3(edges_set, adj):
    """Check if graph contains Q3 as a subgraph.
    Q3 is bipartite with parts of size 4, all 16 cross-edges present.
    We check: for any 4-subset S of vertices, if there exist 4 other vertices
    T such that every vertex in S is adjacent to every vertex in T, AND
    the induced structure matches Q3 (no edges within S, no edges within T,
    and the bipartite graph between S and T is exactly K4,4).
    
    Actually, Q3 as a subgraph means: there exist 8 vertices that induce
    a subgraph containing Q3. Since Q3 is bipartite (4+4) with 12 edges,
    we need 4 vertices in one part, 4 in the other, all 16 cross-edges,
    and no edges within either part (for the induced subgraph to be exactly Q3,
    but as a subgraph we just need the 12 edges of Q3 to be present).
    
    Wait - subgraph means we need the 12 edges of Q3 to exist, not necessarily
    that no other edges exist. So we need: 4 vertices A, 4 vertices B,
    all 16 edges between A and B present. That's it for the edge set.
    The internal edges of A and B in Q3 are 0, so we don't need to check those.
    
    So: Q3 exists iff there exist disjoint 4-sets A, B such that
    every a in A is adjacent to every b in B.
    """
    n = 16
    for A in combinations(range(n), 4):
        # Find all vertices adjacent to all of A
        common = (1 << n) - 1
        for a in A:
            common &= adj[a]
        # Remove A itself
        common &= ~((1 << A[0]) | (1 << A[1]) | (1 << A[2]) | (1 << A[3]))
        # Check if common contains a 4-set
        # Count bits
        if common.bit_count() >= 4:
            # Check if there's a 4-subset of common
            # This is sufficient - if 4+ vertices all connect to all of A,
            # pick any 4 of them
            return True
    return False

def _has_q3_fast(adj):
    """Faster check: for each 4-set A, compute common neighbors, check >= 4 outside A."""
    n = 16
    mask_all = (1 << n) - 1
    for A in combinations(range(n), 4):
        common = mask_all
        for a in A:
            common &= adj[a]
        # Remove A vertices
        a_mask = (1 << A[0]) | (1 << A[1]) | (1 << A[2]) | (1 << A[3])
        common &= ~a_mask
        if common.bit_count() >= 4:
            return True
    return False

def _build_adj(edges):
    """Build adjacency bitmask list from edge list."""
    n = 16
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _edge_count_from_adj(adj):
    total = 0
    for i in range(16):
        total += adj[i].bit_count()
    return total // 2

def construct():
    start_time = time.time()
    random.seed(42)
    
    n = 16
    all_possible = [[i, j] for i in range(n) for j in range(i+1, n)]
    
    # Start with K3,13 (39 edges) as baseline
    best_edges = [[a, b] for a in range(3) for b in range(3, 16)]
    best_adj = _build_adj(best_edges)
    best_count = 39
    
    def is_valid(adj):
        return not _has_q3_fast(adj)
    
    # Verify baseline
    if not is_valid(best_adj):
        # Should not happen, but fallback
        return best_edges
    
    # Strategy: Start with dense graph, remove edges to break Q3s,
    # then try to add edges back.
    
    # Phase 1: Start with complete graph, remove edges to make Q3-free
    # This is hard. Instead, use local search.
    
    # Better approach: Start with a known Q3-free structure and add edges.
    # K3,13 is Q3-free. Try adding edges within the 13-set or within the 3-set.
    
    # Actually, let's do a more systematic search:
    # 1. Try all bipartitions (A, B) of {0..15} into two 8-sets
    # 2. For each, find max edges in bipartite graph avoiding K4,4
    # 3. Add intra-part edges carefully
    
    # Simpler: local search on edge set
    # Start with K3,13, try to add each missing edge if it doesn't create Q3
    
    current_edges = set((min(u,v), max(u,v)) for u, v in best_edges)
    current_adj = _build_adj(best_edges)
    
    missing = [e for e in all_possible if (min(e[0],e[1]), max(e[0],e[1])) not in current_edges]
    
    # Greedy add
    for e in missing:
        if time.time() - start_time > DEADLINE:
            break
        u, v = e
        # Try adding
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        if not _has_q3_fast(current_adj):
            current_edges.add((u, v))
        else:
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
    
    current_count = len(current_edges)
    if current_count > best_count:
        best_count = current_count
        best_adj = current_adj[:]
        best_edges = [[u, v] for u, v in current_edges]
    
    # Phase 2: Local search - try swapping edges
    # Also try different initial structures
    
    # Try: complete bipartite K8,8 minus edges to avoid K4,4
    # K8,8 has 64 edges. Need to remove edges so no K4,4 exists.
    # This is equivalent to: the bipartite graph between A and B has no K4,4.
    # By Zarankiewicz, z(8,8;4,4) is the max edges in 8x8 bipartite graph with no K4,4.
    # z(8,8;4,4) = ? Known: z(8,8;4,4) <= 40 (by Kővári–Sós–Turán or exact values)
    # Actually, let's just search.
    
    # Let's do a more thorough local search with restarts
    
    def local_search(initial_edges, max_iters=50000):
        edges = set((min(u,v), max(u,v)) for u, v in initial_edges)
        adj = _build_adj(list(edges))
        count = len(edges)
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            if it % 1000 == 0:
                pass  # could check deadline
            
            # Try adding a random missing edge
            if random.random() < 0.5:
                # Try to add an edge
                all_e = all_possible
                e = random.choice(all_e)
                key = (min(e[0],e[1]), max(e[0],e[1]))
                if key in edges:
                    continue
                u, v = key
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if not _has_q3_fast(adj):
                    edges.add(key)
                    count += 1
                else:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            else:
                # Try removing an edge and adding two others (or just remove to allow future adds)
                if not edges:
                    continue
                e = random.choice(list(edges))
                u, v = e
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                edges.remove(e)
                count -= 1
                # Try to add back a different edge
                for _ in range(3):
                    e2 = random.choice(all_possible)
                    key2 = (min(e2[0],e2[1]), max(e2[0],e2[1]))
                    if key2 in edges:
                        continue
                    u2, v2 = key2
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    if not _has_q3_fast(adj):
                        edges.add(key2)
                        count += 1
                        break
                    else:
                        adj[u2] ^= (1 << v2)
                        adj[v2] ^= (1 << u2)
        
        return edges, adj, count
    
    # Run multiple local searches from different starts
    starts = [
        best_edges,  # current best
        # K4,12
        [[a, b] for a in range(4) for b in range(4, 16)],
        # K5,11
        [[a, b] for a in range(5) for b in range(5, 16)],
        # Random dense starts
    ]
    
    # Add random starts
    for _ in range(5):
        if time.time() - start_time > DEADLINE:
            break
        random_start = []
        for e in all_possible:
            if random.random() < 0.4:
                random_start.append(e)
        starts.append(random_start)
    
    for init in starts:
        if time.time() - start_time > DEADLINE:
            break
        edges, adj, count = local_search(init, max_iters=30000)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            best_edges = [[u, v] for u, v in edges]
    
    # Final greedy add on best
    current_edges = set((min(u,v), max(u,v)) for u, v in best_edges)
    current_adj = _build_adj(best_edges)
    for e in all_possible:
        if time.time() - start_time > DEADLINE:
            break
        key = (min(e[0],e[1]), max(e[0],e[1]))
        if key in current_edges:
            continue
        u, v = key
        current_adj[u] |= (1 << v)
        current_adj[v] |= (1 << u)
        if not _has_q3_fast(current_adj):
            current_edges.add(key)
        else:
            current_adj[u] ^= (1 << v)
            current_adj[v] ^= (1 << u)
    
    final_count = len(current_edges)
    if final_count > best_count:
        best_count = final_count
        best_edges = [[u, v] for u, v in current_edges]
    
    return best_edges

# Quick self-test
if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Edges: {len(edges)}, Time: {t1-t0:.2f}s")
    adj = _build_adj(edges)
    print(f"Q3-free: {not _has_q3_fast(adj)}")
