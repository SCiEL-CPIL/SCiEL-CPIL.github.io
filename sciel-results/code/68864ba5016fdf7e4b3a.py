import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin
    
    # Target: Maximize n such that C4-free and min_deg >= n - 39
    # Known: n=46 is achievable. n=47 is the upper bound to check.
    
    best_result = None
    best_n = 0
    
    # Try n=47 first, then 46
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        if time.time() > deadline:
            break
            
        min_deg_required = n - 39
        
        if min_deg_required < 0:
            min_deg_required = 0
            
        # Try to construct for this n
        found = try_construct_local_search(n, min_deg_required, deadline)
        if found:
            return found
            
    # Fallback
    return {'n': 39, 'edges': []}

def try_construct_local_search(n, min_deg_required, deadline):
    """
    Uses a local search strategy:
    1. Initialize with a random graph at target density.
    2. Iteratively remove C4s by edge deletion or swapping.
    3. Iteratively increase min degree by adding edges.
    """
    if min_deg_required <= 0:
        return {'n': n, 'edges': []}
    
    max_restarts = 500
    restart_count = 0
    
    while time.time() < deadline and restart_count < max_restarts:
        restart_count += 1
        
        # 1. Initialization: Random graph with expected degree ~ min_deg_required
        # To be safe, start slightly denser and prune, or sparser and grow?
        # Starting at target density is often best for local search.
        p = min_deg_required / (n - 1)
        adj = [[False] * n for _ in range(n)]
        deg = [0] * n
        
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    adj[i][j] = True
                    adj[j][i] = True
                    deg[i] += 1
                    deg[j] += 1
        
        # 2. Phase 1: Eliminate C4s
        # We use a "cooling" approach: repeatedly find C4s and break them.
        # Breaking strategy: 
        #   - Prefer swapping edges to maintain density.
        #   - Fallback to deletion.
        
        stuck_count = 0
        max_stuck = 2000
        
        while time.time() < deadline:
            # Find a C4
            c4 = find_c4(adj, n)
            
            if c4 is None:
                # No C4s left. Check degree.
                if min(deg) >= min_deg_required:
                    # Success
                    edges = []
                    for i in range(n):
                        for j in range(i + 1, n):
                            if adj[i][j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
                else:
                    # No C4s, but degree too low. 
                    # Try to add edges to increase min degree.
                    added = try_add_edges(adj, deg, n, min_deg_required)
                    if added:
                        stuck_count = 0
                        continue
                    else:
                        # Stuck with low degree and no C4s.
                        # This state is a local minimum for C4-free.
                        break
            
            # Found C4: [u, v, w, x]
            u, v, w, x = c4
            # Edges in C4: (u,v), (v,w), (w,x), (x,u)
            # Common neighbors of u and w are v and x.
            # Common neighbors of v and x are u and w.
            
            # Try to break by swapping an edge in the C4.
            # Example: Remove (u,v). Try to add (u, y) where y is not v,w,x and 
            # adding (u,y) doesn't create new C4s? 
            # Actually, simpler: Just remove one edge from the C4.
            # To improve on "random deletion", we can prioritize removing an edge 
            # between vertices with high degree, or try to "swap" if possible.
            
            # For now, let's use a smart deletion:
            # Remove edge (a,b) where deg[a] and deg[b] are highest in the C4.
            # This preserves the low-degree vertices' connectivity better.
            candidates = [(u, v), (v, w), (w, x), (x, u)]
            # Sort by degree sum descending
            candidates.sort(key=lambda e: deg[e[0]] + deg[e[1]], reverse=True)
            
            # Try to swap first?
            # Swap (u,v) with (u, z) or (v, z)?
            # If we remove (u,v), we lose degree for u and v.
            # If we can add (u, z) and (v, z') we maintain degree.
            # This is complex. Let's stick to deletion but be smart about which edge.
            
            a, b = candidates[0]
            
            # Check if we can remove it (degree constraint)
            if deg[a] > min_deg_required and deg[b] > min_deg_required:
                adj[a][b] = False
                adj[b][a] = False
                deg[a] -= 1
                deg[b] -= 1
                stuck_count = 0
            else:
                # Can't remove the best candidate. Try next.
                removed = False
                for a, b in candidates[1:]:
                    if deg[a] > min_deg_required and deg[b] > min_deg_required:
                        adj[a][b] = False
                        adj[b][a] = False
                        deg[a] -= 1
                        deg[b] -= 1
                        removed = True
                        stuck_count = 0
                        break
                if not removed:
                    # Cannot remove any edge in this C4 without violating min degree.
                    # This is a "hard" C4. 
                    # We must fail this restart or try a different C4.
                    # Since we picked a random C4 (effectively), if we can't break it,
                    # the graph is likely stuck in a local minimum where all C4s are "critical".
                    break
        
        # If we exited the loop without returning, this restart failed.
        # Check if we actually succeeded in the last iteration? 
        # The loop breaks if stuck or deadline.
        # We need to verify if the current state is valid.
        
        # Actually, the `return` inside the loop handles success.
        # If we reach here, this restart didn't yield a valid graph.
        continue
        
    return None

def find_c4(adj, n):
    """
    Find a C4 in the graph.
    Returns [u, v, w, x] such that u-v-w-x-u is a cycle, or None.
    Optimized: Check pairs of vertices for common neighbors.
    """
    # For small n (<=47), O(n^3) or O(n^4) is fine if optimized.
    # We can precompute common neighbors? 
    # Let's just iterate.
    
    # To speed up, we can check pairs (u, w) that are NOT adjacent? 
    # No, C4 is u-v-w-x-u. u and w are opposite. v and x are opposite.
    # If u and w are adjacent, it's not a C4 (it's a triangle or something else).
    # So we only need to check non-adjacent pairs (u, w).
    
    for u in range(n):
        for w in range(u + 1, n):
            if adj[u][w]:
                continue # If u and w are connected, they can't be opposite in a C4
            
            # Find common neighbors of u and w
            # These are potential v and x
            common = []
            for v in range(n):
                if v != u and v != w and adj[u][v] and adj[w][v]:
                    common.append(v)
            
            if len(common) >= 2:
                # We found at least two common neighbors.
                # Pick the first two: v and x
                v = common[0]
                x = common[1]
                # Verify v and x are distinct (they are, from list)
                # Verify the cycle: u-v, v-w, w-x, x-u
                # adj[u][v] is True (by common def)
                # adj[w][v] is True (by common def)
                # adj[w][x] is True (by common def)
                # adj[u][x] is True (by common def)
                # We also need to ensure it's a simple cycle (no chords? No, C4 definition in Ramsey usually allows chords? 
                # "Containing no C4" usually means no induced C4? Or no C4 subgraph?
                # Standard Ramsey R(C4, K...) implies no subgraph C4. 
                # A subgraph C4 is a set of 4 vertices with 4 edges forming a cycle. Chords are allowed in the supergraph.
                # So yes, if u,v,w,x form a cycle, it's a C4 subgraph.
                return [u, v, w, x]
                
    return None

def try_add_edges(adj, deg, n, min_deg_required):
    """
    Try to add edges to increase the minimum degree.
    Returns True if an edge was added.
    """
    # Find vertices with degree < min_deg_required
    low_deg_vertices = [i for i in range(n) if deg[i] < min_deg_required]
    
    if not low_deg_vertices:
        return False
        
    # Sort by degree ascending (most in need)
    low_deg_vertices.sort(key=lambda i: deg[i])
    
    for u in low_deg_vertices:
        if deg[u] >= min_deg_required:
            continue
            
        # Try to add an edge (u, v) for some v
        # v should be a non-neighbor of u
        # Adding (u, v) creates a C4 if u and v share >= 2 common neighbors.
        
        # Candidates for v: non-neighbors of u
        # To minimize C4 risk, pick v with fewest common neighbors with u.
        
        best_v = -1
        best_common = n + 1
        
        for v in range(n):
            if v == u or adj[u][v]:
                continue
            if deg[v] >= min_deg_required: # Don't add to high degree nodes if avoidable? 
                # Actually, we want to balance. But adding to high degree node is safer for C4?
                # No, common neighbors matter.
                pass
            
            # Count common neighbors
            common_count = 0
            for w in range(n):
                if adj[u][w] and adj[v][w]:
                    common_count += 1
                    if common_count >= 2:
                        break
            
            if common_count < 2:
                # Safe to add? 
                # Wait, if common_count is 0 or 1, adding (u,v) does NOT create a C4 involving u and v as opposite?
                # A C4 involving the new edge (u,v) would be u-v-w-x-u.
                # This requires w and x to be common neighbors of u and v.
                # If common_count < 2, there are not enough common neighbors to form a C4 with (u,v) as an edge.
                # So it is safe to add (u,v) without creating a NEW C4 that uses (u,v).
                
                if common_count < best_common:
                    best_common = common_count
                    best_v = v
                    
        if best_v != -1:
            # Add edge (u, best_v)
            adj[u][best_v] = True
            adj[best_v][u] = True
            deg[u] += 1
            deg[best_v] += 1
            return True
            
    return False
