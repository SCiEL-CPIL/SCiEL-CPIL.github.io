import time
import random
from itertools import combinations

N = 18
DEADLINE = 115.0  # safety margin under 150s

def _is_k4_free(adj):
    """Check if graph is K4-free using bitmask."""
    for i in range(N):
        for j in range(i+1, N):
            if not (adj[i] & (1 << j)):
                continue
            common = adj[i] & adj[j]
            # Check if common neighborhood has any edge
            c = common
            while c:
                lsb = c & (-c)
                k = lsb.bit_length() - 1
                c ^= lsb
                if k > i and k > j:
                    # Check if k is connected to another vertex in common
                    c2 = common & ~(1 << k)
                    while c2:
                        lsb2 = c2 & (-c2)
                        m = lsb2.bit_length() - 1
                        c2 ^= lsb2
                        if m > k and (adj[k] & (1 << m)):
                            return False
    return True

def _count_tf_4_sets(adj):
    """Count the number of triangle-free induced subgraphs of size 4."""
    count = 0
    # Precompute triangles
    triangles = set()
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j:
                        triangles.add((i, j, k))
    
    # Check all 4-sets
    for combo in combinations(range(N), 4):
        a, b, c, d = combo
        # Check if any triangle exists in {a,b,c,d}
        has_triangle = False
        for tri in triangles:
            if set(tri).issubset(set(combo)):
                has_triangle = True
                break
        if not has_triangle:
            count += 1
    return count

def _find_tf_4_sets(adj):
    """Find all triangle-free induced subgraphs of size 4."""
    tf_sets = []
    triangles = set()
    for i in range(N):
        for j in range(i+1, N):
            if adj[i] & (1 << j):
                common = adj[i] & adj[j]
                c = common
                while c:
                    lsb = c & (-c)
                    k = lsb.bit_length() - 1
                    c ^= lsb
                    if k > j:
                        triangles.add((i, j, k))
    
    for combo in combinations(range(N), 4):
        a, b, c, d = combo
        has_triangle = False
        for tri in triangles:
            if set(tri).issubset(set(combo)):
                has_triangle = True
                break
        if not has_triangle:
            tf_sets.append(combo)
    return tf_sets

def _add_edge_safe(adj, u, v):
    """Try to add edge (u,v) to adj. Return True if successful and K4-free maintained."""
    if adj[u] & (1 << v):
        return True # Already present
    
    # Check if adding (u,v) creates a K4
    # K4 is formed if u and v have a common neighbor w, and w has a common neighbor x with u and v?
    # No, K4 is {u,v,w,x} where all pairs are connected.
    # Adding (u,v) creates a K4 if there exists w such that (u,w) and (v,w) are in G,
    # AND there exists x such that (u,x), (v,x), (w,x) are in G?
    # Actually, if we add (u,v), we form a K4 if there is a triangle {u,v,w} in the NEW graph?
    # No, K4 is 4 vertices.
    # If we add (u,v), we create a K4 if there is a set {u,v,w,x} such that all edges are present.
    # Since (u,v) is new, the other 5 edges must be present.
    # So we need to check if there is a pair (w,x) such that:
    # (u,w), (u,x), (v,w), (v,x), (w,x) are all in G.
    # This is equivalent to checking if the common neighbors of u and v contain an edge.
    
    common = adj[u] & adj[v]
    c = common
    while c:
        lsb = c & (-c)
        w = lsb.bit_length() - 1
        c ^= lsb
        # Check if w is connected to another common neighbor
        c2 = common & ~(1 << w)
        while c2:
            lsb2 = c2 & (-c2)
            x = lsb2.bit_length() - 1
            c2 ^= lsb2
            if (adj[w] & (1 << x)):
                return False # Would create K4 {u,v,w,x}
    
    # Safe to add
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)
    return True

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_adj = None
    best_tf_count = N * (N-1) * (N-2) * (N-3) // 24 # Max possible
    
    # Try to find a graph with 0 triangle-free 4-sets
    # This implies score 3.
    
    for attempt in range(50):
        if time.time() - start_time > DEADLINE:
            break
            
        # Initialize random K4-free graph
        adj = [0] * N
        # Start with empty graph and add edges randomly, checking K4-free
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        rng.shuffle(edges)
        
        for u, v in edges:
            if _add_edge_safe(adj, u, v):
                pass # Edge added
            else:
                pass # Edge not added
        
        # Check if we have a valid K4-free graph
        if not _is_k4_free(adj):
            continue
            
        tf_sets = _find_tf_4_sets(adj)
        current_tf_count = len(tf_sets)
        
        if current_tf_count < best_tf_count:
            best_tf_count = current_tf_count
            best_adj = adj[:]
            
        if best_tf_count == 0:
            break
            
        # Local search: try to reduce tf_count
        # Iterate through tf_sets and try to fix them
        improved = True
        while improved and (time.time() - start_time) < DEADLINE - 10:
            improved = False
            tf_sets = _find_tf_4_sets(adj)
            if not tf_sets:
                break
                
            rng.shuffle(tf_sets)
            
            for S in tf_sets:
                if time.time() - start_time > DEADLINE - 10:
                    break
                    
                a, b, c, d = S
                # Try to add an edge within S to create a triangle
                # Pairs: (a,b), (a,c), (a,d), (b,c), (b,d), (c,d)
                pairs = [(a,b), (a,c), (a,d), (b,c), (b,d), (c,d)]
                rng.shuffle(pairs)
                
                for u, v in pairs:
                    if adj[u] & (1 << v):
                        continue # Edge exists
                    
                    # Check if adding (u,v) creates a triangle in S
                    # Triangle in S requires a third vertex w in S such that (u,w) and (v,w) exist
                    # S = {a,b,c,d}. If u=a, v=b, then w can be c or d.
                    # Check if (a,c) and (b,c) exist, or (a,d) and (b,d) exist.
                    
                    creates_tri = False
                    for w in S:
                        if w != u and w != v:
                            if (adj[u] & (1 << w)) and (adj[v] & (1 << w)):
                                creates_tri = True
                                break
                                
                    if not creates_tri:
                        continue # Adding this edge won't make S triangle-free -> non-triangle-free?
                                 # Wait, S is triangle-free. Adding an edge (u,v) creates a triangle ONLY IF there is a path u-w-v.
                                 # If no such path, S remains triangle-free (just with one more edge, but no triangle).
                                 # Actually, if S was triangle-free, it has no 3-cycle.
                                 # Adding an edge (u,v) creates a 3-cycle iff there is a length-2 path between u and v in S.
                                 # So if creates_tri is False, S is still triangle-free.
                                 # So we should NOT add this edge if it doesn't fix the problem.
                                 # Unless adding it helps other sets?
                                 # For now, only add if it fixes this set.
                                 
                    # Check if safe to add (K4-free)
                    # We need a copy to check
                    adj_copy = adj[:]
                    if _add_edge_safe(adj_copy, u, v):
                        # Apply change
                        adj = adj_copy
                        improved = True
                        break # Move to next set
                    
    # If we found a graph with 0 tf sets, score is 3.
    # Otherwise, we might have a graph with some tf sets.
    # We need to calculate the actual score (max triangle-free set size).
    
    if best_adj is None:
        # Fallback: return a valid K4-free graph
        adj = [0] * N
        edges = [(i, j) for i in range(N) for j in range(i+1, N)]
        rng.shuffle(edges)
        for u, v in edges:
            if _add_edge_safe(adj, u, v):
                pass
        best_adj = adj
        
    # Calculate exact score for best_adj
    # Score is max size of triangle-free induced subgraph.
    # We know it's at least 3.
    # If best_tf_count == 0, score is 3.
    # If best_tf_count > 0, score is at least 4.
    
    if best_tf_count == 0:
        score = 3
    else:
        # Check if there is a triangle-free set of size 5?
        # For now, assume score is 4 if tf_count > 0.
        # We can refine this, but 3 is the target.
        score = 4
        
    # Convert to edge list
    result = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i] & (1 << j):
                result.append([i, j])
                
    return result

if __name__ == "__main__":
    edges = construct()
    print(f"Edges: {len(edges)}")
