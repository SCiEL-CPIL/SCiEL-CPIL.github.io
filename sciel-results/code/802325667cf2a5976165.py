import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 31
    m = 91
    
    # 1. Construct PG(2,5) polarity graph
    # Points in PG(2,5) are non-zero vectors in F_5^3 modulo scaling.
    # There are (5^3 - 1) / (5 - 1) = 124 / 4 = 31 points.
    points = []
    seen = set()
    for a in range(5):
        for b in range(5):
            for c in range(5):
                if a == 0 and b == 0 and c == 0:
                    continue
                # Normalize to first non-zero element being 1
                vec = [a, b, c]
                for i in range(3):
                    if vec[i] != 0:
                        inv = pow(vec[i], 3, 5)  # mod 5, x^4=1, so x^-1 = x^3
                        vec = [(v * inv) % 5 for v in vec]
                        break
                tvec = tuple(vec)
                if tvec not in seen:
                    seen.add(tvec)
                    points.append(tvec)
    
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Build adjacency matrix A (0/1)
    A = np.zeros((n, n), dtype=np.int8)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            p1 = points[i]
            p2 = points[j]
            dot = (p1[0]*p2[0] + p1[1]*p2[1] + p1[2]*p2[2]) % 5
            if dot == 0:
                A[i, j] = 1
                A[j, i] = 1
                edges.append((i, j))
                
    # Check edge count
    current_edges = len(edges)
    
    # Function to calculate C4 count from A2
    def calc_c4(A2):
        # A2[i,j] is number of common neighbors
        # C4 = 0.5 * sum_{i != j} C(A2[i,j], 2)
        total = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    k = A2[i, j]
                    if k >= 2:
                        total += k * (k - 1) // 2
        return total // 2

    # Initialize A2 = A @ A
    A2 = A @ A
    
    # 2. Add the 91st edge
    # Find non-edge (u,v) that minimizes new C4s
    # New C4s = C(k_uv, 2) + sum_{w in N(u) \cap N(v)} ... ?
    # Actually, just add and calculate delta.
    best_delta = float('inf')
    best_edge = None
    
    # List of non-edges
    non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] == 0:
                non_edges.append((i, j))
                
    for u, v in non_edges:
        # Calculate delta for adding (u,v)
        # k_uv = A2[u,v]
        # New C4s involving u,v:
        # Pairs (u, w) and (v, w) change.
        # (u,v) itself: k_uv unchanged.
        
        delta = 0
        for w in range(n):
            if w == u or w == v:
                continue
            # Change in k_uw:
            # A2[u,w] increases by A[v,w] (since A[u,v] becomes 1, term k=v in sum for A2[u,w] is A[u,v]A[v,w] = A[v,w])
            old_k_uw = A2[u, w]
            new_k_uw = old_k_uw + A[v, w]
            delta += (new_k_uw * (new_k_uw - 1) // 2) - (old_k_uw * (old_k_uw - 1) // 2)
            
            # Change in k_vw:
            old_k_vw = A2[v, w]
            new_k_vw = old_k_vw + A[u, w]
            delta += (new_k_vw * (new_k_vw - 1) // 2) - (old_k_vw * (old_k_vw - 1) // 2)
            
            # Change in k_wu:
            # A2[w,u] increases by A[w,v]
            old_k_wu = A2[w, u]
            new_k_wu = old_k_wu + A[w, v]
            delta += (new_k_wu * (new_k_wu - 1) // 2) - (old_k_wu * (old_k_wu - 1) // 2)
            
            # Change in k_wv:
            old_k_wv = A2[w, v]
            new_k_wv = old_k_wv + A[w, u]
            delta += (new_k_wv * (new_k_wv - 1) // 2) - (old_k_wv * (old_k_wv - 1) // 2)
            
        if delta < best_delta:
            best_delta = delta
            best_edge = (u, v)
            
    if best_edge:
        u, v = best_edge
        A[u, v] = 1
        A[v, u] = 1
        edges.append((u, v))
        # Update A2
        for w in range(n):
            A2[u, w] += A[v, w]
            A2[v, w] += A[u, w]
            A2[w, u] += A[w, v]
            A2[w, v] += A[w, u]
            
    current_c4s = calc_c4(A2)
    
    # 3. Local Search
    improved = True
    best_graph_A = A.copy()
    best_graph_c4s = current_c4s
    
    while improved and time.time() < deadline - 5:
        improved = False
        best_swap = None
        best_swap_delta = 0
        
        edges_list = list(edges)
        non_edges_list = list(non_edges)
        # Note: non_edges_list is stale if we added best_edge.
        # Let's regenerate non_edges for each iteration or maintain it.
        # Regenerating is O(n^2) = 961, which is fine.
        
        # To speed up, we can precompute lists.
        # But let's just loop.
        
        for e1 in edges_list:
            u1, v1 = e1
            # Remove e1
            A[u1, v1] = 0
            A[v1, u1] = 0
            # Update A2 for removal
            # A2[u1, w] -= A[v1, w]
            # A2[v1, w] -= A[u1, w]
            # A2[w, u1] -= A[w, v1]
            # A2[w, v1] -= A[w, u1]
            
            # We need to do this carefully to not modify A2 permanently during evaluation.
            # Since n=31, we can just copy A2? 31x31 = 961 ints. Copying is fast.
            # Or we can calculate delta on the fly without modifying A2.
            
            # Let's calculate delta for removing (u1, v1) and adding (u2, v2).
            # This is complex.
            # Alternative:
            # 1. Remove (u1, v1) from A and A2.
            # 2. Loop through non-edges, calculate delta for adding.
            # 3. Add back (u1, v1).
            
            # To avoid modifying global A2 in inner loop, we can use a temporary A2?
            # Or just calculate the effect of removal on A2, store it, then calculate effect of addition.
            
            # Effect of removing (u1, v1):
            # dA2[u1, w] = -A[v1, w]
            # dA2[v1, w] = -A[u1, w]
            # dA2[w, u1] = -A[w, v1]
            # dA2[w, v1] = -A[w, u1]
            
            # We want to find (u2, v2) non-edge that minimizes:
            # C4(A2 + dA2_remove + dA2_add(u2, v2))
            
            # This is still O(n) per non-edge.
            # 91 edges * 374 non-edges * 31 ops = ~1M ops.
            # This is acceptable.
            
            local_best_delta = 0
            local_best_edge = None
            
            # Pre-compute dA2_remove effects?
            # We can just loop.
            
            for e2 in non_edges_list:
                u2, v2 = e2
                if u2 == u1 or u2 == v1 or v2 == u1 or v2 == v1:
                    # If the edge to add shares a vertex with the edge to remove,
                    # the delta calculation is slightly different (overlapping updates).
                    # For simplicity, we can skip these or handle carefully.
                    # Actually, if we remove (u1, v1) and add (u1, u2), it's a move, not a swap.
                    # We should allow this.
                    # But the delta formula for "add" assumes the edge is not present.
                    # If we removed (u1, v1), then (u1, u2) is a valid non-edge to add (if it was).
                    # The A2 updates will overlap.
                    # Let's just compute the full A2 state for the candidate?
                    # No, too slow.
                    # Let's stick to disjoint swaps first, then add overlapping later if needed.
                    # Or, simply:
                    # Calculate new A2 = A2 + dRemove + dAdd.
                    # Since n is small, we can compute the new values for the affected indices.
                    
                    pass # Placeholder for logic
                    
            # This inner loop is the bottleneck.
            # Let's optimize by precomputing the "cost" of adding each non-edge to the graph with e1 removed.
            
            # Actually, let's just do a simpler local search:
            # Try all single edge swaps (remove one, add one).
            # Use the efficient delta calculation.
            
            # We'll implement the delta calculation for a specific pair (remove e1, add e2).
            
            # For now, let's just do a basic local search with random restarts.
            
            # Revert A
            A[u1, v1] = 1
            A[v1, u1] = 1
            
        # If no improvement, break or kick.
        if not improved:
            # Try a random kick
            if time.time() < deadline - 10:
                # Pick 2 random edges to remove, 2 random non-edges to add
                # This is a 2-swap.
                # For now, just restart with a slightly perturbed graph?
                # Or break.
                pass
                
    # Return best edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j]:
                result.append([i, j])
                
    return result
