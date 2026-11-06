import random
import time
import math

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 118

    # 1. Construct the Complement of the Line Graph of K7 (21 vertices)
    # Vertices of K7 are 0..6
    # Vertices of T(7) are edges of K7: (i, j) for 0 <= i < j < 7
    edges_k7 = [(i, j) for i in range(7) for j in range(i + 1, 7)]
    # Map each edge to an index 0..20
    edge_to_idx = {e: i for i, e in enumerate(edges_k7)}
    
    # Build adjacency for T(7)
    # Two edges (i,j) and (k,l) are adjacent in T(7) if they share a vertex
    adj_t7 = [0] * 21
    for idx1, e1 in enumerate(edges_k7):
        for idx2, e2 in enumerate(edges_k7):
            if idx1 == idx2:
                continue
            if e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]:
                adj_t7[idx1] |= (1 << idx2)
                
    # Build adjacency for Complement of T(7), restricted to first 20 vertices (delete vertex 20)
    # We keep vertices 0..19
    adj = [0] * 20
    for u in range(20):
        for v in range(u + 1, 20):
            # If NOT adjacent in T(7), they are adjacent in Complement
            if not (adj_t7[u] & (1 << v)):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)

    # Count edges
    count = 0
    for u in range(20):
        count += bin(adj[u]).count('1')
    count //= 2

    # 2. Repair phase: Remove edges to ensure no pair has > 2 common neighbors
    # We need |N(u) intersection N(v)| <= 2 for all u, v
    # This is equivalent to: for all u, v, the number of w such that adj[u]&adj[w] and adj[v]&adj[w] is <= 2
    
    def common_neighbors_count(u, v):
        # Count bits in adj[u] & adj[v]
        return bin(adj[u] & adj[v]).count('1')

    # Iteratively fix violations
    # To speed up, we can just run a local search that penalizes violations
    # But first, let's try a greedy repair
    for _ in range(100): # Max iterations for repair
        violated = False
        for u in range(20):
            if time.time() > deadline - 50:
                break
            for v in range(u + 1, 20):
                cn = common_neighbors_count(u, v)
                if cn > 2:
                    violated = True
                    # Find a common neighbor w
                    common_mask = adj[u] & adj[v]
                    # Pick the first common neighbor
                    w = (common_mask & -common_mask).bit_length() - 1
                    # Remove edge (u, w) or (v, w)
                    # Randomly choose which to remove to avoid bias
                    if random.random() < 0.5:
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        count -= 1
                    else:
                        adj[v] &= ~(1 << w)
                        adj[w] &= ~(1 << v)
                        count -= 1
        if not violated:
            break

    # 3. Simulated Annealing to maximize edges
    # State: adj (list of int), count (int)
    # Move: Add an edge, or Remove an edge?
    # Since we want to maximize edges, we should try adding edges primarily.
    # But we must maintain K2,3-freeness.
    
    # Precompute all non-edges
    all_pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    
    # Helper to check if adding edge (u,v) is valid
    def can_add(u, v):
        if adj[u] & (1 << v):
            return True # Already exists
        # Check common neighbors of u and v
        # If they have >= 3 common neighbors, adding (u,v) is not directly forbidden by K2,3
        # Wait, the condition is on EXISTING common neighbors.
        # Adding (u,v) makes u and v adjacent.
        # The condition "no two vertices have 3 common neighbors" applies to ALL pairs.
        # So if u and v already have 3 common neighbors, the graph is already invalid?
        # No, the graph must be valid at all times.
        # So if current graph is valid, and we add (u,v), we must ensure:
        # 1. The pair (u,v) does not have 3 common neighbors. (Already true if graph was valid? No, adding edge doesn't change common neighbors of u and v).
        # 2. For any neighbor w of u, the pair (v, w) must not have 3 common neighbors AFTER adding (u,v).
        #    Adding (u,v) increases common neighbors of (v, w) if u is a neighbor of w.
        #    i.e., if u ~ w, then u becomes a common neighbor of v and w.
        #    So we need: for all w in N(u), |N(v) \cap N(w)| < 3 (so that adding u makes it <= 3? No, it must be <= 2).
        #    So we need |N(v) \cap N(w)| <= 1 for all w in N(u).
        #    Similarly for all w in N(v), |N(u) \cap N(w)| <= 1.
        
        # Actually, simpler check:
        # Just add the edge and check if any pair has > 2 common neighbors?
        # That's expensive.
        
        # Optimized check:
        # 1. If |N(u) \cap N(v)| >= 3, we cannot add (u,v)? No, that condition is for the graph to be valid.
        #    If the graph is currently valid, |N(u) \cap N(v)| <= 2.
        #    Adding (u,v) does NOT change |N(u) \cap N(v)|.
        #    So (u,v) remains <= 2.
        # 2. The new constraints are for pairs involving u or v and their neighbors.
        #    For any w in N(u):
        #       The pair (v, w) gains u as a common neighbor (since u~v and u~w).
        #       So we need current |N(v) \cap N(w)| <= 1.
        #    For any w in N(v):
        #       The pair (u, w) gains v as a common neighbor.
        #       So we need current |N(u) \cap N(w)| <= 1.
        
        # Check condition 2
        # Iterate neighbors of u
        mask_u = adj[u]
        while mask_u:
            lsb = mask_u & -mask_u
            w = lsb.bit_length() - 1
            # Check |N(v) \cap N(w)|
            if bin(adj[v] & adj[w]).count('1') > 1:
                return False
            mask_u ^= lsb
            
        mask_v = adj[v]
        while mask_v:
            lsb = mask_v & -mask_v
            w = lsb.bit_length() - 1
            # Check |N(u) \cap N(w)|
            if bin(adj[u] & adj[w]).count('1') > 1:
                return False
            mask_v ^= lsb
            
        return True

    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        return 1

    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return -1

    # SA Loop
    T_start = 1.0
    T_end = 0.01
    total_iters = 0
    # Estimate iterations
    # Each iteration involves O(D) checks
    # D ~ 10. 20*20 pairs.
    # Let's just run until time limit
    
    best_count = count
    best_adj = adj[:]
    
    while time.time() < deadline - 2:
        T = T_start * (T_end / T_start) ** (total_iters / 5000)
        total_iters += 1
        
        # Propose move: Try to add a random non-edge
        # Or remove an edge if we get stuck?
        # Since we want to maximize, bias towards adding.
        
        # Pick random pair
        u = random.randint(0, 19)
        v = random.randint(0, 19)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        if adj[u] & (1 << v):
            # Already an edge. Try to remove?
            # Removing an edge decreases count. Accept with low prob.
            delta = -1
            # Remove
            remove_edge(u, v)
            count += delta
            if delta >= 0 or random.random() < math.exp(delta / T):
                if count > best_count:
                    best_count = count
                    best_adj = adj[:]
            else:
                # Restore
                add_edge(u, v)
                count -= delta
        else:
            # Try to add
            if can_add(u, v):
                add_edge(u, v)
                count += 1
                # Always accept improvement
                if count > best_count:
                    best_count = count
                    best_adj = adj[:]
            else:
                # Cannot add. Do nothing or try different pair
                pass

    # Final polish: Try to add all possible edges
    for u in range(20):
        for v in range(u + 1, 20):
            if not (adj[u] & (1 << v)):
                if can_add(u, v):
                    add_edge(u, v)
                    count += 1
                    if count > best_count:
                        best_count = count
                        best_adj = adj[:]

    # Extract edges from best_adj
    edges = []
    for i in range(20):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges

# Run initial check to ensure no exceptions
# Note: The 'repair' phase in the code above might be slow if implemented with O(N^3) checks inside a loop.
# The 'can_add' function is O(D) which is fast.
# The initial construction is O(1).
# The SA loop is the main part.

# To be safe, let's refine the repair phase to be faster or just rely on SA starting from a valid state?
# The initial state from Complement of T(7) is NOT valid (it has violations).
# So we MUST repair it.
# The repair loop in the code above:
# for u in 20: for v in 20: check common neighbors (O(1) with bit ops).
# If violation, remove edge.
# This is fast.

# One concern: `bin(x).count('1')` is slow.
# Use `int.bit_count()` if Python 3.10+, else fallback.
# The environment likely supports 3.8+. Let's use a helper.

def popcount(x):
    return bin(x).count('1')

# Replace bin(...).count('1') with popcount for clarity/speed if available?
# bin is fine for 20 bits.

if __name__ == "__main__":
    # Dummy run to verify
    pass
