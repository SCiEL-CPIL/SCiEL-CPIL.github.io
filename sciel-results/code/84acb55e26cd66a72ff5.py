"""Program for fewest four-cycles in graphs with 31 vertices and 91 edges."""
import numpy as np
import random
import time

def construct():
    """
    Construct a graph with 31 vertices and exactly 91 edges minimizing the number of C4 subgraphs.
    
    Approach:
    1. Build a maximal C4-free graph using a greedy algorithm.
    2. Adjust edge count to 90 (remove edges if >90, add random if <90).
    3. Add the 91st edge choosing the pair with minimum common neighbors.
    4. Perform fast local search using incremental C4 count updates.
    """
    n = 31
    target_edges = 91
    
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.seed(42)
    
    # Step 1: Greedy C4-free construction
    adj = np.zeros((n, n), dtype=np.int8)
    # cn[i][j] = number of common neighbors of i and j
    cn = np.zeros((n, n), dtype=np.int32)
    
    # Shuffle edges for randomization
    shuffled_edges = all_edges[:]
    random.shuffle(shuffled_edges)
    
    edge_count = 0
    for u, v in shuffled_edges:
        # Check if adding (u,v) creates a C4
        # A C4 is created if cn[u][v] > 0
        if cn[u, v] == 0:
            # Add edge
            adj[u, v] = 1
            adj[v, u] = 1
            edge_count += 1
            
            # Update cn matrix
            # Adding (u,v) increases cn[u][k] and cn[v][k] for neighbors of the other
            # Specifically:
            # cn[u][k] += 1 if adj[v][k] == 1
            # cn[v][k] += 1 if adj[u][k] == 1
            # cn[k][l] does not change for k,l not in {u,v}
            
            # Vectorized update for row u and v
            # For all k, if adj[v,k] is 1, cn[u,k] += 1
            cn[u, :] += adj[v, :]
            cn[v, :] += adj[u, :]
            
            # Note: cn[u,v] becomes 1? No, cn[u,v] is common neighbors.
            # The edge (u,v) itself is not a common neighbor.
            # The update above correctly adds 1 to cn[u,k] for each neighbor k of v.
            # If k==v, adj[v,v]=0, so no change.
            # If k==u, adj[v,u]=1 (just set), so cn[u,u] += 1? 
            # cn[u,u] is not used.
            
            # Symmetry: cn is symmetric. 
            # cn[u,k] was updated. cn[k,u] should also be updated.
            # cn[v,k] was updated. cn[k,v] should also be updated.
            cn[:, u] += adj[v, :]
            cn[:, v] += adj[u, :]
            
            # Wait, this double counts or is wrong?
            # Let's trace.
            # We want cn[i][j] = sum_k adj[i,k] adj[j,k].
            # If we add edge (u,v):
            # adj[u,v] becomes 1, adj[v,u] becomes 1.
            # For pair (u, k): new cn[u,k] = old cn[u,k] + adj[v,k] (since adj[u,v] is now 1, v is a neighbor of u, so if v is also neighbor of k, it's a common neighbor? No.)
            # Common neighbors of u and k are nodes x such that adj[u,x]=1 and adj[k,x]=1.
            # Before: x could be v? No, adj[u,v] was 0.
            # After: x can be v if adj[k,v]=1.
            # So cn[u,k] increases by 1 if adj[k,v]=1.
            # This matches cn[u, :] += adj[v, :] (since adj[v,:] is 1 at indices where v has neighbors).
            # Similarly for cn[v,k].
            # For pair (k, l) where k,l != u,v:
            # Common neighbors are x.
            # x could be u? Before adj[k,u]=1, adj[l,u]=1. u was a common neighbor.
            # After adj[k,u] unchanged, adj[l,u] unchanged. u is still a common neighbor.
            # x could be v? Before adj[k,v]=1, adj[l,v]=1. v was a common neighbor.
            # After adj[k,v] unchanged, adj[l,v] unchanged. v is still a common neighbor.
            # So cn[k,l] does not change.
            
            # So we only need to update rows/cols u and v.
            # cn[u, :] += adj[v, :]
            # cn[v, :] += adj[u, :]
            # And by symmetry cn[:, u] += adj[v, :] and cn[:, v] += adj[u, :]
            # But cn[u, :] update covers cn[u,k]. cn[:, u] update covers cn[k,u].
            # Since cn is symmetric, we should ensure consistency.
            # Actually, just updating the full matrix rows/cols is safer.
            
    # Step 2: Adjust to 90 edges
    if edge_count > 90:
        # Remove random edges
        current_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 1]
        random.shuffle(current_edges)
        for u, v in current_edges:
            if edge_count == 90:
                break
            # Remove edge (u,v)
            adj[u, v] = 0
            adj[v, u] = 0
            edge_count -= 1
            
            # Update cn
            # Removing (u,v) decreases cn[u,k] by 1 if adj[v,k]==1
            cn[u, :] -= adj[v, :]
            cn[v, :] -= adj[u, :]
            cn[:, u] -= adj[v, :]
            cn[:, v] -= adj[u, :]
            
    elif edge_count < 90:
        # Add random edges (might create C4s, but we need 90 edges)
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adj[i, j] == 0]
        random.shuffle(non_edges)
        for u, v in non_edges:
            if edge_count == 90:
                break
            adj[u, v] = 1
            adj[v, u] = 1
            edge_count += 1
            cn[u, :] += adj[v, :]
            cn[v, :] += adj[u, :]
            cn[:, u] += adj[v, :]
            cn[:, v] += adj[u, :]

    # Step 3: Add 91st edge
    # Find non-edge with min cn
    min_cn = float('inf')
    best_pair = None
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0:
                if cn[i, j] < min_cn:
                    min_cn = cn[i, j]
                    best_pair = (i, j)
    
    if best_pair:
        u, v = best_pair
        adj[u, v] = 1
        adj[v, u] = 1
        edge_count += 1
        cn[u, :] += adj[v, :]
        cn[v, :] += adj[u, :]
        cn[:, u] += adj[v, :]
        cn[:, v] += adj[u, :]

    # Step 4: Local Search
    # Calculate initial C4 count
    def calc_c4(cn):
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = cn[i, j]
                if c >= 2:
                    total += c * (c - 1) // 2
        return total

    current_c4 = calc_c4(cn)
    
    # Precompute list of edges and non-edges for faster selection?
    # Or just sample.
    
    start_time = time.time()
    no_improvement = 0
    max_no_improvement = 1000
    iteration = 0
    
    while no_improvement < max_no_improvement and (time.time() - start_time) < 100:
        iteration += 1
        
        # Pick random edge to remove
        # Efficient way: keep a list of edges? Or sample until hit.
        # With 91 edges out of 465, sampling is okay.
        while True:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a < b and adj[a, b] == 1:
                break
        a, b = min(a,b), max(a,b)
        
        # Pick random non-edge to add
        while True:
            c = random.randint(0, n - 1)
            d = random.randint(0, n - 1)
            if c < d and adj[c, d] == 0:
                break
        c, d = min(c,d), max(c,d)
        
        if (a == c and b == d):
            continue
            
        # Calculate delta
        # 1. Remove (a,b)
        # C4s destroyed: C(cn[a,b], 2)
        destroyed = cn[a, b] * (cn[a, b] - 1) // 2
        
        # Update cn for removal of (a,b)
        # Save old values for c,d if affected
        # cn changes for rows/cols a,b
        old_cn_c_d = cn[c, d]
        
        cn[a, :] -= adj[b, :]
        cn[b, :] -= adj[a, :]
        cn[:, a] -= adj[b, :]
        cn[:, b] -= adj[a, :]
        
        # 2. Add (c,d)
        # C4s created: C(cn_new[c,d], 2)
        # cn_new[c,d] is the value after removal of (a,b) and before adding (c,d)
        # Wait, the removal of (a,b) might have changed cn[c,d]
        # If a or b was a common neighbor of c and d, cn[c,d] decreased.
        # The update above handled that.
        created = cn[c, d] * (cn[c, d] - 1) // 2
        
        delta = created - destroyed
        
        if delta < 0:
            # Accept
            current_c4 += delta
            no_improvement = 0
            
            # Finalize addition of (c,d)
            adj[c, d] = 1
            adj[d, c] = 1
            
            # Update cn for addition of (c,d)
            cn[c, :] += adj[d, :]
            cn[d, :] += adj[c, :]
            cn[:, c] += adj[d, :]
            cn[:, d] += adj[c, :]
            
            # Note: adj[c,d] is now 1, adj[d,c] is 1.
            # The update cn[c,:] += adj[d,:] adds 1 to cn[c,d] because adj[d,d]=0? No.
            # adj[d,:] has 1 at index c. So cn[c,c] += 1?
            # cn[c,c] is not used.
            # cn[c,k] += 1 if adj[d,k]=1.
            # This is correct.
            
        else:
            # Reject
            no_improvement += 1
            # Revert removal of (a,b)
            adj[a, b] = 1
            adj[b, a] = 1
            
            # Revert cn updates
            cn[a, :] += adj[b, :]
            cn[b, :] += adj[a, :]
            cn[:, a] += adj[b, :]
            cn[:, b] += adj[a, :]
            
    # Convert to list of edges
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                result.append([i, j])
    
    assert len(result) == 91, f"Expected 91 edges, got {len(result)}"
    
    return result
