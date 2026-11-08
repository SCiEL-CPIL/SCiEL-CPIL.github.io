import numpy as np

def priority(adjacency):
    n = adjacency.shape[0]
    active = set(range(n))
    scores = np.zeros(n, dtype=float)
    
    # Precompute neighbors as lists of indices for faster access?
    # Or just use the matrix. N=200 is small.
    
    step = 0
    while active:
        best_v = -1
        best_score = -1
        
        # Calculate heuristic for all active vertices
        for v in active:
            # Get neighbors of v in active
            # adjacency[v] is a bool array of length n
            # We need neighbors that are in active
            # Since N is small, iterating is fine, but vectorization is better
            pass
        
        # Vectorized approach for the inner loop?
        # It's tricky to vectorize the "remove neighbors" part easily in a while loop with sets.
        # Given N=200, a simple loop is fast enough.
        
        for v in active:
            # Neighbors of v in active
            # adj_v = adjacency[v]
            # neighbors = [u for u in active if adj_v[u]]
            
            # To speed up, we can maintain a list of active nodes and use numpy
            # But let's stick to simple logic first to ensure correctness.
            
            # Heuristic: Max-Product
            # deg = number of active neighbors
            # nn = number of active neighbors of active neighbors
            
            # Using sets for active
            # v_neighs = {u for u in active if adjacency[v][u]}
            # deg = len(v_neighs)
            
            # If deg == 0, score is 0? Or handle separately.
            # If deg > 0:
            #    nn_set = set()
            #    for u in v_neighs:
            #        # add active neighbors of u
            #        # This is O(deg * avg_deg)
            
            # This might be slow if done naively in Python for 200 steps * 200 nodes * neighbors.
            # 200 * 200 * 200 = 8,000,000 operations. Python can handle this in < 1s.
            
            v_neighs = [u for u in active if adjacency[v][u]]
            deg = len(v_neighs)
            
            if deg == 0:
                val = 0
            else:
                # Collect neighbors of neighbors
                nn_set = set()
                for u in v_neighs:
                    # Add active neighbors of u
                    # We can iterate over active and check adjacency[u][w]
                    # Or precompute neighbors of u?
                    # Let's just iterate over active
                    for w in active:
                        if adjacency[u][w]:
                            nn_set.add(w)
                # Exclude v itself? Usually N(N(v)) excludes v?
                # The heuristic is |N(v)| * |N(N(v))|.
                # N(N(v)) typically includes v if there are triangles?
                # Standard definition: N(N(v)) is the set of vertices at distance 2.
                # Let's assume it's the union of neighbors of neighbors.
                # Does it include v? If there is a cycle of length 3, v is in N(N(v)).
                # The heuristic usually counts distinct vertices in the 2-neighborhood.
                # Let's just use len(nn_set).
                
                val = deg * len(nn_set)
            
            if val > best_score:
                best_score = val
                best_v = v
        
        # Assign score
        scores[best_v] = n - step
        
        # Remove best_v and its neighbors from active
        # We need the neighbors of best_v in the CURRENT active set
        # We calculated v_neighs for best_v in the loop above? No, we didn't store it.
        # Let's recalculate or store it.
        
        # Optimization: Store the best v's neighbors during the search
        # Let's refactor the inner loop to store neighbors of the current best candidate.
        
        # Re-doing the loop with storage
        best_v = -1
        best_score = -1
        best_neighs = set()
        
        for v in active:
            v_neighs = [u for u in active if adjacency[v][u]]
            deg = len(v_neighs)
            
            if deg == 0:
                val = 0
            else:
                nn_set = set()
                for u in v_neighs:
                    for w in active:
                        if adjacency[u][w]:
                            nn_set.add(w)
                val = deg * len(nn_set)
            
            if val > best_score:
                best_score = val
                best_v = v
                best_neighs = set(v_neighs)
        
        scores[best_v] = n - step
        
        # Remove
        active.remove(best_v)
        for u in best_neighs:
            if u in active:
                active.remove(u)
                
        step += 1
        
    return scores
