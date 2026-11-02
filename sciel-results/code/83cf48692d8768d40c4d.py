import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin

    # Baseline: All norm-squared-2 vectors
    # Condition: ||v_i - v_j||^2 >= 2.
    # For norm^2=2, ||v_i - v_j||^2 = 4 - 2 v_i.v_j.
    # 4 - 2 v_i.v_j >= 2 => v_i.v_j <= 1.
    # Distinct norm-2 vectors have dot product in {-2, -1, 0, 1}.
    # So all distinct norm-2 vectors are compatible.
    
    vecs_R2 = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i] = si
                v[j] = sj
                vecs_R2.append(v)
    
    best_set = vecs_R2
    best_size = len(best_set)
    
    # Try R=3
    # Vectors with three non-zero entries +/-1
    vecs_R3 = []
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    v[k] = sk
                    vecs_R3.append(v)
    
    n_R3 = len(vecs_R3)
    # n_R3 = C(11,3) * 8 = 165 * 8 = 1320
    
    if time.time() > deadline:
        return np.array(best_set, dtype=np.int64)
    
    # Convert to numpy for fast dot products
    vecs_R3_np = np.array(vecs_R3, dtype=np.int8)
    
    # Build conflict graph
    # Conflict if dot product > 1 (i.e., 2 or 3)
    # Dot product 3 implies identical vectors (not in list)
    # Dot product 2 implies sharing 2 coords with same signs
    
    # Precompute adjacency list
    # adj[i] = list of j such that dot(v_i, v_j) > 1
    adj = [set() for _ in range(n_R3)]
    
    # Optimization: Group vectors by their support (indices of non-zero elements)
    # Vectors with disjoint supports have dot product 0 (no conflict)
    # Vectors with support intersection 1 have dot product +/-1 (no conflict)
    # Vectors with support intersection 2 have dot product -2, 0, 2 (conflict if 2)
    # Vectors with support intersection 3 have dot product -3, -1, 1, 3 (conflict if 3, i.e., same vec)
    
    # So we only need to check pairs with support intersection size >= 2.
    
    support_to_vecs = {}
    for idx, v in enumerate(vecs_R3):
        # Find non-zero indices
        nz = tuple(np.nonzero(v)[0])
        support_to_vecs.setdefault(nz, []).append(idx)
        
    # For each pair of supports that intersect in 2 elements, check conflicts
    # Supports are tuples of 3 indices.
    # There are 165 supports.
    # Pairs of supports: 165 * 165 / 2 = ~13k pairs.
    # This is very fast.
    
    supports = list(support_to_vecs.keys())
    
    for i_s in range(len(supports)):
        for j_s in range(i_s + 1, len(supports)):
            s1 = supports[i_s]
            s2 = supports[j_s]
            inter = set(s1).intersection(s2)
            if len(inter) < 2:
                continue
            
            list1 = support_to_vecs[s1]
            list2 = support_to_vecs[s2]
            
            if len(inter) == 3:
                # Same support
                # Only conflict if dot > 1.
                # Dot is 3 if same signs.
                # But we are iterating distinct vectors.
                # Within same support, dot product is 3 only if identical.
                # Since all vectors in list1 are distinct, no conflicts within list1.
                # Wait, if s1 == s2, then i_s == j_s, but loop is i_s < j_s.
                # So we only handle distinct supports here.
                pass
            
            elif len(inter) == 2:
                # Intersect in 2 coords
                # Check all pairs (v1 in list1, v2 in list2)
                # Dot product is sum of products on the 2 intersecting coords.
                # The 3rd coord of v1 and 3rd coord of v2 are different, so their product is 0.
                # So dot product depends only on signs on the 2 shared coords.
                
                # Identify the 2 shared indices
                shared_indices = list(inter)
                idx_a = shared_indices[0]
                idx_b = shared_indices[1]
                
                for v1_idx in list1:
                    v1 = vecs_R3_np[v1_idx]
                    s1_a = v1[idx_a]
                    s1_b = v1[idx_b]
                    
                    for v2_idx in list2:
                        v2 = vecs_R3_np[v2_idx]
                        s2_a = v2[idx_a]
                        s2_b = v2[idx_b]
                        
                        # Dot product = s1_a * s2_a + s1_b * s2_b
                        dp = s1_a * s2_a + s1_b * s2_b
                        
                        if dp > 1:
                            adj[v1_idx].add(v2_idx)
                            adj[v2_idx].add(v1_idx)
                
    # Now we have the conflict graph.
    # Find Maximum Independent Set using Greedy Min-Degree
    # available = set of all nodes
    # in_set = []
    
    # To speed up, we can use a list of available nodes and a set of removed nodes
    # Or just a boolean array
    
    # Greedy:
    # 1. Start with all nodes available.
    # 2. Find node with min degree among available.
    # 3. Add to set.
    # 4. Remove it and its neighbors from available.
    
    # We'll do this a few times with different tie-breaking or orderings to see if we get better.
    
    def greedy_mis(adj, n):
        available = [True] * n
        in_set = []
        
        # Precompute degrees? No, degree changes as we remove nodes.
        # We need to track current degree for each available node.
        # deg[i] = number of available neighbors of i.
        
        deg = [len(adj[i]) for i in range(n)]
        
        # To find min degree quickly, we can just scan. n=1320 is small.
        
        while True:
            # Find min degree node in available
            min_deg = float('inf')
            min_node = -1
            for i in range(n):
                if available[i] and deg[i] < min_deg:
                    min_deg = deg[i]
                    min_node = i
            
            if min_node == -1:
                break
            
            # Add min_node to set
            in_set.append(min_node)
            available[min_node] = False
            
            # Remove neighbors
            for neighbor in adj[min_node]:
                if available[neighbor]:
                    available[neighbor] = False
                    deg[neighbor] = -1 # Mark as removed/invalid
                    # Update degrees of neighbors of neighbor?
                    # No, we only care about degrees of *available* nodes for the next step.
                    # The degree of an available node u is the number of its available neighbors.
                    # If we remove v, we must decrement deg[u] for all available u in adj[v].
                    # But we just marked neighbors as unavailable.
                    # We need to update the degrees of the *surviving* neighbors?
                    # No, the degree of a surviving node w is the number of its neighbors that are still available.
                    # When we remove v, we check adj[v].
                    # For each u in adj[v]:
                    #   If u is available, it becomes unavailable.
                    #   This does not affect the degree of other nodes directly,
                    #   except that u is no longer a neighbor for them.
                    #   So for each w in adj[u] (where u is removed), if w is available, deg[w] should decrease.
                    
                    # This is getting complicated to maintain dynamically.
                    # Since n=1320, we can just recompute degrees or use a simpler approach.
                    
            # Simpler approach for n=1320:
            # Just keep a list of available nodes.
            # At each step, filter available nodes by checking their current degree in the induced subgraph.
            # Induced subgraph degree of u = count of v in available such that v in adj[u].
            
            # Let's restart the greedy with a cleaner implementation.
            pass

    def greedy_mis_v2(adj, n):
        available = set(range(n))
        in_set = []
        
        while available:
            # Find node with min degree in induced subgraph
            min_deg = float('inf')
            min_node = -1
            
            for u in available:
                # Count available neighbors
                deg = 0
                for v in adj[u]:
                    if v in available:
                        deg += 1
                
                if deg < min_deg:
                    min_deg = deg
                    min_node = u
            
            in_set.append(min_node)
            available.remove(min_node)
            
            # Remove neighbors of min_node
            neighbors_to_remove = []
            for v in adj[min_node]:
                if v in available:
                    neighbors_to_remove.append(v)
            
            for v in neighbors_to_remove:
                available.remove(v)
                
        return in_set

    # Run greedy_mis_v2
    # This is O(N^3) worst case, but N=1320.
    # N * (N * AvgDeg). AvgDeg ~ 50.
    # 1320 * 1320 * 50 ~ 87 million ops. Might be slow in Python.
    
    # Optimization: Use numpy or precomputed bitmasks?
    # Or just a faster greedy.
    
    # Let's try a simpler greedy: Random order, pick if no conflict with picked.
    # This is O(N^2) but fast constant.
    
    def greedy_random(adj, n, rng):
        perm = rng.permutation(n)
        in_set = []
        picked = [False] * n
        
        for u in perm:
            if picked[u]:
                continue
            # Check if u conflicts with any picked
            conflict = False
            for v in in_set:
                if v in adj[u]:
                    conflict = True
                    break
            if not conflict:
                in_set.append(u)
                picked[u] = True
        
        return in_set

    # Let's run a few random greeds
    rng = np.random.default_rng(42)
    
    best_R3_set = []
    best_R3_size = 0
    
    # Try a few restarts
    for i in range(50):
        if time.time() > deadline:
            break
        res = greedy_random(adj, n_R3, rng)
        if len(res) > best_R3_size:
            best_R3_size = len(res)
            best_R3_set = res[:]
            
    if best_R3_size > best_size:
        # Convert indices back to vectors
        final_vecs = [vecs_R3[i] for i in best_R3_set]
        return np.array(final_vecs, dtype=np.int64)
    else:
        return np.array(best_set, dtype=np.int64)
