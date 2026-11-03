import numpy as np
import random
import itertools
import time

def construct():
    D = 11
    # Generate all weight-4 vectors
    # Index mapping: (support_tuple, sign_tuple) -> index
    supports = list(itertools.combinations(range(D), 4))
    n_sup = len(supports)
    
    # Total vectors = n_sup * 2^4 = 330 * 16 = 5280? 
    # Wait. 2^4 = 16 signs.
    # Total vectors = 330 * 16 = 5280.
    # My previous analysis said 660. Why?
    # "weight-4 vector conflict graph".
    # Ah, the current program only uses ONE vector per support?
    # "for signs in itertools.product([1, -1], repeat=4): ... vectors.append(v)"
    # It adds ALL sign combinations.
    # So N = 5280.
    
    # Let's re-verify the conflict condition for N=5280.
    # Norm squared = 4.
    # Conflict if dot > 2.
    # If supports are same:
    #   Same signs: dot = 4. Conflict.
    #   3 signs same, 1 diff: dot = 2. OK.
    #   2 signs same, 2 diff: dot = 0. OK.
    #   1 sign same, 3 diff: dot = -2. OK.
    #   0 signs same, 4 diff: dot = -4. OK.
    #   So on same support, only the identical vector conflicts?
    #   Wait, if $u=v$, dot=4. Conflict.
    #   If $u=-v$, dot=-4. OK.
    #   If $u, v$ differ in 1 sign, dot=2. OK.
    #   So on same support, $u$ conflicts ONLY with itself?
    #   No, "minimum pairwise distance". If $u=v$, dist=0.
    #   If $u \neq v$, dist > 0.
    #   So we just can't pick the same vector twice.
    
    # If supports intersect in 3:
    #   Let $S_1 \cap S_2 = I, |I|=3$.
    #   $u$ has 1 unique, 3 shared.
    #   $v$ has 1 unique, 3 shared.
    #   Dot = sum of products on $I$.
    #   Max dot = 3 (all signs match). Conflict.
    #   If 2 match, 1 diff: dot = 1. OK.
    #   If 1 match, 2 diff: dot = -1. OK.
    #   If 0 match, 3 diff: dot = -3. OK.
    #   So conflict if signs match on all 3 shared indices.
    
    # If supports intersect in 2:
    #   Max dot = 2. OK.
    
    # If supports intersect in 1 or 0:
    #   Dot <= 1 or 0. OK.
    
    # So conflicts are:
    # 1. $u = v$ (Identity)
    # 2. $|S_u \cap S_v| = 3$ AND signs match on the intersection.
    
    # Number of nodes N = 5280.
    # This is larger than I thought.
    # Current program score 2.5455.
    # If seed is 100, k=254.
    # Max kissing number is 265.
    # So we are very close to the theoretical max.
    
    # The current program builds adjacency for supports with intersection 3.
    # It does NOT distinguish signs in the adjacency list construction?
    # `if len(s_i.intersection(s_j)) == 3:`
    # It treats all sign combinations of $i$ as conflicting with all sign combinations of $j$?
    # NO. The current program creates `vectors` from `best_set` of supports.
    # `for idx in best_set: ... for signs in ...: vectors.append(v)`
    # This means if a support is in `best_set`, it adds ALL 16 sign variations.
    # This is WRONG for the conflict graph if different signs don't conflict.
    # If support $S$ is in the set, it adds 16 vectors.
    # Do these 16 vectors conflict with each other?
    # As analyzed above, on the same support, only $u=v$ conflicts.
    # So we can pick all 16 sign variations of a support, EXCEPT we can't pick the same vector twice (which we don't, we pick distinct sign tuples).
    # So if we pick support $S$, we get 16 valid vectors that are mutually compatible (dist >= norm).
    # Wait, is dist >= norm for $u, v$ on same support with 1 sign flip?
    # $u \cdot v = 2$.
    # $d^2 = 4+4-2(2) = 4$. $d=2$.
    # Norm = 2.
    # $2 \ge 2$. Yes.
    
    # So, if we select a set of supports $\mathcal{S}$, and for each support $S \in \mathcal{S}$ we include ALL 16 sign variations, are they all mutually compatible with vectors from other supports?
    # Let $S_1, S_2 \in \mathcal{S}$.
    # We need ALL $u \in V(S_1)$ and $v \in V(S_2)$ to be compatible.
    # Conflict condition: $|S_1 \cap S_2|=3$ AND signs match on intersection.
    # If $|S_1 \cap S_2|=3$, there exist $u, v$ that match signs on intersection.
    # Since we include ALL sign variations, we WILL include the conflicting pair.
    # Therefore, we CANNOT include both supports if they intersect in 3.
    # This matches the current program's logic: Independent Set on the graph of supports where edge means intersection 3.
    
    # So the problem reduces to:
    # Find Max Independent Set of the graph $G_{sup}$ where nodes are supports (330 nodes) and edge if intersection is 3.
    # Score $k = 16 \times |IS|$.
    
    # Current score 2.5455.
    # If seed is 100, $k \approx 254 \implies |IS| \approx 15.8$.
    # Max IS size?
    # If max kissing number is 265, and we use weight-4 vectors...
    # Is it possible to get 265 with weight-4?
    # $265 / 16 \approx 16.5$.
    # So we need an independent set of size 16 or 17.
    # If we get 16, $k=256$. Score $256/100 = 2.56$.
    # Current is 2.5455. So current $k \approx 254$? $254/16 = 15.875$. Not integer.
    # Maybe the seed is different.
    # Or maybe the current program doesn't use all 16 signs?
    # `for signs in itertools.product([1, -1], repeat=4):` -> Yes, it does.
    
    # So the task is to find a Max Independent Set of size ~16-17 in the 330-node graph.
    
    # I will implement a fast SA/QUBO solver for this 330-node graph.
    
    # 1. Build graph of 330 supports.
    # 2. Run SQA.
    # 3. Return 16 * |IS| vectors.

    supports_list = list(itertools.combinations(range(D), 4))
    n_sup = len(supports_list)
    
    # Build adjacency
    adj = [set() for _ in range(n_sup)]
    for i in range(n_sup):
        s_i = set(supports_list[i])
        for j in range(i+1, n_sup):
            s_j = set(supports_list[j])
            if len(s_i.intersection(s_j)) == 3:
                adj[i].add(j)
                adj[j].add(i)
                
    # Convert to list of lists for faster iteration
    adj_list = [list(s) for s in adj]
    
    # SQA Solver
    def solve_sqa(adj_list, n_nodes, time_limit=110.0):
        start_time = time.time()
        
        # Greedy Init
        # Pick node with min degree, remove it and its neighbors.
        def greedy_init():
            remaining = set(range(n_nodes))
            selected = set()
            while remaining:
                # Find min degree in remaining
                min_deg = float('inf')
                best_v = None
                for v in remaining:
                    deg = len(adj_list[v].intersection(remaining))
                    if deg < min_deg:
                        min_deg = deg
                        best_v = v
                        if deg == 0: break
                selected.add(best_v)
                to_remove = {best_v}.union(adj_list[best_v].intersection(remaining))
                remaining.difference_update(to_remove)
            return selected

        S = greedy_init()
        best_S = S.copy()
        best_size = len(S)
        
        # State: x (boolean array)
        x = np.zeros(n_nodes, dtype=bool)
        for i in S:
            x[i] = True
            
        # Active neighbor counts
        # active_neighbors[i] = sum(x[j] for j in adj[i])
        active_neighbors = np.zeros(n_nodes, dtype=int)
        for i in range(n_nodes):
            if x[i]:
                for j in adj_list[i]:
                    active_neighbors[j] += 1
                    
        # SA Parameters
        T = 1.0
        cooling = 0.9995
        gamma = 1.0 # Transverse field
        
        # Precompute neighbors as arrays for speed?
        # adj_list is list of lists.
        
        while time.time() - start_time < time_limit:
            # Pick random node
            v = random.randint(0, n_nodes - 1)
            
            if x[v]:
                # Try to flip to 0
                # Delta E = 1 - M * active_neighbors[v]
                # We want to minimize E = -sum(x) + M * sum(conflicts)
                # Conflict count = sum_{i<j, adj} x_i x_j
                # If we flip v from 1 to 0:
                # sum(x) decreases by 1 -> E increases by 1.
                # conflicts involving v decrease by active_neighbors[v].
                # E decreases by M * active_neighbors[v].
                # Delta E = 1 - M * active_neighbors[v].
                # If active_neighbors[v] > 0, Delta E is negative (good).
                # If active_neighbors[v] == 0, Delta E is positive (bad, unless we want to remove isolated nodes? No, keep them).
                # Actually, if active_neighbors[v] == 0, removing v increases energy (worse).
                # We should only remove if it allows adding more later?
                # In Max IS, we generally don't want to remove isolated nodes.
                # But in SA, we explore.
                
                delta_E = 1 - 10 * active_neighbors[v] # M=10 is enough to prevent conflicts
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    x[v] = False
                    # Update active_neighbors for neighbors
                    for u in adj_list[v]:
                        active_neighbors[u] -= 1
            else:
                # Try to flip to 1
                # Delta E = -1 + M * active_neighbors[v]
                # If active_neighbors[v] == 0, Delta E = -1 (good).
                # If active_neighbors[v] > 0, Delta E > 0 (bad, creates conflict).
                
                delta_E = -1 + 10 * active_neighbors[v]
                
                if delta_E < 0 or random.random() < np.exp(-delta_E / T):
                    # Accept
                    x[v] = True
                    # Update active_neighbors for neighbors
                    for u in adj_list[v]:
                        active_neighbors[u] += 1
            
            # Check current size
            current_size = np.sum(x)
            if current_size > best_size:
                best_size = current_size
                best_S = np.where(x)[0].tolist()
                
            T *= cooling
            gamma *= 0.999
            
            # If T is very small, break or restart?
            if T < 0.01:
                # Restart with best_S
                x = np.zeros(n_nodes, dtype=bool)
                for i in best_S:
                    x[i] = True
                active_neighbors = np.zeros(n_nodes, dtype=int)
                for i in range(n_nodes):
                    if x[i]:
                        for j in adj_list[i]:
                            active_neighbors[j] += 1
                T = 1.0
                gamma = 1.0
                # Randomize slightly?
                # Just restart SA.
                
        return best_S

    best_supports = solve_sqa(adj_list, n_sup)
    
    vectors = []
    for idx in best_supports:
        sup = supports_list[idx]
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for k, s in enumerate(sup):
                v[s] = signs[k]
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
