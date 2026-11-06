import time
import random
import sys

def construct():
    start_time = time.time()
    # Leave margin for evaluator overhead
    deadline = start_time + 115.0

    def find_K4(n, adj):
        """
        Finds a K4 in the graph. Returns a list of 4 vertices or None.
        adj is a list of bitmasks (integers).
        """
        for i in range(n):
            # Neighbors of i with index > i
            # We need to check edges (i, j) where j > i
            # Then common neighbors k, l > j
            # Actually, standard K4 detection:
            # For each pair (i, j) with i < j and edge (i, j):
            #   Find common neighbors k, l with j < k < l
            #   If edge (k, l) exists, then {i, j, k, l} is K4.
            
            # Optimization: iterate i
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    continue
                
                # Common neighbors of i and j with index > j
                common_mask = adj[i] & adj[j]
                # Mask out bits <= j
                common_mask &= ~((1 << (j + 1)) - 1)
                
                # Get list of common neighbors
                # If popcount < 2, no K4 containing edge (i, j) with other two vertices > j
                if (common_mask & (common_mask - 1)) == 0:
                    # 0 or 1 bit set
                    continue
                
                # Extract vertices
                common_verts = []
                m = common_mask
                while m:
                    # get lowest set bit
                    lsb = m & (-m)
                    v = lsb.bit_length() - 1
                    common_verts.append(v)
                    m ^= lsb
                
                # Check pairs in common_verts
                # Since we want efficiency, we can just check if any pair is an edge.
                # common_verts size is usually small in sparse graphs, but could be larger.
                # n=35, max common neighbors is 33. 33*32/2 = 528 checks.
                # This is fast.
                for a_idx in range(len(common_verts)):
                    u = common_verts[a_idx]
                    for b_idx in range(a_idx + 1, len(common_verts)):
                        v = common_verts[b_idx]
                        if (adj[u] >> v) & 1:
                            return [i, j, u, v]
        return None

    def max_independent_set_size(n, adj):
        """
        Computes the size of the maximum independent set in the graph.
        Uses bitmask backtracking with pruning.
        Returns the size (int).
        """
        # adj is list of bitmasks where adj[i] has bit j set if edge (i, j) exists.
        # Independent set: set of vertices with no edges between them.
        # In complement graph, this is a clique.
        # But we work directly on adj.
        
        # We use a recursive function that maintains:
        # candidates: bitmask of vertices available to be added
        # current_size: size of current independent set
        
        # Pruning:
        # 1. If current_size + popcount(candidates) <= best_found, prune.
        # 2. Greedy coloring bound on candidates? 
        #    For independent set in G, it's clique in G_bar.
        #    Coloring G_bar gives an upper bound on clique size.
        #    However, computing coloring every step is expensive.
        #    Simple bound: popcount(candidates) is weak.
        
        # Let's use a simpler but effective backtracking for n<=40.
        # The number of independent sets of size 6 is small if alpha < 6.
        # If we just want to check if alpha >= 6, we can stop early.
        
        best_size = 0
        
        def backtrack(candidates_mask, current_size, current_set_mask):
            nonlocal best_size
            
            if current_size > best_size:
                best_size = current_size
                
            if current_size >= 6:
                return True # Found independent set of size 6, can stop if we only care about validity
                # But for local search we might want exact size? 
                # Actually, for the task, we just need to know if alpha >= 6.
                # If we find one, we return True.
                
            if candidates_mask == 0:
                return False
                
            # Heuristic: pick a vertex
            # Try to pick a vertex with high degree in the subgraph? 
            # Or just pick the first one?
            # To find large independent sets, we should pick vertices that are "isolated" in the subgraph?
            # No, we want to add vertices that are NOT connected to each other.
            
            # Let's iterate through candidates.
            # Optimization: if popcount(candidates) + current_size <= best_size, return.
            if current_size + candidates_mask.bit_count() <= best_size:
                return False
                
            # Get list of candidate vertices
            verts = []
            m = candidates_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
            
            # Sort by degree in the induced subgraph? 
            # High degree in G means low degree in G_bar (complement), so less likely to be in large independent set?
            # Actually, in G, if v has many neighbors in candidates, it can't be with them.
            # So picking a vertex with FEW neighbors in candidates is good for building the set?
            # Yes.
            
            # Calculate degree in candidates for each v
            # deg(v) = popcount(adj[v] & candidates_mask)
            
            # Sort by degree ascending
            verts.sort(key=lambda v: (adj[v] & candidates_mask).bit_count())
            
            for v in verts:
                # If we pick v, new candidates are those in candidates_mask that are NOT adjacent to v
                # and have index > v? 
                # To avoid duplicates, we enforce an ordering. 
                # Standard Bron-Kerbosch style or simple subset enumeration.
                # Since we want independent set, we can just remove v and its neighbors from candidates.
                # And to avoid permutations, we only consider vertices with index > v? 
                # No, that's for cliques in a specific order. 
                # For independent sets, we can just recurse on (candidates_mask without v and neighbors of v).
                # But we must ensure we don't count the same set multiple times.
                # If we iterate through all vertices in candidates, we will generate the same set in different orders.
                # Solution: enforce that we only pick vertices from a specific subset.
                # Let's use the standard technique: 
                # Pick a pivot? Or just iterate and remove v from consideration for subsequent steps in this branch?
                
                # Simple approach: 
                # new_candidates = candidates_mask & ~adj[v] & ~(1 << v)
                # But this allows picking u < v in the next step, leading to duplicates.
                # To fix, we can maintain a "start" index or mask.
                # Let's use the method: 
                # Iterate v in verts.
                # new_mask = candidates_mask & ~((1 << v) | adj[v])
                # But we must ensure we don't pick vertices already "passed" or lower index?
                # Actually, if we just remove v and its neighbors, and recurse, we might pick u (u < v) in the next step if u was in candidates.
                # To prevent this, we can restrict the candidate set to vertices with index > v? 
                # No, that's for finding a specific clique.
                
                # Let's use a different strategy: 
                # We want to find if there is an independent set of size 6.
                # We can use a simple backtracking that picks vertices in increasing order.
                # candidates is a bitmask. 
                # We iterate v in increasing order.
                # If we pick v, the next candidates must be > v.
                
                # This is efficient and avoids duplicates.
                
                # But wait, if we restrict to > v, we might miss sets where a vertex u < v is included?
                # No, because we iterate v in increasing order. If u < v, it was considered in previous iterations of the loop (or current one if we are at u).
                # If we picked u earlier, we wouldn't be at v.
                # If we didn't pick u, we are at v. The set {v, ...} is distinct from {u, ...}.
                # So restricting to > v is correct for generating unique sets in increasing order.
                
                # new_candidates = candidates_mask & ~((1 << v) | adj[v]) & ~((1 << (v + 1)) - 1) 
                # Wait, we want vertices > v.
                # mask_gt_v = candidates_mask & ~((1 << (v + 1)) - 1)
                # new_candidates = mask_gt_v & ~adj[v]
                
                # However, we also need to ensure that the vertices in new_candidates are not adjacent to v.
                # Yes, ~adj[v] handles that.
                
                # But what about vertices < v? They are excluded by mask_gt_v.
                
                # This approach works.
                
                # Let's refine the pruning.
                # If we pick v, the max additional size is bounded by the independence number of the subgraph induced by new_candidates.
                # Upper bound: popcount(new_candidates).
                
                if current_size + 1 + (new_candidates := (candidates_mask & ~((1 << (v + 1)) - 1) & ~adj[v])).bit_count() <= best_size:
                    continue
                
                if backtrack(new_candidates, current_size + 1, current_set_mask | (1 << v)):
                    return True
                    
            return False
        
        # Initial call
        # candidates = all vertices
        # We want to find if size >= 6.
        # So we can set a flag or just let it run.
        # To speed up, we can stop as soon as we find one of size 6.
        
        # Let's modify backtrack to return True if it finds a set of size >= target.
        # We'll pass a target size.
        
        def backtrack_target(candidates_mask, current_size, target):
            if current_size >= target:
                return True
                
            if candidates_mask == 0:
                return False
                
            verts = []
            m = candidates_mask
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                verts.append(v)
                m ^= lsb
                
            # Sort by degree in candidates (ascending) to find independent sets faster?
            # Vertices with fewer neighbors in candidates are better candidates for independent set.
            verts.sort(key=lambda v: (adj[v] & candidates_mask).bit_count())
            
            for v in verts:
                # Mask for vertices > v
                mask_gt_v = candidates_mask & ~((1 << (v + 1)) - 1)
                # Remove v and its neighbors
                new_candidates = mask_gt_v & ~adj[v]
                
                # Pruning: if even taking all remaining vertices doesn't reach target
                if current_size + 1 + new_candidates.bit_count() < target:
                    continue
                
                if backtrack_target(new_candidates, current_size + 1, target):
                    return True
                    
            return False
        
        return backtrack_target((1 << n) - 1, 0, 6)

    def has_independent_set_6(n, adj):
        """
        Returns True if there is an independent set of size 6.
        """
        return max_independent_set_size(n, adj) >= 6
        # Actually, the function above returns True if found.
        # Let's just use the boolean return.

    def is_valid(n, adj):
        """
        Checks if graph is K4-free and has no independent set of size 6.
        """
        if find_K4(n, adj):
            return False
        if max_independent_set_size(n, adj) >= 6:
            return False
        return True

    def local_search(n, max_time):
        """
        Attempts to construct a valid graph on n vertices.
        """
        # Initialize random graph
        # For n=35, p ~ 0.5 is common for Ramsey graphs.
        # But K4-free suggests lower density? 
        # Turan graph T(35, 3) has density ~ 1 - 1/3 = 0.66.
        # But we need alpha < 6.
        # Let's start with p=0.5.
        
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        current_time = time.time()
        steps = 0
        max_steps = 20000 # Safety cap
        
        # We will alternate between fixing K4s and fixing Independent Sets.
        # Or we can check validity, and if invalid, fix one violation.
        
        while current_time < max_time and steps < max_steps:
            # Check K4
            k4 = find_K4(n, adj)
            if k4:
                # Remove a random edge from K4
                pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
                u, v = pairs[random.randrange(6)]
                u, v = k4[u], k4[v]
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                steps += 1
                current_time = time.time()
                continue
            
            # Check Independent Set 6
            # This is the expensive part.
            # We need to find one if it exists.
            # My max_independent_set_size returns boolean for target 6.
            # But to fix it, we need the set.
            # Let's modify the search to return the set.
            
            def find_indep_6_set(n, adj):
                # Similar to max_independent_set_size but returns the set
                def backtrack(candidates_mask, current_set):
                    if len(current_set) == 6:
                        return current_set[:]
                    
                    if candidates_mask == 0:
                        return None
                    
                    verts = []
                    m = candidates_mask
                    while m:
                        lsb = m & (-m)
                        v = lsb.bit_length() - 1
                        verts.append(v)
                        m ^= lsb
                        
                    verts.sort(key=lambda v: (adj[v] & candidates_mask).bit_count())
                    
                    for v in verts:
                        mask_gt_v = candidates_mask & ~((1 << (v + 1)) - 1)
                        new_candidates = mask_gt_v & ~adj[v]
                        
                        # Pruning
                        if len(current_set) + 1 + new_candidates.bit_count() < 6:
                            continue
                        
                        res = backtrack(new_candidates, current_set + [v])
                        if res:
                            return res
                    return None
                
                return backtrack((1 << n) - 1, [])
            
            indep6 = find_indep_6_set(n, adj)
            if indep6:
                # Add a random edge within the independent set
                pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
                u_idx, v_idx = pairs[random.randrange(15)]
                u, v = indep6[u_idx], indep6[v_idx]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                steps += 1
                current_time = time.time()
                continue
            
            # If neither K4 nor Indep 6 found, valid
            # Convert adj to edges list
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (adj[i] >> j) & 1:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        return None

    # Try to construct graphs for n = 35 down to 17
    best_result = None
    
    # We prioritize n=35. If we fail, we try 34, etc.
    # But we should also try to improve n=35 with more time if possible?
    # The loop below tries n=35 first.
    
    for n in range(35, 16, -1):
        if time.time() > deadline:
            break
            
        # Allocate time for this n
        # If n=35, give it most of the time.
        # If n=34, give it remaining time, etc.
        
        # Simple strategy: try n=35 for 90s. If fail, try n=34 for remaining.
        # But we don't know if we'll succeed quickly.
        
        # Let's just run local_search with the global deadline.
        # local_search will return as soon as it finds a valid graph.
        
        result = local_search(n, deadline)
        if result:
            best_result = result
            # If we found a valid graph for n, we can return it immediately?
            # The task says "Maximize n".
            # So if we find n=35, we return it.
            # If we find n=34, we should try to see if we can get n=35?
            # No, we iterate n downwards. So we try 35 first.
            # If we succeed at 35, we return.
            # If we fail at 35 (timeout), we try 34.
            # So we should NOT break on first success if n < 35?
            # Wait, the loop is `for n in range(35, 16, -1)`.
            # So it tries 35, then 34, etc.
            # If local_search(35) returns None (timeout), we go to 34.
            # If local_search(35) returns a result, we should return it.
            # If local_search(34) returns a result, we should return it?
            # No, we want the MAX n.
            # So if we fail 35, and succeed 34, we return 34.
            # But what if we could have succeeded 35 with more time?
            # We gave it the whole deadline.
            
            # So: if result is not None, return it.
            return result

    # Fallback: Paley 17
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

# Ensure random seed is set for reproducibility? 
# The task doesn't require it, but it's good practice.
# random.seed(42) # Uncomment if needed

if __name__ == "__main__":
    # For local testing
    res = construct()
    print(f"n={res['n']}, edges={len(res['edges'])}")
