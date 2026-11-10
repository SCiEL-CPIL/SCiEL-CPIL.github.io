import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_bits(x):
        return bin(x).count("1")

    def local_search_partition(adj, n, seed, time_limit):
        # Initialize with all edges as cliques of size 2
        # We represent the partition as a list of lists of vertices
        # To make it efficient, we maintain a set of active edges and a mapping from edge to clique_id
        
        # 1. Build initial partition: every edge is its own clique
        #    cliques is a list of tuples/sets. 
        #    For fast lookup, we need to know which clique an edge belongs to.
        #    Since we start with size 2, edge (u,v) is in clique i.
        
        edges = []
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    edges.append((u, v))
                mask ^= lsb
        
        if not edges:
            return []
        
        # Initial state: each edge is a clique
        # cliques: list of sets of vertices
        # edge_to_clique: dict from (u,v) to index in cliques
        # clique_edges: list of sets of edges for each clique
        
        cliques = []
        edge_to_clique = {}
        
        for i, (u, v) in enumerate(edges):
            cliques.append({u, v})
            edge_to_clique[(u, v)] = i
            edge_to_clique[(v, u)] = i
            
        # We need to efficiently find merge opportunities.
        # A merge is possible if two edges (a,b) and (b,c) share a vertex 'b' 
        # and the third edge (a,c) exists in the graph AND belongs to the SAME clique? 
        # No, the edges must be in the SAME clique to be merged into a larger clique?
        # Wait. The definition of a clique partition:
        # "Every listed set must be a clique... and every edge must lie in exactly one listed clique."
        # This means the cliques are disjoint in terms of edges.
        # If I have clique {a,b} and clique {b,c}, they share vertex b but no edges.
        # Can I merge them into {a,b,c}?
        # Only if (a,c) is an edge in the graph AND (a,c) is currently in the SAME clique as (a,b) and (b,c)?
        # No. If (a,c) is in a different clique, say {a,c}, then merging {a,b}, {b,c}, {a,c} into {a,b,c}
        # requires that all three edges are present in the partition.
        
        # So the operation is:
        # Find a set of edges in the partition that form a complete graph K_k in the original graph.
        # If they do, merge them into a single clique.
        
        # This is essentially finding cliques in the "line graph" of the partition?
        # Or simpler: 
        # Start with single edges.
        # Look for triangles in the original graph where all 3 edges are present in the partition.
        # If found, merge the 3 cliques into 1.
        # Then look for K4s where all 6 edges are present, etc.
        
        # Since we start with size 2, we only need to look for triangles (K3) to grow to size 3.
        # Once we have size 3, we can look for K4s (adding a vertex connected to all 3).
        
        # Algorithm:
        # 1. Maintain list of cliques.
        # 2. Maintain for each vertex, the list of cliques it belongs to.
        # 3. Iterate:
        #    - Pick a random vertex v.
        #    - Look at cliques containing v.
        #    - Try to merge them with other cliques to form a larger clique.
        
        # Efficient check for merge:
        # If we have clique C = {v1, ..., vk} and we want to add vertex u:
        # - u must be connected to all vi in the original graph.
        # - The edges (u, vi) must exist in the partition.
        # - If they exist, they are currently in some cliques C_1, ..., C_k.
        # - If we merge C, C_1, ..., C_k into {v1, ..., vk, u}, we must ensure no other edges are lost or duplicated.
        # - Actually, if (u, vi) is in C_i, and C_i is just {u, vi} (size 2), then merging is safe.
        # - If C_i is larger, say {u, vi, x}, then x must also be in the new clique?
        #   No, the edge (u, vi) is in C_i. If we take edge (u, vi) out of C_i, C_i loses that edge.
        #   Does C_i remain a valid clique? Yes, removing an edge from a clique doesn't break the clique property for the remaining vertices, 
        #   BUT the partition property requires that *every* edge in the original graph is covered exactly once.
        #   If we move edge (u, vi) to the new big clique, it's still covered.
        #   However, the other edges in C_i (e.g., (vi, x)) must still be covered. They are in C_i.
        #   So C_i becomes {vi, x, ...} minus u? No, C_i is a set of vertices. 
        #   If C_i = {u, vi, x}, the edges are (u,vi), (u,x), (vi,x).
        #   If we move (u,vi) to new clique, C_i still has (u,x) and (vi,x). 
        #   But (u,x) and (vi,x) share x. They are in C_i.
        #   So C_i remains a valid clique {u, vi, x}?? 
        #   Wait, if I remove the edge (u,vi) from the "coverage" of C_i, does C_i still "own" the vertices?
        #   The definition says "Every listed set must be a clique... every edge must lie in exactly one listed clique".
        #   It does NOT say the cliques must be vertex-disjoint.
        #   So if C_i = {u, vi, x}, and I move edge (u,vi) to another clique, C_i still lists vertices {u, vi, x}.
        #   Is {u, vi, x} a clique? Yes, in the original graph.
        #   Does it "contain" the edge (u,vi)? In the sense of the partition, no, that edge is assigned to the other clique.
        #   But the set of vertices is still a clique.
        #   So, we can split the edges of a clique among multiple cliques that share the same vertex set?
        #   No, that would be inefficient and likely not what "minimize number of cliques" intends.
        #   Usually, a clique partition implies the cliques are the sets, and the edges are partitioned.
        #   If two cliques have the same vertex set, they are redundant.
        #   So we should assume that if we merge edges into a new clique, we remove those edges from the old cliques.
        #   If an old clique becomes empty of edges, we remove it.
        #   If an old clique still has edges, it remains.
        
        # This suggests a complex state.
        # Let's simplify: 
        # We will only perform "Clean Merges".
        # A Clean Merge: We take a set of edges E' that form a clique K_m in the original graph.
        # We require that all edges in E' are currently in cliques of size 2 (just themselves).
        # We merge these size-2 cliques into one size-m clique.
        # This is safe and simple.
        # This allows us to build up from edges to triangles to K4s (if we can find 4 edges forming a K4 that are all size-2? No, K4 has 6 edges. 
        # If we merge 3 edges into a triangle, the other 3 edges are still size 2.
        # To form a K4, we need to merge the triangle (size 3) with the other 3 edges?
        # If we allow merging a size-3 clique with size-2 cliques, it's more powerful.
        
        # Let's implement a generic merge:
        # Candidate: A set of vertices S.
        # Check:
        # 1. S is a clique in original graph.
        # 2. For every edge (u,v) in S, find the clique C_uv containing it.
        # 3. If we merge all C_uv into a single clique S, do we break anything?
        #    - The new clique S is valid (it's a clique).
        #    - The edges are now covered by S.
        #    - The old cliques C_uv lose those edges.
        #    - If C_uv had other edges, it remains with those edges.
        #    - If C_uv had only that edge, it disappears.
        #    - This is always valid!
        
        # So the strategy is:
        # 1. Start with all edges as size-2 cliques.
        # 2. Repeatedly find a set of vertices S (size >= 3) such that:
        #    a. S is a clique in the graph.
        #    b. The number of cliques involved in covering the edges of S is > 1.
        #    c. Merging them reduces the total clique count.
        # 3. Perform the merge.
        
        # How to find S?
        # Start with a random edge (u,v) in clique C.
        # Try to add a vertex w to form {u,v,w}.
        # Check if {u,v,w} is a clique.
        # Check edges (u,w) and (v,w).
        # If they exist in the graph, find their cliques C_uw and C_vw.
        # If {C, C_uw, C_vw} are not all the same, we can merge.
        # New clique {u,v,w}.
        # Remove edges (u,v), (u,w), (v,w) from old cliques.
        
        # This is a local search step.
        
        # Data structures:
        # cliques: list of sets of vertices
        # edge_to_clique: dict (u,v) -> index
        # For fast removal, we need to update edge_to_clique for all edges in the merged cliques.
        
        start_time = time.time()
        max_iter = 100000
        
        # Optimization: Precompute neighbor masks
        # adj is already bitmasks.
        
        # Helper: remove edge from clique
        # We don't explicitly store edges in cliques, we derive them from vertices.
        # But we need to know which edges are "assigned" to which clique.
        # The assignment is defined by edge_to_clique.
        
        # When we merge cliques C1, C2, ... into C_new:
        # 1. Identify all edges in the union of vertices of C1, C2... that are in the original graph.
        #    Actually, we only care about the edges that are being moved.
        #    The edges of the new clique S are all pairs in S.
        #    For each edge (u,v) in S:
        #        old_idx = edge_to_clique[(u,v)]
        #        edge_to_clique[(u,v)] = new_idx
        #        edge_to_clique[(v,u)] = new_idx
        #        # Remove (u,v) from old clique's "logical" edge set?
        #        # We don't store edge sets. We store vertex sets.
        #        # But if we remove an edge from a clique, does the vertex set change?
        #        # No, the vertex set remains the same.
        #        # However, if a clique loses ALL its edges, it should be removed.
        #        # How to check if a clique has 0 edges?
        #        # A clique with vertices V has edges if there is at least one pair in V 
        #        # such that edge_to_clique[(u,v)] == idx.
        #        # This is expensive to check every time.
        
        # Alternative: Store cliques as sets of edges?
        # If cliques are sets of edges, then "is a clique" check is harder.
        # Let's stick to vertex sets, but maintain a count of edges per clique.
        
        cliques = [] # list of sets of vertices
        edge_to_clique = {}
        edge_count_in_clique = [] # list of ints
        
        for i, (u, v) in enumerate(edges):
            cliques.append({u, v})
            edge_to_clique[(u, v)] = i
            edge_to_clique[(v, u)] = i
            edge_count_in_clique.append(1)
            
        # To speed up, we can keep a list of "active" clique indices.
        # But removing from a list is O(N).
        # We can use a "dead" flag.
        
        dead = [False] * len(cliques)
        
        def try_merge(vertices):
            # vertices is a set of vertices forming a candidate clique
            if len(vertices) < 3:
                return False
            
            # Check if it's a clique in original graph
            v_list = list(vertices)
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    if not (adj[u] & (1 << v)):
                        return False
            
            # Find all cliques that contain any edge of this candidate clique
            involved_cliques = set()
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    idx = edge_to_clique.get((u, v))
                    if idx is not None and not dead[idx]:
                        involved_cliques.add(idx)
            
            if len(involved_cliques) < 2:
                return False # No merge possible or already merged
            
            # Perform merge
            new_clique_set = set(v_list)
            
            # Remove edges from old cliques
            for idx in involved_cliques:
                for i in range(len(v_list)):
                    for j in range(i+1, len(v_list)):
                        u, v = v_list[i], v_list[j]
                        if edge_to_clique.get((u, v)) == idx:
                            edge_count_in_clique[idx] -= 1
                            if edge_count_in_clique[idx] == 0:
                                dead[idx] = True
                                # Optionally clean up, but we can just leave it dead
                            
            # Add new clique
            new_idx = len(cliques)
            cliques.append(new_clique_set)
            edge_count_in_clique.append(len(v_list) * (len(v_list)-1) // 2)
            
            # Update edge_to_clique for all edges in new clique
            for i in range(len(v_list)):
                for j in range(i+1, len(v_list)):
                    u, v = v_list[i], v_list[j]
                    edge_to_clique[(u, v)] = new_idx
                    edge_to_clique[(v, u)] = new_idx
                    
            return True
            
        # Main loop
        # We need to find candidates.
        # Strategy: Pick a random edge (u,v) from a non-dead clique.
        # Try to extend it to a triangle (u,v,w).
        # If successful, try to extend to K4, etc.
        
        # To pick a random edge, we can pick a random non-dead clique and a random edge from it.
        # Or just pick a random edge from the global list, check if its clique is dead.
        
        # Let's pick random edges from the global list.
        # If the clique is dead, skip.
        
        # We need a list of non-dead clique indices to pick from?
        # Or just iterate and pick.
        
        # Better: Maintain a list of "active edges" (edges in non-dead cliques).
        # This is hard to update.
        
        # Let's just pick a random edge from the full edge list.
        # If it's in a dead clique, skip.
        
        active_cliques_count = len(cliques)
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random edge
            # To avoid bias, pick from a random non-dead clique?
            # Let's pick a random index in the current cliques list.
            # If dead, pick another.
            
            # This is inefficient if many are dead.
            # Let's maintain a list of active clique indices.
            
            # Rebuild active list occasionally?
            # Or just use random.choice on a filtered list?
            
            # Let's use a simpler approach:
            # Pick a random vertex u.
            # Pick a random neighbor v of u (in the original graph).
            # Check if edge (u,v) is in a non-dead clique.
            
            u = random.randint(0, n-1)
            mask = adj[u]
            if mask == 0: continue
            v_bit = (mask >> random.randint(0, n-1)) & 1 # This doesn't work well for sparse
            # Better:
            bits = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                if v > u: # Ensure unique edge representation
                    bits.append(v)
                m ^= lsb
            
            if not bits: continue
            v = random.choice(bits)
            
            idx = edge_to_clique.get((u, v))
            if idx is None or dead[idx]:
                continue
                
            # We have a valid edge (u,v) in clique idx.
            # Try to find a vertex w such that {u,v,w} is a clique.
            # w must be a common neighbor of u and v.
            common = adj[u] & adj[v]
            # Remove u and v from common
            common &= ~(1 << u)
            common &= ~(1 << v)
            
            if common == 0:
                continue
                
            # Pick a random w from common
            w_bits = []
            m = common
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                w_bits.append(w)
                m ^= lsb
                
            if not w_bits: continue
            w = random.choice(w_bits)
            
            # Check if {u,v,w} can be merged.
            # It's a clique in the graph.
            # We need to check if the edges (u,w) and (v,w) are in cliques.
            # They must be, because they are in the graph and the partition covers all edges.
            
            # Try merge
            if try_merge({u, v, w}):
                # Success
                # Optionally try to extend the new clique further?
                # The new clique is {u,v,w}.
                # We could try to add another vertex x to make K4.
                # But let's keep it simple for now.
                pass
            else:
                # Failed, try another w?
                pass
                
        # Final cleanup: return only non-dead cliques
        result = []
        for i, c in enumerate(cliques):
            if not dead[i]:
                result.append(list(c))
                
        return result

    start_time = time.time()
    deadline = start_time + 110
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    # Run local search for each instance
    # Allocate time proportionally or split evenly
    time_per_instance = 50
    
    best_1 = local_search_partition(adj_1, 32, 0, time_per_instance)
    best_2 = local_search_partition(adj_2, 64, 1, time_per_instance)
    
    # If time remains, run more iterations?
    # The function above has a fixed max_iter.
    # We can call it again with a different seed if time allows.
    
    if time.time() < deadline - 10:
        # Try a second run with different seed
        best_1_b = local_search_partition(adj_1, 32, 123, deadline - 10 - time.time())
        if len(best_1_b) < len(best_1):
            best_1 = best_1_b
            
    if time.time() < deadline - 5:
        best_2_b = local_search_partition(adj_2, 64, 456, deadline - 5 - time.time())
        if len(best_2_b) < len(best_2):
            best_2 = best_2_b
            
    return [best_1, best_2]

