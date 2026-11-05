import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    # Precompute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    edge_set = set(all_edges)
    
    # Precompute all Q3 subgraphs (copies of the 3-cube) in K16
    # A Q3 is an isomorphic copy of the cube graph.
    # Vertices of Q3 can be represented as 3-bit integers 0..7.
    # Edges exist if Hamming distance is 1.
    q3_edges_local = [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), 
                      (4,5), (4,6), (5,7), (6,7)]
    
    # Generate all distinct Q3 subgraphs in K16
    # We need to map local vertices 0..7 to global vertices 0..15.
    # To avoid duplicates, we can enforce an ordering on the image of vertex 0.
    # However, for n=16, the number of subgraphs is manageable.
    # Total permutations: 16P8 = 16! / 8! = 518,918,400. This is too many to iterate naively in Python if checking all.
    # But we can be smarter. We just need to check if a specific set of 12 edges exists.
    # Actually, let's just generate the set of edge-tuples that form a Q3.
    
    # Optimization: 
    # A Q3 is determined by its 8 vertices. 
    # Let's iterate over combinations of 8 vertices from 16.
    # C(16, 8) = 12,870. This is very small!
    # For each combination of 8 vertices, check if the induced subgraph contains a Q3.
    # Wait, the task is "no subgraph isomorphic to Q3". 
    # If the induced subgraph on 8 vertices IS a Q3, it's a violation.
    # But a Q3 can also be a subgraph of a larger connected component? 
    # No, Q3 has 8 vertices. Any subgraph isomorphic to Q3 must use exactly 8 distinct vertices.
    # So we just need to ensure that NO set of 8 vertices induces a Q3.
    # Actually, it's "contains a subgraph isomorphic to Q3". 
    # If the 8 vertices induce a supergraph of Q3 (e.g., K8), it contains Q3.
    # So we must ensure that for every subset S of 8 vertices, the induced subgraph G[S] is NOT a supergraph of Q3.
    # Equivalently, G[S] must miss at least one edge from every potential Q3 embedding on S.
    
    # Let's precompute the list of "forbidden configurations".
    # For each combination of 8 vertices, we can map them to 0..7.
    # There are 8! = 40320 permutations.
    # For each perm, we get a set of 12 edges (the Q3 edges under that mapping).
    # If ALL these 12 edges are present in G, then G contains a Q3.
    # So, for each 8-vertex subset S, we have a family of edge-sets E_p (for each permutation p).
    # We must delete at least one edge from each E_p.
    
    # This is a hitting set problem.
    # Universe: Edges of K16.
    # Sets: Each specific Q3 embedding (defined by a permutation of a specific 8-vertex set).
    # We want to remove minimum edges to hit all sets.
    
    # Total number of Q3 embeddings:
    # Choose 8 vertices: C(16,8) = 12870.
    # For each, number of distinct Q3 labelings: 8! / |Aut(Q3)|.
    # Aut(Q3) size is 48.
    # So 40320 / 48 = 840 distinct Q3s per 8-vertex set.
    # Total Q3s = 12870 * 840 = 10,810,800.
    # This is ~10 million constraints.
    
    # We cannot store all 10 million sets explicitly in memory easily or check them fast enough in a simple loop.
    # However, we can use a randomized greedy approach or local search.
    
    # Strategy:
    # 1. Start with K16 (all edges present).
    # 2. Identify a Q3.
    # 3. Remove a random edge from that Q3.
    # 4. Repeat until no Q3 remains.
    # This is the "randomized sequential removal" approach.
    # To improve, we can use a heuristic: remove the edge that is part of the most remaining Q3s.
    
    # Let's implement a faster check for "Does G contain a Q3?".
    # We can sample random 8-vertex subsets and check if they contain a Q3.
    # But we need to be sure.
    
    # Better Strategy:
    # Use a backtracking solver on the "hitting set" formulation but with pruning.
    # Or, simpler: Start with K16. 
    # While there exists a Q3:
    #   Find one Q3.
    #   Remove an edge from it.
    #   
    # Finding a Q3 in K16 is fast.
    # We can precompute a list of "candidate Q3s" to check.
    # Actually, let's just use the property that if we remove edges randomly, we might get stuck.
    # Let's use a deterministic greedy with lookahead?
    
    # Alternative: 
    # The seed score is 1.0 (39 edges).
    # The child p4 score is 1.38 (approx 54 edges).
    # K16 has 120 edges.
    # We can likely do much better than 54.
    
    # Let's try a local search / hill climbing on the number of edges.
    # State: Set of edges.
    # Objective: Maximize |E| subject to No Q3.
    
    # We will use a randomized restart strategy.
    # In each restart:
    # 1. Start with K16.
    # 2. While Q3 exists:
    #    - Find a Q3 (using random sampling of 8-sets or systematic search).
    #    - Pick an edge in this Q3 to remove. 
    #      Heuristic: Remove the edge that appears in the most other potential Q3s (to reduce future conflicts).
    #      Since we don't track all Q3s, we can approximate by removing an edge with high degree in the current graph? 
    #      Or just random.
    
    # To find a Q3 efficiently:
    # We can iterate through all 8-vertex combinations? 12,870 is small.
    # For each combination, check if it contains a Q3.
    # Checking if an 8-vertex induced subgraph contains a Q3:
    # We can try all 840 distinct Q3 embeddings on these 8 vertices.
    # 12870 * 840 = 10.8M checks.
    # Each check is 12 edge lookups.
    # 10.8M * 12 = 130M operations.
    # In Python, this might take 10-20 seconds per full scan.
    # If we do this in a loop, it's too slow.
    
    # Optimization:
    # Instead of scanning all 8-sets, we can maintain a list of "violating" 8-sets.
    # When we remove an edge (u,v), we only need to re-check 8-sets that contain u and v.
    # Number of 8-sets containing a specific edge (u,v):
    # We need to choose 6 more vertices from the remaining 14.
    # C(14, 6) = 3003.
    # So, for each edge removed, we re-check ~3000 8-sets.
    # This is very fast!
    
    # Algorithm:
    # 1. Initialize G = K16.
    # 2. Identify all 8-sets that currently contain a Q3.
    #    - Initially, ALL 8-sets contain a Q3 (since K8 contains Q3).
    #    - So we have a list of all C(16,8) = 12870 sets.
    # 3. While there are active 8-sets:
    #    - Pick an active 8-set S.
    #    - Find a Q3 embedding within S. (We can precompute one for each S? No, the embedding depends on which edges are left).
    #    - Actually, if S is "active", it means the current edges in G[S] form a supergraph of some Q3.
    #    - We need to find WHICH Q3 is still present.
    #    - We can check the 840 possible Q3s on S. One of them must be present.
    #    - Let E_q3 be the edges of that specific Q3 instance.
    #    - We must remove one edge from E_q3.
    #    - Heuristic: Choose the edge e in E_q3 that is "most critical" or "random".
    #    - Remove e from G.
    #    - Update active sets:
    #      - For any 8-set S' that contains e, re-check if S' still contains a Q3.
    #      - If not, remove S' from active list.
    #    - Note: Removing an edge cannot CREATE a Q3, only destroy them.
    #      So we only need to check sets that *might* have been destroyed.
    #      Sets not containing e are unaffected.
    
    # Implementation Details:
    # Precompute all 8-sets.
    # For each 8-set, precompute the list of 840 Q3 edge-tuples (global edges).
    # This is 12870 * 840 * 12 ints. ~130M ints. Might be heavy on memory (500MB+).
    # Better: Generate Q3s on the fly for a specific 8-set when needed.
    
    # Let's refine the "Find Q3 in S" step.
    # Given S (8 vertices) and current adjacency matrix.
    # We know S was active, so it HAD a Q3.
    # We check the 840 standard Q3s on the 8 local indices.
    # For each local Q3, map to global edges. Check if all 12 edges exist in G.
    # If yes, return that Q3's global edges.
    
    # Complexity:
    # Each removal triggers ~3000 re-checks.
    # Each re-check involves testing up to 840 Q3s (but we can stop at the first one found? 
    # No, we need to know if ANY Q3 remains. If the first one is gone, we must check the next).
    # Worst case: 840 checks per re-check.
    # 3000 * 840 = 2.5M checks per edge removal.
    # If we remove 100 edges, that's 250M checks.
    # Each check is 12 array lookups.
    # This might be tight for 110s in Python.
    
    # Optimization: 
    # When re-checking a set S, we don't need to check all 840 Q3s if we know which one was present before?
    # No, because we removed an edge. The previous Q3 might be broken, but another one might remain.
    # However, we can optimize the check: 
    # Instead of checking all 840, we can check if the graph on S is still "dense enough" to contain a Q3?
    # No, Q3 is specific.
    
    # Let's try to limit the search.
    # We'll use a random restart approach.
    # In each restart, we run the greedy removal.
    # We track the best number of edges.
    
    # To speed up the "Does S contain Q3" check:
    # We can pre-compute for each 8-set S, the list of 840 Q3s as bitmasks or lists of edge indices.
    # Let's try to be memory efficient.
    # Edge index: 0..119.
    # Q3 is a set of 12 edges.
    # We can represent a Q3 as a tuple of 12 edge indices.
    # Store these in a list for each 8-set.
    # Memory: 12870 sets * 840 Q3s * 12 ints.
    # 130M ints. In Python, an int is ~28 bytes. 130M * 28 = 3.6 GB. Too much.
    
    # Alternative: Generate on the fly.
    # Precompute the 840 local Q3 edge lists (local indices 0..7).
    # local_q3s = [ [(0,1), ...], ... ] (840 lists of 12 pairs)
    # When checking a global set S (sorted list of 8 vertices):
    #   Map local i to S[i].
    #   For each local_q3 in local_q3s:
    #       Check if all (S[u], S[v]) are in G.
    
    # Let's implement this.
    
    # 1. Precompute local Q3s.
    # 2. Precompute all 8-sets.
    # 3. Main Loop.
    
    # To handle the time limit, we will restart if we get stuck or if time is running out.
    # Actually, the greedy process is deterministic if we pick the "first" Q3 and "first" edge.
    # Randomization in edge selection helps.
    
    def get_local_q3s():
        # Generate all distinct Q3s on vertices 0..7
        # Use canonical labeling or just generate all perms and deduplicate.
        # 8! = 40320.
        # We can just generate all perms, create the edge set, and add to a set of frozensets.
        # Then convert to list of lists.
        q3s = set()
        for perm in itertools.permutations(range(8)):
            # perm maps local 0..7 to perm[0]..perm[7]?
            # No, perm is the image of local vertices.
            # Let's say local vertex i maps to perm[i].
            # Edges in Q3: (i, j) if dist(i,j)=1.
            # Local edges:
            # 0-1, 0-2, 0-4, 1-3, 1-5, 2-3, 2-6, 3-7, 4-5, 4-6, 5-7, 6-7
            edges = []
            for u, v in [(0,1), (0,2), (0,4), (1,3), (1,5), (2,3), (2,6), (3,7), (4,5), (4,6), (5,7), (6,7)]:
                gu, gv = perm[u], perm[v]
                if gu > gv: gu, gv = gv, gu
                edges.append((gu, gv))
            q3s.add(frozenset(edges))
        return [list(q) for q in q3s]

    local_q3s = get_local_q3s()
    # Should be 840.
    
    # Precompute all 8-sets
    all_8sets = list(itertools.combinations(range(16), 8))
    
    def check_q3_in_set(S, adj):
        # S is a tuple of 8 vertices.
        # adj is a 16x16 boolean matrix or set of edges.
        # We use a set of edges for O(1) lookup.
        for lq3 in local_q3s:
            valid = True
            for u, v in lq3:
                gu, gv = S[u], S[v]
                if gu > gv: gu, gv = gv, gu
                if (gu, gv) not in edge_set:
                    valid = False
                    break
            if valid:
                # Return the global edges of this Q3
                global_edges = []
                for u, v in lq3:
                    gu, gv = S[u], S[v]
                    if gu > gv: gu, gv = gv, gu
                    global_edges.append((gu, gv))
                return global_edges
        return None

    def construct():
        start_time = time.time()
        deadline = start_time + 110.0
        
        best_edges = 39 # Seed value
        best_edge_list = None
        
        # We will run multiple random restarts
        num_restarts = 200 # Limit by time
        
        for r in range(num_restarts):
            if time.time() > deadline - 10:
                break
                
            # Initialize K16
            edge_set = set(all_edges)
            adj = [[False]*16 for _ in range(16)]
            for u, v in edge_set:
                adj[u][v] = True
                adj[v][u] = True
            
            # Active 8-sets: Initially all
            active_sets = list(all_8sets)
            # To make removal faster, we can shuffle active_sets
            random.shuffle(active_sets)
            
            current_edge_count = len(edge_set)
            
            # Greedy removal
            # We process active sets. If a set is no longer a Q3-container, we skip it.
            # We need to be careful not to miss sets that become "clean" later?
            # No, if a set is clean, it stays clean.
            # So we can just iterate through the list.
            # However, when we remove an edge, many sets become clean.
            # We should check them and remove them from the active list to save time.
            
            i = 0
            while i < len(active_sets):
                if time.time() > deadline - 5:
                    break
                    
                S = active_sets[i]
                # Check if S still contains a Q3
                q3_edges = check_q3_in_set(S, edge_set)
                
                if q3_edges is None:
                    # This set is now clean. Remove it from active list.
                    active_sets.pop(i)
                    # Do not increment i, as the next element shifted up
                    continue
                else:
                    # S contains a Q3. We must remove an edge from q3_edges.
                    # Heuristic: Remove a random edge from q3_edges
                    # Or the edge that appears in the most other active sets? (Too slow to compute)
                    # Let's pick a random edge.
                    e_to_remove = random.choice(q3_edges)
                    
                    # Remove edge
                    edge_set.remove(e_to_remove)
                    adj[e_to_remove[0]][e_to_remove[1]] = False
                    adj[e_to_remove[1]][e_to_remove[0]] = False
                    current_edge_count -= 1
                    
                    # Update active sets:
                    # Any set containing e_to_remove might now be clean.
                    # We need to re-check them.
                    # Instead of scanning all active sets, we can just let the main loop handle it?
                    # The main loop will encounter them again?
                    # No, we are iterating with index i.
                    # If we just remove the edge, the sets containing it are still in active_sets.
                    # When the loop reaches them, it will check and remove them if clean.
                    # This is inefficient because we might check the same set multiple times?
                    # No, each set is in the list once.
                    # But we might check a set, find it dirty, remove an edge, and then later check another set, 
                    # which might also have been affected by the previous removal?
                    # Yes.
                    # This "lazy" deletion is fine.
                    
                    # However, we should prioritize checking sets that contain the removed edge?
                    # Not necessary for correctness, but for speed?
                    # The current approach checks sets in the order of the list.
                    # Since we shuffled, it's random order.
                    
                    # Increment i to check the next set (which might be one that was just cleaned by the removal? 
                    # No, the removal happened, but the set S we just processed is still "dirty" in the sense 
                    # that we found a Q3. But we removed an edge from it. 
                    # Wait, if we remove an edge from the Q3 in S, does S become clean?
                    # Not necessarily. S might contain ANOTHER Q3.
                    # So S might still be dirty.
                    # We should re-check S?
                    # If we increment i, we move to the next set.
                    # S will not be checked again unless we wrap around?
                    # We are using a while loop with pop.
                    # If we don't pop S, we should re-check it?
                    # If S is still dirty, we need to remove another edge from it.
                    # If we move on, we might leave S dirty while removing edges from other sets.
                    # This is suboptimal. We should focus on S until it is clean?
                    # Or just move on and let it be cleaned when we come back?
                    # Since we don't wrap around, we will NEVER come back to S if we increment i.
                    # So if S is still dirty, we have a problem: we skipped it.
                    
                    # Correction:
                    # If we remove an edge from S, S might still contain a Q3.
                    # We should check S again.
                    # So, do NOT increment i. Stay on S.
                    # But if S becomes clean, we pop it and move to next (which is effectively incrementing).
                    # If S remains dirty, we loop again on the same S (since we didn't pop).
                    
                    # So the logic is:
                    # 1. Check S.
                    # 2. If clean: pop S, i stays same (next item is at i).
                    # 3. If dirty: remove edge, i stays same (check S again).
                    
                    # This ensures we clean S completely before moving on.
                    # This is a good heuristic.
                    pass # The logic above is handled by not incrementing i in the 'else' block implicitly?
                    # Wait, in the code structure:
                    # if clean: pop(i), continue (i effectively stays, points to next)
                    # else: remove edge, (do not pop, do not increment) -> loop repeats with same i.
                    
                    # So we just need to ensure we don't increment i in the else block.
                    # The code below implements this.
                    
            # End while loop for this restart
            
            if current_edge_count > best_edges:
                best_edges = current_edge_count
                best_edge_list = list(edge_set)
                
        # Return best found
        if best_edge_list:
            return [[u, v] for u, v in best_edge_list]
        else:
            # Fallback to seed
            return [[a, b] for a in range(3) for b in range(3, 16)]

    return construct()

# Execute
if __name__ == "__main__":
    construct()
