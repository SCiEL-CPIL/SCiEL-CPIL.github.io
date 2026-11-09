"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

_START = time.time()
_LIMIT = 140.0  # safety margin under 150s wall-clock

def _deadline_ok():
    return (time.time() - _START) < _LIMIT

def _random_edges(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _solve_instance(n, p, seed):
    edges = _random_edges(n, p, seed)
    if not edges:
        return []
    
    # Build adjacency matrix as bitmasks for speed
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute common neighbors for each edge (u, v)
    # common[u][v] is a bitmask of vertices w such that (u,w) and (v,w) are edges
    # We can compute this on the fly or precompute. Given n<=64, on-the-fly is fast enough with bit ops.
    
    def get_common_neighbors(u, v):
        # Vertices connected to both u and v
        # Exclude u and v themselves
        mask = adj[u] & adj[v]
        mask &= ~(1 << u)
        mask &= ~(1 << v)
        return mask

    def expand_clique(start_vertices, current_edges_mask):
        """
        Greedily expand a clique starting from start_vertices.
        current_edges_mask: a set of (u,v) pairs currently unassigned? 
        Actually, for the greedy peeling, we just need to ensure the clique is valid in the original graph.
        The "remaining edges" constraint is handled by removing covered edges.
        But wait, the standard greedy peeling for clique partitioning usually works on the graph structure directly.
        
        Let's refine the strategy:
        1. Represent the set of uncovered edges.
        2. Pick an uncovered edge (u,v).
        3. Find the largest clique containing (u,v) in the subgraph induced by vertices that have only uncovered edges among themselves? 
           No, that's too restrictive and hard.
        
        Alternative Strategy: 
        The current program uses "Greedy Peeling" where it picks an edge, expands to a maximal clique in the *remaining* graph (where remaining graph means edges are removed as they are covered).
        This is effectively finding a clique in the graph of remaining edges.
        
        Let's try a different approach: **Simulated Annealing on the partition**.
        But the state space is huge.
        
        Let's try **Local Search on the Partition**.
        Start with a trivial partition (each edge is a clique).
        Moves:
        1. Merge two cliques if their union is a clique and they share no edges (which is always true if they are distinct edges in a simple graph? No, if C1={u,v} and C2={u,w}, union is {u,v,w}. Is it a clique? Only if (v,w) is an edge. If so, we merge them into one clique {u,v,w}. This reduces clique count by 1.
        2. Move a vertex from one clique to another? Complex.
        
        Let's stick to **Merge-Based Local Search**.
        Initial state: Each edge is a clique. Number of cliques = |E|.
        Move: Pick two cliques C1, C2. If V(C1) U V(C2) is a clique, merge them.
        This is a monotonic improvement (reduces count). We can repeat until no more merges are possible.
        This is a "Hill Climbing" that always goes down. It will get stuck in local minima.
        To escape, we can allow "bad" moves (Simulated Annealing) or restart.
        
        However, checking if V(C1) U V(C2) is a clique is O(n^2).
        With n=64, and potentially thousands of cliques, this might be slow if done naively.
        
        Let's optimize:
        Represent cliques as frozensets of vertices.
        Maintain a list of cliques.
        Randomly pick pairs and check mergeability.
        
        Better: 
        1. Start with all edges as cliques.
        2. Repeatedly scan for mergeable pairs.
        3. To avoid O(N^2) pairs, use a queue or random sampling.
        
        Let's try a **Greedy Merge** first, then **SA**.
        
        Actually, there is a known heuristic for this: 
        **Greedy Maximum Clique Packing**:
        1. While edges remain:
           a. Find a maximum clique in the remaining graph.
           b. Remove its edges.
        This is what the current program approximates.
        
        Let's try **Randomized Local Search with Restart** on the **Merge** idea.
        
        State: List of cliques.
        Objective: Minimize len(cliques).
        
        Algorithm:
        1. Initialize with all edges as cliques.
        2. Repeat until deadline:
           a. Randomly pick two cliques C1, C2.
           b. If V(C1) U V(C2) is a clique, merge them.
           c. If no merges possible, try a "move" operation? 
              Or just restart with a different initial randomization?
              
        The problem with pure merging is that it's deterministic once the initial set is fixed (if we always merge when possible). 
        To introduce randomness, we can shuffle the order of checks.
        
        Let's implement **Simulated Annealing** on the partition.
        State: List of cliques.
        Moves:
        1. Merge C1, C2 if possible (always accept).
        2. Split a clique C into C1, C2 (random partition of vertices such that both are cliques? Hard).
        3. Move a vertex from C1 to C2 if C2 U {v} is a clique and C1 - {v} is still a clique (or empty).
        
        This is getting complex.
        
        Let's look at **Inspiration 1**: "randomized local-search heuristic that starts with a trivial valid partition".
        It mentions `_merge_heuristic`.
        
        Let's implement a robust **Merge-Based Local Search**.
        
        1. Start with all edges as cliques.
        2. Build an index of cliques.
        3. Loop:
           - Pick a random clique C.
           - Try to merge C with other cliques.
           - If a merge is possible, do it.
           - If no merges possible for C, mark it "stuck".
           - If all cliques are stuck, try a "kick" (randomly split a large clique to allow different merges) or restart.
           
        Given the time limit (150s), and n=64, E ~ 1000.
        Initial cliques: 1000.
        Merging reduces count.
        
        Let's try a simpler, effective heuristic: **Greedy with Randomized Tie-Breaking and Multiple Restarts**.
        The current program does this but uses "maximal clique in remaining graph".
        Let's try **Maximum Clique in Remaining Graph** more aggressively.
        
        Instead of greedy expansion, use a proper Max Clique solver (e.g., Bron-Kerbosch with pivoting) on the remaining graph?
        For n=64, Max Clique is NP-hard, but heuristics are fast.
        However, we don't need the *maximum* clique, just a *large* one to reduce the count.
        
        Let's try **Simulated Annealing** on the **Edge Assignment**.
        Each edge is assigned to a clique.
        But the clique structure is induced by the vertices.
        
        Let's go with **Merge-Based Local Search** as it's structurally different from "Peeling".
        Peeling builds cliques from scratch. Merging starts small and combines.
        
        Implementation of Merge-Based LS:
        1. `cliques`: list of sets of vertices.
        2. `edge_to_clique`: map from edge (u,v) to index in `cliques`.
        3. Initially, each edge is its own clique.
        4. `mergeable_pairs`: set of pairs of clique indices that can be merged.
           - Two cliques C1, C2 can be merged if V(C1) U V(C2) is a clique.
           - This requires checking all pairs of vertices in V(C1) and V(C2).
        5. While time remains:
           - If `mergeable_pairs` is not empty:
             - Pick a random pair (i, j) from `mergeable_pairs`.
             - Merge clique j into clique i.
             - Update `cliques`, `edge_to_clique`.
             - Remove all pairs involving i or j from `mergeable_pairs`.
             - Check new pairs involving the new clique i against all other cliques.
           - Else:
             - Local optimum reached.
             - Try a random "kick": Pick a random clique, split it into two random subsets (if possible) that are both cliques? 
               Or just restart with a different random seed for the initial shuffle?
               Since the initial state is deterministic (all edges), the local optimum might be the same.
               To vary the path, we can randomize the *order* in which we check/merge.
               Actually, if we always merge when possible, the result might depend on the order.
               So, randomizing the choice of which mergeable pair to take first leads to different local optima.
        
        This seems promising.
        
        Let's refine the "Mergeable" check.
        C1 = {u, v}, C2 = {u, w}. Union = {u, v, w}.
        Is it a clique? Check (v,w).
        
        Data structures:
        `cliques`: List of lists of vertices.
        `clique_edges`: List of sets of edges (for quick removal? No, we need to know if union is clique).
        
        To check if Union(C1, C2) is a clique:
        Let V1 = set(C1), V2 = set(C2).
        V = V1 U V2.
        For every pair (a, b) in V, check if edge (a, b) exists.
        Since C1 and C2 are cliques, we only need to check edges between V1 and V2.
        
        So, `can_merge(C1, C2)`:
        for u in C1:
            for v in C2:
                if not adj[u][v]: return False
        return True
        
        This is O(|C1| * |C2|).
        
        Complexity:
        Initially, cliques are size 2. Merge check is O(4).
        As cliques grow, it becomes more expensive.
        But n=64, so max clique size is bounded.
        
        Let's implement this.
        
        """
        
        # Build adjacency matrix as bitmasks
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            
        def are_connected(u, v):
            return (adj[u] >> v) & 1
            
        def can_merge(c1, c2):
            # c1, c2 are lists of vertices
            for u in c1:
                for v in c2:
                    if not are_connected(u, v):
                        return False
            return True

        def _solve_instance_merge(n, p, seed):
            edges = _random_edges(n, p, seed)
            if not edges:
                return []
            
            # Build adjacency
            adj = [0] * n
            for u, v in edges:
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
            def can_merge(c1, c2):
                for u in c1:
                    for v in c2:
                        if not ((adj[u] >> v) & 1):
                            return False
                return True

            # Initial state: each edge is a clique
            cliques = [[u, v] for u, v in edges]
            
            # To speed up merge checks, we can maintain a list of cliques and try to merge.
            # Instead of precomputing all pairs, we iterate.
            
            rng = random.Random(seed)
            
            best_cliques = cliques
            best_count = len(cliques)
            
            # We will run multiple iterations of local search.
            # Each iteration: try to merge until stuck.
            # Then, if time permits, restart with a different random seed for the *order* of merges?
            # Actually, the initial state is fixed. The order of merges determines the path.
            # So we can just repeat the process with different random choices.
            
            # But wait, if we always merge when possible, the "randomness" is in which mergeable pair we pick first.
            # This leads to different local optima.
            
            # Let's do this:
            # 1. Create a list of all initial cliques.
            # 2. Shuffle the list.
            # 3. Iterate through the list. For each clique, try to merge it with any other clique.
            #    If a merge is possible, merge and restart the scan (or continue).
            
            # A simpler approach:
            # While there are mergeable pairs:
            #   Pick a random mergeable pair and merge.
            
            # To find mergeable pairs efficiently:
            # We can maintain a set of indices that are "active".
            # When a clique changes, it becomes active.
            # We check active cliques against others.
            
            # Given n is small, let's just do a full scan repeatedly.
            
            current_cliques = [[u, v] for u, v in edges]
            
            # Optimization: If a clique is large, it's harder to merge.
            # We can prioritize merging small cliques.
            
            while _deadline_ok():
                merged_any = False
                # Shuffle indices to randomize merge order
                indices = list(range(len(current_cliques)))
                rng.shuffle(indices)
                
                # We will iterate through the shuffled indices.
                # For each clique i, try to merge with j > i (in the current list).
                # If merged, update list and restart the loop (or continue with updated list).
                
                # To be efficient, we can just try to merge each clique with any other.
                # If successful, mark as merged and continue.
                
                # Let's do a pass.
                i = 0
                while i < len(current_cliques):
                    if not _deadline_ok():
                        break
                    
                    c1 = current_cliques[i]
                    merged = False
                    # Try to merge c1 with any other clique j != i
                    for j in range(len(current_cliques)):
                        if j == i:
                            continue
                        c2 = current_cliques[j]
                        if can_merge(c1, c2):
                            # Merge c2 into c1
                            # New clique is union
                            new_c = list(set(c1 + c2))
                            # Remove c2 from list
                            # Replace c1 with new_c
                            # To avoid index issues, let's build a new list
                            new_list = []
                            for k in range(len(current_cliques)):
                                if k == i or k == j:
                                    continue
                                new_list.append(current_cliques[k])
                            new_list.append(new_c)
                            current_cliques = new_list
                            merged_any = True
                            merged = True
                            break
                    if merged:
                        # Restart the scan from the beginning (or current i)
                        # Since the list changed, restart is safer
                        i = 0
                    else:
                        i += 1
                
                if not merged_any:
                    # Local optimum reached
                    # Try a kick? Or just stop and try a new restart?
                    # Since the initial state is the same, different random shuffles lead to different paths.
                    # Let's count the number of cliques.
                    if len(current_cliques) < best_count:
                        best_count = len(current_cliques)
                        best_cliques = [list(c) for c in current_cliques]
                    
                    # Restart with a new shuffle
                    # To avoid infinite loop if stuck at the same local optimum,
                    # we can just break if we've done many restarts without improvement.
                    # Or just continue shuffling.
                    
                    # Let's limit restarts per instance to avoid wasting time.
                    # If we are stuck, we can try a "split" move to escape.
                    # But for now, let's just restart with a new random seed for the shuffle.
                    pass

            return best_cliques

        # Use the merge-based solver
        return _solve_instance_merge(n, p, seed)

def construct():
    results = []
    for n, p, seed in ((32, 0.6, 601), (64, 0.55, 602)):
        results.append(_solve_instance(n, p, seed))
    return results
# EVOLVE-BLOCK-END
