"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 30, 'edges': [[0, 4], [0, 7], [0, 9], [0, 10], [0, 11], [0, 16], [0, 17], [0, 18], [0, 19], [0, 23], [0, 24], [0, 26], [0, 27], [0, 28], [1, 2], [1, 4], [1, 6], [1, 8], [1, 10], [1, 11], [1, 12], [1, 14], [1, 15], [1, 16], [1, 17], [1, 21], [1, 24], [1, 25], [1, 28], [1, 29], [2, 6], [2, 7], [2, 9], [2, 12], [2, 13], [2, 14], [2, 19], [2, 22], [2, 23], [2, 26], [2, 27], [2, 29], [3, 5], [3, 7], [3, 10], [3, 11], [3, 16], [3, 17], [3, 19], [3, 20], [3, 22], [3, 23], [3, 24], [3, 28], [3, 29], [4, 5], [4, 8], [4, 10], [4, 12], [4, 14], [4, 16], [4, 17], [4, 18], [4, 20], [4, 21], [4, 28], [5, 6], [5, 7], [5, 10], [5, 15], [5, 17], [5, 18], [5, 20], [5, 21], [5, 22], [5, 23], [5, 29], [6, 7], [6, 11], [6, 12], [6, 14], [6, 17], [6, 18], [6, 19], [6, 20], [6, 21], [6, 24], [6, 25], [6, 27], [7, 8], [7, 14], [7, 15], [7, 16], [7, 17], [7, 18], [7, 19], [7, 21], [7, 24], [7, 25], [7, 26], [7, 28], [8, 9], [8, 10], [8, 11], [8, 12], [8, 14], [8, 18], [8, 19], [8, 20], [8, 23], [8, 24], [8, 25], [8, 27], [8, 28], [9, 10], [9, 12], [9, 17], [9, 18], [9, 19], [9, 20], [9, 21], [9, 23], [9, 24], [9, 28], [9, 29], [10, 11], [10, 12], [10, 14], [10, 17], [10, 18], [10, 22], [10, 24], [10, 27], [10, 29], [11, 14], [11, 16], [11, 20], [11, 24], [11, 25], [11, 27], [11, 28], [11, 29], [12, 14], [12, 17], [12, 19], [12, 20], [12, 26], [12, 29], [13, 14], [13, 15], [13, 19], [13, 22], [13, 23], [13, 25], [13, 26], [13, 27], [13, 28], [13, 29], [14, 15], [14, 24], [14, 26], [14, 27], [14, 28], [15, 19], [15, 24], [15, 25], [15, 29], [16, 18], [16, 20], [16, 21], [16, 27], [16, 28], [17, 18], [17, 19], [17, 21], [17, 23], [17, 24], [17, 25], [17, 27], [17, 28], [17, 29], [18, 19], [18, 21], [18, 24], [18, 25], [18, 27], [18, 29], [19, 23], [19, 24], [19, 26], [19, 27], [19, 29], [20, 21], [20, 22], [20, 23], [20, 24], [20, 26], [20, 29], [21, 23], [21, 25], [21, 26], [21, 28], [21, 29], [22, 23], [22, 24], [22, 25], [24, 29], [25, 26], [26, 29], [27, 29]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 90s limit
    
    # Build adjacency sets for quick lookup
    n = P['n']
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
        
    # Normalize edges to sets of frozensets for quick removal
    remaining_edges = set()
    for u, v in P['edges']:
        if u < v:
            remaining_edges.add(frozenset((u, v)))
        else:
            remaining_edges.add(frozenset((v, u)))
            
    best_solution = None
    best_count = float('inf')
    
    # Greedy approach: repeatedly find a maximal biclique in the remaining graph
    # A biclique is defined by (L, R) where all edges between L and R exist in remaining_edges.
    # We want to cover all edges with the fewest such bicliques.
    
    # Strategy: 
    # 1. Pick a random edge (u, v) from remaining.
    # 2. Try to expand it into a maximal biclique.
    #    Start with L={u}, R={v}.
    #    Iteratively add vertices to L or R if they are connected to all vertices in the opposite side.
    # 3. Remove all edges in the biclique from remaining.
    # 4. Repeat until remaining is empty.
    
    # To improve, we can try multiple random restarts or different expansion heuristics.
    # Since n=30 is small, we can afford some search.
    
    for attempt in range(100): # Max attempts
        if time.time() > deadline:
            break
            
        # Shuffle edges to randomize the starting point
        edges_list = list(remaining_edges)
        random.shuffle(edges_list)
        
        current_remaining = set(remaining_edges)
        bicliques = []
        
        # Copy adj for this attempt? No, adj is static. We only care about remaining_edges.
        # But to check if a vertex w can be added to L, we need:
        # For all r in R, (w, r) in current_remaining.
        # And for all l in L, (w, l) is NOT required to be in current_remaining? 
        # Wait, the definition of a biclique in the context of partitioning edges:
        # "Every cross pair must be a host edge." -> This refers to the original graph P.
        # "The cross-edge sets must partition all pinned graph edges exactly once."
        # So, if I select L={0,1} and R={2,3}, I must ensure that (0,2), (0,3), (1,2), (1,3) are ALL in P['edges'].
        # AND, these specific edges must be in current_remaining (not yet covered).
        # Edges within L (0,1) or within R (2,3) are allowed to exist in P, but they are NOT part of the cross edges of this biclique.
        # However, the rule says "cross-edge sets must partition". So edges (0,1) and (2,3) are NOT covered by this biclique. They must be covered by other bicliques or be non-edges.
        # Wait, "Within-side edges are allowed." means they don't invalidate the biclique.
        # But do they need to be covered? "partition all pinned graph edges". Yes, ALL edges in P must be covered.
        # So if (0,1) is an edge in P, it must be in some biclique's cross set.
        # In my greedy loop, I only remove cross edges. So (0,1) stays in current_remaining.
        # Later, a biclique with L'={0}, R'={1} (or similar) will cover it.
        
        valid_solution = True
        
        while current_remaining:
            if time.time() > deadline:
                valid_solution = False
                break
                
            # Pick a random edge to start a new biclique
            edge = random.choice(list(current_remaining))
            u, v = list(edge)
            
            # Initialize biclique sides
            L = {u}
            R = {v}
            
            # Expand greedily
            changed = True
            while changed:
                changed = False
                # Try to add a vertex to L
                # Candidates: vertices w not in L or R such that (w, r) is in current_remaining for all r in R
                # AND (w, r) is in adj[w] (host edge) - actually current_remaining is subset of host edges, so if in current_remaining, it's a host edge.
                # But we also need to ensure we don't "break" the biclique property.
                # If we add w to L, we require edges (w, r) for all r in R to be present in current_remaining.
                # Do we require anything about edges (w, l) for l in L? No, within-side edges are irrelevant for validity of THIS biclique.
                
                # Optimization: Precompute neighbors in current_remaining?
                # For a fixed R, a vertex w is a valid candidate for L if R is a subset of {x | (w,x) in current_remaining}.
                
                # Let's find candidates for L
                candidates_L = set()
                for w in range(n):
                    if w in L or w in R:
                        continue
                    # Check if w is connected to all in R via current_remaining
                    # Note: (w, r) might be stored as frozenset({w, r})
                    valid = True
                    for r in R:
                        if frozenset((w, r)) not in current_remaining:
                            valid = False
                            break
                    if valid:
                        candidates_L.add(w)
                
                # Pick the best candidate for L (e.g., one that covers most edges)
                # Or just pick one to keep it simple?
                # To maximize size, pick w that has max degree into R? 
                # Actually, if it's in candidates, it has ALL edges to R.
                # Let's pick one that maximizes potential future expansion?
                # Simple heuristic: pick w that has most neighbors in current_remaining (outside R)?
                # Or just random?
                if candidates_L:
                    # Pick w that maximizes number of common neighbors with L? No.
                    # Let's pick w that is connected to the most vertices in current_remaining (global degree in remaining graph)
                    # This might help cover more edges later?
                    # Actually, let's just pick one that allows largest R expansion?
                    # Let's stick to simple: pick one.
                    w = random.choice(list(candidates_L))
                    L.add(w)
                    changed = True
                    
                # Try to add a vertex to R
                candidates_R = set()
                for w in range(n):
                    if w in L or w in R:
                        continue
                    valid = True
                    for l in L:
                        if frozenset((w, l)) not in current_remaining:
                            valid = False
                            break
                    if valid:
                        candidates_R.add(w)
                        
                if candidates_R:
                    w = random.choice(list(candidates_R))
                    R.add(w)
                    changed = True
                    
            # We have a maximal (with respect to this greedy order) biclique (L, R)
            # Remove all cross edges from current_remaining
            for l in L:
                for r in R:
                    current_remaining.discard(frozenset((l, r)))
                    
            # Store the biclique
            # Ensure lists are sorted for determinism? Not required, but good.
            bicliques.append([sorted(L), sorted(R)])
            
        if valid_solution:
            # Verify solution
            # 1. All cross pairs are host edges
            # 2. Cross edges partition all edges
            covered_edges = set()
            for left, right in bicliques:
                for l in left:
                    for r in right:
                        e = frozenset((l, r))
                        if e in covered_edges:
                            valid_solution = False
                            break
                        covered_edges.add(e)
                        # Check host edge
                        if e not in remaining_edges: # remaining_edges is the full set
                            valid_solution = False
                            break
                    if not valid_solution:
                        break
                if not valid_solution:
                    break
            if not valid_solution:
                break
            if covered_edges != remaining_edges:
                valid_solution = False
            if valid_solution:
                if len(bicliques) < best_count:
                    best_count = len(bicliques)
                    best_solution = bicliques
                    
        # If we found a solution with 1 biclique, that's optimal (if graph is complete bipartite).
        # Unlikely.
        if best_count == 1:
            break

    if best_solution is None:
        # Fallback to seed strategy (star decomposition) to ensure validity
        # This is guaranteed to be valid and within time limit
        return [[[u],[v for a,v in P['edges'] if a==u]] for u in range(P['n']) if any(a==u for a,v in P['edges'])]
        
    # Convert to required format: list of [left_list, right_list]
    # Ensure integers are Python int
    final_result = []
    for bic in best_solution:
        left = [int(x) for x in bic[0]]
        right = [int(x) for x in bic[1]]
        final_result.append([left, right])
        
    return final_result

# EVOLVE-BLOCK-END
