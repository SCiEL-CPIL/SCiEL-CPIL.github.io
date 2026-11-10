import time
import random
from itertools import combinations

def _is_non_2colorable(n, edges):
    """
    Check if the hypergraph is NOT 2-colorable.
    Returns True if no proper 2-coloring exists.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False
        
    # Deduplicate and normalize edges
    edge_set = [tuple(sorted(e)) for e in edges]
    edge_set = list(set(edge_set))
    if not edge_set:
        return False
        
    m = len(edge_set)
    
    # Map vertices to list of edge indices containing them
    vert_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edge_set):
        for v in e:
            vert_to_edges[v].append(i)
            
    # Colors: -1 unassigned, 0, 1
    colors = [-1] * n
    # Count of 0s and 1s in each edge
    count_0 = [0] * m
    count_1 = [0] * m
    
    def propagate():
        """
        Apply unit propagation.
        Returns False if a conflict is found (edge becomes monochromatic).
        Returns True if no immediate conflict and propagation is stable.
        """
        changed = True
        while changed:
            changed = False
            for i in range(m):
                c0 = count_0[i]
                c1 = count_1[i]
                e = edge_set[i]
                
                # Conflict: all 5 are same color
                if c0 == 5 or c1 == 5:
                    return False
                
                # Unit Propagation: 4 are same color, 1 unassigned
                # The unassigned one MUST be the other color
                if c0 == 4 and c1 == 0:
                    # Find the unassigned vertex
                    for v in e:
                        if colors[v] == -1:
                            # It must be 1
                            colors[v] = 1
                            count_1[i] += 1
                            # Update other edges containing v
                            for j in vert_to_edges[v]:
                                if j != i:
                                    count_1[j] += 1
                            changed = True
                            break
                elif c1 == 4 and c0 == 0:
                    for v in e:
                        if colors[v] == -1:
                            # It must be 0
                            colors[v] = 0
                            count_0[i] += 1
                            for j in vert_to_edges[v]:
                                if j != i:
                                    count_0[j] += 1
                            changed = True
                            break
        return True

    def backtrack():
        # Check for conflicts and propagate
        if not propagate():
            return False
            
        # Check if all vertices assigned
        # We can also check if all edges are satisfied (not mono)
        # If all assigned, we just need to ensure no edge is mono (handled by propagate returning False on mono)
        # Actually, if all assigned and propagate didn't fail, it's a valid coloring.
        
        # Find unassigned vertex
        # Heuristic: Pick vertex with most unassigned neighbors in "tight" edges?
        # Simple heuristic: Pick vertex with highest degree (number of edges) among unassigned.
        # Or: Pick vertex involved in an edge with 3 or 4 assigned?
        
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True # Found a valid coloring
            
        # Score vertices to pick the most constrained one
        best_v = -1
        max_score = -1
        for v in unassigned:
            score = 0
            for j in vert_to_edges[v]:
                c0 = count_0[j]
                c1 = count_1[j]
                # Prefer edges that are close to being unit (4 assigned)
                if c0 + c1 >= 3:
                    score += 1
            if score > max_score:
                max_score = score
                best_v = v
                
        if best_v == -1:
            best_v = unassigned[0]
            
        v = best_v
        
        # Save state
        old_colors = colors[:]
        old_c0 = count_0[:]
        old_c1 = count_1[:]
        
        # Try color 0
        colors[v] = 0
        for j in vert_to_edges[v]:
            count_0[j] += 1
        if backtrack():
            return True
            
        # Restore state
        colors[:] = old_colors
        count_0[:] = old_c0
        count_1[:] = old_c1
        
        # Try color 1
        colors[v] = 1
        for j in vert_to_edges[v]:
            count_1[j] += 1
        if backtrack():
            return True
            
        return False
        
    return not backtrack()

def construct():
    """
    Return a non-2-colorable 5-uniform hypergraph with minimal edges.
    Strategy: 
    1. Try different vertex counts n.
    2. For each n, generate a random hypergraph with a certain density.
    3. If it is non-2-colorable, use a greedy deletion algorithm to minimize edges.
    4. Keep track of the best (smallest) non-2-colorable hypergraph found.
    """
    deadline = time.time() + 110  # Safety margin before 120s limit
    
    best_m = 126 # Fallback: complete hypergraph on 9 vertices
    best_graph = {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    
    # We know m(5) is around 51. Let's search for n in range 10..25
    # And try to find small edge sets.
    
    for n in range(10, 26):
        if time.time() > deadline - 5:
            break
            
        all_edges = list(combinations(range(n), 5))
        total_possible = len(all_edges)
        
        if total_possible < 30:
            continue
            
        # Try a few random initializations with different densities
        # Higher density is more likely to be non-2-colorable
        for density in [0.8, 0.6, 0.4]:
            if time.time() > deadline - 5:
                break
                
            initial_m = int(total_possible * density)
            # Ensure we don't pick too many if total is small
            initial_m = min(initial_m, total_possible)
            
            if initial_m >= best_m:
                continue
                
            # Random sample of edges
            edges = [list(e) for e in random.sample(all_edges, initial_m)]
            
            # Check if non-2-colorable
            if not _is_non_2colorable(n, edges):
                continue
                
            # It is non-2-colorable. Now minimize.
            # Greedy deletion:
            # Shuffle edges to randomize deletion order
            random.shuffle(edges)
            
            i = 0
            while i < len(edges):
                if time.time() > deadline - 2:
                    break
                    
                # Try removing edges[i]
                candidate_edges = edges[:i] + edges[i+1:]
                
                # Check if still non-2-colorable
                if _is_non_2colorable(n, candidate_edges):
                    # Keep it removed
                    edges = candidate_edges
                    # Do not increment i, as the next edge is now at index i
                    # But we must be careful not to loop infinitely if checks are slow
                    # Actually, this is efficient because we only check once per removal
                else:
                    # Put it back (it's already back in 'edges' because we didn't modify 'edges' yet)
                    # Just move to next edge
                    i += 1
                    
            # After deletion loop, we have a minimal (w.r.t deletion) hypergraph
            if len(edges) < best_m:
                best_m = len(edges)
                best_graph = {'n': n, 'edges': edges}
                
                # If we found something very small, we can stop early?
                # Theoretical lower bound is 29 (since 28 is 2-colorable).
                # Best known is 51.
                if best_m <= 51:
                    # We hit the known best. Can we do better?
                    # Unlikely to find < 51 quickly.
                    # But let's continue for a bit to see if we can go lower, or just return.
                    # Given time limits, if we found 51, it's a great result.
                    # Let's try to find even smaller if time permits.
                    pass

    return best_graph
