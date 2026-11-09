import random
import time
from itertools import combinations

def is_2_colorable(n, edges):
    """
    Returns True if the hypergraph is 2-colorable, False otherwise.
    Uses backtracking with unit propagation and state copying for correctness.
    """
    if not edges:
        return True
    
    # Map vertex to list of edge indices
    vertex_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            if v is not None and 0 <= v < n:
                vertex_edges[v].append(i)
            
    E = len(edges)
    
    def assign(idx, color, c, c0, c1):
        if c[idx] != -1:
            return c[idx] == color
        
        c[idx] = color
        for e_idx in vertex_edges[idx]:
            if color == 0:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    return False
            else:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    return False
            
            # Unit propagation
            # If 4 are 0, the last must be 1
            if c0[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 1, c, c0, c1):
                            return False
            # If 4 are 1, the last must be 0
            elif c1[e_idx] == 4:
                for v in edges[e_idx]:
                    if c[v] == -1:
                        if not assign(v, 0, c, c0, c1):
                            return False
        return True

    def solve(c, c0, c1):
        # Find unassigned vertex
        # Heuristic: Pick vertex with highest degree or most constrained edges
        idx = -1
        max_score = -1
        for i in range(n):
            if c[i] == -1:
                # Score: number of edges where this vertex is the last unassigned or near
                score = 0
                for e_idx in vertex_edges[i]:
                    remaining = sum(1 for v in edges[e_idx] if c[v] == -1)
                    if remaining == 1:
                        score += 100
                    elif remaining == 2:
                        score += 10
                    else:
                        score += 1
                if score > max_score:
                    max_score = score
                    idx = i
        if idx == -1:
            return True
            
        # Try 0
        new_c = c[:]
        new_c0 = c0[:]
        new_c1 = c1[:]
        if assign(idx, 0, new_c, new_c0, new_c1):
            if solve(new_c, new_c0, new_c1):
                return True
        
        # Try 1
        new_c = c[:]
        new_c0 = c0[:]
        new_c1 = c1[:]
        if assign(idx, 1, new_c, new_c0, new_c1):
            if solve(new_c, new_c0, new_c1):
                return True
                
        return False

    return solve([-1]*n, [0]*E, [0]*E)

def construct():
    start_time = time.time()
    time_limit = 100  # Keep buffer for overhead
    
    best_edges = None
    best_count = float('inf')
    
    # Known constructions or random search
    # We try to find a non-2-colorable hypergraph with few edges.
    # Theoretical lower bound is 29, known upper is 51.
    
    # Strategy: 
    # 1. Try small n (9-12) and random subsets of edges.
    # 2. If a candidate is found, try to prune it.
    # 3. Keep the best (smallest) one.
    
    for n in range(9, 13):
        all_edges = list(combinations(range(n), 5))
        total_edges = len(all_edges)
        
        # Try edge counts from low to high
        # For n=9, total=126. 
        # For n=10, total=252.
        # We want to minimize edges.
        
        # Start with a reasonable random seed or fixed seed for reproducibility in this attempt?
        # Random is fine.
        
        # We iterate over potential edge counts E
        # If E is too small, it's unlikely to be non-colorable.
        # Let's start from a lower bound. 
        # For n=9, m(5) for n=9 is known to be around ? 
        # Actually, the property B threshold for 5-uniform is global.
        # For small n, the hypergraph is sparse.
        
        # Let's try random subsets of size E.
        # Start E from 30 up to 50.
        
        # If we find one with E edges, we prune it.
        
        for E in range(30, 52):
            if time.time() - start_time > time_limit - 20:
                break
                
            # Number of tries depends on how hard it is.
            # Smaller E is harder to find.
            # Larger E is easier.
            max_tries = 200 if E < 40 else 100
            
            found = False
            for _ in range(max_tries):
                random.shuffle(all_edges)
                candidate = [list(e) for e in all_edges[:E]]
                
                if not is_2_colorable(n, candidate):
                    # Found a non-colorable hypergraph
                    # Try to prune it
                    active_edges = candidate[:]
                    changed = True
                    while changed and time.time() - start_time < time_limit - 5:
                        changed = False
                        # Shuffle for random removal order
                        # We remove from end to avoid index shifting issues? 
                        # Or just iterate indices.
                        indices = list(range(len(active_edges)))
                        random.shuffle(indices)
                        for i in indices:
                            if len(active_edges) <= best_count:
                                continue
                            # Try removing active_edges[i]
                            new_edges = active_edges[:i] + active_edges[i+1:]
                            if not is_2_colorable(n, new_edges):
                                active_edges = new_edges
                                changed = True
                                if len(active_edges) <= best_count:
                                    break
                    
                    if len(active_edges) < best_count:
                        best_count = len(active_edges)
                        best_edges = active_edges[:]
                        found = True
                        
            if found:
                # If we found a solution for this E, do we need to try larger E?
                # No, we want minimum. 
                # But we should try other n's? Yes.
                # And we should try smaller E for this n? 
                # The loop is ascending E. So we checked 30..E-1 and failed.
                # So for this n, E is the best we found.
                # We can break the E loop for this n.
                break
    
    if best_edges:
        return {'n': n, 'edges': best_edges}
    else:
        # Fallback: return a known valid hypergraph if possible, or a dense one.
        # A dense hypergraph on 9 vertices is guaranteed non-2-colorable if it has enough edges?
        # The complete 5-uniform hypergraph on 9 vertices has 126 edges.
        # Is it non-2-colorable? 
        # Yes, because by Ramsey theory or simple counting, it's likely.
        # Actually, for n=9, if we take all edges, is it 2-colorable?
        # A 2-coloring of 9 vertices. By Pigeonhole, one color has at least 5 vertices.
        # If there are 5 vertices of the same color, the edge consisting of those 5 vertices is monochromatic.
        # So the complete hypergraph on 9 vertices is NOT 2-colorable.
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}

# EVOLVE-BLOCK-END
