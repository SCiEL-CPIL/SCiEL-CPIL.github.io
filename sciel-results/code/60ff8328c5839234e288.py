import time
import random
from itertools import combinations

def is_2_colorable(n, edges):
    # Backtracking with unit propagation
    # colors: -1 unassigned, 0/1 assigned
    colors = [-1] * n
    # Precompute edge indices
    edge_indices = [list(e) for e in edges]
    
    def propagate(idx, color):
        # Unit propagation: if an edge has k-1 vertices of one color, the last must be the other?
        # No, if an edge has 5 vertices, and 4 are color 0, the 5th MUST be color 1 to avoid mono.
        # If the 5th is already 0, contradiction.
        # If the 5th is -1, it must be 1.
        # This is 2-SAT like? No, it's a constraint satisfaction problem.
        # Actually, "no proper 2-coloring" means every edge is NOT bichromatic.
        # So every edge is monochromatic.
        # Wait, "no proper 2-coloring (no colouring of the vertices leaves every edge with both colours)".
        # This means: There is NO coloring such that EVERY edge has both colors.
        # Equivalently: For EVERY coloring, there EXISTS an edge that is monochromatic.
        # This is the definition of a non-2-colorable hypergraph (chromatic number > 2).
        
        # So, we are checking if there exists a coloring where ALL edges are bichromatic.
        # If such a coloring exists, it is 2-colorable.
        # If no such coloring exists, it is NOT 2-colorable.
        
        # So `is_2_colorable` should return True if a valid bichromatic coloring exists.
        
        stack = [(idx, color)]
        while stack:
            i, c = stack.pop()
            if colors[i] != -1:
                if colors[i] != c:
                    return False
            else:
                colors[i] = c
                # Check all edges containing i
                for e in edges:
                    if i in e:
                        # Count colors in e
                        cnt0 = 0
                        cnt1 = 0
                        unassigned = -1
                        for v in e:
                            if colors[v] == 0:
                                cnt0 += 1
                            elif colors[v] == 1:
                                cnt1 += 1
                            else:
                                unassigned = v
                        # If we have 5 vertices in the edge:
                        # If cnt0 == 5 or cnt1 == 5, contradiction (mono edge found, but we want bichromatic).
                        # Wait, the condition for "proper 2-coloring" is that EVERY edge is bichromatic.
                        # So if any edge becomes monochromatic, this branch is invalid.
                        # If an edge has 4 of one color, the 5th MUST be the other to be bichromatic.
                        if cnt0 == 4:
                            if unassigned == -1:
                                # All assigned, check if bichromatic
                                if cnt1 == 0: return False
                                # else ok
                            else:
                                if colors[unassigned] == 0: return False
                                # must be 1
                                if colors[unassigned] == -1:
                                    stack.append((unassigned, 1))
                        elif cnt1 == 4:
                            if unassigned == -1:
                                if cnt0 == 0: return False
                            else:
                                if colors[unassigned] == 1: return False
                                if colors[unassigned] == -1:
                                    stack.append((unassigned, 0))
                        elif cnt0 == 5 or cnt1 == 5:
                            return False
        return True

    def backtrack(idx):
        if idx == n:
            return True
        # Try 0
        if propagate(idx, 0):
            if backtrack(idx + 1):
                return True
            # Undo? propagate modifies colors. We need to restore.
            # This implementation is flawed because propagate doesn't return a state to undo.
            # We need a proper backtracking with state restoration.
        # Try 1
        if propagate(idx, 1):
            if backtrack(idx + 1):
                return True
        return False

    # The above propagate is destructive and doesn't handle backtracking well.
    # Let's use a simpler DFS for small n.
    
    def dfs(idx):
        if idx == n:
            return True
        for c in [0, 1]:
            colors[idx] = c
            # Check if any edge is now monochromatic
            valid = True
            for e in edges:
                if idx in e:
                    # Check if e is mono
                    cnt0 = sum(1 for v in e if colors[v] == 0)
                    cnt1 = sum(1 for v in e if colors[v] == 1)
                    # If all assigned in e
                    if all(colors[v] != -1 for v in e):
                        if cnt0 == 5 or cnt1 == 5:
                            valid = False
                            break
            if valid:
                if dfs(idx + 1):
                    return True
            colors[idx] = -1
        return False

    return dfs(0)

def construct():
    # Seed: 126 edges
    best_n = 9
    best_edges = [list(e) for e in combinations(range(9), 5)]
    best_m = len(best_edges)
    
    start_time = time.time()
    deadline = 110.0
    
    # Search for smaller non-2-colorable hypergraphs
    # Try n from 5 to 15
    for n in range(5, 16):
        if time.time() > deadline:
            break
        # Max edges to try for this n
        # We want to find a small m
        # Start with small m, go up
        # But random search for specific m is hard.
        # Instead, try random hypergraphs with increasing density.
        
        # For each n, try to find a non-2-colorable hypergraph with m < best_m
        # We will sample random hypergraphs with m edges.
        # Start with m = 29 (lower bound) up to best_m - 1
        
        # To be efficient, we can try a few random samples for each m.
        # But 29 to 125 is a lot.
        # Let's focus on m around 30-50.
        
        # Actually, let's just try random hypergraphs with m edges.
        # If we find one with m < best_m, update best.
        
        # We will iterate m from 29 to best_m - 1
        # But this is too slow if we do it for all n.
        
        # Strategy:
        # For n in 5..15:
        #   For m in 29..min(60, best_m-1):
        #     Try 10 random hypergraphs with m edges.
        #     If any is non-2-colorable, update best.
        
        max_m_to_try = min(60, best_m - 1)
        for m in range(29, max_m_to_try + 1):
            if time.time() > deadline:
                break
            # Try 5 random samples
            for _ in range(5):
                if time.time() > deadline:
                    break
                # Generate random m edges
                all_edges = list(combinations(range(n), 5))
                random.shuffle(all_edges)
                edges = [list(e) for e in all_edges[:m]]
                
                # Check if 2-colorable
                if not is_2_colorable(n, edges):
                    # Found a non-2-colorable hypergraph
                    if m < best_m:
                        best_m = m
                        best_n = n
                        best_edges = edges
                    # We can break inner loop, maybe even outer m loop if we want smaller m
                    # But we want to find even smaller m?
                    # The loop goes m from 29 up. So we found one with current m.
                    # We can break the m loop to try smaller n? No, we are iterating n outer.
                    # We should break the m loop to try next n? 
                    # Actually, if we found one with m, we don't need to try larger m for this n.
                    break
            # If we found one with m, we don't need to try larger m for this n
            # But we might find a smaller m with a different n?
            # The m loop starts at 29. So we try 29, 30, ...
            # If we find one at m=30, we don't try m=31 for this n.
            # Then we go to next n.
            # This is good.
        
    return {'n': best_n, 'edges': best_edges}
