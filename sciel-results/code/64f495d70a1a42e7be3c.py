import time
import random
from itertools import combinations
from collections import deque

def _is_2_colorable(n, edges):
    """
    Exact 2-colorability check using backtracking with unit propagation.
    Returns True if colorable, False otherwise.
    """
    if not edges:
        return True

    m = len(edges)
    edge_verts = [tuple(e) for e in edges]
    
    # Precompute vertex incidence
    v_to_edges = [[] for _ in range(n)]
    for i, edge in enumerate(edges):
        for v in edge:
            v_to_edges[v].append(i)

    # State: colors (0 or 1 or -1), e0_count, e1_count
    colors = [-1] * n
    e0_count = [0] * m
    e1_count = [0] * m

    def propagate():
        """
        Apply unit propagation. If an edge has 4 vertices of one color,
        the 5th must be the other color.
        Returns False if a conflict is found.
        """
        queue = deque()
        # Initialize queue with edges that have 4 colored vertices
        for ei in range(m):
            colored = e0_count[ei] + e1_count[ei]
            if colored == 4:
                queue.append(ei)
        
        while queue:
            ei = queue.popleft()
            # Check if this edge is still in a "forceable" state
            # It might have changed since being queued
            if e0_count[ei] + e1_count[ei] != 4:
                continue
            
            if e0_count[ei] == 4:
                forced_color = 1
            elif e1_count[ei] == 4:
                forced_color = 0
            else:
                continue # 2 and 2, no force

            # Find the uncolored vertex in this edge
            uncolored_v = -1
            for v in edge_verts[ei]:
                if colors[v] == -1:
                    uncolored_v = v
                    break
            
            if uncolored_v == -1:
                # All colored, but counts were 4? Impossible if consistent.
                # Unless 5 are colored and one count is 4? No, sum is 5.
                # If sum is 4, exactly one is uncolored.
                continue

            # Check for conflict: is uncolored_v already colored? No, we checked.
            # Is it forced to a different color?
            # It's uncolored, so we assign it.
            
            # But wait, what if uncolored_v is already forced by another edge?
            # We just assign it here.
            
            # Assign color
            colors[uncolored_v] = forced_color
            
            # Update counts for all edges containing uncolored_v
            for ej in v_to_edges[uncolored_v]:
                if forced_color == 0:
                    e0_count[ej] += 1
                else:
                    e1_count[ej] += 1
                
                # Check for immediate conflict
                if e0_count[ej] == 5 or e1_count[ej] == 5:
                    return False
                
                # If this edge is now forceable (4 colored), add to queue
                if e0_count[ej] + e1_count[ej] == 4:
                    queue.append(ej)
        
        return True

    def backtrack():
        # Check if all vertices are colored
        # Actually, we just need to ensure no conflicts and all edges are satisfied.
        # If a vertex is uncolored, it can be 0 or 1.
        # But if all edges are "safe" (not monochromatic), we are good.
        
        # Heuristic: Pick an uncolored vertex.
        # Prefer vertices in edges with the most colored vertices (closest to conflict).
        
        best_v = -1
        best_priority = -1
        
        # Find a vertex to assign
        for v in range(n):
            if colors[v] != -1:
                continue
            
            # Calculate priority: max colored count in any incident edge
            max_colored = 0
            for ei in v_to_edges[v]:
                colored = e0_count[ei] + e1_count[ei]
                if colored > max_colored:
                    max_colored = colored
            
            # If max_colored is 3, it's critical (next step might force).
            # If 2, less so.
            priority = max_colored
            if priority > best_priority:
                best_priority = priority
                best_v = v
        
        if best_v == -1:
            # All vertices colored. Check if any edge is monochromatic.
            # This should have been caught by propagation/conflict checks.
            for ei in range(m):
                if e0_count[ei] == 5 or e1_count[ei] == 5:
                    return False
            return True

        # Try both colors for best_v
        for c in [0, 1]:
            # Save state
            old_colors = colors[:]
            old_e0 = e0_count[:]
            old_e1 = e1_count[:]

            # Assign
            colors[best_v] = c
            
            # Update counts
            conflict = False
            for ei in v_to_edges[best_v]:
                if c == 0:
                    e0_count[ei] += 1
                else:
                    e1_count[ei] += 1
                
                if e0_count[ei] == 5 or e1_count[ei] == 5:
                    conflict = True
                    break

            if not conflict:
                # Propagate
                if propagate():
                    if backtrack():
                        return True
            
            # Restore state
            colors[:] = old_colors
            e0_count[:] = old_e0
            e1_count[:] = old_e1

        return False

    # Initial propagation
    if not propagate():
        return False

    return backtrack()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Known valid hypergraph: Random search with correct checker
    # We'll try to find a small one.
    
    best_edges_count = float('inf')
    best_result = None

    # Fallback: Complete graph on 9 vertices (126 edges) - Always non-2-colorable
    edges_9 = [list(e) for e in combinations(range(9), 5)]
    best_result = {'n': 9, 'edges': edges_9}
    best_edges_count = len(edges_9)

    # Try to find smaller ones
    # n=10, 11, 12...
    for n in range(10, 18):
        if time.time() > deadline:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        # Try different edge counts
        # Start with small, increase
        for target_size in range(10, 60):
            if time.time() > deadline:
                break
            
            # Try a few random samples
            for trial in range(20):
                if time.time() > deadline:
                    break
                
                current_edges = set(random.sample(all_possible, min(target_size, len(all_possible))))
                
                if not _is_2_colorable(n, list(current_edges)):
                    # Found non-2-colorable
                    # Try to prune
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        edges_list = list(current_edges)
                        random.shuffle(edges_list)
                        for e in edges_list:
                            if time.time() > deadline:
                                break
                            temp_edges = current_edges - {e}
                            if not _is_2_colorable(n, list(temp_edges)):
                                current_edges = temp_edges
                                improved = True
                                break
                            
                    if len(current_edges) < best_edges_count:
                        best_edges_count = len(current_edges)
                        best_result = {'n': n, 'edges': [list(e) for e in current_edges]}
                        if best_edges_count <= 30:
                            return best_result

    return best_result
