import time
import random
import sys
from itertools import combinations

def _is_non_2_colorable(n, edges):
    """
    Returns True if the hypergraph is NOT 2-colorable (no proper 2-coloring exists).
    Returns False if it IS 2-colorable.
    Uses backtracking with unit propagation.
    """
    if not edges:
        return False  # Empty hypergraph is 2-colorable
    
    # Precompute vertex to edges mapping
    v_to_e = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_to_e[v].append(i)
    
    # State
    colors = [-1] * n
    e_c0 = [0] * len(edges)
    e_c1 = [0] * len(edges)
    
    # We'll use a recursive search with state copying for simplicity and correctness.
    # n <= 26, E <= 200. State copy is O(n+E) ~ 226.
    # Depth <= n.
    # To speed up, we use a heuristic to pick the next vertex.
    
    def search(colors, e_c0, e_c1):
        # 1. Check for conflicts and propagate
        changed = True
        while changed:
            changed = False
            # Check for conflicts
            for i in range(len(edges)):
                if e_c0[i] == 5 or e_c1[i] == 5:
                    return False  # Conflict found, not 2-colorable path
            
            # Propagate
            for i in range(len(edges)):
                c0 = e_c0[i]
                c1 = e_c1[i]
                unassigned = 5 - c0 - c1
                if unassigned == 0:
                    continue
                if c0 == 4:
                    # Must assign 1 to the unassigned vertex in this edge
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 1
                            e_c1[i] += 1
                            for j in v_to_e[v]:
                                if j != i:
                                    e_c1[j] += 1
                            changed = True
                            break
                elif c1 == 4:
                    # Must assign 0 to the unassigned vertex in this edge
                    for v in edges[i]:
                        if colors[v] == -1:
                            colors[v] = 0
                            e_c0[i] += 1
                            for j in v_to_e[v]:
                                if j != i:
                                    e_c0[j] += 1
                            changed = True
                            break
        
        # After propagation, check conflicts again (in case propagation caused a conflict)
        for i in range(len(edges)):
            if e_c0[i] == 5 or e_c1[i] == 5:
                return False
        
        # Find unassigned vertices
        unassigned = [v for v in range(n) if colors[v] == -1]
        if not unassigned:
            return True  # All assigned, no conflict -> 2-colorable
        
        # Heuristic: Pick a vertex that appears in the most "critical" edges?
        # Or just pick the first one.
        # Better heuristic: Pick a vertex in an edge with the most assigned vertices?
        # Let's try: pick vertex v that maximizes the number of edges where v is unassigned
        # but the edge has 3 or 4 assigned vertices.
        # Simpler: just pick the first unassigned vertex.
        v = unassigned[0]
        
        # Try color 0
        new_colors = colors[:]
        new_e_c0 = e_c0[:]
        new_e_c1 = e_c1[:]
        new_colors[v] = 0
        for j in v_to_e[v]:
            new_e_c0[j] += 1
        if search(new_colors, new_e_c0, new_e_c1):
            return True  # Found a 2-coloring
        
        # Try color 1
        new_colors = colors[:]
        new_e_c0 = e_c0[:]
        new_e_c1 = e_c1[:]
        new_colors[v] = 1
        for j in v_to_e[v]:
            new_e_c1[j] += 1
        if search(new_colors, new_e_c0, new_e_c1):
            return True  # Found a 2-coloring
        
        return False  # No 2-coloring found in this branch
    
    # Start search
    return not search(colors, e_c0, e_c1)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Search for small non-2-colorable hypergraphs
    # The best known is 51 edges.
    # We want to find something <= 51.
    
    # Strategy:
    # 1. Generate random hypergraphs with n vertices and m edges.
    # 2. Check if non-2-colorable.
    # 3. If yes, try to minimize edges by greedy removal.
    # 4. Keep the best (fewest edges) found.
    
    # To increase chances of finding small ones, we should try a range of n and m.
    # Small n (e.g., 10-15) might have fewer edges needed?
    # Actually, the property B threshold m(5) is about the number of edges.
    # Known: m(5) <= 51.
    
    # Let's try n from 10 to 20.
    # For each n, try different numbers of edges, starting from low?
    # No, random generation with low edge count is likely 2-colorable.
    # It's better to generate dense ones and reduce.
    
    for n in range(10, 21):
        if time.time() > deadline:
            break
        
        # Try a few random instances for this n
        for trial in range(10):
            if time.time() > deadline:
                break
            
            # Number of edges to start with
            # C(n, 5) is the max.
            # For n=10, C(10,5)=252.
            # For n=15, C(15,5)=3003.
            # We want to find instances with ~50 edges.
            # So we should pick a subset of edges.
            
            all_edges = list(combinations(range(n), 5))
            random.shuffle(all_edges)
            
            # Pick a random number of edges, say 60-80, to ensure high chance of being non-2-colorable
            # If it's 2-colorable, we discard.
            # If it's non-2-colorable, we reduce.
            
            m_start = 60 + random.randint(0, 20)
            if m_start > len(all_edges):
                m_start = len(all_edges)
            
            edges = [list(e) for e in all_edges[:m_start]]
            
            # Check if non-2-colorable
            if _is_non_2_colorable(n, edges):
                # It is non-2-colorable. Now reduce.
                current_edges = edges[:]
                
                # Greedy reduction
                # Shuffle edges to make reduction order random
                random.shuffle(current_edges)
                
                i = 0
                while i < len(current_edges):
                    if time.time() > deadline:
                        break
                    
                    e_to_remove = current_edges[i]
                    test_edges = current_edges[:i] + current_edges[i+1:]
                    
                    if _is_non_2_colorable(n, test_edges):
                        # Removal successful
                        current_edges = test_edges
                        # Do not increment i, check next edge at same index
                    else:
                        i += 1
                
                # Update best
                if not best_edges or len(current_edges) < len(best_edges):
                    best_n = n
                    best_edges = current_edges[:]
                    # print(f"Found non-2-colorable with {len(best_edges)} edges on {best_n} vertices", file=sys.stderr)
                    
                    # If we found 51 or less, we can stop early?
                    # The task says best known is 51.
                    # If we find 50, that's an improvement.
                    # But 51 is the target.
                    # Let's keep searching for a bit to see if we can do better or just confirm 51.
                    # Actually, the score is based on the number of edges.
                    # So we want the minimum.
                    
                    # If we hit 51, we might still want to try to find 50.
                    # But 50 might be rare.
                    
                    # Let's continue searching if we have time.
    
    if not best_edges:
        # Fallback: Return a known non-2-colorable hypergraph
        # K_9^5 has 126 edges.
        return {'n': 9, 'edges': [list(e) for e in combinations(range(9), 5)]}
    else:
        return {'n': best_n, 'edges': best_edges}
