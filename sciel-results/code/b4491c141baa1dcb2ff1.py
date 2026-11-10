import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    d = 7
    n_vertices = 1 << d
    all_edges = []
    edge_index = {}
    
    # Generate all undirected edges of Q7
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                e = (x, y)
                all_edges.append(e)
                edge_index[e] = len(all_edges) - 1
                
    m = len(all_edges) # 448
    
    # Precompute for each edge, the other edges in the squares it belongs to.
    # An edge (u, v) with diff bit i belongs to squares for each j != i.
    # The square consists of (u,v), (u^2^j, v^2^j), (u, u^2^j), (v, v^2^j) ? No.
    # Let u and v differ in bit i.
    # Let j be another bit.
    # The 4 vertices are u, v, u^2^j, v^2^j.
    # The 4 edges are:
    # 1. (u, v) -- current
    # 2. (u^2^j, v^2^j) -- opposite edge in the face
    # 3. (u, u^2^j) -- side edge
    # 4. (v, v^2^j) -- side edge
    
    # For each edge index, store the indices of the 3 other edges for each of the 2 squares?
    # Wait, an edge in Q7 belongs to how many squares?
    # In Qd, an edge belongs to d-1 squares.
    # For d=7, each edge belongs to 6 squares.
    # Correction: My previous thought of 2 squares was for Q3 (cube faces).
    # In Q7, an edge is part of 6 distinct 2D faces (squares).
    
    # So for each edge, we need to check 6 squares.
    # If any square has the other 3 edges selected, we cannot add this edge.
    
    edge_squares = [[] for _ in range(m)]
    
    for x in range(n_vertices):
        for i in range(d):
            y = x ^ (1 << i)
            if x >= y: continue
            e_idx = edge_index[(x, y)]
            
            for j in range(d):
                if j == i: continue
                # The square is defined by bits i and j
                u = x
                v = y
                u2 = u ^ (1 << j)
                v2 = v ^ (1 << j)
                
                # Edges in this square:
                # e1 = (u, v) -> current
                # e2 = (u2, v2)
                # e3 = (u, u2)
                # e4 = (v, v2)
                
                e2 = (min(u2, v2), max(u2, v2))
                e3 = (min(u, u2), max(u, u2))
                e4 = (min(v, v2), max(v, v2))
                
                # Store indices of the other 3 edges
                other_edges = [edge_index[e2], edge_index[e3], edge_index[e4]]
                edge_squares[e_idx].append(other_edges)
                
    # Helper to check if edge can be added
    # selected is a set of edge indices
    def can_add(e_idx, selected):
        for square in edge_squares[e_idx]:
            if all(o in selected for o in square):
                return False
        return True

    best_count = 0
    best_set = set()
    
    # Multiple restarts
    while time.time() < deadline:
        # Randomized Greedy Initialization
        selected = set()
        indices = list(range(m))
        random.shuffle(indices)
        
        for idx in indices:
            if can_add(idx, selected):
                selected.add(idx)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_set = set(selected)
            
        # Local Search Improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each edge in selected and adding as many as possible
            # To speed up, we can randomize the order of removal candidates
            candidates = list(selected)
            random.shuffle(candidates)
            
            for e_to_remove in candidates:
                if e_to_remove not in selected:
                    continue
                
                # Temporarily remove
                selected.remove(e_to_remove)
                
                # Try to add all non-selected edges that are now valid
                added = []
                for idx in range(m):
                    if idx not in selected:
                        if can_add(idx, selected):
                            selected.add(idx)
                            added.append(idx)
                
                # Net gain
                gain = len(added) - 1
                if gain > 0:
                    improved = True
                    if len(selected) > best_count:
                        best_count = len(selected)
                        best_set = set(selected)
                    # If we improved, we can break and restart the scan
                    # or continue. Restarting is usually more effective to find new states.
                    break
                else:
                    # Revert
                    selected.add(e_to_remove)
                    for idx in added:
                        selected.remove(idx)
                        
            # If no improvement in a full pass, try a different strategy or stop
            # To avoid getting stuck, we might just restart the whole process if stuck
            # But the 'break' above handles improvement. If no improvement, 'improved' stays False.
            
        # If we are stuck in a local optimum, the next restart with different shuffle will help.
        # To diversify, we can also try "k-opt" moves but that's expensive.
        # Simple restarts with greedy + 1-opt (remove 1, add many) is decent.
        
    # Convert best_set back to list of edges
    result = [all_edges[i] for i in best_set]
    return result

if __name__ == "__main__":
    # Small test
    res = construct()
    print(f"Found {len(res)} edges")
