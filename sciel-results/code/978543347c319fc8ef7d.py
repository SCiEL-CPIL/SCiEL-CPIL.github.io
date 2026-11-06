import time
import random
from itertools import combinations

def is_2_colorable_fast(n, edges):
    """
    Check if a 5-uniform hypergraph is 2-colorable.
    Returns True if 2-colorable, False otherwise.
    """
    if not edges:
        return True
    
    num_edges = len(edges)
    
    # Convert edges to bitmasks
    edge_masks = []
    for e in edges:
        mask = 0
        for v in e:
            mask |= (1 << v)
        edge_masks.append(mask)
    
    # Precompute vertex -> list of edges
    vertex_edges = [[] for _ in range(n)]
    for i, em in enumerate(edge_masks):
        for v in range(n):
            if em & (1 << v):
                vertex_edges[v].append(i)
    
    # State
    c0 = [0] * num_edges
    c1 = [0] * num_edges
    color = [-1] * n
    
    # Use a stack for unit propagation to avoid re-scanning all edges repeatedly
    def propagate():
        # Initialize stack with all edges
        stack = list(range(num_edges))
        
        while stack:
            i = stack.pop()
            
            # Check for immediate conflict
            if c0[i] == 5 or c1[i] == 5:
                return False
            
            # Unit propagation
            # If 4 are 0, the 5th MUST be 1.
            if c0[i] == 4 and c1[i] == 0:
                # Find the unassigned vertex in this edge
                unassigned_v = -1
                for v in range(n):
                    if color[v] == -1 and (edge_masks[i] & (1 << v)):
                        unassigned_v = v
                        break
                if unassigned_v != -1:
                    color[unassigned_v] = 1
                    c1[i] += 1
                    # Update other edges and push them to stack
                    for other_e in vertex_edges[unassigned_v]:
                        c1[other_e] += 1
                        stack.append(other_e)
                        if c1[other_e] == 5:
                            return False
                    # If c1[i] becomes 5, it's a conflict, but we already checked c0[i]==4, c1[i]==0 -> c1[i] becomes 1.
                    # Wait, c0[i]==4, c1[i]==0. Assigning last vertex to 1 makes c1[i]=1. No conflict on this edge.
                    # But we need to check if any other edge becomes 5.
                    
            elif c1[i] == 4 and c0[i] == 0:
                unassigned_v = -1
                for v in range(n):
                    if color[v] == -1 and (edge_masks[i] & (1 << v)):
                        unassigned_v = v
                        break
                if unassigned_v != -1:
                    color[unassigned_v] = 0
                    c0[i] += 1
                    for other_e in vertex_edges[unassigned_v]:
                        c0[other_e] += 1
                        stack.append(other_e)
                        if c0[other_e] == 5:
                            return False
                    
        return True

    def backtrack():
        # Find the most constrained edge
        best_edge = -1
        best_unassigned = 6
        best_max_color = 0
        
        for i in range(num_edges):
            if c0[i] == 5 or c1[i] == 5:
                return False
            unassigned = 5 - c0[i] - c1[i]
            max_color = max(c0[i], c1[i])
            
            # Prioritize edges with fewer unassigned vertices
            if unassigned < best_unassigned or (unassigned == best_unassigned and max_color > best_max_color):
                best_unassigned = unassigned
                best_max_color = max_color
                best_edge = i
        
        if best_edge == -1:
            # All edges fully assigned
            return True
            
        if best_unassigned == 0:
            return False

        # Get unassigned vertices in best_edge
        candidates = []
        for v in range(n):
            if color[v] == -1 and (edge_masks[best_edge] & (1 << v)):
                candidates.append(v)
                
        if not candidates:
            return True
        
        # Try assigning each candidate
        for v in candidates:
            # Try color 0
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            saved_color = color[v]
            
            color[v] = 0
            ok = True
            for e_idx in vertex_edges[v]:
                c0[e_idx] += 1
                if c0[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            # Unwind
            color[v] = saved_color
            c0[:] = saved_c0
            c1[:] = saved_c1
            
            # Try color 1
            saved_c0 = c0[:]
            saved_c1 = c1[:]
            saved_color = color[v]
            
            color[v] = 1
            ok = True
            for e_idx in vertex_edges[v]:
                c1[e_idx] += 1
                if c1[e_idx] == 5:
                    ok = False
                    break
            
            if ok and propagate():
                if backtrack():
                    return True
            
            # Unwind
            color[v] = saved_color
            c0[:] = saved_c0
            c1[:] = saved_c1
            
        return False

    return backtrack()

def construct():
    start_time = time.time()
    # Hard limit 120s, leave 10s buffer for overhead/return
    deadline = start_time + 110
    
    best_edges = None
    best_n = 0
    best_m = 10**9
    
    # Strategy:
    # 1. Start with complete 5-uniform hypergraphs for n=9..14
    # 2. Greedily remove edges using randomized order to minimize count while keeping non-2-colorable
    # 3. Repeat with different random seeds to escape local optima
    
    # We can increase the number of attempts since the checker is faster
    for n in range(9, 15):
        if time.time() > deadline:
            break
            
        all_possible = list(combinations(range(n), 5))
        
        # Check if complete hypergraph is non-2-colorable
        if is_2_colorable_fast(n, [list(e) for e in all_possible]):
            continue
            
        current_edges = [list(e) for e in all_possible]
        current_m = len(current_edges)
        
        # Multiple restarts for each n
        # Increase attempts from 5 to 20
        for attempt in range(20):
            if time.time() > deadline:
                break
                
            # Use a weighted random order based on vertex degrees
            # Edges with lower total vertex degree are "easier" to remove
            vertex_degrees = [0] * n
            for edge in current_edges:
                for v in edge:
                    vertex_degrees[v] += 1
            
            # Score each edge by sum of vertex degrees (lower = easier to remove)
            edge_scores = []
            for edge in current_edges:
                score = sum(vertex_degrees[v] for v in edge)
                edge_scores.append(score)
            
            # Create order: sort by score (ascending), then shuffle within groups
            # This gives priority to "easier" edges while maintaining randomness
            order = sorted(range(len(current_edges)), key=lambda i: edge_scores[i])
            # Add some randomness to avoid getting stuck
            for _ in range(5):
                i = random.randint(0, len(order) - 1)
                j = random.randint(0, len(order) - 1)
                order[i], order[j] = order[j], order[i]
            
            improved = True
            while improved:
                improved = False
                for idx in order:
                    if time.time() > deadline:
                        break
                    if idx >= len(current_edges):
                        continue
                        
                    # Try removing edge at index idx
                    new_edges = current_edges[:idx] + current_edges[idx+1:]
                    
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        improved = True
                        # Reshuffle remaining indices to try different order
                        order = list(range(len(current_edges)))
                        random.shuffle(order)
                        break # Restart inner loop with new state
        
        if current_m < best_m:
            best_m = current_m
            best_edges = current_edges[:]
            best_n = n
    
    # If we found a very good solution, try to push further with local search on the best one
    if best_edges is not None and best_m > 30:
        n = best_n
        current_edges = best_edges[:]
        current_m = len(current_edges)
        
        # Phase 1: Greedy removal
        for attempt in range(50):
            if time.time() > deadline:
                break
            order = list(range(len(current_edges)))
            random.shuffle(order)
            
            improved = True
            while improved:
                improved = False
                for idx in order:
                    if time.time() > deadline:
                        break
                    if idx >= len(current_edges):
                        continue
                    new_edges = current_edges[:idx] + current_edges[idx+1:]
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        improved = True
                        order = list(range(len(current_edges)))
                        random.shuffle(order)
                        break
        
        if current_m < best_m:
            best_m = current_m
            best_edges = current_edges[:]
            best_n = n
        
        # Phase 2: Edge swap local search - try removing one edge and adding another
        if best_edges is not None and best_m > 30:
            n = best_n
            current_edges = best_edges[:]
            current_m = len(current_edges)
            edge_set = set(map(tuple, current_edges))
            
            # Generate candidate edges not in current set
            all_edges = list(combinations(range(n), 5))
            candidate_edges = [e for e in all_edges if tuple(e) not in edge_set]
            
            for attempt in range(30):
                if time.time() > deadline:
                    break
                
                # Try random edge swaps
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random edge to remove
                    idx = random.randint(0, len(current_edges) - 1)
                    removed_edge = current_edges[idx]
                    
                    # Pick a random candidate edge to add
                    if not candidate_edges:
                        break
                    add_edge = random.choice(candidate_edges)
                    
                    # Create new edge list
                    new_edges = current_edges[:idx] + current_edges[idx+1:] + [list(add_edge)]
                    
                    if not is_2_colorable_fast(n, new_edges):
                        current_edges = new_edges
                        current_m = len(current_edges)
                        edge_set = set(map(tuple, current_edges))
                        candidate_edges = [e for e in all_edges if tuple(e) not in edge_set]
                        break
                
                if current_m < best_m:
                    best_m = current_m
                    best_edges = current_edges[:]
                    best_n = n
            
    # Fallback if nothing found (should not happen for n=9)
    if best_edges is None:
        n = 9
        best_edges = [list(e) for e in combinations(range(n), 5)]
        best_n = n
        
    return {'n': best_n, 'edges': best_edges}

if __name__ == "__main__":
    res = construct()
    print(f"Found hypergraph with n={res['n']}, m={len(res['edges'])}")
