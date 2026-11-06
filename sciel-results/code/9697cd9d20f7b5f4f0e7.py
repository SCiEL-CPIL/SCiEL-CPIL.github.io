import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2
    
    # 1. Build edges
    edges = []
    edge_to_idx = {}
    for u in range(nverts):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                idx = len(edges)
                edges.append((u, v))
                edge_to_idx[(u, v)] = idx
                edge_to_idx[(v, u)] = idx

    # 2. Build squares (faces)
    # A square is defined by two vertices differing in 2 bits.
    # Or by a center and two axes.
    # Let's iterate over all pairs of dimensions i, j and all vertices x where bits i,j are 0 (to avoid duplicates)
    squares = [] # List of sets of 4 edge indices
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(nverts):
                # Only process if bits i and j are 0 to ensure uniqueness
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ mask
                    
                    e1 = edge_to_idx[(min(v1, v2), max(v1, v2))]
                    e2 = edge_to_idx[(min(v2, v4), max(v2, v4))]
                    e3 = edge_to_idx[(min(v4, v3), max(v4, v3))]
                    e4 = edge_to_idx[(min(v3, v1), max(v3, v1))]
                    
                    squares.append({e1, e2, e3, e4})

    # 3. Map edge to squares
    edge_to_squares = [[] for _ in range(total_edges)]
    for s_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_to_squares[e_idx].append(s_idx)

    # 4. Precompute for each edge, for each square it's in, the other 3 edges
    # edge_other_edges[e_idx] is a list of tuples (s_idx, [other_e1, other_e2, other_e3])
    edge_other_edges = [[] for _ in range(total_edges)]
    for s_idx, sq in enumerate(squares):
        sq_list = list(sq)
        for k, e_idx in enumerate(sq_list):
            others = [sq_list[m] for m in range(4) if m != k]
            edge_other_edges[e_idx].append((s_idx, others))

    def can_add(e_idx, selected):
        for s_idx, others in edge_other_edges[e_idx]:
            # Check if all others are in selected
            if all(o in selected for o in others):
                return False
        return True

    def get_blocking_square(e_idx, selected):
        for s_idx, others in edge_other_edges[e_idx]:
            if all(o in selected for o in others):
                return s_idx, others
        return None, None

    best_count = 0
    best_set = set()
    
    # Use a fixed seed for reproducibility in local runs, but vary for search
    # The prompt implies we should just return the best found.
    
    while time.time() < deadline:
        selected = set()
        # Random greedy initialization
        order = list(range(total_edges))
        random.shuffle(order)
        
        for e_idx in order:
            if can_add(e_idx, selected):
                selected.add(e_idx)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_set = set(selected)
            
        # Local Search
        # We try to improve the current selected set
        # Strategy: Try to add rejected edges. If blocked, try to swap.
        # We need a list of edges not in selected.
        # Since total edges is small, we can just iterate.
        
        # To make it faster, we can pick random edges to try.
        # Let's do a few passes of random sampling.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try adding any edge
            # Randomly permute all edges to try them in random order
            # Or just pick random edges not in selected
            
            # For efficiency, let's just iterate through all edges not in selected
            # 448 edges is small.
            
            # We'll try to add edges in random order
            candidates = [e for e in range(total_edges) if e not in selected]
            random.shuffle(candidates)
            
            for e_idx in candidates:
                if can_add(e_idx, selected):
                    selected.add(e_idx)
                    improved = True
                else:
                    # Try to swap
                    # Find a blocking square
                    s_idx, others = get_blocking_square(e_idx, selected)
                    if others:
                        # Randomly pick one of the 'others' to remove
                        # This is a heuristic: removing a random edge from the blocking set
                        # might unblock e_idx, but we need to check if e_idx is still blocked by other squares
                        # after removal.
                        
                        # Actually, removing one edge from 'others' breaks the C4 for this square.
                        # But e_idx might be blocked by another square.
                        # So we should check if e_idx becomes addable.
                        
                        # Let's try removing each of the 'others' one by one?
                        # Or just pick one randomly.
                        # If we pick randomly, we might fail often.
                        # Let's try all 3 options for the blocking square.
                        
                        for f_idx in others:
                            # Remove f_idx
                            selected.remove(f_idx)
                            
                            # Check if e_idx can now be added
                            if can_add(e_idx, selected):
                                selected.add(e_idx)
                                # Net change: -1 + 1 = 0. Count doesn't increase.
                                # But structure changed. Might allow further adds.
                                # We mark improved to restart the candidate loop?
                                # Or just continue.
                                # Since count didn't increase, we shouldn't update best_count yet.
                                # But we might have enabled other adds.
                                # Let's break out of the 'others' loop and continue to next candidate.
                                # Actually, if we swap, we should probably restart the candidate search
                                # because the set of 'candidates' (edges not in selected) has changed.
                                # And the blocking status of other edges has changed.
                                
                                # For simplicity and to avoid complex state management, 
                                # let's just accept the swap and break to restart the pass.
                                improved = True
                                break
                            else:
                                # Revert
                                selected.add(f_idx)
                                
                        if improved:
                            break # Restart the candidate loop with new state
            
            if not improved:
                # If no adds or swaps happened, try a more aggressive move?
                # Or just stop this iteration.
                pass

        if len(selected) > best_count:
            best_count = len(selected)
            best_set = set(selected)

    # Convert best_set to list of [u, v]
    result = []
    for e_idx in best_set:
        u, v = edges[e_idx]
        result.append([u, v])
        
    return result
