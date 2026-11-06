import random
import time
import sys

def construct():
    d = 7
    n_vertices = 1 << d
    edges = []
    edge_map = {}
    idx = 0
    for u in range(n_vertices):
        for i in range(d):
            v = u ^ (1 << i)
            if u < v:
                edge_map[(u, v)] = idx
                edges.append((u, v))
                idx += 1
    
    n_edges = len(edges)
    
    # Find all squares
    squares = []
    # A square is defined by two non-adjacent bits i, j and a vertex u.
    # The vertices are u, u^(1<<i), u^(1<<j), u^(1<<i)^(1<<j).
    # To avoid duplicates, we can iterate over u, i, j with i < j.
    # But we need to ensure we don't count the same square 4 times.
    # A square has 4 vertices. We can identify it by its 4 edges.
    # Or simply: iterate i < j. For each u, the square exists.
    # There are 21 pairs of bits. 128 vertices. 21*128 = 2704 squares?
    # Wait, each square is counted 4 times (once for each vertex).
    # So 21 * 128 / 4 = 672 squares. Correct.
    
    # We will store squares as tuples of 4 edge indices.
    square_set = set()
    for i in range(d):
        for j in range(i+1, d):
            mask_i = 1 << i
            mask_j = 1 << j
            for u in range(n_vertices):
                # Only process if u is the "minimal" vertex in some sense to avoid duplicates?
                # No, just generate and use a set of frozensets of edge indices.
                v1 = u
                v2 = u ^ mask_i
                v3 = u ^ mask_j
                v4 = u ^ mask_i ^ mask_j
                
                e1 = edge_map[(min(v1, v2), max(v1, v2))]
                e2 = edge_map[(min(v3, v4), max(v3, v4))]
                e3 = edge_map[(min(v1, v3), max(v1, v3))]
                e4 = edge_map[(min(v2, v4), max(v2, v4))]
                
                sq = frozenset([e1, e2, e3, e4])
                square_set.add(sq)
                
    squares = [list(sq) for sq in square_set]
    n_squares = len(squares)
    
    # Precompute edge to squares mapping
    edge_squares = [[] for _ in range(n_edges)]
    for s_idx, sq in enumerate(squares):
        for e in sq:
            edge_squares[e].append(s_idx)
    
    # State
    in_set = [False] * n_edges
    blockers = [0] * n_edges
    
    def can_add(e_idx):
        return blockers[e_idx] == 0
    
    def add_edge(e_idx):
        in_set[e_idx] = True
        for s_idx in edge_squares[e_idx]:
            sq = squares[s_idx]
            # For each other edge in this square, check if the other 3 are in set
            for f in sq:
                if f == e_idx:
                    continue
                # Check if all other 3 edges in sq (excluding f) are in set
                # Since e_idx is in set, we just need to check the other 2
                others = [g for g in sq if g != f and g != e_idx]
                if in_set[others[0]] and in_set[others[1]]:
                    blockers[f] += 1
    
    def remove_edge(e_idx):
        in_set[e_idx] = False
        for s_idx in edge_squares[e_idx]:
            sq = squares[s_idx]
            for f in sq:
                if f == e_idx:
                    continue
                others = [g for g in sq if g != f and g != e_idx]
                if in_set[others[0]] and in_set[others[1]]:
                    blockers[f] -= 1
    
    def count_edges():
        return sum(in_set)
    
    def polish():
        # Try to add any edge that is currently addable
        improved = True
        while improved:
            improved = False
            for e in range(n_edges):
                if not in_set[e] and can_add(e):
                    add_edge(e)
                    improved = True
    
    best_count = 0
    best_in_set = [False] * n_edges
    
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Multi-restart SA
    restart_time = 10.0 # 10 seconds per restart
    max_restarts = int((deadline - start_time) / restart_time)
    
    for restart in range(max_restarts):
        # Initialize
        # Random greedy
        in_set = [False] * n_edges
        blockers = [0] * n_edges
        
        # Greedy initialization: shuffle and add
        order = list(range(n_edges))
        random.shuffle(order)
        for e in order:
            if can_add(e):
                add_edge(e)
        
        polish()
        cur_count = count_edges()
        
        if cur_count > best_count:
            best_count = cur_count
            best_in_set = in_set[:]
        
        # SA
        T = 5.0
        cooling = 0.9999
        min_T = 0.1
        
        end_time = start_time + (restart + 1) * restart_time
        if end_time > deadline:
            end_time = deadline - 1.0 # Safety margin
        
        while time.time() < end_time:
            move_type = random.random()
            
            if move_type < 0.4:
                # Add
                e_idx = random.randrange(n_edges)
                if not in_set[e_idx] and can_add(e_idx):
                    add_edge(e_idx)
                    cur_count += 1
                    if cur_count > best_count:
                        best_count = cur_count
                        best_in_set = in_set[:]
            elif move_type < 0.7:
                # Remove
                # Find an edge in set
                # Optimization: maintain a list of in_set edges?
                # For now, rejection sampling
                while True:
                    e_idx = random.randrange(n_edges)
                    if in_set[e_idx]:
                        break
                remove_edge(e_idx)
                cur_count -= 1
                # Accept with prob exp(-1/T)
                if random.random() > (2.718281828459045 ** (-1.0 / T)):
                    # Revert
                    add_edge(e_idx)
                    cur_count += 1
            else:
                # Swap
                # Pick e_out in set, e_in not in set
                while True:
                    e_out = random.randrange(n_edges)
                    if in_set[e_out]:
                        break
                while True:
                    e_in = random.randrange(n_edges)
                    if not in_set[e_in]:
                        break
                
                # Try swap
                remove_edge(e_out)
                if can_add(e_in):
                    add_edge(e_in)
                    # Swap is neutral in count, but might enable future adds
                    # Accept with prob 1 (or exp(0/T)=1)
                    # Actually, standard SA accepts neutral moves to mix
                else:
                    # Revert
                    add_edge(e_out)
                
                T *= cooling
                if T < min_T:
                    T = min_T
        
        # Polish after SA
        polish()
        cur_count = count_edges()
        if cur_count > best_count:
            best_count = cur_count
            best_in_set = in_set[:]
        
        # Check time
        if time.time() > deadline - 2.0:
            break
    
    # Final polish
    # Restore best state
    in_set = best_in_set[:]
    blockers = [0] * n_edges
    # Recalculate blockers for the best state?
    # It's safer to just reconstruct the state from scratch for the final polish
    # to ensure consistency.
    
    in_set = [False] * n_edges
    blockers = [0] * n_edges
    for e in range(n_edges):
        if best_in_set[e]:
            # We need to add them in a way that doesn't violate constraints?
            # But best_in_set is a valid set.
            # We can just set in_set and recalculate blockers.
            pass
    
    # Recalculate blockers for best_in_set
    in_set = best_in_set[:]
    blockers = [0] * n_edges
    for e in range(n_edges):
        if in_set[e]:
            # This add_edge function assumes blockers are updated.
            # We need a bulk initialization.
            pass
            
    # Let's just do a simple bulk check for the final polish
    # Actually, the `polish` function relies on `can_add` which relies on `blockers`.
    # If we don't have correct `blockers`, `polish` will fail.
    
    # Recalculate blockers correctly for the current `in_set`
    blockers = [0] * n_edges
    for s_idx, sq in enumerate(squares):
        # Count how many edges in sq are in set
        cnt = 0
        for e in sq:
            if in_set[e]:
                cnt += 1
        if cnt == 3:
            # The missing edge is blocked by this square
            for e in sq:
                if not in_set[e]:
                    blockers[e] += 1
        # If cnt == 4, it's an invalid state, but best_in_set should be valid.
        
    polish()
    
    result = []
    for e_idx in range(n_edges):
        if in_set[e_idx]:
            u, v = edges[e_idx]
            result.append([u, v])
    
    return result
