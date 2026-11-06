import time
import random
import numpy as np

def construct():
            start_time = time.time()
            deadline = start_time + 110  # 10s safety margin
            
            d = 7
            n_edges = 448
            n_squares = 672
            
            # Precompute edge -> squares mapping
            # Edge index e corresponds to (vertex, dim)
            # We need a list of squares for each edge.
            # Square index s corresponds to (dims i, j, fixed_bits)
            
            # Let's build the incidence lists
            edge_squares = [[] for _ in range(n_edges)]
            square_edges = [[] for _ in range(n_squares)]
            
            # Generate edges
            edges = []
            for x in range(128):
                for i in range(7):
                    y = x ^ (1 << i)
                    if x < y:
                        edges.append((x, y, i))
                        
            # Map edge to index
            edge_index = {}
            for idx, (x, y, i) in enumerate(edges):
                edge_index[(x, y)] = idx
                edge_index[(y, x)] = idx
                
            # Generate squares
            # Square defined by two dims i, j and a vertex v (with i, j bits set to 0? No, just the base vertex)
            # Actually, a square is defined by the set of 4 vertices.
            # Vertices: v, v^2^i, v^2^j, v^2^(i|j)
            # To avoid duplicates, we can iterate over v and i<j.
            # But v must be such that the square is unique?
            # If we fix v to have bits i and j as 0, we cover each square exactly once?
            # Yes.
            square_idx = 0
            for i in range(7):
                for j in range(i+1, 7):
                    mask = (1 << i) | (1 << j)
                    # v must have 0 in bits i and j
                    # Iterate over all v, skip if v has bit i or j set
                    for v in range(128):
                        if (v & mask) == 0:
                            v1 = v ^ (1 << i)
                            v2 = v ^ (1 << j)
                            v3 = v ^ (1 << i) ^ (1 << j)
                            
                            # Edges of this square
                            e1 = edge_index[(v, v1)]
                            e2 = edge_index[(v1, v3)]
                            e3 = edge_index[(v, v2)]
                            e4 = edge_index[(v2, v3)]
                            
                            square_edges[square_idx] = [e1, e2, e3, e4]
                            for e in [e1, e2, e3, e4]:
                                edge_squares[e].append(square_idx)
                            square_idx += 1
                            
            # Verify counts
            assert square_idx == 672
            for e in range(n_edges):
                assert len(edge_squares[e]) == 6
                
            # Initial State: Greedy
            selected = [False] * n_edges
            counts = [0] * n_squares
            
            for e in range(n_edges):
                # Check if adding e is valid
                valid = True
                for s in edge_squares[e]:
                    if counts[s] >= 3:
                        valid = False
                        break
                if valid:
                    selected[e] = True
                    for s in edge_squares[e]:
                        counts[s] += 1
                        
            current_size = sum(selected)
            best_size = current_size
            best_selected = selected[:]
            
            # Simulated Annealing
            # Penalty lambda
            lambda_val = 10.0  # High penalty to keep feasibility mostly
            T = 10.0
            cooling = 0.9999
            min_T = 0.01
            
            rng = random.Random(42)
            
            while time.time() < deadline and T > min_T:
                # Pick random edge
                e = rng.randint(0, n_edges - 1)
                
                # Calculate delta
                # If adding
                if not selected[e]:
                    delta_size = 1
                    delta_penalty = 0
                    for s in edge_squares[e]:
                        if counts[s] == 3:
                            delta_penalty += 1
                        # If counts[s] > 3, it's already bad, adding makes it worse?
                        # Penalty is max(0, count - 3).
                        # If count=4, pen=1. Add -> count=5, pen=2. Delta=1.
                        # If count=3, pen=0. Add -> count=4, pen=1. Delta=1.
                        # So delta_penalty is number of squares where counts[s] >= 3.
                        else:
                            if counts[s] > 3:
                                delta_penalty += 1
                else:
                    delta_size = -1
                    delta_penalty = 0
                    for s in edge_squares[e]:
                        # If count=4, pen=1. Remove -> count=3, pen=0. Delta=-1.
                        # If count=5, pen=2. Remove -> count=4, pen=1. Delta=-1.
                        # So delta_penalty is - (number of squares where counts[s] > 3).
                        if counts[s] > 3:
                            delta_penalty -= 1
                            
                delta_F = delta_size - lambda_val * delta_penalty
                
                if delta_F > 0 or rng.random() < np.exp(delta_F / T):
                    # Accept move
                    if not selected[e]:
                        selected[e] = True
                        for s in edge_squares[e]:
                            counts[s] += 1
                    else:
                        selected[e] = False
                        for s in edge_squares[e]:
                            counts[s] -= 1
                            
                    current_size += delta_size
                    
                    if current_size > best_size:
                        best_size = current_size
                        best_selected = selected[:]
                        
                T *= cooling
                
            # Final Polish: Greedy add
            # Try to add any edge that is currently valid
            changed = True
            while changed:
                changed = False
                for e in range(n_edges):
                    if not best_selected[e]:
                        valid = True
                        for s in edge_squares[e]:
                            if counts[s] >= 3: # Note: counts correspond to best_selected?
                                # Wait, counts are for current state, not best_selected.
                                # We should update counts to match best_selected or recompute.
                                # Better: run polish on the current state if it's the best, or just recompute counts for best_selected.
                                pass
            # Simpler: Reconstruct counts for best_selected and do one pass of greedy add.
            final_selected = best_selected[:]
            final_counts = [0] * n_squares
            for e in range(n_edges):
                if final_selected[e]:
                    for s in edge_squares[e]:
                        final_counts[s] += 1
                        
            # Greedy Add
            for e in range(n_edges):
                if not final_selected[e]:
                    valid = True
                    for s in edge_squares[e]:
                        if final_counts[s] >= 3:
                            valid = False
                            break
                    if valid:
                        final_selected[e] = True
                        for s in edge_squares[e]:
                            final_counts[s] += 1
            
            # Construct output
            out = []
            for e in range(n_edges):
                if final_selected[e]:
                    x, y, _ = edges[e]
                    out.append([x, y])
                    
            return out
