import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110.0
    d = 7
    nverts = 1 << d
    total_edges = d * nverts // 2  # 448
    
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
    squares = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(nverts):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ mask
                    
                    e1 = edge_to_idx[(min(v1, v2), max(v1, v2))]
                    e2 = edge_to_idx[(min(v2, v4), max(v2, v4))]
                    e3 = edge_to_idx[(min(v4, v3), max(v4, v3))]
                    e4 = edge_to_idx[(min(v3, v1), max(v3, v1))]
                    
                    squares.append([e1, e2, e3, e4])
    
    num_squares = len(squares) # 672
    
    # 3. Precompute edge_to_squares and square_edges
    edge_to_squares = [[] for _ in range(total_edges)]
    square_edges = [set(sq) for sq in squares]
    
    for s_idx, sq in enumerate(squares):
        for e_idx in sq:
            edge_to_squares[e_idx].append(s_idx)
            
    # Helper to compute cost
    def compute_uncovered(R):
        """
        Returns (uncovered_count_array, total_uncovered)
        """
        cover_count = [0] * num_squares
        for e in R:
            for f in edge_to_squares[e]:
                cover_count[f] += 1
        
        total_uncovered = 0
        for f in range(num_squares):
            if cover_count[f] == 0:
                total_uncovered += 1
        return cover_count, total_uncovered

    def greedy_hitting_set(rng):
        """
        Greedy Set Cover to find a hitting set R for all squares.
        """
        uncovered = set(range(num_squares))
        uncovered_count = [0] * total_edges
        
        for e in range(total_edges):
            uncovered_count[e] = len(edge_to_squares[e])
            
        R = set()
        
        while uncovered:
            max_count = -1
            best_edges = []
            for e in range(total_edges):
                if e in R:
                    continue
                if uncovered_count[e] > max_count:
                    max_count = uncovered_count[e]
                    best_edges = [e]
                elif uncovered_count[e] == max_count:
                    best_edges.append(e)
            
            if not best_edges:
                break
                
            e = rng.choice(best_edges)
            R.add(e)
            
            for f in edge_to_squares[e]:
                if f in uncovered:
                    uncovered.remove(f)
                    for e_other in square_edges[f]:
                        if e_other != e:
                            uncovered_count[e_other] -= 1
                            
        return R

    def simulated_annealing(R_init, rng, time_budget=5.0):
        """
        Simulated Annealing on the hitting set with soft constraints.
        Cost = |R| + mu * uncovered_squares
        """
        R = set(R_init)
        cover_count, total_uncovered = compute_uncovered(R)
        
        # Parameters
        T_start = 10.0
        T_end = 0.1
        mu = 100.0 # Penalty for uncovered squares
        
        best_R = set(R)
        best_cost = len(R) + mu * total_uncovered
        best_valid_size = len(R) if total_uncovered == 0 else float('inf')
        
        start_time = time.time()
        T = T_start
        
        # Precompute edge indices for faster access
        in_R = [False] * total_edges
        for e in R:
            in_R[e] = True
            
        while (time.time() - start_time) < time_budget:
            # Choose a random move
            # 1. Add random edge
            # 2. Remove random edge from R
            # 3. Swap random edge in R with random edge not in R
            
            move_type = rng.random()
            
            if move_type < 0.3:
                # Add
                e_new = rng.randint(0, total_edges - 1)
                if in_R[e_new]:
                    continue
                # Delta R: +1
                # Delta Uncovered: - (number of squares e_new covers that are currently uncovered)
                delta_uncovered = 0
                for f in edge_to_squares[e_new]:
                    if cover_count[f] == 0:
                        delta_uncovered -= 1
                delta_cost = 1 + mu * delta_uncovered
            elif move_type < 0.6:
                # Remove
                if not R:
                    continue
                e_old = rng.choice(list(R))
                # Delta R: -1
                # Delta Uncovered: + (number of squares e_old covers that are uniquely covered by e_old)
                delta_uncovered = 0
                for f in edge_to_squares[e_old]:
                    if cover_count[f] == 1:
                        delta_uncovered += 1
                delta_cost = -1 + mu * delta_uncovered
            else:
                # Swap
                if not R:
                    continue
                e_old = rng.choice(list(R))
                e_new = rng.randint(0, total_edges - 1)
                if e_new == e_old or in_R[e_new]:
                    continue
                # Delta R: 0
                # Delta Uncovered: complex
                # Squares losing coverage: f in edge_to_squares[e_old] where cover_count[f] == 1
                # Squares gaining coverage: f in edge_to_squares[e_new] where cover_count[f] == 0
                # But if f is in both, net change 0?
                # Let's calculate exactly.
                delta_uncovered = 0
                # Mark squares affected
                affected = set()
                for f in edge_to_squares[e_old]:
                    affected.add(f)
                for f in edge_to_squares[e_new]:
                    affected.add(f)
                
                # Simulate
                for f in affected:
                    old_cc = cover_count[f]
                    new_cc = old_cc
                    if f in edge_to_squares[e_old]:
                        new_cc -= 1
                    if f in edge_to_squares[e_new]:
                        new_cc += 1
                    if old_cc == 0 and new_cc > 0:
                        delta_uncovered -= 1
                    elif old_cc > 0 and new_cc == 0:
                        delta_uncovered += 1
                
                delta_cost = mu * delta_uncovered
            
            # Acceptance
            if delta_cost <= 0:
                accept = True
            else:
                # Metropolis criterion
                if T > 0:
                    prob = np.exp(-delta_cost / T)
                    accept = (rng.random() < prob)
                else:
                    accept = False
            
            if accept:
                # Apply move
                if move_type < 0.3:
                    R.add(e_new)
                    in_R[e_new] = True
                    for f in edge_to_squares[e_new]:
                        cover_count[f] += 1
                    total_uncovered += delta_uncovered
                elif move_type < 0.6:
                    R.remove(e_old)
                    in_R[e_old] = False
                    for f in edge_to_squares[e_old]:
                        cover_count[f] -= 1
                    total_uncovered += delta_uncovered
                else:
                    R.remove(e_old)
                    in_R[e_old] = False
                    R.add(e_new)
                    in_R[e_new] = True
                    for f in edge_to_squares[e_old]:
                        cover_count[f] -= 1
                    for f in edge_to_squares[e_new]:
                        cover_count[f] += 1
                    total_uncovered += delta_uncovered
                
                # Update best
                current_cost = len(R) + mu * total_uncovered
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_R = set(R)
                    if total_uncovered == 0:
                        best_valid_size = len(R)
            
            # Cooling
            T = max(T_end, T * 0.99)
            
            # If we found a valid solution, we might want to restart with lower T or different mu?
            # For now, just keep going.
        
        # Return the best valid solution found, or the best overall if none valid?
        # We must return a valid hitting set.
        if total_uncovered == 0:
            return R
        else:
            return best_R

    best_R = None
    best_size = total_edges + 1
    rng = random.Random(42)
    
    num_restarts = 50 # Reduced to allow more SA time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        R = greedy_hitting_set(rng)
        
        # Run SA
        R = simulated_annealing(R, rng, time_budget=1.5)
        
        size = len(R)
        if size < best_size:
            best_size = size
            best_R = set(R)
            
        if 448 - best_size >= 308:
            break

    # Final polish on best
    if best_R is not None and time.time() < deadline - 5:
        best_R = simulated_annealing(best_R, rng, time_budget=10.0)
        best_size = len(best_R)

    if best_R is None:
        best_R = set()
        
    selected_edges = []
    for e_idx in range(total_edges):
        if e_idx not in best_R:
            u, v = edges[e_idx]
            selected_edges.append([u, v])
            
    return selected_edges
