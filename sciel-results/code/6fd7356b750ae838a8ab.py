"""Strong edge colourings via Local Search on Conflict Graph."""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        # Generate random graph with max degree d
        # The seed program uses a specific generation method, we must replicate it exactly.
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_graph(edges, n):
        """
        Build a conflict graph where nodes are edge indices.
        Two nodes are connected if the corresponding edges in the original graph
        share a vertex or are joined by an edge (i.e., they cannot have the same color).
        Returns both set-based and bitmask-based adjacency for speed.
        """
        m = len(edges)
        # Map vertex to list of incident edge indices
        vertex_to_edges = defaultdict(list)
        for i, (u, v) in enumerate(edges):
            vertex_to_edges[u].append(i)
            vertex_to_edges[v].append(i)
        
        # Build adjacency sets for conflict graph
        conflict_sets = [set() for _ in range(m)]
        
        # 1. Edges sharing a vertex conflict
        for u in range(n):
            inc_edges = vertex_to_edges[u]
            for i in range(len(inc_edges)):
                for j in range(i + 1, len(inc_edges)):
                    e_i = inc_edges[i]
                    e_j = inc_edges[j]
                    conflict_sets[e_i].add(e_j)
                    conflict_sets[e_j].add(e_i)
        
        # 2. Edges joined by an edge conflict
        vertex_adj = [set() for _ in range(n)]
        for u, v in edges:
            vertex_adj[u].add(v)
            vertex_adj[v].add(u)
        
        for i, (u, v) in enumerate(edges):
            # Neighbors of u (excluding v)
            for nu in vertex_adj[u]:
                if nu == v:
                    continue
                for j in vertex_to_edges[nu]:
                    if j != i:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)
            # Neighbors of v (excluding u)
            for nv in vertex_adj[v]:
                if nv == u:
                    continue
                for j in vertex_to_edges[nv]:
                    if j != i:
                        conflict_sets[i].add(j)
                        conflict_sets[j].add(i)
        
        # Convert to bitmask representation for fast operations
        conflict_masks = [0] * m
        for i in range(m):
            mask = 0
            for j in conflict_sets[i]:
                mask |= (1 << j)
            conflict_masks[i] = mask
        
        return conflict_sets, conflict_masks

    def greedy_color(m, conflict_masks, order=None, rng=None):
        """Greedy coloring with optional custom order using bitmask conflict checks."""
        if m == 0:
            return []
        if order is None:
            # Order by conflict degree (highest first)
            order = sorted(range(m), key=lambda i: -bin(conflict_masks[i]).count('1'))
        
        colours = [-1] * m
        for idx in order:
            # Find blocked colors using bitmask
            blocked = 0
            for nb in range(m):
                if (conflict_masks[idx] >> nb) & 1:
                    if colours[nb] >= 0:
                        blocked |= (1 << colours[nb])
            c = 0
            while (blocked >> c) & 1:
                c += 1
            colours[idx] = c
        return colours

    def try_remove_color_fast(colours, conflict_masks, c_remove, rng, m):
        """Try to remove color c_remove by reassigning all edges with that color using bitmask checks."""
        edges_with_c = [i for i in range(m) if colours[i] == c_remove]
        if not edges_with_c:
            return True  # Already removed
        
        old_colours = colours[:]
        rng.shuffle(edges_with_c)
        
        for idx in edges_with_c:
            # Build blocked mask for this vertex
            blocked = 0
            for nb in range(m):
                if (conflict_masks[idx] >> nb) & 1:
                    if colours[nb] >= 0:
                        blocked |= (1 << colours[nb])
            
            # Find available colors below c_remove
            available = []
            for c_new in range(c_remove):
                if not ((blocked >> c_new) & 1):
                    available.append(c_new)
            
            if not available:
                colours[:] = old_colours
                return False
            else:
                colours[idx] = rng.choice(available)
        
        return True

    def multi_swap_search(colours, conflict_masks, rng, m, time_limit):
        """
        Enhanced local search with multi-swap moves and better heuristics.
        """
        improved = True
        no_improve_count = 0
        max_no_improve = 50
        
        while improved and time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False
            no_improve_count += 1
            max_c = max(colours) if m > 0 else -1
            
            if max_c < 1:
                break
            
            # Try to remove colors from max_c down to 1
            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    break
                
                edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                if not edges_with_c:
                    continue
                
                # First try direct reassignment with multiple attempts
                success = False
                for attempt in range(10):
                    if try_remove_color_fast(colours, conflict_masks, c_remove, rng, m):
                        success = True
                        break
                
                if success:
                    improved = True
                    no_improve_count = 0
                    break
                
                # If direct reassignment fails, try swap-based approach
                # For each edge with color c_remove, try to find a swap that makes it reassignable
                for idx in edges_with_c:
                    if time.time() > time_limit:
                        break
                    
                    # Find which colors are blocked for idx
                    blocked = 0
                    for nb in range(m):
                        if (conflict_masks[idx] >> nb) & 1:
                            if colours[nb] >= 0:
                                blocked |= (1 << colours[nb])
                    
                    # Try to unblock one of the colors < c_remove by swapping
                    for c_target in range(c_remove):
                        if not ((blocked >> c_target) & 1):
                            continue
                        
                        # Find an edge j with color c_target that conflicts with idx
                        for nb in range(m):
                            if time.time() > time_limit:
                                break
                            if not ((conflict_masks[idx] >> nb) & 1):
                                continue
                            if colours[nb] != c_target:
                                continue
                            
                            # Try to move nb to another color
                            old_colours = colours[:]
                            for c_new in range(c_remove):
                                if c_new == c_target:
                                    continue
                                # Check if c_new is valid for nb
                                valid = True
                                for nb2 in range(m):
                                    if (conflict_masks[nb] >> nb2) & 1:
                                        if colours[nb2] == c_new:
                                            valid = False
                                            break
                                if valid:
                                    colours[nb] = c_new
                                    # Now try to reassign idx
                                    available = []
                                    for c_new2 in range(c_remove):
                                        valid2 = True
                                        for nb3 in range(m):
                                            if (conflict_masks[idx] >> nb3) & 1:
                                                if colours[nb3] == c_new2:
                                                    valid2 = False
                                                    break
                                        if valid2:
                                            available.append(c_new2)
                                    
                                    if available:
                                        colours[idx] = rng.choice(available)
                                        success = True
                                        break
                                    else:
                                        colours[:] = old_colours
                            if success:
                                break
                        if success:
                            break
                    if success:
                        break
                
                if success:
                    improved = True
                    no_improve_count = 0
                    break
        
        return colours

    def high_precision_polish(colours, conflict_masks, rng, m, time_limit):
        """Final high-precision polish pass with more iterations and tighter heuristics."""
        improved = True
        no_improve_count = 0
        max_no_improve = 100
        
        while improved and time.time() < time_limit and no_improve_count < max_no_improve:
            improved = False
            no_improve_count += 1
            max_c = max(colours) if m > 0 else -1
            
            if max_c < 1:
                break
            
            for c_remove in range(max_c, 0, -1):
                if time.time() > time_limit:
                    break
                
                edges_with_c = [i for i in range(m) if colours[i] == c_remove]
                if not edges_with_c:
                    continue
                
                # Try multiple random reassignment attempts with different strategies
                for attempt in range(50):
                    old_colours = colours[:]
                    rng.shuffle(edges_with_c)
                    
                    success = True
                    for idx in edges_with_c:
                        # Build blocked mask
                        blocked = 0
                        for nb in range(m):
                            if (conflict_masks[idx] >> nb) & 1:
                                if colours[nb] >= 0:
                                    blocked |= (1 << colours[nb])
                        
                        # Find available colors
                        available = []
                        for c_new in range(c_remove):
                            if not ((blocked >> c_new) & 1):
                                available.append(c_new)
                        
                        if not available:
                            success = False
                            break
                        else:
                            # Strategy: prefer lower colors, but with some randomness
                            available.sort()
                            if attempt % 3 == 0:
                                # Always pick lowest
                                colours[idx] = available[0]
                            elif attempt % 3 == 1:
                                # Pick random from lower half
                                half = len(available) // 2
                                colours[idx] = rng.choice(available[:max(1, half)])
                            else:
                                # Pick random
                                colours[idx] = rng.choice(available)
                    
                    if success:
                        improved = True
                        no_improve_count = 0
                        break
                    else:
                        colours[:] = old_colours
                
                if improved:
                    break
        
        return colours

    def solve_instance(edges, n, d, seed, time_limit):
        m = len(edges)
        if m == 0:
            return []
        
        conflict_sets, conflict_masks = build_conflict_graph(edges, n)
        
        best_colours = None
        best_count = float('inf')
        
        rng = random.Random(seed)
        
        max_restarts = 100
        restart_count = 0
        
        # Precompute degree order
        degree_order = sorted(range(m), key=lambda i: -bin(conflict_masks[i]).count('1'))
        
        while restart_count < max_restarts and time.time() < time_limit:
            # 1. Greedy Color with different ordering strategies
            strategy = restart_count % 4
            if strategy == 0:
                order = degree_order[:]
            elif strategy == 1:
                order = list(range(m))
                rng.shuffle(order)
            elif strategy == 2:
                # DSATUR-like: start with high degree, then randomize slightly
                order = degree_order[:]
                for i in range(min(5, m)):
                    j = rng.randint(0, m - 1)
                    order[i], order[j] = order[j], order[i]
            else:
                # Reverse degree order
                order = degree_order[::-1]
            
            colours = greedy_color(m, conflict_masks, order, rng)
            
            # 2. Enhanced Local Search with multi-swap moves
            colours = multi_swap_search(colours, conflict_masks, rng, m, time_limit)
            
            # 3. High-precision polish
            colours = high_precision_polish(colours, conflict_masks, rng, m, time_limit)
            
            current_count = max(colours) + 1 if m > 0 else 0
            
            if current_count < best_count:
                best_count = current_count
                best_colours = colours[:]
            
            restart_count += 1
            
            # If we've reached the theoretical lower bound, stop early
            # Lower bound: max degree of conflict graph + 1
            max_conflict_degree = max(bin(conflict_masks[i]).count('1') for i in range(m)) if m > 0 else 0
            if best_count <= max_conflict_degree + 1:
                break
        
        if best_colours is None:
            # Fallback
            best_colours = greedy_color(m, conflict_masks)
        
        return best_colours

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        
        # Allocate time per instance
        # Total 140s. 3 instances. ~45s each.
        inst_start = time.time()
        inst_limit = inst_start + 45.0
        
        colours = solve_instance(edges, n, d, seed, inst_limit)
        results.append(colours)
        
        # Check global deadline
        if time.time() > deadline:
            # If we run out of time, we just return what we have.
            # The loop above handles individual limits.
            pass

    return results
# EVOLVE-BLOCK-END
