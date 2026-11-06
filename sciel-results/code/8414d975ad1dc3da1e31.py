import random
import time
import itertools

def construct():
        start_time = time.time()
        deadline = start_time + 115.0
        n = 20
        num_colors = 20
        
        # Precompute all 4-cycles in K_n
        all_cycles = []
        for comb in itertools.combinations(range(n), 4):
            u, v, w, x = comb
            c1 = [(u,v), (v,w), (w,x), (x,u)]
            c2 = [(u,v), (v,x), (x,w), (w,u)]
            c3 = [(u,w), (w,v), (v,x), (x,u)]
            for cycle in [c1, c2, c3]:
                norm_cycle = tuple(sorted((min(e[0], e[1]), max(e[0], e[1])) for e in cycle))
                if len(set(norm_cycle)) == 4:
                    all_cycles.append(norm_cycle)
        
        # Map edge to cycles
        edge_to_cycles = {e: [] for e in itertools.combinations(range(n), 2)}
        for i, c in enumerate(all_cycles):
            for e in c:
                edge_to_cycles[e].append(i)
        
        # Strategy: Try to find a valid coloring for graphs with decreasing edge count
        # Start with K_20, then K_20 - 1 edge, etc.
        # But K_20 is likely impossible.
        # Start with K_20 - Perfect Matching (180 edges)
        
        def solve_for_edge_set(edges_set):
            # edges_set: set of (u, v)
            edges_list = list(edges_set)
            # Initialize colors
            colors = {e: random.randint(0, num_colors - 1) for e in edges_list}
            
            # Calculate initial cost
            def get_cost():
                cost = 0
                for c in all_cycles:
                    if all(e in edges_set for e in c):
                        cols = [colors[e] for e in c]
                        if len(set(cols)) < 4:
                            cost += 1
                return cost
            
            current_cost = get_cost()
            if current_cost == 0:
                return [(u, v, colors[(u,v)]) for u, v in edges_list]
            
            # Simulated Annealing
            T = 1.0
            step = 0
            while time.time() < deadline - 5:
                step += 1
                if step % 1000 == 0:
                    T *= 0.95
                    
                # Pick a random edge to recolor
                e = random.choice(edges_list)
                old_color = colors[e]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                
                # Calculate delta cost
                # Only cycles containing e are affected
                delta = 0
                for c_idx in edge_to_cycles[e]:
                    c = all_cycles[c_idx]
                    if not all(ee in edges_set for ee in c):
                        continue
                    
                    # Old state
                    old_cols = [colors[ee] for ee in c]
                    old_rainbow = (len(set(old_cols)) == 4)
                    
                    # New state
                    # Replace color of e
                    new_cols = old_cols.copy()
                    for i, ee in enumerate(c):
                        if ee == e:
                            new_cols[i] = new_color
                            break
                    new_rainbow = (len(set(new_cols)) == 4)
                    
                    if old_rainbow and not new_rainbow:
                        delta += 1
                    elif not old_rainbow and new_rainbow:
                        delta -= 1
                
                # Accept
                if delta < 0 or random.random() < pow(2.718, -delta / T):
                    colors[e] = new_color
                    current_cost += delta
                    if current_cost == 0:
                        return [(u, v, colors[(u,v)]) for u, v in edges_list]
                else:
                    # Rollback not needed as we didn't change colors[e]
                    pass
                    
            return None

        # Try different edge sets
        # 1. K_20 - Perfect Matching
        all_edges = set(itertools.combinations(range(n), 2))
        
        # Generate a random perfect matching
        vertices = list(range(n))
        random.shuffle(vertices)
        matching = set()
        for i in range(0, n, 2):
            matching.add((min(vertices[i], vertices[i+1]), max(vertices[i], vertices[i+1])))
        
        target_edges = all_edges - matching
        result = solve_for_edge_set(target_edges)
        if result:
            return result
            
        # If failed, try removing more edges
        # Add a few more random edges to the removed set
        removed = list(matching)
        for _ in range(10):
            if time.time() > deadline - 10: break
            # Remove 5 more random edges
            extra_removed = set()
            while len(extra_removed) < 5:
                e = random.choice(list(all_edges - set(removed)))
                extra_removed.add(e)
            removed.extend(extra_removed)
            target_edges = all_edges - set(removed)
            result = solve_for_edge_set(target_edges)
            if result:
                return result
                
        # Fallback: Return the best found or a random valid small graph
        # If no valid graph found, return empty or a known valid small graph
        # A complete graph on 4 vertices with 6 edges can be colored?
        # No, we need to return a valid graph.
        # If we can't find a large one, we should return a smaller valid one to get some score.
        # But the task says "Maximize".
        
        # Let's try one more time with a different matching
        random.shuffle(vertices)
        matching = set()
        for i in range(0, n, 2):
            matching.add((min(vertices[i], vertices[i+1]), max(vertices[i], vertices[i+1])))
        target_edges = all_edges - matching
        result = solve_for_edge_set(target_edges)
        if result:
            return result

        # If all else fails, return a valid 4-cycle free graph?
        # A forest is 4-cycle free.
        # Max edges in a forest on 20 vertices is 19.
        # This is a very low score, but better than 0.
        forest_edges = []
        for i in range(1, n):
            forest_edges.append([i-1, i, i % num_colors])
        return forest_edges
