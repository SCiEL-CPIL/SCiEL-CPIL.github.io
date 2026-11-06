import random
import time
import itertools
from collections import defaultdict
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 112.0
    
    n = 20
    num_colors = 20
    
    # Precompute cycles
    all_cycles = []
    edge_to_cycles = defaultdict(list)
    
    for u, v, w, x in itertools.combinations(range(n), 4):
        cycles = [
            [(u, v), (v, w), (w, x), (x, u)],
            [(u, v), (v, x), (x, w), (w, u)],
            [(u, w), (w, v), (v, x), (x, u)],
        ]
        for cycle in cycles:
            norm_cycle = tuple(sorted((min(e), max(e)) for e in cycle))
            if len(set(norm_cycle)) == 4:
                idx = len(all_cycles)
                all_cycles.append(norm_cycle)
                for e in norm_cycle:
                    edge_to_cycles[e].append(idx)
    
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    num_edges = len(all_edges)
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # Build conflict structure: for each edge, which other edges share a cycle with it
    # conflict_same_cycle[e1][e2] = True if e1 and e2 are in a common 4-cycle
    # This is the key structural constraint: same color can't appear on two edges in same cycle
    
    # Precompute: for each edge, the set of edges that share a cycle with it
    shared_cycle = defaultdict(set)
    for cycle in all_cycles:
        for i in range(4):
            for j in range(i+1, 4):
                e1, e2 = cycle[i], cycle[j]
                if e1 != e2:
                    shared_cycle[e1].add(e2)
                    shared_cycle[e2].add(e1)
    
    # For fast conflict checking: given edge e and color c, which existing colored edges conflict?
    # An existing edge (e', c') conflicts with (e, c) iff:
    #   - e' == e (same edge, different color) -> but we only assign one color per edge
    #   - c' == c and e' shares a cycle with e
    # So for a given (e, c), the conflicting existing edges are those with color c in shared_cycle[e]
    
    # Track color_to_edges: for each color, which edges have that color
    color_to_edges = defaultdict(set)
    
    def is_conflict(e, c, edge_colors):
        """Check if adding (e, c) conflicts with any existing edge."""
        # Check if any edge with color c shares a cycle with e
        for e_prime in color_to_edges.get(c, ()):
            if e_prime != e and e_prime in shared_cycle[e]:
                return True
        return False
    
    def find_valid_colors(e, edge_colors):
        """Return list of valid colors for edge e."""
        valid = []
        for c in range(num_colors):
            if not is_conflict(e, c, edge_colors):
                valid.append(c)
        return valid
    
    # Phase 1: Greedy construction with randomization
    # Sort edges by "constraint degree" (number of edges sharing a cycle)
    edge_constraint_degree = {e: len(shared_cycle[e]) for e in all_edges}
    sorted_edges = sorted(all_edges, key=lambda e: -edge_constraint_degree[e])
    
    def greedy_construct():
        edge_colors = {}
        color_to_edges_local = defaultdict(set)
        
        # Random shuffle for diversity
        random.shuffle(sorted_edges)
        
        for e in sorted_edges:
            valid = []
            for c in range(num_colors):
                conflict = False
                for e_prime in color_to_edges_local.get(c, ()):
                    if e_prime != e and e_prime in shared_cycle[e]:
                        conflict = True
                        break
                if not conflict:
                    valid.append(c)
            
            if valid:
                # Prefer colors that are less "popular" in shared cycles
                # Score each color by how many existing edges with that color are in shared_cycle[e]
                scored = []
                for c in valid:
                    count = sum(1 for e_prime in color_to_edges_local.get(c, ()) if e_prime in shared_cycle[e])
                    scored.append((count, c))
                scored.sort()
                # Pick from top few for diversity
                top_k = min(3, len(scored))
                chosen_c = scored[random.randint(0, top_k-1)][1]
                edge_colors[e] = chosen_c
                color_to_edges_local[chosen_c].add(e)
        
        return edge_colors, color_to_edges_local
    
    # Phase 2: Local search with subgradient guidance
    best_count = 0
    best_state = {}
    
    def save_if_better(edge_colors, color_to_edges):
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)
    
    # Subgradient multipliers for colors (penalty for overuse in dense areas)
    lambda_color = [0.0] * num_colors
    
    def update_lagrangian(edge_colors, color_to_edges, step_size=0.05):
        """Update lambda_color based on color usage density."""
        # For each color, count how many edges have it
        for c in range(num_colors):
            count = len(color_to_edges.get(c, ()))
            target = num_edges / num_colors  # ~9.5
            lambda_color[c] += step_size * (count - target)
            lambda_color[c] = max(0.0, min(3.0, lambda_color[c]))
    
    def score_color_for_edge(e, c, color_to_edges, edge_colors):
        """Score a color for an edge. Lower is better (more preferred)."""
        if is_conflict(e, c, edge_colors):
            return float('inf')
        # Penalty: how many existing edges with color c are in shared_cycle[e]
        pressure = sum(1 for e_prime in color_to_edges.get(c, ()) if e_prime in shared_cycle[e])
        # Lagrangian penalty
        lagrangian = 0.3 * lambda_color[c]
        # Slight bonus for using rare colors
        rarity_bonus = -0.05 * (num_colors - len(color_to_edges.get(c, ())))
        return pressure + lagrangian + rarity_bonus
    
    def local_search(edge_colors, color_to_edges, time_limit):
        """Local search: try to add edges, recolor, swap."""
        missing_set = set(e for e in all_edges if e not in edge_colors)
        iteration = 0
        lagrangian_interval = 50
        
        while time.time() < time_limit:
            iteration += 1
            
            if iteration % lagrangian_interval == 0:
                update_lagrangian(edge_colors, color_to_edges)
            
            # 1. Try to add a missing edge
            if missing_set:
                # Sample a subset of missing edges for speed
                sample_size = min(len(missing_set), 50)
                sample = random.sample(list(missing_set), sample_size)
                
                best_add = None
                best_score = float('inf')
                for e in sample:
                    for c in range(num_colors):
                        s = score_color_for_edge(e, c, color_to_edges, edge_colors)
                        if s < best_score:
                            best_score = s
                            best_add = (e, c)
                
                if best_add is not None and best_score < float('inf'):
                    e, c = best_add
                    edge_colors[e] = c
                    color_to_edges[c].add(e)
                    missing_set.discard(e)
                    save_if_better(edge_colors, color_to_edges)
                    continue
            
            # 2. Recolor a random edge
            if edge_colors and random.random() < 0.7:
                e = random.choice(list(edge_colors.keys()))
                old_c = edge_colors[e]
                # Find better color
                best_c = old_c
                best_s = score_color_for_edge(e, old_c, color_to_edges, edge_colors)
                for c in range(num_colors):
                    if c == old_c:
                        continue
                    s = score_color_for_edge(e, c, color_to_edges, edge_colors)
                    if s < best_s:
                        best_s = s
                        best_c = c
                if best_c != old_c and best_s < float('inf'):
                    color_to_edges[old_c].discard(e)
                    edge_colors[e] = best_c
                    color_to_edges[best_c].add(e)
            
            # 3. Swap: remove one edge, add another
            if edge_colors and missing_set and random.random() < 0.5:
                e_remove = random.choice(list(edge_colors.keys()))
                c_remove = edge_colors[e_remove]
                del edge_colors[e_remove]
                color_to_edges[c_remove].discard(e_remove)
                missing_set.add(e_remove)
                
                # Try to add a missing edge
                sample_size = min(len(missing_set), 30)
                sample = random.sample(list(missing_set), sample_size)
                best_add = None
                best_score = float('inf')
                for e in sample:
                    for c in range(num_colors):
                        s = score_color_for_edge(e, c, color_to_edges, edge_colors)
                        if s < best_score:
                            best_score = s
                            best_add = (e, c)
                
                if best_add is not None and best_score < float('inf'):
                    e_add, c_add = best_add
                    edge_colors[e_add] = c_add
                    color_to_edges[c_add].add(e_add)
                    missing_set.discard(e_add)
                    save_if_better(edge_colors, color_to_edges)
                
                # Try to re-add the removed edge
                valid_remove = []
                for c in range(num_colors):
                    if not is_conflict(e_remove, c, edge_colors):
                        valid_remove.append(c)
                if valid_remove:
                    # Pick best color
                    best_c = min(valid_remove, key=lambda c: score_color_for_edge(e_remove, c, color_to_edges, edge_colors))
                    edge_colors[e_remove] = best_c
                    color_to_edges[best_c].add(e_remove)
                    missing_set.discard(e_remove)
                    save_if_better(edge_colors, color_to_edges)
                else:
                    # Restore original
                    edge_colors[e_remove] = c_remove
                    color_to_edges[c_remove].add(e_remove)
                    missing_set.discard(e_remove)
            
            # 4. Double swap
            if random.random() < 0.15 and len(edge_colors) >= 2 and len(missing_set) >= 2:
                e1, e2 = random.sample(list(edge_colors.keys()), 2)
                c1, c2 = edge_colors[e1], edge_colors[e2]
                del edge_colors[e1]
                del edge_colors[e2]
                color_to_edges[c1].discard(e1)
                color_to_edges[c2].discard(e2)
                missing_set.update([e1, e2])
                
                # Try to add two missing edges
                added = []
                sample_size = min(len(missing_set), 40)
                sample = random.sample(list(missing_set), sample_size)
                for e in sample:
                    if len(added) >= 2:
                        break
                    best_c = None
                    best_s = float('inf')
                    for c in range(num_colors):
                        s = score_color_for_edge(e, c, color_to_edges, edge_colors)
                        if s < best_s:
                            best_s = s
                            best_c = c
                    if best_c is not None and best_s < float('inf'):
                        edge_colors[e] = best_c
                        color_to_edges[best_c].add(e)
                        missing_set.discard(e)
                        added.append(e)
                
                if added:
                    save_if_better(edge_colors, color_to_edges)
                
                # Try to re-add removed edges
                for e_rem in [e1, e2]:
                    if e_rem in missing_set:
                        valid = []
                        for c in range(num_colors):
                            if not is_conflict(e_rem, c, edge_colors):
                                valid.append(c)
                        if valid:
                            best_c = min(valid, key=lambda c: score_color_for_edge(e_rem, c, color_to_edges, edge_colors))
                            edge_colors[e_rem] = best_c
                            color_to_edges[best_c].add(e_rem)
                            missing_set.discard(e_rem)
                            save_if_better(edge_colors, color_to_edges)
                        else:
                            # Restore original color
                            orig_c = c1 if e_rem == e1 else c2
                            edge_colors[e_rem] = orig_c
                            color_to_edges[orig_c].add(e_rem)
                            missing_set.discard(e_rem)
    
    # Run multiple restarts with different random seeds
    num_restarts = 5
    edges_per_restart = max(1, (int(deadline - time.time()) - 5) // num_restarts)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Greedy construction
        edge_colors, color_to_edges = greedy_construct()
        save_if_better(edge_colors, color_to_edges)
        
        # Local search with time budget
        local_time = min(edges_per_restart, deadline - time.time() - 8)
        if local_time > 2:
            local_search(edge_colors, color_to_edges, time.time() + local_time)
    
    # Final polish: try to add remaining edges in multiple passes
    if best_state:
        edge_colors = dict(best_state)
        color_to_edges = defaultdict(set)
        for e, c in edge_colors.items():
            color_to_edges[c].add(e)
        
        missing_set = set(e for e in all_edges if e not in edge_colors)
        
        # Multiple passes in different orders
        for pass_num in range(3):
            if time.time() > deadline - 3:
                break
            if not missing_set:
                break
            
            if pass_num == 0:
                order = sorted(missing_set, key=lambda e: -edge_constraint_degree[e])
            elif pass_num == 1:
                order = sorted(missing_set, key=lambda e: edge_constraint_degree[e])
            else:
                order = list(missing_set)
                random.shuffle(order)
            
            for e in order:
                if time.time() > deadline - 3:
                    break
                if e not in missing_set:
                    continue
                valid = []
                for c in range(num_colors):
                    if not is_conflict(e, c, edge_colors):
                        valid.append(c)
                if valid:
                    # Pick best color
                    best_c = min(valid, key=lambda c: score_color_for_edge(e, c, color_to_edges, edge_colors))
                    edge_colors[e] = best_c
                    color_to_edges[best_c].add(e)
                    missing_set.discard(e)
    
    # Use best state if we have one
    if best_state and len(best_state) > len(edge_colors):
        edge_colors = best_state
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
