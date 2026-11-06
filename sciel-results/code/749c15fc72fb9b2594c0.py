import random
import time
import itertools
from collections import defaultdict
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
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
    
    # Lagrangian multipliers for each color (penalty for overuse)
    lambda_color = [0.0] * num_colors
    # Lagrangian multipliers for each edge (penalty for being in many tight cycles)
    lambda_edge = {e: 0.0 for e in all_edges}
    
    def can_add_edge(e, c, edge_colors):
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            other_colors = []
            valid = True
            for oe in cycle:
                if oe == e:
                    continue
                if oe not in edge_colors:
                    valid = False
                    break
                other_colors.append(edge_colors[oe])
            if not valid:
                continue
            if c in other_colors:
                return False
            if len(set(other_colors)) < 3:
                return False
        return True

    def get_valid_colors(e, edge_colors):
        """Return list of all valid colors for edge e, sorted by preference."""
        valid_colors = []
        color_freq = defaultdict(int)
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            for oe in cycle:
                if oe != e and oe in edge_colors:
                    color_freq[edge_colors[oe]] += 1
        
        for c in range(num_colors):
            if can_add_edge(e, c, edge_colors):
                tau = 2.0
                penalty = color_freq[c] / tau + 0.5 * lambda_color[c]
                valid_colors.append((penalty, c))
        
        valid_colors.sort()
        return [c for _, c in valid_colors]

    def compute_edge_score(e, edge_colors):
        """
        Compute a Lagrangian-inspired score for edge e.
        Higher score means more desirable to add.
        Base score: 1.0 for each edge we want to add.
        Penalty: sum of lambda_color[c] for colors that would be used,
                 plus lambda_edge[e] for the edge itself.
        Bonus: -sum of lambda_color[c] for colors NOT in tight cycles with e
               (encourages using rare colors)
        """
        # Count how many cycles e participates in that are "tight"
        # (all other 3 edges present)
        tight_count = 0
        color_pressure = defaultdict(float)
        for idx in edge_to_cycles[e]:
            cycle = all_cycles[idx]
            all_present = True
            for oe in cycle:
                if oe != e and oe not in edge_colors:
                    all_present = False
                    break
            if all_present:
                tight_count += 1
                for oe in cycle:
                    if oe != e and oe in edge_colors:
                        c_oe = edge_colors[oe]
                        color_pressure[c_oe] += 1.0
        
        # Base score
        score = 1.0
        
        # Subtract pressure from colors already heavily used in tight cycles
        for c, pressure in color_pressure.items():
            score -= 0.1 * pressure * (1.0 + lambda_color[c])
        
        # Subtract edge-specific penalty
        score -= 0.05 * lambda_edge[e]
        
        # Bonus for edges that are in few tight cycles (more flexible)
        score += 0.02 * max(0, 5 - tight_count)
        
        return score

    def find_best_color(e, edge_colors):
        """Find a valid color, using Lagrangian guidance to prefer less-conflicting colors."""
        valid_colors = get_valid_colors(e, edge_colors)
        
        if not valid_colors:
            return None
        
        # Pick randomly from the top few to maintain diversity
        top_k = min(3, len(valid_colors))
        c = random.choice(valid_colors[:top_k])
        return c

    def update_lagrangian(edge_colors, missing_set, step_size=0.1):
        """
        Update Lagrangian multipliers using subgradient method.
        For each color c: if c is overused (appears in many tight cycles), increase lambda.
        For each edge e: if e is in many tight cycles, increase lambda_edge[e].
        """
        # Compute subgradient for each color
        color_tight_count = defaultdict(int)
        for idx in range(len(all_cycles)):
            cycle = all_cycles[idx]
            all_present = True
            for e in cycle:
                if e not in edge_colors:
                    all_present = False
                    break
            if all_present:
                # This cycle is tight (all 4 edges present)
                colors_in_cycle = [edge_colors[e] for e in cycle]
                # If this cycle is rainbow (4 distinct colors), it's satisfied
                # But we still track pressure
                for c in colors_in_cycle:
                    color_tight_count[c] += 1
        
        # Update lambda_color: increase if color is in many tight cycles
        for c in range(num_colors):
            count = color_tight_count.get(c, 0)
            # Target: each color should be in at most ~len(all_cycles)/num_colors tight cycles
            target = len(all_cycles) / num_colors
            subgrad = count - target
            lambda_color[c] += step_size * subgrad
            lambda_color[c] = max(0.0, min(5.0, lambda_color[c]))
        
        # Update lambda_edge: increase for edges in many tight cycles
        edge_tight_count = defaultdict(int)
        for idx in range(len(all_cycles)):
            cycle = all_cycles[idx]
            all_present = True
            for e in cycle:
                if e not in edge_colors:
                    all_present = False
                    break
            if all_present:
                for e in cycle:
                    edge_tight_count[e] += 1
        
        for e in all_edges:
            count = edge_tight_count.get(e, 0)
            # Target: each edge in at most ~5 tight cycles
            target = 5.0
            subgrad = count - target
            lambda_edge[e] += step_size * subgrad
            lambda_edge[e] = max(0.0, min(5.0, lambda_edge[e]))

    # Phase 1: Greedy Random Initialization with Lagrangian guidance
    # Sort edges by degree (number of cycles they participate in) descending
    # Edges in more cycles are more constraining, so place them first
    edge_cycle_counts = {e: len(edge_to_cycles[e]) for e in all_edges}
    sorted_edges = sorted(all_edges, key=lambda e: -edge_cycle_counts[e])
    random.shuffle(sorted_edges)  # Add randomness to break ties
    
    edge_colors = {}
    for e in sorted_edges:
        c = find_best_color(e, edge_colors)
        if c is not None:
            edge_colors[e] = c

    # Phase 2: Local Search with Lagrangian updates
    best_count = len(edge_colors)
    best_state = dict(edge_colors)
    missing_set = set(e for e in all_edges if e not in edge_colors)
    
    def save_if_better():
        nonlocal best_count, best_state
        if len(edge_colors) > best_count:
            best_count = len(edge_colors)
            best_state = dict(edge_colors)

    # Move types
    lagrangian_update_interval = 100  # Update every 100 iterations
    iteration_count = 0
    
    while time.time() < deadline:
        iteration_count += 1
        
        # Periodically update Lagrangian multipliers
        if iteration_count % lagrangian_update_interval == 0:
            update_lagrangian(edge_colors, missing_set)
        
        # 1. Add (weighted by Lagrangian score)
        if missing_set:
            # Score a larger sample of missing edges
            sample_size = min(len(missing_set), 80)
            sample = random.sample(list(missing_set), sample_size)
            scored_edges = []
            for e in sample:
                score = compute_edge_score(e, edge_colors)
                c = find_best_color(e, edge_colors)
                if c is not None:
                    scored_edges.append((score, e, c))
            
            if scored_edges:
                # Sort by score descending, pick from top
                scored_edges.sort(key=lambda x: -x[0])
                top_k = min(5, len(scored_edges))
                _, e, c = random.choice(scored_edges[:top_k])
                edge_colors[e] = c
                missing_set.discard(e)
                save_if_better()
                continue

        # 2. Recolor
        if edge_colors:
            e_recolor = random.choice(list(edge_colors.keys()))
            old_c = edge_colors[e_recolor]
            new_c = find_best_color(e_recolor, edge_colors)
            if new_c is not None and new_c != old_c:
                edge_colors[e_recolor] = new_c

        # 3. Swap (1 remove, 1 add)
        if edge_colors and missing_set and random.random() < 0.5:
            e_remove = random.choice(list(edge_colors.keys()))
            e_add = random.choice(list(missing_set))
            
            old_c = edge_colors[e_remove]
            del edge_colors[e_remove]
            missing_set.add(e_remove)
            
            c_add = find_best_color(e_add, edge_colors)
            if c_add is not None:
                edge_colors[e_add] = c_add
                missing_set.discard(e_add)
                save_if_better()
                
                c_rem = find_best_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    missing_set.discard(e_remove)
                    save_if_better()
            else:
                c_rem = find_best_color(e_remove, edge_colors)
                if c_rem is not None:
                    edge_colors[e_remove] = c_rem
                    missing_set.discard(e_remove)
                else:
                    edge_colors[e_remove] = old_c

        # 4. Double Swap (2 remove, 2 add)
        if random.random() < 0.2 and len(edge_colors) >= 2 and len(missing_set) >= 2:
            e1, e2 = random.sample(list(edge_colors.keys()), 2)
            e3, e4 = random.sample(list(missing_set), 2)
            
            c1 = edge_colors[e1]
            c2 = edge_colors[e2]
            del edge_colors[e1]
            del edge_colors[e2]
            missing_set.add(e1)
            missing_set.add(e2)
            
            success = False
            c3 = find_best_color(e3, edge_colors)
            if c3 is not None:
                edge_colors[e3] = c3
                missing_set.discard(e3)
                c4 = find_best_color(e4, edge_colors)
                if c4 is not None:
                    edge_colors[e4] = c4
                    missing_set.discard(e4)
                    save_if_better()
                    success = True
                    
                    # Try to re-add removed edges
                    c1_new = find_best_color(e1, edge_colors)
                    if c1_new is not None:
                        edge_colors[e1] = c1_new
                        missing_set.discard(e1)
                        save_if_better()
                    c2_new = find_best_color(e2, edge_colors)
                    if c2_new is not None:
                        edge_colors[e2] = c2_new
                        missing_set.discard(e2)
                        save_if_better()
                else:
                    del edge_colors[e3]
                    missing_set.add(e3)
            
            # If swap didn't succeed, restore original edges
            if not success:
                edge_colors[e1] = c1
                missing_set.discard(e1)
                edge_colors[e2] = c2
                missing_set.discard(e2)

    # Restore best state
    edge_colors = best_state
    
    # Final Polish: Multiple passes trying to add remaining edges
    missing_set = set(e for e in all_edges if e not in edge_colors)
    
    # Sort missing edges by desirability (cycle count) for better ordering
    missing_sorted = sorted(missing_set, key=lambda e: -edge_cycle_counts[e])
    
    for _ in range(5):  # Multiple polish passes
        if time.time() > deadline - 5:
            break
        progress = False
        for e in missing_sorted:
            if e not in missing_set:
                continue
            if time.time() > deadline - 5:
                break
            c = find_best_color(e, edge_colors)
            if c is not None:
                edge_colors[e] = c
                missing_set.discard(e)
                progress = True
        if not progress:
            break
    
    # One more pass in reverse order for diversity
    if time.time() < deadline - 5:
        for e in reversed(missing_sorted):
            if e not in missing_set:
                continue
            if time.time() > deadline - 5:
                break
            c = find_best_color(e, edge_colors)
            if c is not None:
                edge_colors[e] = c
                missing_set.discard(e)
    
    result = []
    for (u, v), c in edge_colors.items():
        result.append([u, v, c])
    return result
