"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
from collections import defaultdict
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill
    
    n = 20
    num_colors = 20
    
    # Pre-compute all 4-cycles in K_n
    # We store them as tuples of 4 edge indices
    edge_index = {}
    edge_list = []
    for u in range(n):
        for v in range(u + 1, n):
            edge_index[(u, v)] = len(edge_list)
            edge_list.append((u, v))
    m = len(edge_list)
    
    def get_edge_idx(u, v):
        if u > v:
            u, v = v, u
        return edge_index.get((u, v), -1)

    all_4cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        # 3 distinct 4-cycles on these 4 vertices
        # Cycle 1: a-b-c-d-a
        e1 = [get_edge_idx(a,b), get_edge_idx(b,c), get_edge_idx(c,d), get_edge_idx(d,a)]
        if all(e != -1 for e in e1):
            all_4cycles.append(tuple(sorted(set(e1))))
        # Cycle 2: a-b-d-c-a
        e2 = [get_edge_idx(a,b), get_edge_idx(b,d), get_edge_idx(d,c), get_edge_idx(c,a)]
        if all(e != -1 for e in e2):
            all_4cycles.append(tuple(sorted(set(e2))))
        # Cycle 3: a-c-b-d-a
        e3 = [get_edge_idx(a,c), get_edge_idx(c,b), get_edge_idx(b,d), get_edge_idx(d,a)]
        if all(e != -1 for e in e3):
            all_4cycles.append(tuple(sorted(set(e3))))

    # Map edge index to the list of 4-cycle indices it belongs to
    edge_to_cycles = defaultdict(list)
    for i, cycle in enumerate(all_4cycles):
        for e in cycle:
            edge_to_cycles[e].append(i)

    def is_rainbow(cycle_idx, colors, active_edges):
        """Check if the 4-cycle is rainbow. Returns True if valid."""
        cycle = all_4cycles[cycle_idx]
        # Check if all edges are active
        if not all(e in active_edges for e in cycle):
            return True # Vacuously true if not a cycle in current graph
        
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        return len({c0, c1, c2, c3}) == 4

    def count_violations(colors, active_edges):
        """Count number of non-rainbow 4-cycles in the active graph."""
        violations = 0
        for i, cycle in enumerate(all_4cycles):
            if all(e in active_edges for e in cycle):
                c0 = colors[cycle[0]]
                c1 = colors[cycle[1]]
                c2 = colors[cycle[2]]
                c3 = colors[cycle[3]]
                if len({c0, c1, c2, c3}) < 4:
                    violations += 1
        return violations

    best_edge_count = 0
    best_edges_list = []
    
    # Try multiple random initializations
    for seed in [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]:
        if time.time() - t0 > DEADLINE:
            break
            
        random.seed(seed)
        
        # Initialize with a dense random graph
        # Start with ~70% density to have many edges to save
        p_add = 0.7
        active_edges = set()
        colors = [0] * m
        
        for e_idx in range(m):
            if random.random() < p_add:
                active_edges.add(e_idx)
                colors[e_idx] = random.randint(0, num_colors - 1)
        
        # Local Search Phase 1: Repair violations
        # We iterate through cycles, and if a violation is found, try to fix by recoloring.
        # If recoloring is impossible, remove an edge.
        
        max_iter = 1000
        for _ in range(max_iter):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            # Find a random violation
            # To be faster, we can maintain a list of bad cycles, but for n=20, 
            # checking all ~1000 cycles is fast enough in Python if optimized?
            # Let's just pick a random cycle and check it.
            
            found_violation = False
            for _ in range(50): # Try a few random cycles
                if not all_4cycles:
                    break
                c_idx = random.randint(0, len(all_4cycles) - 1)
                cycle = all_4cycles[c_idx]
                
                # Check if active
                if not all(e in active_edges for e in cycle):
                    continue
                
                c0, c1, c2, c3 = (colors[e] for e in cycle)
                if len({c0, c1, c2, c3}) == 4:
                    continue # This cycle is fine
                
                found_violation = True
                # We have a violation. Try to fix by changing color of one edge.
                # An edge e in the cycle can take a color c if:
                # 1. c is not in {colors of other 3 edges in this cycle}
                # 2. Changing colors[e] to c does NOT create a violation in any other cycle containing e.
                
                fixed = False
                # Try edges in random order
                edges_in_cycle = list(cycle)
                random.shuffle(edges_in_cycle)
                
                for e in edges_in_cycle:
                    # Determine forbidden colors for this edge in THIS cycle
                    other_colors = {colors[other_e] for other_e in cycle if other_e != e}
                    
                    # Candidate colors: 0..19 not in other_colors
                    candidates = [c for c in range(num_colors) if c not in other_colors]
                    random.shuffle(candidates)
                    
                    for new_c in candidates:
                        # Check if new_c is safe for ALL other cycles containing e
                        safe = True
                        for other_c_idx in edge_to_cycles[e]:
                            if other_c_idx == c_idx:
                                continue
                            other_cycle = all_4cycles[other_c_idx]
                            if not all(x in active_edges for x in other_cycle):
                                continue
                            # Simulate change
                            # Get colors of other 3 edges in other_cycle
                            oc = [colors[x] if x != e else new_c for x in other_cycle]
                            if len(set(oc)) < 4:
                                safe = False
                                break
                        
                        if safe:
                            colors[e] = new_c
                            fixed = True
                            break
                    if fixed:
                        break
                
                if not fixed:
                    # Cannot fix by recoloring. Remove one edge from the cycle.
                    # Choose the edge with the highest degree in the "conflict graph" 
                    # or just random to break symmetry.
                    # Removing an edge reduces edge count, so we want to avoid this.
                    # Pick random edge to remove.
                    e_to_remove = random.choice(edges_in_cycle)
                    active_edges.remove(e_to_remove)
                    # No need to reset color, edge is inactive.
                
                break # Restart the random sampling loop with the updated state
            
            if not found_violation:
                # No violations found in this pass. 
                # We can try to ADD edges back.
                # Local Search Phase 2: Add edges
                added_any = True
                while added_any:
                    added_any = False
                    # Try adding a random inactive edge
                    inactive = [e for e in range(m) if e not in active_edges]
                    if not inactive:
                        break
                    random.shuffle(inactive)
                    
                    for e in inactive[:10]: # Try a few
                        # Check if adding e creates a violation
                        # It only affects cycles containing e
                        safe = True
                        for c_idx in edge_to_cycles[e]:
                            cycle = all_4cycles[c_idx]
                            # Check if other 3 edges are active
                            others = [x for x in cycle if x != e]
                            if not all(x in active_edges for x in others):
                                continue
                            
                            # We need to assign a color to e such that the cycle is rainbow.
                            other_colors = {colors[x] for x in others}
                            candidates = [c for c in range(num_colors) if c not in other_colors]
                            
                            if not candidates:
                                safe = False
                                break
                            
                            # Check if any candidate color is safe for other cycles
                            found_color = False
                            for new_c in candidates:
                                # Check other cycles containing e
                                ok = True
                                for other_c_idx in edge_to_cycles[e]:
                                    if other_c_idx == c_idx:
                                        continue
                                    other_cycle = all_4cycles[other_c_idx]
                                    if not all(x in active_edges for x in other_cycle):
                                        continue
                                    oc = [colors[x] if x != e else new_c for x in other_cycle]
                                    if len(set(oc)) < 4:
                                        ok = False
                                        break
                                if ok:
                                    found_color = True
                                    colors[e] = new_c
                                    break
                            
                            if not found_color:
                                safe = False
                                break
                        
                        if safe:
                            active_edges.add(e)
                            added_any = True
                            # Continue trying to add more, but restart inner loop logic?
                            # For simplicity, just break and restart the while loop to re-shuffle
                            break
        
        # Final check and record
        # Verify no violations
        valid = True
        for c_idx, cycle in enumerate(all_4cycles):
            if all(e in active_edges for e in cycle):
                c0, c1, c2, c3 = (colors[e] for e in cycle)
                if len({c0, c1, c2, c3}) < 4:
                    valid = False
                    break
        
        if valid:
            if len(active_edges) > best_edge_count:
                best_edge_count = len(active_edges)
                best_edges_list = [(edge_list[e][0], edge_list[e][1], colors[e]) for e in active_edges]
    
    # If no valid solution found (shouldn't happen, empty graph is valid), return empty
    if not best_edges_list:
        return []
        
    return best_edges_list

# EVOLVE-BLOCK-END
