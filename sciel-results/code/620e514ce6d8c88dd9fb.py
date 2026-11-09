"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    t0 = time.time()
    DEADLINE = 115.0  # safety margin under 150s kill
    
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles (as tuples of 4 vertices in order)
    # A 4-cycle is a sequence v0-v1-v2-v3-v0.
    # We only need to check distinct sets of 4 vertices.
    # For any 4 vertices {a,b,c,d}, there are 3 distinct 4-cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # Note: (a,c,d,b) is same as (a,b,c,d) reversed. (a,d,b,c) is same as (a,b,c,d) reversed. (a,d,c,b) is same as (a,c,b,d) reversed.
    
    all_cycles = []
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        all_cycles.append((a, b, c, d))
        all_cycles.append((a, b, d, c))
        all_cycles.append((a, c, b, d))
        
    # Total number of 4-cycles is C(20,4) * 3 = 4845 * 3 = 14535
    
    # Strategy: Start with K20 (190 edges).
    # We need to remove edges such that every 4-cycle has 4 distinct colors.
    # Wait, the colors are fixed? No, the task says "edge-coloured with 20 colours".
    # The input is the colored edges. We control the colors too?
    # "construct() -> coloured edges [[u, v, c], ...]"
    # Yes, we choose the graph AND the coloring.
    
    # The current program uses a 1-factorization which is a specific valid coloring of K20 minus a 1-factor?
    # No, 1-factorization of K20 covers all edges. K20 has 190 edges.
    # The current program returns 190 edges.
    # If K20 with a valid rainbow coloring exists, that is the maximum possible (190 edges).
    # The current program claims to have found one using 1-factorization.
    # Let's verify if the current program is actually valid.
    # The current program uses a 1-factorization of K20.
    # A 1-factorization of K_{2m} partitions edges into 2m-1 perfect matchings.
    # If we color each matching with a distinct color, then any 4-cycle...
    # In a 1-factorization, a 4-cycle consists of 4 edges.
    # Do these 4 edges belong to 4 distinct matchings?
    # Consider a 4-cycle a-b-c-d-a.
    # Edges are (a,b), (b,c), (c,d), (d,a).
    # In a 1-factorization, (a,b) is in matching M_i.
    # (b,c) is in matching M_j. Since b is in M_i with a, b cannot be in M_i with c. So i != j.
    # (c,d) is in M_k. c is in M_j with b, so c not in M_j with d. j != k.
    # (d,a) is in M_l. d is in M_k with c, so d not in M_k with a. k != l.
    # Also a is in M_i with b, so a not in M_i with d. i != l.
    # What about b and d? (b,c) is M_j, (d,a) is M_l.
    # Is it possible that M_j == M_l?
    # If M_j == M_l, then (b,c) and (d,a) are in the same matching.
    # This implies b is matched with c, and d is matched with a in the same perfect matching.
    # This is possible!
    # Example: Matching M0 contains (0,1), (2,3).
    # Cycle 0-1-2-3-0.
    # (0,1) in M0.
    # (1,2) in M1 (must be, since 1 is in M0 with 0).
    # (2,3) in M0.
    # (3,0) in M2 (must be, since 3 is in M0 with 2).
    # Colors: M0, M1, M0, M2.
    # Duplicate color M0. Not rainbow.
    
    # So the current program's logic "This maintains the rainbow property... because the new color is unique... and the other 3 edges were already distinct" is FLAWED.
    # The current program likely fails the validity check in the evaluator?
    # Or does it?
    # Let's look at the current program's verification code.
    # It checks `if len({e1, e2, e3, e4}) != 4: valid = False`.
    # If the construction is invalid, `valid` becomes False, and it returns a sparse graph `[[0, v, 0] for v in range(1, n)]`.
    # That sparse graph has 19 edges.
    # The score is 1.0.
    # If the sparse graph is valid (it is, no 4-cycles), the score is based on edge count?
    # "The evaluator measures the raw value (higher is better)."
    # "Maximize the number of edges."
    # So if the current program returns 19 edges, and I can return more, I win.
    # But the score is 1.0.
    # Maybe the seed score is normalized? "search score is value/seed".
    # If seed returns 19 edges, and value is 19, score is 1.
    # If I return 190 edges, score is 190/19 = 10.
    # So the current program is likely failing the K20 construction and falling back to the sparse graph.
    
    # My goal: Construct a valid dense graph.
    # Can we have K20?
    # We need a coloring of K20 such that every 4-cycle is rainbow.
    # This is equivalent to a graph where no 4-cycle has two edges of the same color.
    # This is a "proper" edge coloring with respect to 4-cycles? No, just distinct colors on the 4 edges.
    
    # Let's try a random greedy approach.
    # Start with empty graph. Add edges one by one.
    # For each new edge (u,v), assign a color.
    # Check all 4-cycles involving this edge.
    # A 4-cycle involving (u,v) looks like u-w-x-v-u or u-x-w-v-u.
    # So we need to check all pairs (w,x) such that (u,w), (w,x), (x,v) are in the graph.
    # The new edge (u,v) must have a color distinct from (u,w), (w,x), (x,v).
    
    # This is a CSP.
    # With 190 edges and 20 colors, is it solvable?
    # The constraint is local.
    
    # Let's try to construct K20 using a greedy algorithm with backtracking or just greedy repair.
    # Given the time limit, a simple greedy might work if we order edges well.
    
    # Alternative: Use a 1-factorization but FIX the conflicts.
    # Conflicts arise when a 4-cycle uses two edges from the same matching.
    # This happens if the 4-cycle is a "rectangle" in the matching structure?
    # In a 1-factorization of K20, the edges of a specific matching form a perfect matching.
    # A 4-cycle a-b-c-d-a has edges (a,b), (b,c), (c,d), (d,a).
    # Conflict if (a,b) and (c,d) are in the same matching M_i.
    # This means a is matched to b, and c is matched to d in M_i.
    # This is always possible for any pair of disjoint edges.
    # So K20 with 1-factorization coloring has many conflicts.
    
    # Can we recolor?
    # We have 20 colors.
    # We can try to find a valid coloring of K20.
    # This is a hard problem.
    
    # What if we remove some edges?
    # We want to maximize edges.
    # Maybe we can keep most edges and remove a few to break all "bad" 4-cycles.
    
    # Let's try a random search for a valid dense graph.
    # 1. Start with K20.
    # 2. Identify all 4-cycles.
    # 3. Randomly assign colors to edges.
    # 4. Check for violations.
    # 5. If violation, change color of one of the edges in the violating cycle.
    # 6. Repeat until valid or timeout.
    
    # This is a local search.
    
    def check_validity(edge_colors):
        # edge_colors is a dict (u,v) -> color
        # Check all 4-cycles
        for cycle in all_cycles:
            a, b, c, d = cycle
            e1 = edge_colors.get((min(a,b), max(a,b)))
            e2 = edge_colors.get((min(b,c), max(b,c)))
            e3 = edge_colors.get((min(c,d), max(c,d)))
            e4 = edge_colors.get((min(d,a), max(d,a)))
            if e1 is None or e2 is None or e3 is None or e4 is None:
                continue
            if len({e1, e2, e3, e4}) != 4:
                return False, cycle
        return True, None

    # Initialize with K20 and random colors
    edges = []
    edge_dict = {}
    for u in range(n):
        for v in range(u+1, n):
            c = random.randint(0, num_colors-1)
            edges.append([u, v, c])
            edge_dict[(u, v)] = c
            edge_dict[(v, u)] = c
            
    # Local search
    best_edges = edges
    best_valid = False
    
    for _ in range(10000): # Iterations
        if time.time() > DEADLINE:
            break
            
        valid, conflict_cycle = check_validity(edge_dict)
        if valid:
            best_valid = True
            best_edges = [list(e) for e in edges]
            break
            
        if conflict_cycle:
            a, b, c, d = conflict_cycle
            # Get the 4 edges
            e1_key = (min(a,b), max(a,b))
            e2_key = (min(b,c), max(b,c))
            e3_key = (min(c,d), max(c,d))
            e4_key = (min(d,a), max(d,a))
            
            # Pick a random edge from the cycle to recolor
            edge_keys = [e1_key, e2_key, e3_key, e4_key]
            target_key = random.choice(edge_keys)
            old_color = edge_dict[target_key]
            new_color = random.randint(0, num_colors-1)
            
            # Apply change
            edge_dict[target_key] = new_color
            # Update edges list
            for i, e in enumerate(edges):
                if (e[0], e[1]) == target_key or (e[1], e[0]) == target_key:
                    edges[i][2] = new_color
                    break
                    
            # If no improvement in next few steps, maybe backtrack?
            # For now, just random walk.
            
    if not best_valid:
        # Fallback to sparse graph if local search fails
        # We can try to keep more edges by removing conflicting ones?
        # For now, return the sparse graph which is guaranteed valid.
        return [[0, v, 0] for v in range(1, n)]
        
    return best_edges
# EVOLVE-BLOCK-END
