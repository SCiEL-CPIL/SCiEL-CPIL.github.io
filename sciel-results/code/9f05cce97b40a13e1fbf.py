"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Build conflict graph: edge (a,b) if a+b is a perfect square (a<=b)
    # Also include diagonal: a+a = 2a must NOT be a perfect square
    n = 256
    squares = set()
    for i in range(1, int(math.isqrt(2*n)) + 1):
        squares.add(i*i)
    
    # Conflict matrix: conflict[i][j] = True if i+j is a perfect square (i<=j)
    # We want to select max subset with no conflicts
    # This is maximum independent set = maximum clique in complement
    # But ILP is cleaner: maximize sum(x_i) s.t. x_i + x_j <= 1 for all conflicting pairs
    
    # Build list of conflicting pairs (i, j) with i <= j
    conflicts = []
    for i in range(1, n+1):
        # Diagonal: 2*i must not be square
        if 2*i in squares:
            # This element alone is invalid (can't include it)
            # Actually wait - if 2i is a square, then a=b=i gives a+b=2i which is square
            # So we simply cannot include i
            pass
        for j in range(i+1, n+1):
            if i + j in squares:
                conflicts.append((i, j))
    
    # Elements that are self-conflicting (2*i is a perfect square)
    self_conflict = set()
    for i in range(1, n+1):
        if 2*i in squares:
            self_conflict.add(i)
    
    # Valid elements are those not in self_conflict
    valid_elements = [i for i in range(1, n+1) if i not in self_conflict]
    elem_to_idx = {e: idx for idx, e in enumerate(valid_elements)}
    m = len(valid_elements)
    
    # Filter conflicts to only include pairs where both elements are valid
    filtered_conflicts = []
    for (a, b) in conflicts:
        if a in elem_to_idx and b in elem_to_idx:
            filtered_conflicts.append((elem_to_idx[a], elem_to_idx[b]))
    
    # Try ILP using scipy if available, otherwise use heuristic
    best_solution = None
    best_size = 0
    
    # Heuristic: greedy + local search
    # Start with all valid elements, remove conflicts
    # Use a greedy approach: process elements, add if no conflict with already added
    
    def greedy_local_search(valid_elems, conflict_pairs, deadline):
        """Greedy construction + local search improvement."""
        n_elem = len(valid_elems)
        # Build adjacency: adj[i] = set of indices j where (i,j) is a conflict
        adj = [set() for _ in range(n_elem)]
        for (i, j) in conflict_pairs:
            adj[i].add(j)
            adj[j].add(i)
        
        # Greedy: sort by degree (ascending), add if no conflict
        order = sorted(range(n_elem), key=lambda x: len(adj[x]))
        selected = set()
        for idx in order:
            # Check if any selected neighbor conflicts
            conflict_found = False
            for nb in adj[idx]:
                if nb in selected:
                    conflict_found = True
                    break
            if not conflict_found:
                selected.add(idx)
        
        # Local search: try to improve by swaps
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n_elem):
                if i in selected:
                    continue
                # Try adding i: must remove all its neighbors in selected
                neighbors_in_selected = adj[i] & selected
                if len(neighbors_in_selected) < 1:
                    # Can add directly
                    selected.add(i)
                    improved = True
                    break
                # Try swap: add i, remove one neighbor, see if net gain
                # Net gain = 1 - len(neighbors_in_selected) if we remove all
                # Better: try removing just one neighbor at a time
                for nb in list(neighbors_in_selected):
                    # Remove nb, add i
                    # Check if i conflicts with any other selected (besides nb)
                    other_conflicts = adj[i] & (selected - {nb})
                    if not other_conflicts:
                        # Check if removing nb causes no issue (it doesn't, we're removing)
                        # But we need to make sure nb wasn't needed for others
                        # Actually, removing nb and adding i: net change = +1 - 1 = 0
                        # Only useful if we can then add more
                        pass
            # Try removals to enable additions
            for i in range(n_elem):
                if i not in selected:
                    continue
                # Try removing i to see if it enables adding multiple others
                # Skip for now, keep simple
        
        # Convert back to element values
        result = [valid_elems[idx] for idx in selected]
        return sorted(result)
    
    # Run heuristic
    heuristic_result = greedy_local_search(valid_elements, filtered_conflicts, deadline - 5)
    best_solution = heuristic_result
    best_size = len(heuristic_result)
    
    # Try ILP if time permits
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import linprog
            import numpy as np
            
            # ILP: maximize sum(x_i) s.t. x_i + x_j <= 1 for all conflicts, x_i in {0,1}
            # scipy linprog doesn't do integer programming directly
            # Use pulp if available, otherwise skip
            pass
        except ImportError:
            pass
    
    # Try a more sophisticated approach: branch and bound with time limit
    if time.time() < deadline - 5 and m <= 200:
        # Branch and bound for maximum independent set
        # Use bitsets for speed
        n_elem = m
        adj_bitset = [0] * n_elem
        for (i, j) in filtered_conflicts:
            adj_bitset[i] |= (1 << j)
            adj_bitset[j] |= (1 << i)
        
        # Greedy upper bound: for a set of candidates, max independent set <= |candidates|
        # Better bound: greedy coloring of the subgraph
        
        def greedy_color_bound(candidates_mask):
            """Upper bound on MIS size using greedy coloring."""
            # Extract candidate nodes
            cand = []
            for i in range(n_elem):
                if (candidates_mask >> i) & 1:
                    cand.append(i)
            if not cand:
                return 0
            # Greedy coloring
            color = {}
            for node in cand:
                # Get colors of neighbors in cand
                neighbor_colors = set()
                for nb in cand:
                    if nb != node and (adj_bitset[node] >> nb) & 1:
                        if nb in color:
                            neighbor_colors.add(color[nb])
                # Assign smallest available color
                c = 0
                while c in neighbor_colors:
                    c += 1
                color[node] = c
            return max(color.values()) + 1 if color else 0
        
        # Branch and bound
        best_bb = 0
        best_bb_set = set()
        
        def branch_and_bound(candidates_mask, current_set, depth):
            nonlocal best_bb, best_bb_set
            if time.time() > deadline - 2:
                return
            if candidates_mask == 0:
                if len(current_set) > best_bb:
                    best_bb = len(current_set)
                    best_bb_set = set(current_set)
                return
            
            # Upper bound
            ub = len(current_set) + greedy_color_bound(candidates_mask)
            if ub <= best_bb:
                return
            
            # Pick a vertex to branch on
            # Heuristic: pick vertex with highest degree in candidate subgraph
            best_v = -1
            best_deg = -1
            for v in range(n_elem):
                if (candidates_mask >> v) & 1:
                    deg = bin(adj_bitset[v] & candidates_mask).count('1')
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
            
            if best_v == -1:
                if len(current_set) > best_bb:
                    best_bb = len(current_set)
                    best_bb_set = set(current_set)
                return
            
            # Branch 1: include best_v
            new_set = current_set | {best_v}
            # Remove best_v and its neighbors from candidates
            new_candidates = candidates_mask & ~(1 << best_v) & ~adj_bitset[best_v]
            branch_and_bound(new_candidates, new_set, depth + 1)
            
            # Branch 2: exclude best_v
            new_candidates = candidates_mask & ~(1 << best_v)
            branch_and_bound(new_candidates, current_set, depth + 1)
        
        initial_mask = (1 << n_elem) - 1
        branch_and_bound(initial_mask, set(), 0)
        
        if best_bb > best_size:
            best_solution = [valid_elements[idx] for idx in best_bb_set]
            best_size = best_bb
    
    # Final validation
    if best_solution:
        result_set = set(best_solution)
        # Verify no a+b is a perfect square
        valid = True
        for a in best_solution:
            for b in best_solution:
                if a <= b:
                    if a + b in squares:
                        valid = False
                        break
            if not valid:
                break
        if valid:
            return sorted(best_solution)
    
    # Fallback: return heuristic result (already validated implicitly)
    return sorted(heuristic_result)
# EVOLVE-BLOCK-END
