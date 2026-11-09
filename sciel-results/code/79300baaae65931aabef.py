"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 85  # 85s safety margin under 90s limit
    
    # Precompute all edge indices and the C4 constraints
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            edges.append((i, j))
    num_edges = len(edges)  # 120
    
    # edge_index[(i,j)] -> position in edges list
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
    
    # For each 4-vertex subset, precompute the 3 C4s (each is a set of 4 edge indices)
    # A C4 on vertices {a,b,c,d} (sorted) has 3 possible cycles:
    # (a,b,c,d), (a,b,d,c), (a,c,b,d)
    # Each cycle uses 4 specific edges
    c4_constraints = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d-a
                    c1 = [edge_idx[(a,b)], edge_idx[(b,c)], edge_idx[(c,d)], edge_idx[(d,a)]]
                    # Cycle 2: a-b-d-c-a
                    c2 = [edge_idx[(a,b)], edge_idx[(b,d)], edge_idx[(d,c)], edge_idx[(c,a)]]
                    # Cycle 3: a-c-b-d-a
                    c3 = [edge_idx[(a,c)], edge_idx[(c,b)], edge_idx[(b,d)], edge_idx[(d,a)]]
                    c4_constraints.append(c1)
                    c4_constraints.append(c2)
                    c4_constraints.append(c3)
    
    num_c4s = len(c4_constraints)
    
    # For each edge, which C4s does it participate in?
    edge_to_c4s = [[] for _ in range(num_edges)]
    for c4_idx, c4 in enumerate(c4_constraints):
        for e in c4:
            edge_to_c4s[e].append(c4_idx)
    
    def is_valid(coloring):
        """Check if coloring has no C4 with 4 distinct colors."""
        for c4 in c4_constraints:
            colors = set()
            for e in c4:
                colors.add(coloring[e])
                if len(colors) > 3:
                    return False
        return True
    
    def count_colors(coloring):
        return len(set(coloring))
    
    def greedy_construct(seed):
        """Build coloring by assigning edges in random order, preferring new colors."""
        random.seed(seed)
        coloring = [-1] * num_edges
        order = list(range(num_edges))
        random.shuffle(order)
        
        for e in order:
            # Try to assign a color that doesn't create a C4 with 4 distinct colors
            # First try: find if we can use an existing color
            # Then try: use a new color
            # We want to maximize distinct colors, so prefer new colors when possible
            
            # Check if assigning a new unique color would be valid
            # A new color is always safe for that edge itself, but we need to check
            # all C4s containing this edge: if the other 3 edges already have 3 distinct
            # colors, then adding a 4th distinct color would violate.
            
            new_color_valid = True
            for c4_idx in edge_to_c4s[e]:
                other_colors = set()
                for oe in c4_constraints[c4_idx]:
                    if oe != e:
                        if coloring[oe] != -1:
                            other_colors.add(coloring[oe])
                        else:
                            # Unassigned edge - we'll handle later, assume it could be anything
                            # For now, be conservative: if there are unassigned edges, 
                            # a new color might still be OK since they could match
                            pass
                if len(other_colors) >= 3:
                    new_color_valid = False
                    break
            
            if new_color_valid:
                # Assign a new unique color
                coloring[e] = max(coloring) + 1 if max(coloring) >= 0 else 0
            else:
                # Must reuse an existing color
                # Try each existing color
                existing_colors = set(c for c in coloring if c != -1)
                random_colors = list(existing_colors)
                random.shuffle(random_colors)
                
                assigned = False
                for c in random_colors:
                    # Check if assigning color c to edge e is valid
                    valid = True
                    for c4_idx in edge_to_c4s[e]:
                        other_colors = set()
                        for oe in c4_constraints[c4_idx]:
                            if oe != e:
                                if coloring[oe] != -1:
                                    other_colors.add(coloring[oe])
                                else:
                                    other_colors.add(c)  # worst case: unassigned takes same color
                        other_colors.add(c)
                        if len(other_colors) > 3:
                            valid = False
                            break
                    if valid:
                        coloring[e] = c
                        assigned = True
                        break
                
                if not assigned:
                    # Fallback: assign color 0 (shouldn't happen often)
                    coloring[e] = 0
        
        # Fill any remaining -1 (shouldn't happen)
        for e in range(num_edges):
            if coloring[e] == -1:
                coloring[e] = 0
        
        return coloring
    
    def local_search(coloring, max_iters=5000):
        """Try to improve by changing colors of individual edges."""
        coloring = list(coloring)
        current_score = count_colors(coloring)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            e = random.randint(0, num_edges - 1)
            old_color = coloring[e]
            
            # Try changing to a new color
            new_color = max(coloring) + 1
            coloring[e] = new_color
            
            # Check validity
            valid = True
            for c4_idx in edge_to_c4s[e]:
                colors = set()
                for oe in c4_constraints[c4_idx]:
                    colors.add(coloring[oe])
                if len(colors) > 3:
                    valid = False
                    break
            
            if valid:
                new_score = count_colors(coloring)
                if new_score >= current_score:
                    current_score = new_score
                # Keep the change
            else:
                coloring[e] = old_color
        
        return coloring, current_score
    
    # Main search
    best_coloring = None
    best_score = 0
    
    # Try multiple random seeds
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        coloring = greedy_construct(attempt)
        score = count_colors(coloring)
        
        if score > best_score:
            best_score = score
            best_coloring = list(coloring)
        
        # Local search improvement
        if time.time() < deadline - 5:
            improved, improved_score = local_search(coloring, max_iters=2000)
            if improved_score > best_score:
                best_score = improved_score
                best_coloring = list(improved)
    
    if best_coloring is None:
        best_coloring = [0] * num_edges
    
    # Renumber colors to 0..k-1 for cleanliness (not required but clean)
    color_map = {}
    next_color = 0
    for c in best_coloring:
        if c not in color_map:
            color_map[c] = next_color
            next_color += 1
    return [color_map[c] for c in best_coloring]
# EVOLVE-BLOCK-END

