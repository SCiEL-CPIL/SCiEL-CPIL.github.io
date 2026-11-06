"""
Greedy Set-Cover / Maximal Independent Set strategy for 'many edge colors without a rainbow C4' on K_16.

This program treats the problem as a hypergraph independent set problem.
Vertices in the hypergraph are the 120 edges of K_16.
Hyperedges are the sets of 4 edges that form a rainbow 4-cycle (i.e., the 4 edges must have distinct colors).
We seek a partition of the 120 edges into the maximum number of color classes such that no hyperedge is fully contained within a single class.
Equivalently, we are looking for a coloring where the set of monochromatic edges for each color is an independent set in the hypergraph of "bad" 4-cycles.

The algorithm uses a greedy set-cover approach:
1. Initialize all edges as uncolored.
2. While uncolored edges remain:
   a. Find the largest subset of uncolored edges that does not contain any "bad" 4-cycle (where a bad cycle is one where all 4 edges are in the subset).
   b. Assign a new color to this subset.
3. Return the coloring.

This is a structural change from the seed's O(1) constructive min(i,j) coloring, which is provably suboptimal for maximizing color count in this specific context if we allow global rearrangement (though min(i,j) is a strong baseline, it uses only 15 colors. The theoretical max is higher, potentially up to ~24-30 depending on the exact extremal graph theory bounds for C4-free colorings).
"""

import time
import random

N = 16
M = N * (N - 1) // 2  # 120 edges

def _edge_index(i, j):
    """Lexicographic index of edge (i,j), i<j, in the list of all C(N,2) edges."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _get_edge_vertices(idx):
    """Reverse map from index to (i, j)."""
    # Inverse of _edge_index
    # idx = i*(N-1) - i*(i-1)//2 + (j-i-1)
    # Let k = N-1-i. The number of edges starting with vertices < i is sum_{a=0}^{i-1} (N-1-a) = i*(N-1) - i*(i-1)//2.
    # We can iterate i from 0 to N-1 to find the correct block.
    for i in range(N):
        start = i * (N - 1) - i * (i - 1) // 2
        end = start + (N - 1 - i)
        if start <= idx < end:
            j = i + 1 + (idx - start)
            return i, j
    raise ValueError("Invalid index")

# Precompute all 4-cycles and their edge indices
# A 4-cycle is defined by 4 distinct vertices {a,b,c,d}. 
# There are 3 ways to form a cycle on 4 vertices: (a,b,c,d), (a,b,d,c), (a,c,b,d).
# Actually, the task says "all three cycles on every four-vertex subset are checked".
# The edges involved in any cycle on {a,b,c,d} (sorted a<b<c<d) are:
# Cycle 1: (a,b), (b,c), (c,d), (d,a) -> indices: ab, bc, cd, da
# Cycle 2: (a,b), (b,d), (d,c), (c,a) -> indices: ab, bd, dc, ca
# Cycle 3: (a,c), (c,b), (b,d), (d,a) -> indices: ac, cb, bd, da

def _precompute_cycles():
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = [_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(d, a)]
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = [_edge_index(a, b), _edge_index(b, d), _edge_index(d, c), _edge_index(c, a)]
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = [_edge_index(a, c), _edge_index(c, b), _edge_index(b, d), _edge_index(d, a)]
                    cycles.append(e3)
    return cycles

CYCLES = _precompute_cycles()

def _is_valid_subset(edges_in_subset):
    """Check if a set of edges contains no rainbow 4-cycle where all 4 edges are in the set."""
    # This check is expensive if done naively for every candidate.
    # We will use a heuristic greedy approach instead of checking all subsets.
    pass

def construct():
    deadline = time.time() + 80.0  # Safety margin
    
    # Fallback to the known valid 15-color solution if we run out of time or fail
    def _fallback():
        colors = [0] * M
        for i in range(N):
            base = _edge_index(i, i + 1)
            for j in range(i + 1, N):
                colors[base + (j - i - 1)] = i
        return colors

    # Greedy Set Cover / Maximal Independent Set
    # We want to partition edges into max number of classes.
    # Each class must be "C4-rainbow-free" internally.
    # A class is C4-rainbow-free if for every 4-cycle, not all 4 edges are in the class.
    
    uncolored = set(range(M))
    colors = [-1] * M
    current_color = 0
    
    while uncolored and time.time() < deadline:
        # We want to find a large subset of uncolored edges that is valid.
        # Heuristic: Start with all uncolored edges. 
        # Remove edges that participate in "conflicts".
        # A conflict arises if a 4-cycle has all 4 edges in the candidate set.
        # We want to remove the minimum number of edges to break all such cycles.
        # This is a hitting set problem on the cycles.
        
        candidate = set(uncolored)
        
        # Iteratively remove edges that are in many "active" cycles
        # An active cycle is one where all 4 edges are in candidate.
        
        # To speed this up, we can just try to build a valid set greedily.
        # Start with empty set S.
        # Add edges from uncolored one by one if they don't create a bad cycle.
        # This might result in small sets. 
        # Better: Start with all uncolored, and remove edges that cause the most conflicts.
        
        # Let's try a simple greedy removal:
        # 1. Count for each edge in candidate, how many 4-cycles are "complete" (all 4 edges in candidate).
        # 2. Remove the edge with the highest count.
        # 3. Repeat until no complete cycles exist.
        
        # Optimization: Only check cycles relevant to the current candidate.
        # Since N=16, M=120, number of cycles is C(16,4)*3 = 1820*3 = 5460.
        # This is small enough to check repeatedly.
        
        changed = True
        while changed and time.time() < deadline:
            changed = False
            # Count conflicts per edge
            conflict_count = {e: 0 for e in candidate}
            
            for cycle in CYCLES:
                # Check if all 4 edges are in candidate
                if all(e in candidate for e in cycle):
                    for e in cycle:
                        conflict_count[e] += 1
                    # If we found any conflict, we need to resolve them.
                    # We will remove the edge with max conflict in the next step.
                    # But we should count all conflicts first to pick the best removal.
            
            # Find edge with max conflicts
            if any(c > 0 for c in conflict_count.values()):
                # Pick the edge with the most conflicts
                max_edge = max(conflict_count, key=conflict_count.get)
                candidate.remove(max_edge)
                changed = True
        
        # Now 'candidate' is a valid set (no rainbow C4 fully inside).
        # Assign current_color to all edges in candidate.
        if not candidate:
            # Fallback: assign random uncolored edges one by one? 
            # If candidate is empty, it means we couldn't find any valid set from uncolored?
            # That shouldn't happen unless uncolored is empty or we made a mistake.
            # Actually, a single edge is always a valid set.
            # So candidate should never be empty if uncolored is not.
            # If it is, we have a bug in the conflict resolution.
            # Fallback: just take one edge.
            candidate = {next(iter(uncolored))}
        
        for e in candidate:
            colors[e] = current_color
            uncolored.remove(e)
        
        current_color += 1
        
        # Optional: Randomization to escape local optima?
        # The greedy removal might be too deterministic.
        # We can shuffle the order of edges considered for removal if counts are tied.
        # Or we can restart the whole process with a different random seed if we want to explore more.
        # Given the 60s limit, we might only have time for one pass.
        # Let's add a simple randomization: when multiple edges have the max conflict count, pick one randomly.
        
    # If we still have uncolored edges (due to time limit), color them uniquely (1 edge per color)
    # This is always valid because a single edge cannot form a 4-cycle.
    for e in uncolored:
        colors[e] = current_color
        current_color += 1
        
    # Final verification (safety net)
    # Check if the coloring is valid.
    # If not, return fallback.
    
    # We can't afford a full O(Cycles) check for every edge if we did many colors, but we can check the whole thing once.
    # The check is: for every cycle, len({colors[e] for e in cycle}) < 4.
    
    valid = True
    for cycle in CYCLES:
        if len({colors[e] for e in cycle}) == 4:
            valid = False
            break
    
    if not valid:
        return _fallback()
    
    return colors

# Note: The above greedy approach might be slow if the "remove max conflict" loop iterates many times.
# However, with 120 edges, it should be fast.
# Let's refine the conflict resolution to be faster.
# Instead of recalculating all conflicts from scratch, we can just do it once per color class.
# The number of colors will be roughly M / (avg class size).
# If avg class size is 10, we have 12 iterations.
# Each iteration checks 5460 cycles. 12 * 5460 = ~65k checks. Very fast.

# Let's double check the logic.
# "No four-cycle may have four distinct edge colors."
# This is equivalent to saying: For every 4-cycle, the set of colors on its edges has size <= 3.
# My greedy algorithm ensures that for every color class (set of edges with same color), there is no 4-cycle entirely contained in that class.
# Wait. If a 4-cycle has edges colored 1, 2, 3, 4, then each color class contains exactly 1 edge of that cycle.
# The condition "no 4-cycle entirely contained in a color class" means that for any color c, there is no cycle where all 4 edges have color c.
# This is a NECESSARY condition for the coloring to be valid?
# NO.
# If a cycle has colors 1, 1, 2, 2, it is valid.
# If a cycle has colors 1, 2, 3, 4, it is INVALID.
# My algorithm prevents a cycle from having colors 1, 1, 1, 1.
# It does NOT prevent a cycle from having colors 1, 2, 3, 4.
# 
# RE-READING THE PROBLEM:
# "No four-cycle may have four distinct edge colors."
# This means we must AVOID the case where a 4-cycle uses 4 different colors.
# It DOES NOT say "No color class can contain a 4-cycle".
# 
# My previous interpretation was WRONG.
# The constraint is: For every 4-cycle C, |{color(e) : e in C}| != 4.
# 
# The seed program uses color(i,j) = i.
# Cycle a-b-c-d-a: colors a, b, c, a. Distinct: {a,b,c}. Size 3. Valid.
# Cycle a-b-d-c-a: colors a, b, d, a. Distinct: {a,b,d}. Size 3. Valid.
# Cycle a-c-b-d-a: colors a, c, b, a. Distinct: {a,b,c}. Size 3. Valid.
# 
# So the constraint is about the *diversity* of colors on a cycle, not about monochromatic cycles.
# 
# Therefore, the "Set Cover" approach I was planning (partitioning into C4-free subgraphs) is solving the WRONG problem.
# The "C4-free subgraph" constraint applies to the *complement* of the color classes?
# No.
# 
# Let's rethink the structural change.
# The problem is: Maximize number of colors such that no 4-cycle is rainbow.
# This is equivalent to: The coloring is NOT a proper coloring of the "4-cycle hypergraph" in the sense of avoiding 4 distinct colors.
# 
# A different strategy: **Local Search / Simulated Annealing**.
# Start with a random coloring or the seed coloring.
# Define energy = number of rainbow 4-cycles.
# We want to minimize energy to 0, while maximizing the number of colors used.
# We can use a penalty method: Score = (Number of Colors Used) - Lambda * (Number of Rainbow Cycles).
# Or, simply search for a valid coloring with many colors.
# 
# Since the seed is valid and has 15 colors, we can start from the seed and try to *increase* the number of colors by splitting color classes, ensuring we don't create rainbow cycles.
# 
# Algorithm:
# 1. Start with seed coloring (15 colors, valid).
# 2. While time remains:
#    a. Pick a random color class C with size > 1.
#    b. Try to split C into two subsets C1 and C2.
#    c. Check if the new coloring (where C1 has color A, C2 has color B, others unchanged) is still valid.
#    d. If valid, accept the split (increases color count by 1).
#    e. If invalid, try a different split or different color class.
# 
# This is a "Grow the number of colors" heuristic.
# It is structurally different from the static construction.
# 
# How to check validity efficiently?
# When we split color class C into C1 and C2, we only need to check 4-cycles that are affected.
# A 4-cycle is affected if it contains at least one edge from C.
# Specifically, if a 4-cycle has edges from C1 and C2, its color set might change.
# Previously, all edges in C had the same color. Now they have two different colors.
# This can only INCREASE the number of distinct colors on that cycle.
# If the cycle had 3 distinct colors before, and we split a color into two, it might become 4.
# If the cycle had 2 distinct colors, it might become 3 or 4.
# If the cycle had 1 distinct color (monochromatic), it might become 2.
# 
# So, we only need to check 4-cycles that contain at least one edge from the split class C.
# 
# Implementation details:
# 1. Precompute for each edge, the list of 4-cycles it belongs to.
# 2. Start with seed coloring.
# 3. Loop:
#    - Pick a random color class with size > 1.
#    - Randomly partition its edges into C1 and C2.
#    - For each cycle that intersects C:
#        - Calculate the new set of colors on the cycle.
#        - If size == 4, the split is invalid.
#    - If valid, apply the split.
#    - If invalid, revert and try again.
# 
# This should be fast and allows us to potentially exceed 15 colors.

def _precompute_edge_cycles():
    edge_to_cycles = {i: [] for i in range(M)}
    for idx, cycle in enumerate(CYCLES):
        for e in cycle:
            edge_to_cycles[e].append(idx)
    return edge_to_cycles

EDGE_TO_CYCLES = _precompute_edge_cycles()

def _check_split_validity(colors, split_edges, new_color):
    """
    Check if assigning 'new_color' to 'split_edges' (which previously had 'old_color')
    results in a valid coloring.
    'colors' is the current coloring array.
    'split_edges' is the list of edges to be reassigned.
    'new_color' is the new color value.
    """
    old_color = colors[split_edges[0]]
    
    # Identify all cycles that contain any edge from split_edges
    affected_cycles = set()
    for e in split_edges:
        for cycle_idx in EDGE_TO_CYCLES[e]:
            affected_cycles.add(cycle_idx)
    
    for cycle_idx in affected_cycles:
        cycle = CYCLES[cycle_idx]
        # Calculate the set of colors on this cycle
        # We can do this by iterating the 4 edges
        col_set = set()
        for e in cycle:
            if e in split_edges:
                col_set.add(new_color)
            else:
                col_set.add(colors[e])
        
        if len(col_set) == 4:
            return False
            
    return True

def construct():
    deadline = time.time() + 80.0
    
    # Initialize with seed coloring
    colors = [0] * M
    for i in range(N):
        base = _edge_index(i, i + 1)
        for j in range(i + 1, N):
            colors[base + (j - i - 1)] = i
            
    # Map color to list of edges
    color_map = {i: [] for i in range(N-1)}
    for e in range(M):
        color_map[colors[e]].append(e)
        
    # List of active colors
    active_colors = list(color_map.keys())
    
    # Random seed for reproducibility in exploration, but we want to find a good one.
    # Using random module.
    
    while time.time() < deadline:
        # Pick a random color class to split
        if not active_colors:
            break
            
        # Prefer larger classes? Or random?
        # Random is fine.
        color_to_split = random.choice(active_colors)
        edges_in_class = color_map[color_to_split]
        
        if len(edges_in_class) < 2:
            # Cannot split
            continue
            
        # Try to split
        # Random permutation of edges in class
        shuffled = edges_in_class[:]
        random.shuffle(shuffled)
        
        # Try different split sizes? 
        # Start with small splits (1 edge) to be safe, or random?
        # Splitting 1 edge is safest.
        # Let's try splitting 1 edge first.
        split_size = 1
        # If 1 edge split fails, try larger? Or just move on?
        # Let's try a few random split sizes.
        successful_split = False
        
        for attempt in range(10):
            if attempt == 0:
                s_size = 1
            else:
                s_size = random.randint(1, len(edges_in_class) // 2)
                
            split_edges = shuffled[:s_size]
            remaining_edges = shuffled[s_size:]
            
            new_color = max(active_colors) + 1
            
            if _check_split_validity(colors, split_edges, new_color):
                # Apply split
                for e in split_edges:
                    colors[e] = new_color
                    color_map[color_to_split].remove(e)
                
                color_map[new_color] = split_edges
                active_colors.append(new_color)
                successful_split = True
                break
                
        if not successful_split:
            # Maybe try a different color next iteration
            continue
            
    return colors
