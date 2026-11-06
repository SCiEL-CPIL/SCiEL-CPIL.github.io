"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from itertools import combinations
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-vertex subset, the 3 cycles
def precompute_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}:
        # (a-b-c-d-a): edges (a,b), (b,c), (c,d), (d,a)
        e1 = (edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a))
        # (a-b-d-c-a): edges (a,b), (b,d), (d,c), (c,a)
        e2 = (edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a))
        # (a-c-b-d-a): edges (a,c), (c,b), (b,d), (d,a)
        e3 = (edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

ALL_CYCLES = precompute_cycles()
NUM_CYCLES = len(ALL_CYCLES)  # 5460

# Precompute: for each edge, which cycles contain it
EDGE_TO_CYCLES = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(ALL_CYCLES):
    for e in cycle:
        EDGE_TO_CYCLES[e].append(ci)

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring):
    """Check if no 4-cycle has 4 distinct colors."""
    for e1, e2, e3, e4 in ALL_CYCLES:
        if len({coloring[e1], coloring[e2], coloring[e3], coloring[e4]}) == 4:
            return False
    return True

def count_violations(coloring):
    """Count number of 4-cycles with 4 distinct colors."""
    v = 0
    for e1, e2, e3, e4 in ALL_CYCLES:
        if len({coloring[e1], coloring[e2], coloring[e3], coloring[e4]}) == 4:
            v += 1
    return v

# Fast incremental violation check for a single edge color change
def check_edge_change_valid(coloring, edge, new_color):
    """Check if changing coloring[edge] to new_color would create any 4-distinct-color cycle."""
    old_color = coloring[edge]
    if old_color == new_color:
        return True
    
    for ci in EDGE_TO_CYCLES[edge]:
        e1, e2, e3, e4 = ALL_CYCLES[ci]
        # Get the other three edges in this cycle
        other_edges = []
        for e in (e1, e2, e3, e4):
            if e != edge:
                other_edges.append(e)
        
        # Check if with new_color, the four colors would all be distinct
        other_colors = {coloring[e] for e in other_edges}
        if len(other_colors) == 3 and new_color not in other_colors:
            return False
    
    return True

def try_add_new_color(coloring, edge, new_color):
    """Try to set coloring[edge] = new_color, return True if valid."""
    if not check_edge_change_valid(coloring, edge, new_color):
        return False
    coloring[edge] = new_color
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    random.seed(42)
    
    # Start with all edges color 0
    best = [0] * NUM_EDGES
    best_score = 1  # 1 color used
    
    # Strategy: Build valid coloring, then aggressively add new colors via local search
    
    # Phase 1: Generate initial valid colorings using group-based approach
    initial_colorings = []
    
    # Try different partition structures
    for num_groups in range(2, 10):
        if time.time() > deadline - 30:
            break
        
        # Multiple random assignments
        for trial in range(3):
            if time.time() > deadline - 30:
                break
            
            # Random balanced assignment
            groups = list(range(num_groups))
            while len(groups) < N:
                # Add vertices to smallest group
                group_sizes = [groups.count(g) for g in range(num_groups)]
                min_size = min(group_sizes)
                candidates = [g for g in range(num_groups) if group_sizes[g] == min_size]
                groups.append(random.choice(candidates))
            random.shuffle(groups)
            
            # Build coloring based on group pairs
            group_pairs = {}
            color_counter = 1  # start at 1, 0 is default
            
            for g1 in range(num_groups):
                for g2 in range(g1 + 1, num_groups):
                    group_pairs[(g1, g2)] = color_counter
                    color_counter += 1
            
            # Build coloring
            coloring = [0] * NUM_EDGES
            for i in range(N):
                for j in range(i + 1, N):
                    g1, g2 = groups[i], groups[j]
                    key = (min(g1, g2), max(g1, g2))
                    coloring[edge_idx(i, j)] = group_pairs.get(key, 0)
            
            # Check validity
            if count_violations(coloring) == 0:
                score = count_colors(coloring)
                if score > best_score:
                    best_score = score
                    best = coloring[:]
                initial_colorings.append((coloring[:], score))
    
    # If no valid coloring found, use all-zeros
    if not initial_colorings:
        initial_colorings = [([0] * NUM_EDGES, 1)]
    
    # Phase 2: Aggressive local search to add new colors
    # Use the best initial coloring as starting point
    coloring = best[:]
    current_score = best_score
    
    # Track used colors
    used_colors = set(coloring)
    
    # Local search: repeatedly try to introduce new colors
    max_iterations = 50000
    no_improve_count = 0
    
    for iteration in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Find edges that can accept a new color
        # Prioritize edges in colors that appear multiple times
        color_counts = Counter(coloring)
        
        # Get edges sorted by color frequency (most frequent first)
        edges_by_freq = []
        for color, count in color_counts.items():
            if count > 1:
                edges_with_color = [i for i in range(NUM_EDGES) if coloring[i] == color]
                for e in edges_with_color:
                    edges_by_freq.append((count, e))
        
        if not edges_by_freq:
            break
        
        # Pick a random edge from the list, weighted by frequency
        random.shuffle(edges_by_freq)
        
        # Try several candidate edges
        found_improvement = False
        for _ in range(min(20, len(edges_by_freq))):
            if time.time() > deadline - 10:
                break
            
            freq, e = random.choice(edges_by_freq)
            old_color = coloring[e]
            
            # Try a new unused color
            available_colors = [c for c in range(120) if c not in used_colors]
            if not available_colors:
                break
            
            # Try a few random new colors
            for new_color in random.sample(available_colors, min(5, len(available_colors))):
                if check_edge_change_valid(coloring, e, new_color):
                    # Valid! Apply the change
                    coloring[e] = new_color
                    used_colors.add(new_color)
                    current_score += 1
                    found_improvement = True
                    no_improve_count = 0
                    
                    if current_score > best_score:
                        best_score = current_score
                        best = coloring[:]
                    break
            
            if found_improvement:
                break
        
        if not found_improvement:
            no_improve_count += 1
            if no_improve_count > 200:
                # Try a different strategy: randomly change some edges to break symmetry
                if time.time() > deadline - 10:
                    break
                
                # Make a random valid perturbation
                for _ in range(10):
                    e = random.randint(0, NUM_EDGES - 1)
                    old_color = coloring[e]
                    # Try to change to another existing color
                    other_colors = [c for c in used_colors if c != old_color]
                    if other_colors:
                        new_color = random.choice(other_colors)
                        if check_edge_change_valid(coloring, e, new_color):
                            coloring[e] = new_color
                            # Remove old color if no longer used
                            if coloring.count(old_color) == 0:
                                used_colors.discard(old_color)
                                current_score -= 1
                            if new_color not in used_colors:
                                used_colors.add(new_color)
                                current_score += 1
                            if current_score > best_score:
                                best_score = current_score
                                best = coloring[:]
                            break
                
                no_improve_count = 0
    
    # Phase 3: Final polish - try to add more colors with more aggressive search
    if time.time() < deadline - 5:
        coloring = best[:]
        current_score = best_score
        used_colors = set(coloring)
        
        # More targeted search: for each edge, try all possible new colors
        for e in range(NUM_EDGES):
            if time.time() > deadline - 3:
                break
            
            old_color = coloring[e]
            available_colors = [c for c in range(120) if c not in used_colors]
            
            if not available_colors:
                break
            
            for new_color in available_colors:
                if check_edge_change_valid(coloring, e, new_color):
                    coloring[e] = new_color
                    used_colors.add(new_color)
                    current_score += 1
                    if current_score > best_score:
                        best_score = current_score
                        best = coloring[:]
                    break
    
    return best
# EVOLVE-BLOCK-END
