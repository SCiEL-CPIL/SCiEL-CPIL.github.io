"""Edge-colourings of K16 in which every K4 sees five colours.
construct() -> 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16, in lexicographic order.
Every set of four vertices must span edges of at least five distinct colours. Minimize the number of colours used.
"""
import time
import random
import itertools
import sys

_START = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0  # seconds margin

# Precompute K4s and Edge Indices
N = 16
ALL_EDGES = [(i, j) for i in range(N) for j in range(i + 1, N)]
EDGE_MAP = {e: i for i, e in enumerate(ALL_EDGES)}
NUM_EDGES = len(ALL_EDGES)

# Precompute K4s: list of tuples (a, b, c, d)
ALL_K4S = [(a, b, c, d) for a in range(N) for b in range(a + 1, N)
           for c in range(b + 1, N) for d in range(c + 1, N)]
NUM_K4S = len(ALL_K4S)

# Map each K4 to the indices of its 6 edges
K4_EDGES = []
for k4 in ALL_K4S:
    a, b, c, d = k4
    edges = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
    K4_EDGES.append([EDGE_MAP[e] for e in edges])

# Map each edge to the K4s it belongs to
EDGE_K4S = [[] for _ in range(NUM_EDGES)]
for k4_idx in range(NUM_K4S):
    for e_idx in K4_EDGES[k4_idx]:
        EDGE_K4S[e_idx].append(k4_idx)

def _check_k4_validity(edge_idx, new_color, current_colors):
    """Check if assigning new_color to edge_idx keeps all incident K4s valid."""
    for k4_idx in EDGE_K4S[edge_idx]:
        k4_e = K4_EDGES[k4_idx]
        colors = [0] * 6
        for pos, e in enumerate(k4_e):
            if e == edge_idx:
                colors[pos] = new_color
            else:
                colors[pos] = current_colors[e]
        if len(set(colors)) < 5:
            return False
    return True

def _greedy_coloring(order, seed=0):
    """
    Attempt to color the conflict graph using a greedy approach.
    The 'conflict graph' is defined implicitly: two edges conflict if they
    cannot share a color.
    We try to assign colors 0..K-1.
    """
    rng = random.Random(seed)
    colors = [-1] * NUM_EDGES
    
    # We will try to find the minimum K such that a valid coloring exists.
    # We start with a lower bound.
    # Lower bound: Max degree in the "structural" conflict graph?
    # Actually, let's just try K from a reasonable lower bound upwards.
    # Theoretical lower bound is 13.
    
    # To make this efficient, we use a "List Coloring" approach or simple greedy with backtracking?
    # Backtracking is too slow for 120 nodes usually, but we have a strong constraint.
    # Let's use a greedy heuristic that picks the most constrained edge first.
    
    # Precompute a static order based on degree in the "share vertex" graph?
    # Or just use the provided order.
    
    # We will implement a simple backtracking solver that is optimized with forward checking.
    # Since N=120 is small, and constraints are tight, this might work.
    
    # However, full backtracking might be too slow if the search space is huge.
    # Let's try a hybrid: Greedy assignment with random restarts and local repair.
    
    # Actually, the prompt suggests: "greedy set-cover heuristic... selecting the edge with the highest degree in the remaining conflict graph".
    # This implies we are building the coloring incrementally.
    
    # Let's try a simpler approach first: 
    # 1. Randomly shuffle edges.
    # 2. Greedily assign the smallest available color.
    # 3. If stuck, backtrack or restart.
    
    # Given the 120s limit, we can afford many fast restarts.
    
    best_colors = None
    best_k = 100
    
    # Heuristic: The number of colors must be at least 13.
    # Let's try to find a valid coloring with K colors.
    
    # We'll run a loop trying to find a valid coloring with increasing K.
    # But we want to MINIMIZE K.
    
    # Strategy:
    # 1. Try K=13 with many random restarts.
    # 2. If found, try to improve? No, 13 is the lower bound.
    # 3. If not found quickly, try K=14, etc.
    
    # To speed up, we use a "Most Constrained Variable" heuristic.
    # At each step, pick the uncolored edge that has the fewest available colors.
    
    # Available colors for an edge e given current partial coloring C:
    # A color c is unavailable if there exists a K4 containing e where the other 5 edges
    # already have 5 distinct colors and c is one of them? No.
    # A color c is unavailable if there exists a K4 containing e where the other 5 edges
    # have 4 distinct colors, and c is NOT one of them? No.
    # Condition: Every K4 must have >= 5 colors.
    # If we assign color c to edge e in K4, the other 5 edges must NOT have <= 4 colors that exclude c?
    # Actually, if the other 5 edges have 5 distinct colors, e can be any color.
    # If the other 5 edges have 4 distinct colors, e MUST be a new color (not in those 4).
    # If the other 5 edges have 3 distinct colors, e MUST be a new color (not in those 3)?
    # Wait. If other 5 have 3 colors, total distinct is max(3, 1+3) = 4 if e matches one, or 4 if e is new?
    # If e is new, total is 4. Still < 5.
    # So, if other 5 edges have < 4 distinct colors, it's IMPOSSIBLE to satisfy the K4 constraint
    # regardless of e's color?
    # Let's check:
    # Other 5 edges have colors {1, 2, 3}.
    # If e is 4, total {1,2,3,4} -> 4 colors. Fail.
    # If e is 1, total {1,2,3} -> 3 colors. Fail.
    # So, if any K4 has its other 5 edges with < 4 distinct colors, the problem is UNSATISFIABLE
    # for that partial assignment.
    
    # Therefore, a necessary condition for a valid coloring is that for every K4,
    # the 5 edges OTHER than any given edge e must have at least 4 distinct colors?
    # No, that's not right. The constraint is on the whole set of 6.
    # But it implies that we cannot get into a state where a K4's other 5 edges have < 4 colors.
    
    # Let's refine the "available colors" check:
    # For edge e and color c:
    # For each K4 containing e:
    #   Let S be the set of colors of the other 5 edges.
    #   If len(S) < 4: FAIL (impossible to reach 5 distinct colors with one more edge).
    #   If len(S) == 4: c MUST NOT be in S. (If c in S, total distinct is 4. If c not in S, total is 5).
    #   If len(S) >= 5: c can be anything.
    
    # This allows for very fast pruning!
    
    def get_available_colors(e_idx, current_colors):
        avail = set(range(best_k))
        for k4_idx in EDGE_K4S[e_idx]:
            k4_e = K4_EDGES[k4_idx]
            other_colors = []
            for e in k4_e:
                if e != e_idx:
                    if current_colors[e] != -1:
                        other_colors.append(current_colors[e])
                    else:
                        # If other edges are uncolored, we can't determine the constraint yet.
                        # We must be conservative or handle this in the search.
                        # For greedy forward checking, if other edges are uncolored,
                        # we assume they can be anything, so no restriction from this K4 yet?
                        # This makes greedy weak.
                        # Backtracking is better.
                        pass
            
            # Only apply restriction if all other edges are colored?
            # Or if we have enough info?
            # For a simple greedy, we only check if the constraint is VIOLATED by the current partial assignment.
            # If all 5 others are colored:
            if len(other_colors) == 5:
                s = set(other_colors)
                if len(s) < 4:
                    return set() # Unsat
                if len(s) == 4:
                    avail = avail - s
                # if len(s) >= 5: no restriction
        
        return avail

    # Given the complexity of dynamic constraints, let's stick to a robust local search on the full coloring.
    # The "conflict graph" idea is good for initialization.
    
    # Let's generate a random valid coloring first using a simple randomized greedy that checks validity at the end?
    # No, let's use the local search from the seed but improved.
    
    # Actually, let's try to construct a valid coloring using a specific algebraic structure first,
    # as the seed did, but then use the "conflict graph" view to prune the local search.
    
    # The seed's local search was slow because it checked all K4s.
    # My _check_k4_validity is faster.
    
    # Let's implement a fast local search.
    
    pass

def construct():
    deadline = _START + _DEADLINE - _SAFETY
    
    # 1. Generate initial valid colorings from algebraic families (fast)
    best_col = None
    best_count = 100
    
    def _build(m, shift):
        col = [-1] * NUM_EDGES
        for idx, (i, j) in enumerate(ALL_EDGES):
            col[idx] = (i + j + shift) % m
        return col
    
    def _valid_col(col):
        for k4_idx in range(NUM_K4S):
            cs = [col[e] for e in K4_EDGES[k4_idx]]
            if len(set(cs)) < 5:
                return False
        return True
    
    def _count_colours(col):
        return len(set(col))
    
    # Try algebraic families
    for m in range(13, 18):
        if time.time() > deadline: break
        for shift in range(m):
            if time.time() > deadline: break
            col = _build(m, shift)
            if _valid_col(col):
                c = _count_colours(col)
                if c < best_count:
                    best_count = c
                    best_col = col[:]
    
    if best_col is None:
        # Fallback: random valid coloring
        # This is unlikely to be needed if m=17 works, but just in case.
        # Use a simple repair heuristic on a random assignment.
        pass
        
    # 2. Local Search to minimize color count
    # We represent the coloring as a list of 120 ints.
    # We want to reduce the number of distinct values.
    
    rng = random.Random(42)
    
    # Precompute for speed:
    # For each edge, list of K4s it's in. (EDGE_K4S)
    # For each K4, list of edges. (K4_EDGES)
    
    current_col = best_col[:]
    current_used = best_count
    
    # We will try to change an edge's color to one that is already used by another edge,
    # effectively merging color classes.
    
    # To do this efficiently, we can pick a color c that is used, and try to change
    # some edges of a DIFFERENT color d to c.
    
    # Strategy:
    # 1. Identify all colors used.
    # 2. Pick a target color T.
    # 3. Pick an edge e with color C != T.
    # 4. Try to change e to T.
    # 5. If valid, accept. This reduces the number of colors if C is no longer used.
    
    # To make it faster, we only check the K4s containing e.
    
    # Optimization: 
    # We can maintain a count of how many edges have each color.
    # If we change e from C to T, and count[C] becomes 0, we reduced the palette size.
    
    def _local_search(col, palette_limit, time_limit=10.0):
        start_t = time.time()
        cur_col = col[:]
        cur_used = len(set(cur_col))
        best_local = cur_col[:]
        best_local_used = cur_used
        
        # Count colors
        color_counts = {}
        for c in cur_col:
            color_counts[c] = color_counts.get(c, 0) + 1
            
        edges_by_color = {}
        for i, c in enumerate(cur_col):
            if c not in edges_by_color:
                edges_by_color[c] = []
            edges_by_color[c].append(i)
            
        for _ in range(1000000): # Large loop, break on time
            if time.time() > start_t + time_limit:
                break
            if time.time() > deadline:
                break
                
            # Pick a random edge
            e_idx = rng.randint(0, NUM_EDGES - 1)
            old_c = cur_col[e_idx]
            
            # Pick a target color. 
            # Prefer colors that are already used to merge classes.
            # Randomly pick from used colors.
            used_colors = list(color_counts.keys())
            if not used_colors:
                break
            
            # Try a few random target colors
            # To encourage merging, pick a color that is NOT old_c
            candidates = [c for c in used_colors if c != old_c]
            if not candidates:
                # All edges same color? Impossible for valid K4s.
                continue
                
            # Shuffle candidates to try different merges
            rng.shuffle(candidates)
            
            improved = False
            for new_c in candidates[:5]: # Try top 5
                # Check validity
                if _check_k4_validity(e_idx, new_c, cur_col):
                    # Apply change
                    cur_col[e_idx] = new_c
                    
                    # Update counts
                    color_counts[old_c] -= 1
                    if color_counts[old_c] == 0:
                        del color_counts[old_c]
                        edges_by_color[old_c] = [] # Cleanup
                    else:
                        # Remove e_idx from edges_by_color[old_c]
                        # This is O(N) which is slow. 
                        # For simplicity, we don't maintain edges_by_color strictly, 
                        # we just use color_counts for the "did we reduce palette" check.
                        pass
                    
                    if new_c not in color_counts:
                        color_counts[new_c] = 0
                    color_counts[new_c] += 1
                    
                    # Update used count
                    new_used = len(color_counts)
                    if new_used < best_local_used:
                        best_local_used = new_used
                        best_local = cur_col[:]
                        improved = True
                    break # Accept first valid move
            
            # If no move was accepted, we might want to allow a neutral move (same used count)
            # to escape local minima.
            # But for now, let's stick to strict improvement or neutral.
            # Actually, if we don't accept neutral moves, we get stuck.
            # Let's add a 20% chance to accept a neutral valid move.
            
            if not improved:
                if rng.random() < 0.2:
                    # Try again with neutral acceptance
                    for new_c in candidates[:5]:
                        if _check_k4_validity(e_idx, new_c, cur_col):
                            cur_col[e_idx] = new_c
                            color_counts[old_c] -= 1
                            if color_counts[old_c] == 0:
                                del color_counts[old_c]
                            if new_c not in color_counts:
                                color_counts[new_c] = 0
                            color_counts[new_c] += 1
                            break
    
    # Run local search with decreasing palette limits?
    # No, just run it and see how low it goes.
    
    # We have ~100s left.
    # Run multiple restarts?
    # The initial algebraic coloring is likely good (13 or 14 colors).
    # Local search can reduce it if 14->13 is possible.
    
    # Let's run local search for the remaining time.
    # We'll do it in chunks to allow for restarts if it gets stuck.
    
    current_best = best_col
    current_best_count = best_count
    
    while time.time() < deadline:
        # Run local search for 5 seconds
        start_t = time.time()
        # We pass the current best as start
        # We want to allow it to explore.
        # If it finds something better, we update.
        
        # To avoid getting stuck in the same local minimum,
        # we can slightly perturb the initial coloring before each run.
        # But for the first run, use the algebraic one.
        
        # Perturb: pick 5 random edges and change their color to a random valid one?
        # Or just run LS on the best found so far.
        
        ls_col = current_best[:]
        
        # Run LS
        # We need a version of LS that returns the best found.
        
        # Let's implement a simple LS loop here for clarity.
        
        cur_col = ls_col[:]
        cur_used = len(set(cur_col))
        local_best = cur_col[:]
        local_best_used = cur_used
        
        color_counts = {}
        for c in cur_col:
            color_counts[c] = color_counts.get(c, 0) + 1
            
        while time.time() < start_t + 5.0 and time.time() < deadline:
            e_idx = rng.randint(0, NUM_EDGES - 1)
            old_c = cur_col[e_idx]
            
            used_colors = list(color_counts.keys())
            candidates = [c for c in used_colors if c != old_c]
            if not candidates:
                continue
            
            rng.shuffle(candidates)
            
            accepted = False
            for new_c in candidates[:3]:
                if _check_k4_validity(e_idx, new_c, cur_col):
                    cur_col[e_idx] = new_c
                    color_counts[old_c] -= 1
                    if color_counts[old_c] == 0:
                        del color_counts[old_c]
                    if new_c not in color_counts:
                        color_counts[new_c] = 0
                    color_counts[new_c] += 1
                    
                    new_used = len(color_counts)
                    if new_used < local_best_used:
                        local_best_used = new_used
                        local_best = cur_col[:]
                    accepted = True
                    break
            
            if not accepted and rng.random() < 0.1:
                # Neutral move
                for new_c in candidates[:3]:
                    if _check_k4_validity(e_idx, new_c, cur_col):
                        cur_col[e_idx] = new_c
                        color_counts[old_c] -= 1
                        if color_counts[old_c] == 0:
                            del color_counts[old_c]
                        if new_c not in color_counts:
                            color_counts[new_c] = 0
                        color_counts[new_c] += 1
                        break
        
        if local_best_used < current_best_count:
            current_best_count = local_best_used
            current_best = local_best[:]
            
        # If we hit 13, we can stop early?
        # The lower bound is 13.
        if current_best_count == 13:
            break
            
    out = current_best
    # Ensure colours are within 0..119
    # The algebraic construction uses small numbers.
    # Local search only reassigns to existing colors, so max color is bounded by initial max.
    # Initial max < 18. So we are safe.
    
    return out

if __name__ == "__main__":
    r = construct()
    print("colours used:", len(set(r)))
