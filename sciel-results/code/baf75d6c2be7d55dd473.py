import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map pair (i,j) with 0<=i<j<16 to 0..119 in lexicographic order."""
    # Lexicographic order of pairs (i,j): for each i, j goes from i+1 to 15
    # Count of edges before vertex i: sum_{k=0}^{i-1} (15-k) = 15*i - i*(i-1)//2
    count_before_i = 15 * i - i * (i - 1) // 2
    return count_before_i + (j - i - 1)

def _build_color_list(colors_2d):
    """Build the 120-element list from a 2D array."""
    result = [0] * 120
    for i in range(16):
        base = 15 * i - i * (i - 1) // 2
        for j in range(i + 1, 16):
            result[base + (j - i - 1)] = colors_2d[i][j]
    return result

def _check_k4(color_list):
    """Check if every K4 has at least 5 distinct colours. Return True if valid."""
    for quad in combinations(range(16), 4):
        a, b, c, d = quad
        cols = set()
        cols.add(color_list[_edge_index(a, b)])
        cols.add(color_list[_edge_index(a, c)])
        cols.add(color_list[_edge_index(a, d)])
        cols.add(color_list[_edge_index(b, c)])
        cols.add(color_list[_edge_index(b, d)])
        cols.add(color_list[_edge_index(c, d)])
        if len(cols) < 5:
            return False
    return True

def _count_colours(color_list):
    return len(set(color_list))

def _initial_construct():
    """Seed: (i+j) mod 17 gives 17 colours."""
    colors = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin: 5s before 150s kill
    
    # Start with the known valid 17-colour solution
    best_colors = _initial_construct()
    best_num_colors = _count_colours(best_colors)  # should be 17
    
    # Precompute all K4s and their edge indices for fast checking
    all_k4s = list(combinations(range(16), 4))
    k4_edges = []
    for quad in all_k4s:
        a, b, c, d = quad
        k4_edges.append([
            _edge_index(a, b), _edge_index(a, c), _edge_index(a, d),
            _edge_index(b, c), _edge_index(b, d), _edge_index(c, d)
        ])
    
    # Greedy colour reduction: try to reduce number of colours
    # Strategy: iteratively pick a colour, try to merge it into another colour
    # if doing so preserves the constraint for all K4s
    
    def try_merge(color_a, color_b, colors):
        """Try to merge colour color_a into color_b. Return new list or None."""
        new_colors = list(colors)
        for idx in range(120):
            if new_colors[idx] == color_a:
                new_colors[idx] = color_b
        # Check validity
        for edges in k4_edges:
            cols = set(new_colors[e] for e in edges)
            if len(cols) < 5:
                return None
        return new_colors
    
    # Greedy: repeatedly try to merge pairs of colours
    def greedy_reduce(colors, time_limit):
        """Try to reduce colours by merging pairs."""
        colors = list(colors)
        colors_used = sorted(set(colors))
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for i in range(len(colors_used)):
                for j in range(i + 1, len(colors_used)):
                    ca, cb = colors_used[i], colors_used[j]
                    result = try_merge(ca, cb, colors)
                    if result is not None:
                        colors = result
                        colors_used = sorted(set(colors))
                        improved = True
                        break
                if improved:
                    break
        return colors
    
    # Also try local search: randomly change colours to reduce total used
    def local_search(colors, time_limit):
        """Randomly perturb and try to reduce colour count."""
        colors = list(colors)
        best_local = colors
        best_local_count = _count_colours(colors)
        
        current = list(colors)
        current_count = best_local_count
        colors_in_use = set(current)
        
        while time.time() < time_limit:
            # Pick a random edge and change its colour to a random other colour
            idx = random.randint(0, 119)
            old_color = current[idx]
            # Try a colour not currently in use, or one in use
            if random.random() < 0.5:
                # Try to use a colour already in use (to potentially reduce count)
                other_colors = [c for c in colors_in_use if c != old_color]
                if other_colors:
                    new_color = random.choice(other_colors)
                else:
                    new_color = old_color
            else:
                # Random colour in range
                new_color = random.randint(0, max(colors_in_use) if colors_in_use else 10)
            
            if new_color == old_color:
                continue
            
            current[idx] = new_color
            
            # Quick check: only K4s containing this edge matter
            # But for simplicity, check all (1820 K4s)
            valid = True
            for edges in k4_edges:
                cols = set(current[e] for e in edges)
                if len(cols) < 5:
                    valid = False
                    break
            
            if valid:
                new_count = _count_colours(current)
                if new_count < best_local_count:
                    best_local = list(current)
                    best_local_count = new_count
                    colors_in_use = set(current)
            else:
                # Revert
                current[idx] = old_color
        
        return best_local, best_local_count
    
    # Phase 1: Greedy reduction from seed
    if time.time() < deadline:
        reduced = greedy_reduce(best_colors, deadline - 5)
        if _count_colours(reduced) < best_num_colors:
            best_colors = reduced
            best_num_colors = _count_colours(reduced)
    
    # Phase 2: Local search with multiple restarts
    if time.time() < deadline:
        remaining = deadline - time.time()
        num_restarts = max(1, int(remaining / 3))  # ~3s per restart
        for restart in range(num_restarts):
            if time.time() >= deadline - 2:
                break
            # Start from best found or seed with perturbation
            if restart == 0:
                start = list(best_colors)
            else:
                start = list(_initial_construct())
                # Perturb
                for _ in range(20):
                    idx = random.randint(0, 119)
                    start[idx] = random.randint(0, 16)
            
            time_left = deadline - time.time() - 1
            if time_left < 0.5:
                break
            
            found, found_count = local_search(start, time.time() + time_left)
            if found_count < best_num_colors:
                best_colors = found
                best_num_colors = found_count
    
    return best_colors

# EVOLVE-BLOCK-END
