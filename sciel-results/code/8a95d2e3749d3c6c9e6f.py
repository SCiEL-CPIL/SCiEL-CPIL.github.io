"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def _count_distinct_colors_in_progression(colors, a, d):
    """Count distinct colors in the four-term progression a, a+d, a+2d, a+3d."""
    # positions are 1-indexed in the problem, but our list is 0-indexed
    # so position x corresponds to colors[x-1]
    c0 = colors[a - 1]
    c1 = colors[a + d - 1]
    c2 = colors[a + 2 * d - 1]
    c3 = colors[a + 3 * d - 1]
    return len({c0, c1, c2, c3})


def _is_valid(colors):
    """Check if every nonconstant four-term progression uses at least 3 distinct colors."""
    n = len(colors)
    for a in range(1, n - 2):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            if _count_distinct_colors_in_progression(colors, a, d) < 3:
                return False
    return True


def _num_colors_used(colors):
    """Count the number of distinct color labels used."""
    return len(set(colors))


def _greedy_construct():
    """
    Greedy construction: assign each position the smallest color label
    that satisfies all constraints with previously assigned positions.
    """
    n = 128
    colors = [0] * n
    colors[0] = 0  # position 1
    
    for x in range(1, n):
        # position x+1 (1-indexed)
        pos = x + 1
        found = False
        for color in range(128):
            valid = True
            # Check all four-term progressions ending at pos
            # a + 3d = pos => a = pos - 3d, need a >= 1 and a + d <= pos (trivially true)
            # d ranges from 1 to (pos-1)//3
            for d in range(1, (pos) // 3 + 1):
                a = pos - 3 * d
                if a < 1:
                    continue
                # Progression: a, a+d, a+2d, a+3d=pos
                # All these positions are < pos, so already assigned
                c0 = colors[a - 1]
                c1 = colors[a + d - 1]
                c2 = colors[a + 2 * d - 1]
                # c3 = color (the new one)
                distinct = len({c0, c1, c2, color})
                if distinct < 3:
                    valid = False
                    break
            # Also check progressions where pos is not the last term
            # pos = a, a+d, a+2d
            # Case: pos = a (first term): a=pos, need a+3d <= n
            # But those progressions don't involve future unassigned positions yet in a way that constrains this choice
            # Actually, we only need to check constraints that involve the current position and already-assigned ones.
            # A four-term progression involving pos and 3 already-assigned positions:
            # pos can be at index 0, 1, 2, or 3 in the progression.
            # We already checked pos at index 3 (last term).
            # Check pos at index 2: a, a+d, pos, a+3d => a+2d=pos, so a=pos-2d, and a+3d = pos+d
            # But pos+d > pos, so a+3d is not yet assigned. So this constraint involves a future position.
            # We don't need to check it now because when we assign pos+d, we'll check the full progression.
            # Check pos at index 1: a, pos, a+2d, a+3d => a+d=pos, so a=pos-d, a+2d=pos+d, a+3d=pos+2d
            # Both a+2d and a+3d are > pos, so not yet assigned.
            # Check pos at index 0: pos, pos+d, pos+2d, pos+3d => all others > pos, not yet assigned.
            # So we only need to check progressions where pos is the LAST term.
            if valid:
                colors[x] = color
                found = True
                break
        if not found:
            # Should not happen if we have enough colors
            colors[x] = 0
    
    return colors


def _local_search(colors, max_iters=50000, timeout=60):
    """
    Local search: start from greedy solution, try to reduce number of colors used.
    """
    start_time = time.time()
    n = len(colors)
    
    # First, relabel to use consecutive colors 0..k-1
    unique_colors = sorted(set(colors))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    colors = [color_map[c] for c in colors]
    k = len(unique_colors)
    
    best_colors = colors[:]
    best_k = k
    
    for iteration in range(max_iters):
        if time.time() - start_time > timeout - 5:
            break
        
        # Try to recolor a random position with a different color to see if we can reduce k
        # Or try swapping colors to merge two color classes
        
        # Strategy: pick a random position, try assigning it a different color
        # such that validity is maintained, and see if we can eliminate a color class
        
        pos = random.randint(0, n - 1)
        old_color = colors[pos]
        
        # Try all other colors
        for new_color in range(k):
            if new_color == old_color:
                continue
            
            colors[pos] = new_color
            
            # Check validity for all progressions involving pos
            valid = True
            # pos is at index 0, 1, 2, or 3 in a four-term progression
            # We need to check all progressions that include position pos+1 (1-indexed)
            
            p = pos + 1  # 1-indexed position
            
            # pos at index 3 (last): a + 3d = p
            for d in range(1, (p) // 3 + 1):
                a = p - 3 * d
                if a < 1:
                    continue
                if _count_distinct_colors_in_progression(colors, a, d) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # pos at index 2: a + 2d = p => a = p - 2d, a+3d = p+d
            for d in range(1, (p) // 2 + 1):
                a = p - 2 * d
                if a < 1:
                    continue
                a3 = p + d
                if a3 > n:
                    continue
                # Check progression a, a+d, a+2d, a+3d
                c0 = colors[a - 1]
                c1 = colors[a + d - 1]
                c2 = colors[a + 2 * d - 1]
                c3 = colors[a + 3 * d - 1]
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # pos at index 1: a + d = p => a = p - d, a+2d = p+d, a+3d = p+2d
            for d in range(1, (p) // 1 + 1):
                a = p - d
                if a < 1:
                    continue
                a3 = p + 2 * d
                if a3 > n:
                    continue
                c0 = colors[a - 1]
                c1 = colors[a + d - 1]
                c2 = colors[a + 2 * d - 1]
                c3 = colors[a + 3 * d - 1]
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # pos at index 0: a = p, a+d = p+d, a+2d = p+2d, a+3d = p+3d
            for d in range(1, (n - p + 1) // 3 + 1):
                a3 = p + 3 * d
                if a3 > n:
                    continue
                c0 = colors[a - 1] if False else colors[p - 1]  # a = p
                c1 = colors[p + d - 1]
                c2 = colors[p + 2 * d - 1]
                c3 = colors[p + 3 * d - 1]
                if len({c0, c1, c2, c3}) < 3:
                    valid = False
                    break
            
            if not valid:
                continue
            
            # Valid move. Check if this reduces the number of colors used.
            new_colors = set(colors)
            if len(new_colors) < k:
                # Relabel to consecutive
                unique = sorted(new_colors)
                cmap = {c: i for i, c in enumerate(unique)}
                colors = [cmap[c] for c in colors]
                k = len(unique)
                best_colors = colors[:]
                best_k = k
            # Keep the change even if not reducing k (for exploration)
            break
        
        # Occasionally try a more aggressive move: swap all occurrences of one color with another
        if iteration % 10 == 0 and k > 3:
            c1 = random.randint(0, k - 1)
            c2 = random.randint(0, k - 1)
            if c1 != c2:
                # Swap c1 and c2
                for i in range(n):
                    if colors[i] == c1:
                        colors[i] = c2
                    elif colors[i] == c2:
                        colors[i] = c1
                # Check validity
                if _is_valid(colors):
                    new_colors = set(colors)
                    if len(new_colors) < k:
                        unique = sorted(new_colors)
                        cmap = {c: i for i, c in enumerate(unique)}
                        colors = [cmap[c] for c in colors]
                        k = len(unique)
                        best_colors = colors[:]
                        best_k = k
    
    return best_colors, best_k


def construct():
    """
    Construct a coloring of positions 1..128 with at least 3 distinct colors
    in every nonconstant four-term progression, minimizing the number of colors used.
    """
    start_time = time.time()
    deadline = 80  # safety margin before 90s kill
    
    # Start with greedy construction
    colors = _greedy_construct()
    k = _num_colors_used(colors)
    best_colors = colors[:]
    best_k = k
    
    # Try local search to reduce number of colors
    if time.time() - start_time < deadline - 10:
        improved_colors, improved_k = _local_search(colors, max_iters=100000, timeout=deadline - time.time() - 5)
        if improved_k < best_k:
            best_colors = improved_colors
            best_k = improved_k
    
    # Return the best solution found
    return best_colors

# EVOLVE-BLOCK-END
