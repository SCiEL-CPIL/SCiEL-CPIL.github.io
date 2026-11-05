"""Colorings with three colors in every four-term progression."""
import time
import random
from functools import lru_cache

# EVOLVE-BLOCK-START

def _all_4taps(n):
    """Generate all (a, d) such that a, a+d, a+2d, a+3d are in [0, n)."""
    taps = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            taps.append((a, a + d, a + 2 * d, a + 3 * d))
    return taps

def _check_validity(colors):
    """Check if every nonconstant 4-term progression uses at least 3 distinct colors."""
    for a, b, c, d in _all_4taps(len(colors)):
        if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
            return False
    return True

def _greedy_construct(n, max_colors, rng):
    """
    Greedy coloring: assign colors 0..max_colors-1 one by one.
    Each position gets the smallest color that doesn't violate the constraint
    given previously assigned positions.
    Returns (colors, num_colors_used) or (None, None) if stuck.
    """
    colors = [-1] * n
    # Precompute all 4-term progressions
    taps = _all_4taps(n)
    
    for pos in range(n):
        # Find all 4-term progressions that include this position
        # and check which colors are forbidden
        forbidden = set()
        can_assign = True
        
        for (a, b, c, d) in taps:
            pts = [a, b, c, d]
            if pos not in pts:
                continue
            idx = pts.index(pos)
            others = [p for j, p in enumerate(pts) if j != idx]
            other_colors = [colors[p] for p in others]
            if -1 in other_colors:
                # Some positions not yet assigned; we can't fully check now.
                # But in greedy left-to-right, if pos is the rightmost, all others are assigned.
                # If pos is not the rightmost, some future positions are unassigned.
                # We can only check the constraint partially.
                # For now, skip this check if any other is unassigned.
                continue
            
            # All other three are assigned
            # The constraint: len({color[pos]} ∪ {other_colors}) >= 3
            # i.e., we need at least one color in {color[pos]} ∪ other_colors that is new,
            # or equivalently, the set must have >= 3 elements.
            other_set = set(other_colors)
            
            # If other_set already has >= 3 distinct colors, any color works for pos.
            if len(other_set) >= 3:
                continue
            
            # If other_set has 2 distinct colors, say {x, y}, then:
            #   We need len({color[pos], x, y}) >= 3, so color[pos] must be != x and != y.
            if len(other_set) == 2:
                forbidden.update(other_set)
            
            # If other_set has 1 distinct color, say {x}, then:
            #   We need len({color[pos], x}) >= 3, which is impossible since max is 2.
            #   Wait, no: len({color[pos], x}) where color[pos] is one value.
            #   If color[pos] == x: set is {x}, size 1.
            #   If color[pos] != x: set is {x, color[pos]}, size 2.
            #   We need size >= 3. So this progression CANNOT be satisfied!
            #   Unless... wait, the progression has 4 positions. If three of them all have color x,
            #   and the fourth has color y != x, the set is {x, y}, size 2. Still < 3.
            #   So if three positions in a 4-term progression all have the same color, it's invalid
            #   no matter what the fourth color is (since max distinct is 2).
            if len(other_set) == 1:
                # This means three assigned positions all have the same color.
                # No matter what we assign to pos, we'll have at most 2 distinct colors.
                can_assign = False
                break
        
        if not can_assign:
            return None, None
        
        # Find the smallest color in 0..max_colors-1 not in forbidden
        for c in range(max_colors):
            if c not in forbidden:
                colors[pos] = c
                break
        else:
            return None, None
    
    # Final validation
    if not _check_validity(colors):
        return None, None
    
    used = set(colors)
    return colors, len(used)

def _backtracking_search(n, max_colors, rng, time_limit):
    """
    Backtracking search with heuristics. Tries to find a valid coloring
    using at most max_colors colors.
    """
    start_time = time.time()
    colors = [-1] * n
    taps = _all_4taps(n)
    
    # For each position, precompute which taps it participates in
    pos_taps = [[] for _ in range(n)]
    for i, (a, b, c, d) in enumerate(taps):
        for p in (a, b, c, d):
            pos_taps[p].append(i)
    
    def is_partial_valid(pos, color):
        """Check if assigning color to pos is consistent with all taps where pos is involved
        and all other positions in those taps are already assigned."""
        for ti in pos_taps[pos]:
            a, b, c, d = taps[ti]
            pts = [a, b, c, d]
            others = [p for p in pts if p != pos]
            other_colors = [colors[p] for p in others]
            if -1 in other_colors:
                continue  # Can't fully check yet
            other_set = set(other_colors)
            full_set = other_set | {color}
            if len(full_set) < 3:
                return False
        return True
    
    def backtrack(pos):
        if time.time() - start_time > time_limit:
            return False
        if pos == n:
            return True
        
        # Try colors in random order for diversity
        color_order = list(range(max_colors))
        rng.shuffle(color_order)
        
        for c in color_order:
            if is_partial_valid(pos, c):
                colors[pos] = c
                if backtrack(pos + 1):
                    return True
                colors[pos] = -1
        return False
    
    if backtrack(0):
        used = set(colors)
        return colors, len(used)
    return None, None

def _local_search(n, start_colors, max_iters=5000, time_limit=30.0, rng=None):
    """
    Local search: start from a partial or full coloring, try to fix violations
    by reassigning colors.
    """
    if rng is None:
        rng = random.Random()
    
    start_time = time.time()
    colors = list(start_colors)
    if -1 in colors:
        # Fill unassigned positions greedily
        taps = _all_4taps(n)
        for pos in range(n):
            if colors[pos] == -1:
                forbidden = set()
                for (a, b, c, d) in taps:
                    if pos not in (a, b, c, d):
                        continue
                    idx = (a, b, c, d).index(pos)
                    others = [(a, b, c, d)[j] for j in range(4) if j != idx]
                    other_colors = [colors[p] for p in others]
                    if -1 in other_colors:
                        continue
                    other_set = set(other_colors)
                    if len(other_set) == 2:
                        forbidden.update(other_set)
                    elif len(other_set) == 1:
                        # Stuck
                        return None
                for c in range(128):
                    if c not in forbidden:
                        colors[pos] = c
                        break
                else:
                    return None
    
    # Compute initial violations
    def count_violations():
        viol = 0
        for (a, b, c, d) in _all_4taps(n):
            if len({colors[a], colors[b], colors[c], colors[d]}) < 3:
                viol += 1
        return viol
    
    current_viol = count_violations()
    best_colors = list(colors)
    best_viol = current_viol
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if current_viol == 0:
            break
        
        # Pick a random violation and try to fix it
        # For efficiency, just pick a random position and try a random color
        pos = rng.randint(0, n - 1)
        old_color = colors[pos]
        new_color = rng.randint(0, 127)
        if new_color == old_color:
            continue
        
        colors[pos] = new_color
        new_viol = count_violations()
        
        if new_viol < current_viol:
            current_viol = new_viol
            if new_viol < best_viol:
                best_viol = new_viol
                best_colors = list(colors)
        else:
            colors[pos] = old_color
    
    if best_viol == 0:
        used = set(best_colors)
        return best_colors, len(used)
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s
    
    n = 128
    rng = random.Random(42)
    
    # Strategy 1: Try greedy with increasing number of colors
    for max_colors in [3, 4, 5, 6, 7, 8]:
        if time.time() > deadline:
            break
        result, used = _greedy_construct(n, max_colors, rng)
        if result is not None:
            return result
    
    # Strategy 2: Backtracking search
    for max_colors in [4, 5, 6]:
        if time.time() > deadline:
            break
        time_limit = min(10.0, deadline - time.time() - 1.0)
        if time_limit < 1.0:
            break
        result, used = _backtracking_search(n, max_colors, rng, time_limit)
        if result is not None:
            return result
    
    # Strategy 3: Local search from random initial colorings
    for attempt in range(50):
        if time.time() > deadline:
            break
        time_limit = min(5.0, deadline - time.time() - 1.0)
        if time_limit < 0.5:
            break
        init = [rng.randint(0, 7) for _ in range(n)]
        result, used = _local_search(n, init, max_iters=3000, time_limit=time_limit, rng=rng)
        if result is not None:
            return result
    
    # Fallback: return the seed program's output
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x - 1 - j * d] for j in (1, 2, 3)}) >= 3 
                    for d in range(1, (x - 1) // 3 + 1))
                and all(any(colors[x - 1 - j * d] != color for j in (1, 2)) 
                        for d in range(1, (x - 1) // 2 + 1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
