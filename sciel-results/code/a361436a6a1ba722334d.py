import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill

    n = 128

    # Precompute 4-term APs
    # For each position i, store all 4-term APs (a, a+d, a+2d, a+3d) where i is one of the positions
    # Actually, let's precompute all 4-term APs as tuples
    all_4aps = []
    for a in range(n):
        for d in range(1, n):
            if a + 3 * d < n:
                all_4aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Precompute for each position, which APs it participates in
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(all_4aps):
        for pos in ap:
            pos_aps[pos].append(idx)

    # For each AP, precompute which positions are which
    # ap_positions[ap_idx] = (p0, p1, p2, p3)
    ap_positions = all_4aps  # already tuples

    def is_valid(colors):
        """Check if the coloring is valid for all 4-term APs."""
        for ap in all_4aps:
            vals = {colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}
            if len(vals) < 3:
                return False
        return True

    def count_used(colors):
        """Count distinct colors used."""
        return len(set(colors))

    def greedy_assign(colors, k, pos_order=None, lookahead=False):
        """
        Greedy assignment with optional lookahead.
        colors: initial list (None for unassigned)
        k: number of colors to try
        pos_order: order to assign positions
        lookahead: if True, use 3-term AP lookahead
        """
        if pos_order is None:
            pos_order = list(range(n))
        
        assigned = [c is not None for c in colors]
        
        for i in pos_order:
            if time.time() > deadline:
                return None
            
            if colors[i] is not None:
                continue
            
            # Find valid colors for position i
            valid_colors = []
            for c in range(k):
                # Check all 4-term APs containing position i where other positions are assigned
                ok = True
                for ap_idx in pos_aps[i]:
                    ap = ap_positions[ap_idx]
                    # Check if all other positions in this AP are assigned
                    if all(assigned[pos] for pos in ap if pos != i):
                        vals = {colors[pos] for pos in ap if pos != i}
                        vals.add(c)
                        if len(vals) < 3:
                            ok = False
                            break
                
                if not ok:
                    continue
                
                # Lookahead: check 3-term APs where i is the 3rd term
                if lookahead:
                    # For AP (a, a+d, a+2d, a+3d) where a+2d = i
                    # a = i - 2d, a+d = i - d, a+3d = i + d
                    for d in range(1, n):
                        a = i - 2 * d
                        mid = i - d
                        last = i + d
                        if a >= 0 and mid >= 0 and last < n:
                            # Check if a, mid are assigned and last is not
                            if assigned[a] and assigned[mid] and not assigned[last]:
                                c0 = colors[a]
                                c1 = colors[mid]
                                # The 4th term at position last will need to form a valid 4-term AP
                                # We need that c is different from at least one of c0 or c1
                                if c == c0 and c == c1:
                                    ok = False
                                    break
                
                if ok:
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Pick the smallest valid color
            colors[i] = valid_colors[0]
            assigned[i] = True
        
        # Final verification
        if is_valid(colors):
            return colors
        return None

    # Strategy 1: Simple greedy from left to right with increasing colors
    for k in [3, 4, 5, 6, 8]:
        colors = [None] * n
        result = greedy_assign(colors, k, pos_order=list(range(n)), lookahead=False)
        if result is not None:
            return result

    # Strategy 2: Greedy with lookahead
    for k in [4, 5, 6, 8]:
        colors = [None] * n
        result = greedy_assign(colors, k, pos_order=list(range(n)), lookahead=True)
        if result is not None:
            return result

    # Strategy 3: Randomized greedy with different position orders
    rng = random.Random(42)
    
    # Try different position orderings
    orders = [
        list(range(n)),  # left to right
        list(range(n-1, -1, -1)),  # right to left
        list(range(0, n, 2)) + list(range(1, n, 2)),  # even then odd
        list(range(1, n, 2)) + list(range(0, n, 2)),  # odd then even
    ]
    
    for k in [4, 5, 6, 8]:
        for order in orders:
            colors = [None] * n
            result = greedy_assign(colors, k, pos_order=order, lookahead=True)
            if result is not None:
                return result

    # Strategy 4: Backtracking with limited depth
    def backtrack_limited(colors, k, depth_limit=50):
        """Backtracking search with depth limit."""
        if time.time() > deadline:
            return None
        
        # Find next unassigned position
        next_pos = None
        for i in range(n):
            if colors[i] is None:
                next_pos = i
                break
        
        if next_pos is None:
            # All assigned, verify
            if is_valid(colors):
                return colors.copy()
            return None
        
        # Find valid colors for next_pos
        for c in range(k):
            # Check if c is valid at next_pos
            ok = True
            for ap_idx in pos_aps[next_pos]:
                ap = ap_positions[ap_idx]
                if all(colors[pos] is not None for pos in ap if pos != next_pos):
                    vals = {colors[pos] for pos in ap if pos != next_pos}
                    vals.add(c)
                    if len(vals) < 3:
                        ok = False
                        break
            
            if not ok:
                continue
            
            colors[next_pos] = c
            
            # Check if this leads to immediate contradiction
            valid = True
            for ap_idx in pos_aps[next_pos]:
                ap = ap_positions[ap_idx]
                if all(colors[pos] is not None for pos in ap):
                    vals = {colors[pos] for pos in ap}
                    if len(vals) < 3:
                        valid = False
                        break
            
            if valid:
                result = backtrack_limited(colors, k, depth_limit - 1)
                if result is not None:
                    return result
            
            colors[next_pos] = None
        
        return None

    for k in [4, 5, 6]:
        colors = [None] * n
        result = backtrack_limited(colors, k)
        if result is not None:
            return result

    # Strategy 5: Random restarts with greedy
    for attempt in range(20):
        if time.time() > deadline:
            break
        for k in [5, 6, 8]:
            colors = [None] * n
            # Randomize position order slightly
            order = list(range(n))
            rng.shuffle(order)
            # But keep some structure: try to assign in blocks
            # Actually, just use the shuffled order
            result = greedy_assign(colors, k, pos_order=order, lookahead=True)
            if result is not None:
                return result

    # Fallback: return all distinct colors
    return list(range(n))

if __name__ == "__main__":
    res = construct()
    print(f"Used {len(set(res))} colors")
    print(res[:20])
