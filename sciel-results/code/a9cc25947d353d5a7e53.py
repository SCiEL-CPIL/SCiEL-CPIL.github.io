import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # Precompute constraints for 4-APs ending at x
    # For each position x (0-indexed), store list of (i1, i2, i3) such that
    # colors[i1], colors[i2], colors[i3], colors[x] form a 4-AP in positions.
    # Positions are 1-indexed in the AP definition: a, a+d, a+2d, a+3d = x+1.
    # So a = x+1 - 3d.
    # i1 = a-1, i2 = a+d-1, i3 = a+2d-1.
    
    constraints = [[] for _ in range(128)]
    for x in range(128):
        # x is 0-indexed, so position is x+1
        pos = x + 1
        for d in range(1, (pos - 1) // 3 + 1):
            a = pos - 3 * d
            if a < 1:
                continue
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            constraints[x].append((i1, i2, i3))

    def check_color(x, c, colors):
        """Check if assigning color c to position x violates any 4-AP constraint."""
        for i1, i2, i3 in constraints[x]:
            c1, c2, c3 = colors[i1], colors[i2], colors[i3]
            # We need len({c1, c2, c3, c}) >= 3
            # This is equivalent to: not (c1==c2==c3 and c==c1) and not (c1==c2==c3) and not (c1==c2 and c3==c) ... 
            # Simpler: count distinct values.
            # If c1==c2==c3, then set is {c1, c}. Size is 1 or 2. Always invalid.
            if c1 == c2 and c2 == c3:
                return False
            # If c1==c2 and c3 != c1:
            #   Set is {c1, c3, c}. Size is 2 if c==c1 or c==c3. Size is 3 otherwise.
            #   So invalid if c==c1 or c==c3.
            if c1 == c2 and c3 != c1:
                if c == c1 or c == c3:
                    return False
            # If c1 != c2 and c2 == c3:
            #   Set is {c1, c2, c}. Size is 2 if c==c1 or c==c2.
            elif c2 == c3 and c1 != c2:
                if c == c1 or c == c2:
                    return False
            # If c1 == c3 and c2 != c1:
            #   Set is {c1, c2, c}. Size is 2 if c==c1 or c==c2.
            elif c1 == c3 and c2 != c1:
                if c == c1 or c == c2:
                    return False
            # If c1, c2, c3 are all distinct:
            #   Set is {c1, c2, c3, c}. Size is 3 if c is any of them. Size is 4 if new.
            #   Always valid since size >= 3.
            else:
                # c1, c2, c3 distinct.
                pass
        return True

    def try_construct(max_c, seed_val=0):
        random.seed(seed_val)
        colors = [None] * 128
        used = set()
        for x in range(128):
            if time.time() > deadline:
                return None
            
            valid_colors = []
            for c in range(max_c):
                if check_color(x, c, colors):
                    valid_colors.append(c)
            
            if not valid_colors:
                return None
            
            # Heuristic: prefer used colors to minimize count
            # But sometimes we need to pick a new color to avoid future dead ends?
            # The task says "Minimize the number of labels actually used".
            # Greedy "prefer used" is good for minimizing count, but might lead to dead ends.
            # If we hit a dead end, we return None and try higher max_c.
            
            # To be safe, we can randomize the choice among valid colors,
            # but bias towards used colors.
            
            used_valid = [c for c in valid_colors if c in used]
            new_valid = [c for c in valid_colors if c not in used]
            
            if used_valid:
                # Pick randomly from used valid colors? Or smallest?
                # Smallest is deterministic. Random might help escape local minima.
                # Let's pick randomly from used_valid to explore.
                chosen = random.choice(used_valid)
            else:
                # Must pick a new color. Pick smallest to keep labels low?
                # Or random?
                # Since we are minimizing count, we are forced to use a new one.
                # The label value doesn't matter for the count, only distinctness.
                # But "Minimize the number of labels actually used" is the objective.
                # The labels themselves can be anything in 0..127.
                # So picking the smallest new one is fine.
                chosen = new_valid[0]
            
            colors[x] = chosen
            used.add(chosen)
            
        return colors

    # Try 3 colors with several random seeds
    for seed in range(10):
        res = try_construct(3, seed)
        if res:
            return res
            
    # Try 4 colors
    for seed in range(10):
        res = try_construct(4, seed)
        if res:
            return res
            
    # Try 5 colors
    for seed in range(10):
        res = try_construct(5, seed)
        if res:
            return res
            
    # Fallback: Safety net (seed logic)
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
