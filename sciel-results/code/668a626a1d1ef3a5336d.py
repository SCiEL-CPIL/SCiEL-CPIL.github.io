import random

def construct():
    n = 128
    
    def is_valid_assignment(colors, i, c):
        """Check if assigning color c to position i is valid given current colors."""
        # Check all 4-term APs containing i where all other positions are assigned
        for d in range(1, (i // 3) + 1):
            # i is 4th term
            a = i - 3 * d
            if a >= 0:
                if colors[a] != -1 and colors[a+d] != -1 and colors[a+2*d] != -1:
                    if len({colors[a], colors[a+d], colors[a+2*d], c}) < 3:
                        return False
        # i is 3rd term: a, a+d, i, a+3d
        for d in range(1, n):
            a = i - 2 * d
            b = i + d
            if a >= 0 and b < n:
                if colors[a] != -1 and colors[b] != -1:
                    if len({colors[a], colors[a+d], colors[b], c}) < 3:
                        return False
        # i is 2nd term: a, i, a+2d, a+3d
        for d in range(1, n):
            a = i - d
            b = i + d
            c2 = i + 2 * d
            if a >= 0 and b < n and c2 < n:
                if colors[a] != -1 and colors[b] != -1 and colors[c2] != -1:
                    if len({colors[a], colors[b], colors[c2], c}) < 3:
                        return False
        # i is 1st term: i, i+d, i+2d, i+3d
        for d in range(1, n):
            b = i + d
            c2 = i + 2 * d
            d2 = i + 3 * d
            if b < n and c2 < n and d2 < n:
                if colors[b] != -1 and colors[c2] != -1 and colors[d2] != -1:
                    if len({colors[b], colors[c2], colors[d2], c}) < 3:
                        return False
        return True
    
    best_colors = None
    best_num_colors = n + 1
    
    for attempt in range(1000):
        colors = [-1] * n
        success = True
        for i in range(n):
            # Try colors in random order
            candidates = list(range(n))
            random.shuffle(candidates)
            placed = False
            for c in candidates:
                if is_valid_assignment(colors, i, c):
                    colors[i] = c
                    placed = True
                    break
            if not placed:
                success = False
                break
        if success:
            num_colors = len(set(colors))
            if num_colors < best_num_colors:
                best_num_colors = num_colors
                best_colors = colors[:]
            if best_num_colors == 3:
                break
    
    return best_colors
