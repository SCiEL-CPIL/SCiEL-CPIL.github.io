"""Colorings with three colors in every four-term progression."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 80  # safety margin within 90s wall-clock

    n = 128

    def check_validity(colors):
        """Check if coloring satisfies the constraint."""
        for a in range(1, n - 2):
            max_d = (n - a) // 3
            for d in range(1, max_d + 1):
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                c4 = colors[a + 3 * d - 1]
                if len({c1, c2, c3, c4}) < 3:
                    return False
        return True

    def greedy_lookahead(num_colors, lookahead_k=4):
        """
        Deterministic greedy with look-ahead: assign the smallest valid color to each position,
        but simulate the next k steps to prune choices that lead to dead-ends.
        """
        colors = [0] * n
        colors[0] = 0  # Start with color 0

        for x in range(2, n + 1):
            valid_colors = set(range(num_colors))

            # APs where x is the 4th term: a = x-3d, a+d, a+2d, x
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                existing = {c1, c2, c3}
                if len(existing) >= 3:
                    pass
                elif len(existing) == 2:
                    for c in list(valid_colors):
                        if c in existing:
                            valid_colors.discard(c)
                else:
                    for c in list(valid_colors):
                        if c == list(existing)[0]:
                            valid_colors.discard(c)

            # APs where x is the 3rd term: a, a+d, x, x+d
            for d in range(1, (n - x) // 1 + 1):
                if x + d > n:
                    break
                a = x - 2 * d
                if a < 1:
                    continue
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c4 = colors[x + d - 1]  # This is 0 (unassigned), so we can't check yet
                # Skip for now, will be checked when x+d is assigned

            # APs where x is the 2nd term: a, x, x+d, x+2d
            for d in range(1, (n - x) // 2 + 1):
                if x + 2 * d > n:
                    break
                a = x - d
                if a < 1:
                    continue
                c1 = colors[a - 1]
                c3 = colors[x + d - 1]  # Unassigned
                c4 = colors[x + 2 * d - 1]  # Unassigned
                # Skip for now

            # APs where x is the 1st term: x, x+d, x+2d, x+3d
            # All others unassigned, skip

            if not valid_colors:
                return None

            # Look-ahead: simulate next k steps for each candidate color
            best_color = None
            for c in sorted(valid_colors):
                colors[x - 1] = c
                if _lookahead_simulate(colors, x, num_colors, lookahead_k, n):
                    best_color = c
                    break

            if best_color is None:
                # No color passes look-ahead, fall back to first valid
                colors[x - 1] = min(valid_colors)

        if check_validity(colors):
            return colors
        return None

    def _lookahead_simulate(colors, current_x, num_colors, k, n):
        """Simulate k steps forward from current_x to check for dead-ends."""
        for step in range(1, k + 1):
            x = current_x + step
            if x > n:
                return True  # No more positions, no conflict

            valid = set(range(num_colors))

            # Check APs where x is the 4th term
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                existing = {c1, c2, c3}
                if len(existing) >= 3:
                    pass
                elif len(existing) == 2:
                    for c in list(valid):
                        if c in existing:
                            valid.discard(c)
                else:
                    for c in list(valid):
                        if c == list(existing)[0]:
                            valid.discard(c)

            if not valid:
                return False  # Dead-end detected

        return True

    # Try with 3 colors using look-ahead greedy
    if time.time() < deadline - 10:
        result = greedy_lookahead(3, lookahead_k=4)
        if result is not None:
            return result

    # Try with 3 colors using simple greedy (fallback)
    def greedy_simple(num_colors):
        colors = [0] * n
        colors[0] = 0
        for x in range(2, n + 1):
            valid_colors = set(range(num_colors))
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                c1 = colors[a - 1]
                c2 = colors[a + d - 1]
                c3 = colors[a + 2 * d - 1]
                existing = {c1, c2, c3}
                if len(existing) >= 3:
                    pass
                elif len(existing) == 2:
                    for c in list(valid_colors):
                        if c in existing:
                            valid_colors.discard(c)
                else:
                    for c in list(valid_colors):
                        if c == list(existing)[0]:
                            valid_colors.discard(c)
            if not valid_colors:
                return None
            colors[x - 1] = min(valid_colors)
        if check_validity(colors):
            return colors
        return None

    if time.time() < deadline - 5:
        result = greedy_simple(3)
        if result is not None:
            return result

    # Try with 4 colors
    if time.time() < deadline - 5:
        result = greedy_lookahead(4, lookahead_k=4)
        if result is not None:
            return result

    if time.time() < deadline - 5:
        result = greedy_simple(4)
        if result is not None:
            return result

    # Fallback: use the seed program's approach (should always work)
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color} | {colors[x-1-j*d] for j in (1,2,3)}) >= 3 for d in range(1, (x-1)//3+1))
                and all(any(colors[x-1-j*d] != color for j in (1,2)) for d in range(1, (x-1)//2+1))):
                colors.append(color)
                break
    return colors

# EVOLVE-BLOCK-END
