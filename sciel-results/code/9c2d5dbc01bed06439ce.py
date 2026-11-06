"""Combined: Legendre-symbol seeding + fast simulated annealing + color-merging polish."""
import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    n = 128

    # Precompute all 4-term APs (0-indexed): (a, a+d, a+2d, a+3d)
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))

    num_aps = len(aps)

    # For each position, list of AP indices containing it
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)

    # Precompute for each AP, the other 3 positions for each of its 4 positions
    # ap_other[ap_idx][j] = list of 3 other positions in ap_idx
    ap_other = []
    for ap in aps:
        others = []
        for j in range(4):
            others.append([ap[k] for k in range(4) if k != j])
        ap_other.append(others)

    def count_violations(colors):
        """Count APs with < 3 distinct colors."""
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                v += 1
            elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                v += 1
        return v

    def is_valid(colors):
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if c0 == c1 == c2 == c3:
                return False
            if c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                return False
        return True

    def num_colors(colors):
        return len(set(colors))

    # Legendre symbol: (a/p) for prime p
    def legendre_symbol(a, p):
        if a % p == 0:
            return 0
        ls = pow(a, (p - 1) // 2, p)
        return 1 if ls == 1 else -1

    # Generate a good initial coloring using number-theoretic structure
    def generate_legendre_coloring():
        """Use Legendre symbols mod several primes to create diverse colorings."""
        colorings = []
        primes = [127, 113, 109, 107, 103, 97, 89, 83]
        for p in primes:
            colors = []
            for pos in range(1, n + 1):
                ls = legendre_symbol(pos, p)
                # Map -1, 0, 1 to colors 0, 1, 2
                if ls == 1:
                    colors.append(0)
                elif ls == -1:
                    colors.append(1)
                else:
                    colors.append(2)
            colorings.append(colors)

        # Also try combinations of two primes for more colors
        for i in range(min(4, len(primes))):
            for j in range(i + 1, min(4, len(primes))):
                p1, p2 = primes[i], primes[j]
                colors = []
                for pos in range(1, n + 1):
                    ls1 = legendre_symbol(pos, p1)
                    ls2 = legendre_symbol(pos, p2)
                    # Combine: (ls1+1, ls2+1) -> 4 colors
                    colors.append((ls1 + 1) * 2 + (ls2 + 1))
                colorings.append(colors)

        # Add some random colorings
        for seed in range(20):
            rng = random.Random(seed * 7919)
            colors = [rng.randint(0, 3) for _ in range(n)]
            colorings.append(colors)

        return colorings

    def simulated_annealing(initial_colors, k, max_steps=50000):
        """Simulated annealing to minimize violations, then minimize colors."""
        colors = list(initial_colors)
        current_violations = count_violations(colors)

        if current_violations == 0:
            return colors, current_violations

        best_colors = colors[:]
        best_violations = current_violations
        best_nc = num_colors(colors)

        temp = 2.0
        cooling = 0.99998
        steps = 0

        while steps < max_steps:
            if time.time() > deadline - 5.0:
                break
            steps += 1

            i = random.randint(0, n - 1)
            old_color = colors[i]
            new_color = random.randint(0, k - 1)
            if new_color == old_color:
                continue

            # Compute delta in violations
            delta = 0
            for ap_idx in pos_to_aps[i]:
                ap = aps[ap_idx]
                # Check before
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                was_violation = False
                if c0 == c1 == c2 == c3:
                    was_violation = True
                elif c0 == c1 == c2 or c0 == c1 == c3 or c0 == c2 == c3 or c1 == c2 == c3:
                    was_violation = True

                # Set new color
                j = ap.index(i)
                temp_colors = list(colors)
                temp_colors[i] = new_color
                nc0, nc1, nc2, nc3 = temp_colors[ap[0]], temp_colors[ap[1]], temp_colors[ap[2]], temp_colors[ap[3]]
                now_violation = False
                if nc0 == nc1 == nc2 == nc3:
                    now_violation = True
                elif nc0 == nc1 == nc2 or nc0 == nc1 == nc3 or nc0 == nc2 == nc3 or nc1 == nc2 == nc3:
                    now_violation = True

                if was_violation and not now_violation:
                    delta -= 1
                elif not was_violation and now_violation:
                    delta += 1

            new_violations = current_violations + delta

            if new_violations <= current_violations or random.random() < math.exp(-(new_violations - current_violations) / temp):
                colors[i] = new_color
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                    best_nc = num_colors(colors)
                if current_violations == 0:
                    break

            temp *= cooling

        return best_colors, best_violations

    def merge_colors(colors, k_target, max_steps=20000):
        """Try to reduce the number of distinct colors to k_target."""
        colors = list(colors)
        current_nc = num_colors(colors)
        if current_nc <= k_target:
            return colors, current_nc

        best_colors = colors[:]
        best_nc = current_nc

        steps = 0
        while steps < max_steps:
            if time.time() > deadline - 3.0:
                break
            steps += 1

            if num_colors(colors) <= k_target:
                return colors, num_colors(colors)

            # Pick a position with a rare color
            color_counts = {}
            for c in colors:
                color_counts[c] = color_counts.get(c, 0) + 1

            # Find the color with fewest positions
            rarest_color = min(color_counts, key=color_counts.get)
            positions_with_rarest = [i for i in range(n) if colors[i] == rarest_color]

            if not positions_with_rarest:
                break

            i = random.choice(positions_with_rarest)
            old_color = colors[i]

            # Try changing to another existing color
            other_colors = [c for c in color_counts if c != rarest_color]
            if not other_colors:
                break

            new_color = random.choice(other_colors)

            # Check if this change keeps validity
            colors[i] = new_color
            if is_valid(colors):
                nc = num_colors(colors)
                if nc < best_nc:
                    best_nc = nc
                    best_colors = colors[:]
                if nc <= k_target:
                    return colors, nc
            else:
                colors[i] = old_color

        return best_colors, best_nc

    # Main search
    best_result = None
    best_nc = 128

    # Generate initial colorings
    initial_colorings = generate_legendre_coloring()

    # Try each initial coloring with simulated annealing
    for init_colors in initial_colorings:
        if time.time() > deadline - 10.0:
            break

        # Determine k: try small k first
        for k in [3, 4, 5, 6, 8]:
            if time.time() > deadline - 8.0:
                break

            # Clamp initial colors to [0, k-1]
            clamped = [c % k for c in init_colors]

            # Quick check
            if is_valid(clamped):
                nc = num_colors(clamped)
                if nc < best_nc:
                    best_nc = nc
                    best_result = clamped[:]

            # Simulated annealing
            sa_colors, sa_violations = simulated_annealing(clamped, k, max_steps=30000)

            if sa_violations == 0:
                nc = num_colors(sa_colors)
                if nc < best_nc:
                    best_nc = nc
                    best_result = sa_colors[:]

                # Try to merge colors further
                if nc > 3 and time.time() < deadline - 5.0:
                    for target_k in range(3, nc):
                        if time.time() > deadline - 4.0:
                            break
                        merged_colors, merged_nc = merge_colors(sa_colors, target_k, max_steps=10000)
                        if merged_nc < best_nc:
                            best_nc = merged_nc
                            best_result = merged_colors[:]
                        if merged_nc <= 3:
                            break

    # If no valid result found, use a greedy fallback
    if best_result is None:
        colors = []
        for x in range(n):
            found = False
            for c in range(128):
                valid = True
                for d in range(1, (x) // 3 + 1):
                    i3 = x - 3 * d
                    i2 = x - 2 * d
                    i1 = x - d
                    if i3 < 0:
                        continue
                    if len({colors[i3], colors[i2], colors[i1], c}) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(c)
                    found = True
                    break
            if not found:
                colors.append(0)
        best_result = colors
        best_nc = num_colors(colors)

    # Final polish: try to reduce colors one more time
    if best_nc > 3 and time.time() < deadline - 2.0:
        for target_k in range(3, best_nc):
            if time.time() > deadline - 1.0:
                break
            merged_colors, merged_nc = merge_colors(best_result, target_k, max_steps=5000)
            if merged_nc < best_nc:
                best_nc = merged_nc
                best_result = merged_colors[:]
            if merged_nc <= 3:
                break

    return best_result
