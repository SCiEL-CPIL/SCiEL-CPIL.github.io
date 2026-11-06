import time
import random

def construct():
    deadline = time.time() + 80.0
    n = 128

    # Precompute all 4-term progressions
    progressions = []
    for a in range(1, n + 1):
        for d in range(1, n + 1):
            e = a + 3 * d
            if e <= n:
                progressions.append((a - 1, a + d - 1, a + 2 * d - 1, e - 1))

    num_progressions = len(progressions)

    # For each element, store which progressions it belongs to
    elem_to_progressions = [[] for _ in range(n)]
    for pi, p in enumerate(progressions):
        for idx in p:
            elem_to_progressions[idx].append(pi)

    def is_valid(colors):
        for (a, b, c, e) in progressions:
            if len({colors[a], colors[b], colors[c], colors[e]}) < 3:
                return False
        return True

    random.seed(98765)

    best_colors = None
    best_num = 999

    # Start with 4-color modular pattern
    colors4 = [(i % 4) for i in range(n)]
    if is_valid(colors4):
        best_colors = colors4[:]
        best_num = 4

    # Try to reduce to 3 colors
    for restart in range(500):
        if time.time() > deadline:
            break

        current = best_colors[:] if best_colors is not None else [(i % 4) for i in range(n)]
        cur_num = len(set(current))

        # Try to merge: pick a color and reassign its elements
        for merge_iter in range(100):
            if time.time() > deadline:
                break
            used = sorted(set(current))
            if len(used) <= 3:
                break

            cc = {c: 0 for c in used}
            for x in current:
                cc[x] += 1

            # Try each color in order of fewest elements
            for target_color in sorted(used, key=lambda c: cc[c]):
                if len(used) <= 3:
                    break
                elements = [i for i in range(n) if current[i] == target_color]
                if not elements:
                    continue

                new_current = current[:]
                success = True
                for i in elements:
                    if time.time() > deadline:
                        success = False
                        break
                    found = False
                    # Try other colors in random order
                    other_colors = [c for c in used if c != target_color]
                    random.shuffle(other_colors)
                    for new_c in other_colors:
                        new_current[i] = new_c
                        ok = True
                        for pi in elem_to_progressions[i]:
                            a, b, c, e = progressions[pi]
                            if len({new_current[a], new_current[b], new_current[c], new_current[e]}) < 3:
                                ok = False
                                break
                        if ok:
                            found = True
                            break
                        else:
                            new_current[i] = current[i]
                    if not found:
                        success = False
                        break

                if success:
                    current = new_current
                    cur_num = len(set(current))
                    if cur_num < best_num:
                        best_num = cur_num
                        best_colors = current[:]
                    break  # restart merge loop with new state

            if cur_num <= 3:
                break

        if best_num == 3:
            break

    if best_colors is not None and is_valid(best_colors):
        return best_colors
    return [(i % 4) for i in range(n)]
