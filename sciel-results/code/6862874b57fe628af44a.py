def construct():
    n = 128
    colors = [-1] * (n + 1)  # 1-indexed

    def check_last_progressions(pos, color, arr):
        """Check 4-term APs where pos is the last element and all prior are assigned."""
        for d in range(1, (pos - 1) // 3 + 1):
            p1 = pos - 3 * d
            if p1 < 1:
                break
            c1 = arr[p1]
            c2 = arr[p1 + d]
            c3 = arr[p1 + 2 * d]
            if c1 == -1 or c2 == -1 or c3 == -1:
                continue
            if len({c1, c2, c3, color}) < 3:
                return False
        return True

    def greedy_pick(pos, arr, max_c=127):
        for c in range(max_c + 1):
            if check_last_progressions(pos, c, arr):
                return c
        return -1

    def lookahead_ok(pos, color, arr, k=4, max_c=127):
        sim = arr[:]
        sim[pos] = color
        for i in range(pos + 1, min(pos + 1 + k, n + 1)):
            c = greedy_pick(i, sim, max_c)
            if c == -1:
                return False
            sim[i] = c
        return True

    for pos in range(1, n + 1):
        chosen = -1
        for c in range(128):
            if not check_last_progressions(pos, c, colors):
                continue
            if lookahead_ok(pos, c, colors, k=4, max_c=127):
                chosen = c
                break
        if chosen == -1:
            # Fallback: smallest valid color without lookahead
            for c in range(128):
                if check_last_progressions(pos, c, colors):
                    chosen = c
                    break
        if chosen == -1:
            chosen = 0
        colors[pos] = chosen

    return colors[1:n + 1]
