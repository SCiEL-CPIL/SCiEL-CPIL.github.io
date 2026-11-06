import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = 128
    constraints = [[] for _ in range(n)]
    for i in range(n):
        max_d = i // 3
        for d in range(1, max_d + 1):
            a = i - 3 * d
            b = i - 2 * d
            c = i - d
            constraints[i].append((a, b, c))

    def solve(k):
        colors = [-1] * n
        
        def is_valid(pos, color):
            for a, b, c in constraints[pos]:
                ca, cb, cc = colors[a], colors[b], colors[c]
                if color == ca and color == cb and color == cc:
                    return False
                # Check set size logic
                if color in (ca, cb, cc):
                    if ca == cb and cb == cc:
                        return False
                    # If 2 distinct in ca,cb,cc, and color is one of them, set size is 2. Invalid.
                    # If 3 distinct, valid.
                    # So if color in S, we need |S| >= 3.
                    if len({ca, cb, cc}) < 3:
                        return False
                else:
                    # If color not in S, set size is |S| + 1.
                    # Need |S| + 1 >= 3 => |S| >= 2.
                    if len({ca, cb, cc}) < 2:
                        return False
            return True

        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            
            # Heuristic: Try colors in order 0, 1, ..., k-1
            for color in range(k):
                if is_valid(pos, color):
                    colors[pos] = color
                    if backtrack(pos + 1):
                        return True
                    colors[pos] = -1
            return False

        if backtrack(0):
            return colors
        return None

    sol = solve(3)
    if sol is not None:
        return sol

    # If 3 fails, try 4
    if time.time() < deadline - 5:
        sol4 = solve(4)
        if sol4 is not None:
            return sol4

    # Fallback to greedy 4
    def greedy(k):
        colors = []
        for x in range(1, n + 1):
            idx = x - 1
            found = False
            for color in range(k):
                valid = True
                for a, b, c in constraints[idx]:
                    ca, cb, cc = colors[a], colors[b], colors[c]
                    if color in (ca, cb, cc):
                        if len({ca, cb, cc}) < 3:
                            valid = False
                            break
                    else:
                        if len({ca, cb, cc}) < 2:
                            valid = False
                            break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                colors.append(0)
        return colors

    return greedy(4)
