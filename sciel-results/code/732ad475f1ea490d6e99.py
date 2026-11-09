import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 90s kill

    P = 27720  # LCM(1..12)
    n = 12
    max_L = 144  # upper bound from contract

    # Precompute offset[i][s] = (i-1-s) % i
    offset = [[0]*13 for _ in range(13)]
    for i in range(1, 13):
        for s in range(i):
            offset[i][s] = (i - 1 - s) % i

    best_L = 0
    best_r = 0

    for r in range(P):
        if time.time() > deadline:
            break

        for L in range(12, max_L + 1):
            # Build valid positions for each i
            valid = [None] * 13
            feasible = True
            for i in range(1, 13):
                c = offset[i][r % i]
                if c >= L:
                    feasible = False
                    break
                valid[i] = list(range(c, L, i))

            if not feasible:
                continue

            # Order by constraint (fewest valid positions first)
            order = tuple(sorted(range(1, 13), key=lambda i: len(valid[i])))

            match_to = [-1] * L

            def dfs(i, visited):
                for j in valid[i]:
                    if not visited[j]:
                        visited[j] = True
                        if match_to[j] == -1 or dfs(match_to[j], visited):
                            match_to[j] = i
                            return True
                return False

            found = True
            for i in order:
                visited = [False] * L
                if not dfs(i, visited):
                    found = False
                    break

            if found:
                if L > best_L:
                    best_L = L
                    best_r = r
                break

    return int(best_r)
