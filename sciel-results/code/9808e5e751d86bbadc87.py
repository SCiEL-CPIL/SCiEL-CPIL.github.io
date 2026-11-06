"""Combined program: structural backtracking with symmetry breaking and deadline guards."""
import time

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin before 90s kill

    # Precompute APs ending at each index x (1-indexed)
    # APs are (i, j, k, x) such that i, j, k, x form a 4-AP with i = x-3d
    aps_ending = [[] for _ in range(N + 1)]
    for x in range(1, N + 1):
        for d in range(1, (x - 1) // 3 + 1):
            i = x - 3 * d
            if i >= 1:
                j = i + d
                k = i + 2 * d
                aps_ending[x].append((i, j, k))

    def solve_with_deadline(k, dl):
        """
        Backtracking solver that tries to color positions 1..N with colors 0..k-1
        such that every 4-AP uses at least 3 distinct colors.
        """
        colors = [None] * (N + 1)
        colors[1] = 0  # symmetry breaking

        def backtrack(x):
            if time.time() > dl:
                return None

            if x > N:
                return colors[1:].copy()

            # Try each color c in 0..k-1
            for c in range(k):
                valid = True
                for (i, j, k_idx) in aps_ending[x]:
                    ci = colors[i]
                    cj = colors[j]
                    ck = colors[k_idx]
                    # Check if {ci, cj, ck, c} has >= 3 distinct colors
                    if len({ci, cj, ck, c}) < 3:
                        valid = False
                        break

                if valid:
                    colors[x] = c
                    res = backtrack(x + 1)
                    if res is not None:
                        return res
                    colors[x] = None

            return None

        return backtrack(2)

    # Try k=3 first with a generous time budget
    sol = solve_with_deadline(3, start_time + 45.0)
    if sol:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]

    # Try k=4
    sol = solve_with_deadline(4, start_time + 75.0)
    if sol:
        used = sorted(set(sol))
        mapping = {c: i for i, c in enumerate(used)}
        return [mapping[c] for c in sol]

    # Fallback: seed program logic
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
