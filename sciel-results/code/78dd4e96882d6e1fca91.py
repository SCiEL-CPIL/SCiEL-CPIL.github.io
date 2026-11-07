import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    n = 10
    # L1: cyclic group Z_10
    L1 = [[(i + j) % n for j in range(n)] for i in range(n)]

    squares = [L1]

    # Search for a second Latin square L2 orthogonal to L1.
    # Orthogonality: for each symbol a in L1, the positions where L1[i][j]==a
    # must have all distinct values in L2.
    # L1[i][j] = (i+j)%n, so symbol a appears at positions (i, (a-i)%n) for i in 0..n-1.
    # This forms a "diagonal" in the grid.

    # Use bitmask-based backtracking for speed.
    def search_l2():
        L2 = [[-1] * n for _ in range(n)]
        row_mask = [0] * n  # bitmask of used symbols per row
        col_mask = [0] * n  # bitmask of used symbols per column
        class_mask = [0] * n  # for each symbol a in L1, bitmask of used values in L2 on that diagonal

        def backtrack(num_filled):
            if time.time() > deadline:
                return False

            if num_filled == n * n:
                return True

            # MRV: find the empty cell with fewest candidates
            min_count = n + 1
            min_i = -1
            min_j = -1
            min_cands = []

            for i in range(n):
                for j in range(n):
                    if L2[i][j] == -1:
                        a = L1[i][j]
                        allowed = (~row_mask[i] & ~col_mask[j] & ~class_mask[a]) & ((1 << n) - 1)
                        cands = []
                        for v in range(n):
                            if allowed & (1 << v):
                                cands.append(v)
                        if len(cands) < min_count:
                            min_count = len(cands)
                            min_i = i
                            min_j = j
                            min_cands = cands
                            if min_count == 0:
                                return False
                            if min_count == 1:
                                break
                if min_count == 1:
                    break

            if min_i == -1:
                return True

            for v in min_cands:
                L2[min_i][min_j] = v
                a = L1[min_i][min_j]
                row_mask[min_i] |= (1 << v)
                col_mask[min_j] |= (1 << v)
                class_mask[a] |= (1 << v)

                if backtrack(num_filled + 1):
                    return True

                L2[min_i][min_j] = -1
                row_mask[min_i] &= ~(1 << v)
                col_mask[min_j] &= ~(1 << v)
                class_mask[a] &= ~(1 << v)

            return False

        if backtrack(0):
            # Final verification
            for i in range(n):
                if sorted(L2[i]) != list(range(n)):
                    return None
            for j in range(n):
                if sorted([L2[i][j] for i in range(n)]) != list(range(n)):
                    return None
            # Check orthogonality
            pairs = set()
            for i in range(n):
                for j in range(n):
                    p = (L1[i][j], L2[i][j])
                    if p in pairs:
                        return None
                    pairs.add(p)
            return L2
        return None

    L2 = search_l2()
    if L2:
        squares.append(L2)

    # Ensure types
    for s in squares:
        for i in range(n):
            for j in range(n):
                if not isinstance(s[i][j], int) or isinstance(s[i][j], bool):
                    s[i][j] = int(s[i][j])
            if not isinstance(s[i], list):
                s[i] = list(s[i])
        if not isinstance(s, list):
            squares = list(squares)

    if not isinstance(squares, list):
        squares = list(squares)

    return squares
