import math
import time
import sys

def construct():
    """
    Construct a nonempty subset of [1, 255] containing 255,
    with gcd(a,b) > 1 for all distinct pairs. Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0

    # Build candidate set: all x in [1,255] with gcd(x,255) > 1
    candidates = []
    for x in range(1, 256):
        if math.gcd(x, 255) > 1:
            candidates.append(x)

    n = len(candidates)

    # Build adjacency using integer bitsets for speed
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    idx_255 = candidates.index(255)

    # Initialize with a known large clique: all multiples of 3
    init_clique = set()
    for i, val in enumerate(candidates):
        if val % 3 == 0:
            init_clique.add(i)

    best_clique = init_clique
    best_size = len(best_clique)

    # Tomita-style maximum clique with greedy coloring upper bound
    def bk(R, P, X):
        nonlocal best_clique, best_size
        if time.time() > deadline:
            return

        if P == 0:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = set(R)
            return

        P_list = [i for i in range(n) if (P >> i) & 1]

        # Greedy coloring for upper bound
        color = {}
        max_c = 0
        for v in P_list:
            used = set()
            for u in P_list:
                if u == v:
                    break
                if (adj[u] >> v) & 1 and u in color:
                    used.add(color[u])
            c = 1
            while c in used:
                c += 1
            color[v] = c
            if c > max_c:
                max_c = c

        if len(R) + max_c <= best_size:
            return

        # Pivot selection
        U = P | X
        if U:
            U_list = [i for i in range(n) if (U >> i) & 1]
            pivot = max(U_list, key=lambda u: bin(adj[u] & P).count('1'))
            cand = [v for v in P_list if not ((adj[pivot] >> v) & 1)]
        else:
            cand = P_list

        # Sort by color descending for better pruning
        cand.sort(key=lambda v: color.get(v, 0), reverse=True)

        for v in cand:
            if time.time() > deadline:
                return
            if len(R) + 1 + (max_c - 1) <= best_size:
                return
            bk(R | {v}, P & adj[v], X & adj[v])

    bk({idx_255}, adj[idx_255], 0)

    # Greedy local improvement: try adding vertices
    if time.time() < deadline - 5.0:
        result_set = set(candidates[i] for i in best_clique)
        improved = True
        while improved:
            improved = False
            for x in range(1, 256):
                if x in result_set or math.gcd(x, 255) == 1:
                    continue
                can_add = True
                for y in result_set:
                    if math.gcd(x, y) == 1:
                        can_add = False
                        break
                if can_add:
                    result_set.add(x)
                    improved = True
        best_clique = set()
        for i, val in enumerate(candidates):
            if val in result_set:
                best_clique.add(i)
        best_size = len(best_clique)

    result_set = set(candidates[i] for i in best_clique)
    if 255 not in result_set:
        result_set.add(255)

    res_list = list(result_set)
    if not res_list:
        return [255]

    for i in range(len(res_list)):
        for j in range(i + 1, len(res_list)):
            if math.gcd(res_list[i], res_list[j]) == 1:
                return list(range(3, 256, 3))

    if 255 not in res_list:
        return list(range(3, 256, 3))

    return res_list

# For quick testing
if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Contains 255: {255 in result}")
    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if math.gcd(result[i], result[j]) == 1:
                valid = False
                break
    print(f"Valid: {valid}")
