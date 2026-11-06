import time
import random
import numpy as np

def construct():
    """
    Construct a valid 2-coloring of K_n without monochromatic K2,2,2.
    Returns {'n': n, 'colors': [...]} where colors[i] is 0 or 1 for edge (i,j) in lexicographic order.
    """
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit

    # The known Ramsey number R(K2,2,2) = 30, so max valid n is 29.
    # We use the Paley coloring with prime p=29 to get a valid 29-vertex coloring.
    n = 29
    p = 29  # 29 is prime and 29 ≡ 1 (mod 4)

    # Compute quadratic residues mod 29
    squares = set((x * x) % p for x in range(1, p))

    # Build coloring in lexicographic order
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)

    # Verify validity (quick check using bitsets)
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1

    # Verify no K2,2,2 in either color class
    def count_k222(adj, n):
        count = 0
        for c in range(n):
            for d in range(c + 1, n):
                if adj[c] & (1 << d):
                    continue
                common = adj[c] & adj[d]
                verts = []
                mask = common
                while mask:
                    v = (mask & (-mask)).bit_length() - 1
                    verts.append(v)
                    mask &= mask - 1
                m = len(verts)
                if m < 4:
                    continue
                for i in range(m - 1):
                    a = verts[i]
                    for j in range(i + 1, m):
                        b = verts[j]
                        ab_common = adj[a] & adj[b] & common
                        if ab_common & (ab_common - 1):
                            count += 1
        return count

    if count_k222(adj0, n) == 0 and count_k222(adj1, n) == 0:
        return {'n': n, 'colors': colors}

    # Fallback: if Paley doesn't work (shouldn't happen), try simulated annealing
    if time.time() < deadline - 30:
        for attempt in range(5):
            colors_try = [random.randint(0, 1) for _ in range(n * (n - 1) // 2)]
            adj0_t = [0] * n
            adj1_t = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors_try[idx] == 0:
                        adj0_t[i] |= (1 << j)
                        adj0_t[j] |= (1 << i)
                    else:
                        adj1_t[i] |= (1 << j)
                        adj1_t[j] |= (1 << i)
                    idx += 1
            if count_k222(adj0_t, n) == 0 and count_k222(adj1_t, n) == 0:
                return {'n': n, 'colors': colors_try}

    # Absolute fallback
    return {'n': 1, 'colors': [0]}
