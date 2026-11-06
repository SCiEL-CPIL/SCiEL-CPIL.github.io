import time
import random
from math import gcd
from collections import deque
import numpy as np

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85  # 5s safety margin

    n = limit
    # Build adjacency using bitsets (Python ints for fast bitwise ops)
    bitsets = [0] * n
    for u in range(n):
        num_u = u + 1
        mask = 1 << u  # self
        for v in range(u + 1, n):
            num_v = v + 1
            g = gcd(num_u, num_v)
            lcm_val = (num_u * num_v) // g
            if lcm_val <= limit:
                mask |= (1 << v)
                bitsets[v] |= (1 << u)
        bitsets[u] |= mask
    for i in range(n):
        bitsets[i] |= (1 << i)

    # Degrees for ordering
    degrees = [bin(bitsets[i]).count('1') for i in range(n)]

    # Bron-Kerbosch with greedy coloring (Tomita's algorithm)
    best_clique = []
    best_size = 0
    node_count = [0]
    timeout_flag = [False]

    def check_timeout():
        if time.time() > deadline:
            timeout_flag[0] = True
            return True
        return False

    def greedy_color_and_order(P_mask):
        """Returns (ordered_list, color_of) where ordered_list is sorted by decreasing color,
        and color_of[v] is the color assigned to v. The max color gives the upper bound."""
        P_list = []
        mask = P_mask
        while mask:
            lsb = mask & -mask
            idx = lsb.bit_length() - 1
            P_list.append(idx)
            mask ^= lsb

        if not P_list:
            return [], {}

        # Order by degree descending for better coloring
        P_set = set(P_list)
        deg = {v: len([u for u in P_list if u != v and (bitsets[v] & (1 << u))]) for v in P_list}
        ordered = sorted(P_list, key=lambda x: -deg[x])

        color_of = {}
        colors_used = 0
        for v in ordered:
            neighbor_colors = set()
            for u in P_list:
                if u != v and (bitsets[v] & (1 << u)):
                    if u in color_of:
                        neighbor_colors.add(color_of[u])
            c = 1
            while c in neighbor_colors:
                c += 1
            color_of[v] = c
            colors_used = max(colors_used, c)

        # Sort by decreasing color
        color_order = sorted(P_list, key=lambda x: -color_of[x])
        return color_order, color_of

    def bron_kerbosch(P, X, R):
        nonlocal best_clique, best_size, node_count
        node_count[0] += 1
        if node_count[0] % 500 == 0:
            if check_timeout():
                return False

        if not P and not X:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return True

        color_order, color_of = greedy_color_and_order(P)
        if not color_order:
            return True

        for i in range(len(color_order) - 1, -1, -1):
            if check_timeout():
                return False
            v = color_order[i]
            # Upper bound: |R| + (number of vertices from i to end) = |R| + (len - i)
            if len(R) + (len(color_order) - i) <= best_size:
                break

            R.append(v)
            P_new = P & bitsets[v]
            X_new = X & bitsets[v]

            result = bron_kerbosch(P_new, X_new, R)
            if result is False:
                R.pop()
                return False

            R.pop()
            P &= ~(1 << v)
            X |= (1 << v)

        return True

    # Initial P = all nodes
    all_mask = (1 << n) - 1
    P = all_mask
    X = 0
    R = []

    result = bron_kerbosch(P, X, R)

    if not best_clique:
        return [1]

    # Final verification
    final_list = sorted(best_clique)
    is_valid = True
    for i in range(len(final_list)):
        for j in range(i + 1, len(final_list)):
            a = final_list[i] + 1
            b = final_list[j] + 1
            g = gcd(a, b)
            if (a * b) // g > limit:
                is_valid = False
                break
        if not is_valid:
            break

    if not is_valid:
        return [1]

    return [x + 1 for x in final_list]

construct()
