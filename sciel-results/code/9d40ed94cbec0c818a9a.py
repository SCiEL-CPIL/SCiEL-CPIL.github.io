import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256
    nodes = list(range(2, N + 1))
    n = len(nodes)

    # Build compatibility bitmask: compat[i] has bit j set if lcm(nodes[i],nodes[j]) > 256
    compat = [0] * n
    for i in range(n):
        a = nodes[i]
        mask = 0
        for j in range(n):
            if i == j:
                continue
            b = nodes[j]
            g = gcd(a, b)
            if a * b // g > N:
                mask |= (1 << j)
        compat[i] = mask

    weights = [1.0 / nodes[i] for i in range(n)]

    # Order by descending weight for better BnB pruning
    order = sorted(range(n), key=lambda i: -weights[i])
    new_weights = [weights[order[k]] for k in range(n)]

    # new_compat in reordered space
    new_compat = [0] * n
    for k in range(n):
        orig = order[k]
        mask = compat[orig]
        nm = 0
        for j in range(n):
            if mask & (1 << order[j]):
                nm |= (1 << j)
        new_compat[k] = nm

    best_value = 0.0
    best_clique = []

    def simple_bound(P):
        s = 0.0
        for k in range(n):
            if P & (1 << k):
                s += new_weights[k]
        return s

    def color_bound_fast(P):
        """Greedy coloring upper bound for max weight clique."""
        if P == 0:
            return 0.0
        color_of = [-1] * n
        color_max_w = []
        for k in range(n):
            if not (P & (1 << k)):
                continue
            ck = new_compat[k] & P
            forbidden = [False] * 64
            tmp = ck
            while tmp:
                lb = tmp & (-tmp)
                j = lb.bit_length() - 1
                c = color_of[j]
                if c >= 0:
                    if c < 64:
                        forbidden[c] = True
                    else:
                        while len(forbidden) <= c:
                            forbidden.append(False)
                        forbidden[c] = True
                tmp ^= lb
            c = 0
            while c < len(forbidden) and forbidden[c]:
                c += 1
            color_of[k] = c
            if c >= len(color_max_w):
                color_max_w.append(0.0)
            if new_weights[k] > color_max_w[c]:
                color_max_w[c] = new_weights[k]
        return sum(color_max_w)

    def bnb(P, clique, current_w):
        nonlocal best_value, best_clique
        if time.time() > deadline:
            return
        if P == 0:
            if current_w > best_value:
                best_value = current_w
                best_clique = list(clique)
            return

        ub_quick = current_w + simple_bound(P)
        if ub_quick <= best_value + 1e-15:
            return

        ub = current_w + color_bound_fast(P)
        if ub <= best_value + 1e-15:
            return

        # Find highest weight node in P
        best_k = -1
        best_w = -1.0
        for k in range(n):
            if P & (1 << k):
                if new_weights[k] > best_w:
                    best_w = new_weights[k]
                    best_k = k

        k = best_k
        bit = 1 << k

        # Include k
        new_P = P & new_compat[k] & ~bit
        clique.append(k)
        bnb(new_P, clique, current_w + new_weights[k])
        clique.pop()

        # Exclude k
        bnb(P & ~bit, clique, current_w)

    all_mask = (1 << n) - 1
    bnb(all_mask, [], 0.0)

    result = [nodes[order[k]] for k in best_clique]
    if not result:
        result = [nodes[order[0]]]

    # Verify
    valid = True
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            g = gcd(a, b)
            if a * b // g <= N:
                valid = False
                break
        if not valid:
            break

    if not valid:
        remaining = set(range(n))
        greedy_sol = []
        while remaining:
            bk = max(remaining, key=lambda i: weights[i])
            greedy_sol.append(bk)
            to_rm = {bk}
            for j in remaining:
                if j != bk and not (compat[bk] & (1 << j)):
                    to_rm.add(j)
            remaining -= to_rm
        result = [nodes[i] for i in greedy_sol]

    return result

if __name__ == "__main__":
    construct()
