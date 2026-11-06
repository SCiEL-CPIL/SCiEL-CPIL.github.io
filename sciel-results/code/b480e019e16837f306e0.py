import time
import sys
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 256

    def lcm(a, b):
        return a // gcd(a, b) * b

    adj_val = [set() for _ in range(N + 1)]
    for a in range(2, N + 1):
        for b in range(a + 1, N + 1):
            if lcm(a, b) <= N:
                adj_val[a].add(b)
                adj_val[b].add(a)

    def solve_mwis(values):
        if not values:
            return 0.0, []

        verts = sorted(values)
        n = len(verts)
        weights = [1.0 / v for v in verts]

        adj = [set() for _ in range(n)]
        for i in range(n):
            vi = verts[i]
            for j in range(i + 1, n):
                vj = verts[j]
                if vj in adj_val[vi]:
                    adj[i].add(j)
                    adj[j].add(i)

        # Greedy init
        chosen = []
        available = set(range(n))
        order = sorted(range(n), key=lambda i: verts[i])
        for i in order:
            if i in available:
                chosen.append(i)
                available.remove(i)
                for j in adj[i]:
                    if j in available:
                        available.remove(j)

        current_set = set(chosen)
        current_sum = sum(weights[i] for i in current_set)

        while time.time() < deadline - 5.0:
            improved = False

            # Add phase
            for i in sorted(set(range(n)) - current_set, key=lambda x: verts[x]):
                if i in current_set:
                    continue
                if not (adj[i] & current_set):
                    current_set.add(i)
                    current_sum += weights[i]
                    improved = True

            if not improved:
                # 1-swap
                best_gain = 0
                best_i = -1
                best_j = -1
                for j in set(range(n)) - current_set:
                    C_j = adj[j] & current_set
                    if not C_j:
                        continue
                    min_w = min(weights[i] for i in C_j)
                    gain = weights[j] - min_w
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j
                        for i in C_j:
                            if weights[i] == min_w:
                                best_i = i
                                break
                if best_gain > 0:
                    current_set.remove(best_i)
                    current_set.add(best_j)
                    current_sum += best_gain
                    improved = True

            if not improved:
                break

        return current_sum, [verts[i] for i in current_set]

    V0 = set(range(17, 257))
    sum0, set0 = solve_mwis(V0)
    best_score = sum0
    best_candidate = set0

    for x in range(2, 17):
        Vx = []
        for y in range(2, 257):
            if y == x:
                continue
            if lcm(x, y) > 256:
                Vx.append(y)

        if not Vx:
            score = 1.0 / x
            set_x = [x]
        else:
            sum_x, set_x = solve_mwis(Vx)
            score = 1.0 / x + sum_x
            set_x.append(x)

        if score > best_score:
            best_score = score
            best_candidate = set_x

        if time.time() > deadline - 5.0:
            break

    return best_candidate
