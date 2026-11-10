import itertools
import time
import random

P = {'n': 15}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 55.0

    A = list(range(7))
    B = list(range(7, 15))

    base_edges = set()
    for a in A:
        for b in B:
            base_edges.add((a, b))

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    def get_degrees(edge_set):
        deg = [0] * n
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def build_adj(edge_set):
        adj = [set() for _ in range(n)]
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def is_valid_full(edge_set):
        deg = get_degrees(edge_set)
        adj = build_adj(edge_set)
        for u, v in edge_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    def check_add(edge_set, new_edge, deg, adj):
        """Check if adding new_edge to edge_set (with given deg/adj of edge_set) is valid.
        Only new paths involving new_edge need checking. deg/adj are for edge_set WITHOUT new_edge."""
        u, v = new_edge
        # New degrees: u and v each get +1
        new_deg = list(deg)
        new_deg[u] += 1
        new_deg[v] += 1

        # New adjacency: u and v are connected
        # Paths s-u-v-t: s in N(u)\{v}, t in N(v)\{u,s}
        for s in adj[u]:
            if s == v:
                continue
            for t in adj[v]:
                if t == u or t == s:
                    continue
                if new_deg[s] == new_deg[t]:
                    return False
        # Paths s-v-u-t: s in N(v)\{u}, t in N(u)\{v,s}
        for s in adj[v]:
            if s == u:
                continue
            for t in adj[u]:
                if t == v or t == s:
                    continue
                if new_deg[s] == new_deg[t]:
                    return False
        return True

    def check_add_two(edge_set, e1, e2, deg, adj):
        """Check if adding both e1 and e2 is valid."""
        test_set = edge_set | {e1, e2}
        return is_valid_full(test_set)

    # Initial best solution
    best_edges = set(base_edges)
    best_count = len(best_edges)

    # All possible non-base edges (internal to A or B)
    candidates = []
    for i in range(len(A)):
        for j in range(i + 1, len(A)):
            candidates.append((A[i], A[j]))
    for i in range(len(B)):
        for j in range(i + 1, len(B)):
            candidates.append((B[i], B[j]))

    def greedy_add(edges, candidates_order):
        deg = get_degrees(edges)
        adj = build_adj(edges)
        for c in candidates_order:
            if c in edges:
                continue
            if check_add(edges, c, deg, adj):
                edges = edges | {c}
                deg = get_degrees(edges)
                adj = build_adj(edges)
        return edges

    def local_search(edges, time_limit):
        deg = get_degrees(edges)
        adj = build_adj(edges)
        improved = True
        while improved:
            if time.time() > time_limit:
                break
            improved = False
            edge_list = list(edges)
            random.shuffle(edge_list)

            # Try: add one edge
            avail = [c for c in candidates if c not in edges]
            random.shuffle(avail)
            for add_e in avail[:50]:
                if time.time() > time_limit:
                    break
                if check_add(edges, add_e, deg, adj):
                    edges = edges | {add_e}
                    deg = get_degrees(edges)
                    adj = build_adj(edges)
                    improved = True
                    break

            if improved:
                continue

            # Try: remove one, add one (swap)
            for rem in edge_list:
                if time.time() > time_limit:
                    break
                without_rem = edges - {rem}
                deg2 = get_degrees(without_rem)
                adj2 = build_adj(without_rem)
                avail2 = [c for c in candidates if c not in without_rem and c != rem]
                random.shuffle(avail2)
                for add_e in avail2[:35]:
                    if time.time() > time_limit:
                        break
                    if check_add(without_rem, add_e, deg2, adj2):
                        edges = without_rem | {add_e}
                        deg = get_degrees(edges)
                        adj = build_adj(edges)
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Try: remove one, add two
            for rem in edge_list[:15]:
                if time.time() > time_limit:
                    break
                without_rem = edges - {rem}
                avail3 = [c for c in candidates if c not in without_rem and c != rem]
                random.shuffle(avail3)
                small = avail3[:10]
                found = False
                for i in range(len(small)):
                    if time.time() > time_limit:
                        break
                    for j in range(i + 1, len(small)):
                        if time.time() > time_limit:
                            break
                        test = without_rem | {small[i], small[j]}
                        if is_valid_full(test):
                            edges = test
                            deg = get_degrees(edges)
                            adj = build_adj(edges)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if improved:
                    break

        return edges

    # Initial greedy
    current_edges = set(base_edges)
    shuffled_cands = candidates[:]
    random.shuffle(shuffled_cands)
    current_edges = greedy_add(current_edges, shuffled_cands)

    # Local search on initial result
    current_edges = local_search(current_edges, start_time + 25.0)

    if len(current_edges) > best_count:
        best_edges = set(current_edges)
        best_count = len(best_edges)

    # Random restarts
    for restart in range(300):
        if time.time() > deadline:
            break

        if restart % 3 == 0:
            curr = set(base_edges)
        elif restart % 3 == 1:
            curr = set(best_edges)
            to_remove = random.sample(list(curr), min(random.randint(1, 3), len(curr)))
            for e in to_remove:
                curr.discard(e)
        else:
            curr = set(base_edges)
            random_cands = candidates[:]
            random.shuffle(random_cands)
            deg_t = get_degrees(curr)
            adj_t = build_adj(curr)
            for c in random_cands[:random.randint(0, 5)]:
                if check_add(curr, c, deg_t, adj_t):
                    curr = curr | {c}
                    deg_t = get_degrees(curr)
                    adj_t = build_adj(curr)

        shuffled = candidates[:]
        random.shuffle(shuffled)
        curr = greedy_add(curr, shuffled)

        remaining_time = deadline - time.time()
        if remaining_time > 2:
            curr = local_search(curr, deadline - 2.0)

        if len(curr) > best_count:
            best_edges = set(curr)
            best_count = len(curr)

    result = [list(e) for e in best_edges]
    result.sort()

    for i in range(len(result)):
        result[i][0] = int(result[i][0])
        result[i][1] = int(result[i][1])

    return result
