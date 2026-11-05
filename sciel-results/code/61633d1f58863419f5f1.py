import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 78  # Safety margin before 90s limit

    N = 180
    all_elements = set(range(1, N + 1))

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
    # 1/a = 1/b + 1/c => bc = a(b+c) => bc - ab - ac = 0 => (b-a)(c-a) = a^2
    triples = []
    triples_by_a = defaultdict(list)
    for a in range(1, N + 1):
        a_sq = a * a
        for u in range(1, int(a_sq ** 0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u == v:
                    continue
                b = a + u
                c = a + v
                if b > N or c > N:
                    continue
                if b == c:
                    continue
                if b < c:
                    triples.append((a, b, c))
                    triples_by_a[a].append((b, c))
                else:
                    triples.append((a, c, b))
                    triples_by_a[a].append((c, b))

    # For each element, compute which triples it participates in
    triples_by_element = defaultdict(list)
    for idx, (a, b, c) in enumerate(triples):
        triples_by_element[a].append(idx)
        triples_by_element[b].append(idx)
        triples_by_element[c].append(idx)

    # Fast can_add check using precomputed structures
    def can_add(x, S_set):
        # x as small element
        for b, c in triples_by_a[x]:
            if b in S_set and c in S_set:
                return False
        # x as large element
        for a in S_set:
            if a >= x:
                continue
            den = x - a
            num = a * x
            if num % den != 0:
                continue
            c = num // den
            if c == a or c == x:
                continue
            if c in S_set:
                return False
        return True

    # Greedy solution with given ordering
    def greedy_solution(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    # Try multiple greedy orderings
    best_S = set()
    best_size = 0

    # Order 1: descending (prefer large numbers - they rarely serve as 'a')
    order1 = list(range(N, 0, -1))
    S1 = greedy_solution(order1)
    if len(S1) > best_size:
        best_S = set(S1)
        best_size = len(S1)

    # Order 2: ascending
    order2 = list(range(1, N + 1))
    S2 = greedy_solution(order2)
    if len(S2) > best_size:
        best_S = set(S2)
        best_size = len(S2)

    # Order 3: by conflict degree ascending (least constrained first)
    conflict_degree = {x: len(triples_by_element[x]) for x in range(1, N + 1)}
    order3 = sorted(range(1, N + 1), key=lambda x: conflict_degree[x])
    S3 = greedy_solution(order3)
    if len(S3) > best_size:
        best_S = set(S3)
        best_size = len(S3)

    # Order 4: by conflict degree descending (most constrained first)
    order4 = sorted(range(1, N + 1), key=lambda x: -conflict_degree[x])
    S4 = greedy_solution(order4)
    if len(S4) > best_size:
        best_S = set(S4)
        best_size = len(S4)

    # Random orderings
    import random
    for _ in range(30):
        order_r = list(range(1, N + 1))
        random.shuffle(order_r)
        Sr = greedy_solution(order_r)
        if len(Sr) > best_size:
            best_S = set(Sr)
            best_size = len(Sr)

    # Local search: try to add elements
    S = set(best_S)
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        for y in range(1, N + 1):
            if y not in S and can_add(y, S):
                S.add(y)
                improved = True

    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Branch and bound for improvement
    # Process elements in order of conflict degree (most constrained first for better pruning)
    element_order = sorted(range(1, N + 1), key=lambda x: -conflict_degree[x])
    
    lower_bound = best_size
    best_found = set(best_S)
    best_found_size = best_size
    nodes_explored = 0
    max_nodes = 2000000

    def branch_and_bound(pos, current_set, remaining_count):
        nonlocal best_found, best_found_size, lower_bound, nodes_explored

        if time.time() > deadline - 5:
            return

        if pos >= len(element_order):
            if len(current_set) > best_found_size:
                best_found_size = len(current_set)
                best_found = set(current_set)
                lower_bound = best_found_size
            return

        # Upper bound: current size + remaining elements
        if len(current_set) + remaining_count <= lower_bound:
            return

        if nodes_explored > max_nodes:
            return

        x = element_order[pos]
        nodes_explored += 1

        # Try including x
        if can_add(x, current_set):
            current_set.add(x)
            branch_and_bound(pos + 1, current_set, remaining_count - 1)
            current_set.discard(x)

        # Try excluding x
        branch_and_bound(pos + 1, current_set, remaining_count - 1)

    # Run B&B with time limit
    if time.time() < deadline - 20:
        branch_and_bound(0, set(), N)

    if len(best_found) > best_size:
        best_S = set(best_found)
        best_size = len(best_found)

    # Final local search on best found
    S = set(best_S)
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for y in range(1, N + 1):
            if y not in S and can_add(y, S):
                S.add(y)
                improved = True

    if len(S) > len(best_S):
        best_S = set(S)

    return list(best_S)
