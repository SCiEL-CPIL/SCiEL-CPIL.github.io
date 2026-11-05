import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s limit

    N = 180
    all_elements = set(range(1, N + 1))

    # Precompute all forbidden triples (a, b, c) with a < b < c and 1/a = 1/b + 1/c
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

    # Precompute for each element x, the set of "partner" elements it conflicts with
    # If x is added, and both partners are already in S, we have a triple
    # For x as 'small': conflict with pairs (b,c) where (x,b,c) is a triple
    # For x as 'large': if x is b, conflict with (a,c) where (a,x,c) is a triple
    #                  if x is c, conflict with (a,b) where (a,b,x) is a triple
    
    # Precompute: for each x, list of (a, y) such that adding x with a,y in S creates triple
    # If x is small: for each (b,c) in triples_by_a[x], if b in S and c in S -> conflict
    # If x is large (as b): for each a < x, c = ax/(x-a), if a in S and c in S -> conflict
    # If x is large (as c): for each a < x, b = ax/(x-a), if a in S and b in S -> conflict

    # For fast checking, precompute for each x the list of "witness" pairs
    # witness_pairs[x] = list of (a, b) such that {x, a, b} forms a forbidden triple
    # When adding x to S, we need to check: for any (a,b) in witness_pairs[x], not (a in S and b in S)

    witness_pairs = defaultdict(set)  # x -> set of frozenset({a,b})
    for (a, b, c) in triples:
        # a is small, b and c are large
        witness_pairs[a].add(frozenset((b, c)))
        witness_pairs[b].add(frozenset((a, c)))
        witness_pairs[c].add(frozenset((a, b)))

    # Convert to list for faster iteration
    witness_list = {}
    for x in range(1, N + 1):
        witness_list[x] = list(witness_pairs.get(x, set()))

    def can_add(x, S_set):
        """Check if x can be added to S_set without creating a forbidden triple."""
        for pair in witness_list[x]:
            # pair is frozenset of two elements
            # Check if both are in S_set
            it = iter(pair)
            e1 = next(it)
            e2 = next(it)
            if e1 in S_set and e2 in S_set:
                return False
        return True

    # Greedy initial solution
    def greedy_solution(order):
        S_set = set()
        for x in order:
            if can_add(x, S_set):
                S_set.add(x)
        return S_set

    # Try multiple greedy orderings
    best_S = set()
    best_size = 0

    # Order 1: descending (prefer large numbers)
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

    # Order 3: random shuffles
    for _ in range(30):
        order3 = list(range(1, N + 1))
        random.shuffle(order3)
        S3 = greedy_solution(order3)
        if len(S3) > best_size:
            best_S = set(S3)
            best_size = len(S3)

    # Order 4: sorted by number of conflicts (most constrained first)
    conflict_count = {x: len(witness_list[x]) for x in range(1, N + 1)}
    order4 = sorted(range(1, N + 1), key=lambda x: -conflict_count[x])
    S4 = greedy_solution(order4)
    if len(S4) > best_size:
        best_S = set(S4)
        best_size = len(S4)

    # Local search: try to add elements
    S = set(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for y in range(1, N + 1):
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    improved = True

    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Swap-based local search: remove one element, try to add two
    # This can help escape local optima
    improved = True
    while improved and time.time() < deadline - 15:
        improved = False
        S_list = list(S)
        for i, rem in enumerate(S_list):
            if time.time() > deadline - 15:
                break
            S_temp = set(S)
            S_temp.discard(rem)
            # Try to add elements
            added = []
            for y in range(1, N + 1):
                if y not in S_temp and y != rem:
                    if can_add(y, S_temp):
                        S_temp.add(y)
                        added.append(y)
            if len(S_temp) > len(S):
                S = S_temp
                improved = True
                break

    if len(S) > best_size:
        best_S = set(S)
        best_size = len(S)

    # Branch and bound for improvement
    lower_bound = best_size

    # Order elements by conflict degree (most constrained first)
    element_order = sorted(range(1, N + 1), key=lambda x: -len(witness_list[x]))

    best_found = set(best_S)
    best_found_size = best_size
    nodes_explored = 0
    max_nodes = 1000000

    def branch_and_bound(pos, current_set, remaining_count):
        nonlocal best_found, best_found_size, nodes_explored, lower_bound

        if time.time() > deadline - 5:
            return

        if pos >= len(element_order):
            if len(current_set) > best_found_size:
                best_found_size = len(current_set)
                best_found = set(current_set)
                lower_bound = best_found_size
            return

        # Upper bound
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
    if time.time() < deadline - 10:
        branch_and_bound(0, set(), N)

    # Final local search on best found
    S = set(best_found)
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for y in range(1, N + 1):
            if y not in S:
                if can_add(y, S):
                    S.add(y)
                    improved = True

    if len(S) > len(best_found):
        best_found = set(S)

    return list(best_found)
