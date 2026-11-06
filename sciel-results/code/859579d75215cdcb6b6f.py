import time
import random
import sys
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill

    N = 256
    all_nodes = list(range(1, N + 1))

    # 1. Build the conflict graph
    adj = [set() for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            if (a * b) % (a + b) == 0:
                adj[a].add(b)
                adj[b].add(a)

    # Precompute degrees
    degs = [0] + [len(adj[i]) for i in range(1, N + 1)]

    def greedy_min_degree(nodes):
        """Greedy independent set: pick min-degree node repeatedly."""
        available = set(nodes)
        independent_set = set()
        while available:
            if time.time() > deadline:
                break
            min_deg = float('inf')
            selected = None
            for node in available:
                d = len(adj[node] & available)
                if d < min_deg:
                    min_deg = d
                    selected = node
            if selected is None:
                break
            independent_set.add(selected)
            available.difference_update(adj[selected])
            available.discard(selected)
        return independent_set

    def greedy_max_degree(nodes):
        """Greedy independent set: pick max-degree node repeatedly."""
        available = set(nodes)
        independent_set = set()
        while available:
            if time.time() > deadline:
                break
            max_deg = -1
            selected = None
            for node in available:
                d = len(adj[node] & available)
                if d > max_deg:
                    max_deg = d
                    selected = node
            if selected is None:
                break
            independent_set.add(selected)
            available.difference_update(adj[selected])
            available.discard(selected)
        return independent_set

    def greedy_random(nodes):
        """Greedy independent set: pick random node repeatedly."""
        available = set(nodes)
        independent_set = set()
        while available:
            if time.time() > deadline:
                break
            selected = random.choice(list(available))
            independent_set.add(selected)
            available.difference_update(adj[selected])
            available.discard(selected)
        return independent_set

    def greedy_weighted(nodes, weights):
        """Greedy independent set: pick node with best weight/degree ratio."""
        available = set(nodes)
        independent_set = set()
        while available:
            if time.time() > deadline:
                break
            best_ratio = -1
            selected = None
            for node in available:
                d = len(adj[node] & available)
                if d == 0:
                    ratio = float('inf')
                else:
                    ratio = weights[node] / d
                if ratio > best_ratio:
                    best_ratio = ratio
                    selected = node
            if selected is None:
                break
            independent_set.add(selected)
            available.difference_update(adj[selected])
            available.discard(selected)
        return independent_set

    def local_search_improve(S, max_iters=200):
        """Try to improve S by adding nodes or swapping."""
        S = set(S)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            if time.time() > deadline:
                break
            improved = False
            iters += 1
            # Try adding a node
            for node in all_nodes:
                if node in S:
                    continue
                if not (adj[node] & S):
                    S.add(node)
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one node, add one or more
            for remove_node in list(S):
                if time.time() > deadline:
                    break
                S_temp = S - {remove_node}
                # Find nodes that can be added
                addable = [n for n in all_nodes if n not in S_temp and not (adj[n] & S_temp)]
                if len(addable) > 1:
                    # Add multiple nodes
                    S = S_temp
                    for n in addable:
                        if not (adj[n] & S):
                            S.add(n)
                    improved = True
                    break
        return S

    def hill_climbing(S, max_iters=500):
        """Hill climbing: try single swaps to improve size."""
        S = set(S)
        best_size = len(S)
        for it in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try adding
            for node in all_nodes:
                if node in S:
                    continue
                if not (adj[node] & S):
                    S.add(node)
                    best_size = len(S)
                    improved = True
                    break
            if improved:
                continue
            # Try swap: remove a, add b (net 0), but might enable more additions
            for remove_node in list(S):
                if time.time() > deadline:
                    break
                S_without = S - {remove_node}
                # What can we add now?
                new_addable = [n for n in all_nodes if n not in S_without and not (adj[n] & S_without)]
                # If removing one node lets us add 2+, that's a win
                if len(new_addable) >= 2:
                    S = S_without
                    added = 0
                    for n in new_addable:
                        if not (adj[n] & S):
                            S.add(n)
                            added += 1
                    if len(S) > best_size:
                        best_size = len(S)
                        improved = True
                        break
        return S

    # Multi-strategy search
    best_set = set()
    best_size = 0

    # Phase 1: Try various greedy strategies
    strategies = [
        lambda: greedy_min_degree(all_nodes),
        lambda: greedy_max_degree(all_nodes),
        lambda: greedy_random(all_nodes),
    ]

    # Weighted greedy with different weight functions
    weight_funcs = [
        lambda n: 1.0,  # uniform
        lambda n: 1.0 / (degs[n] + 1),  # prefer low degree
        lambda n: n,  # prefer large numbers
        lambda n: 256 - n,  # prefer small numbers
        lambda n: 1.0 if degs[n] == 0 else 0.5,  # prefer isolated
    ]

    for _ in range(30):
        if time.time() > deadline:
            break
        for strat in strategies:
            if time.time() > deadline:
                break
            S = strat()
            if len(S) > best_size:
                best_size = len(S)
                best_set = S
        for wf in weight_funcs:
            if time.time() > deadline:
                break
            weights = [0] + [wf(n) for n in all_nodes]
            S = greedy_weighted(all_nodes, weights)
            if len(S) > best_size:
                best_size = len(S)
                best_set = S

    # Phase 2: Local search / hill climbing on best found
    if time.time() < deadline - 5:
        improved = local_search_improve(best_set, max_iters=100)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved

    # Phase 3: More hill climbing with random restarts
    for _ in range(20):
        if time.time() > deadline - 5:
            break
        # Start from a random greedy solution
        S = greedy_random(all_nodes)
        # Improve it
        S = hill_climbing(S, max_iters=200)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S

    # Phase 4: Final polish on the best set
    if time.time() < deadline - 2:
        final = local_search_improve(best_set, max_iters=300)
        if len(final) > best_size:
            best_size = len(final)
            best_set = final

    return list(best_set)
