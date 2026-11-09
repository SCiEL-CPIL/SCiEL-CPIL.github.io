"""Rewritten program: maximum independent set on square-sum conflict graph."""
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin before 90s kill

    N = 256

    # Perfect squares up to 2*256 = 512
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1

    # Valid nodes: 2*v is NOT a perfect square (diagonal constraint)
    valid_nodes = []
    for v in range(1, N + 1):
        if 2 * v not in squares:
            valid_nodes.append(v)

    node_set = set(valid_nodes)
    node_index = {v: i for i, v in enumerate(valid_nodes)}
    m = len(valid_nodes)

    # Build conflict adjacency: conflicts[i] = set of indices j where valid_nodes[i]+valid_nodes[j] is a square
    conflicts = [set() for _ in range(m)]
    for i_idx in range(m):
        v = valid_nodes[i_idx]
        for j_idx in range(i_idx + 1, m):
            u = valid_nodes[j_idx]
            if v + u in squares:
                conflicts[i_idx].add(j_idx)
                conflicts[j_idx].add(i_idx)

    def greedy_mis():
        """Greedy: repeatedly pick node with fewest conflicts among remaining."""
        remaining = set(range(m))
        chosen = set()
        while remaining:
            # Pick node with minimum degree in remaining subgraph
            best = min(remaining, key=lambda x: len(conflicts[x] & remaining))
            chosen.add(best)
            remaining.discard(best)
            remaining -= conflicts[best]
        return chosen

    def local_search(seed_set):
        """Try to improve by adding nodes or swapping."""
        chosen = set(seed_set)
        improved = True
        while improved:
            improved = False
            # Try adding each excluded node
            for x in range(m):
                if x not in chosen:
                    if not (conflicts[x] & chosen):
                        chosen.add(x)
                        improved = True
            # Try removing a node to allow two others
            for x in list(chosen):
                if len(chosen) <= 1:
                    continue
                # Check if removing x allows adding more
                without_x = chosen - {x}
                addable = set()
                for y in range(m):
                    if y not in without_x and not (conflicts[y] & without_x):
                        addable.add(y)
                if len(addable) > 1:
                    chosen = without_x | addable
                    improved = True
        return chosen

    def random_restart_mis():
        """Randomized greedy with different orderings."""
        order = list(range(m))
        random.shuffle(order)
        remaining = set(range(m))
        chosen = set()
        for x in order:
            if x not in remaining:
                continue
            if not (conflicts[x] & chosen):
                chosen.add(x)
                remaining.discard(x)
                remaining -= conflicts[x]
        return chosen

    # Initial solution
    best = greedy_mis()
    best = local_search(best)
    best_size = len(best)

    # Multiple restarts with randomization
    while time.time() < deadline - 2:
        candidate = random_restart_mis()
        candidate = local_search(candidate)
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)

    # Convert back to values
    result = [valid_nodes[i] for i in sorted(best)]

    # Final verification
    for i in range(len(result)):
        a = result[i]
        if 2 * a in squares:
            raise ValueError(f"Diagonal violation: {a}")
        for j in range(i, len(result)):
            b = result[j]
            if a + b in squares:
                raise ValueError(f"Sum violation: {a}+{b}")

    return result

# EVOLVE-BLOCK-END
