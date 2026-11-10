import time
import random

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin
    N = 128

    # Pre-compute forbidden hyperedges
    edges = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            max_c = min(N - b, N - a)
            if max_c <= b:
                continue
            for c in range(b + 1, max_c + 1):
                s1 = a + b
                s2 = a + c
                s3 = b + c
                v = frozenset({a, b, c, s1, s2, s3})
                edges.append(v)

    unique_edges = list(set(edges))

    # Build index: element -> list of edge indices containing it
    elem_to_edges = {i: [] for i in range(1, N + 1)}
    for i, e in enumerate(unique_edges):
        for v in e:
            elem_to_edges[v].append(i)

    def is_valid(S_set):
        for e in unique_edges:
            if e.issubset(S_set):
                return False
        return True

    def check_add(S_set, x):
        """Check if adding x to S_set creates any violation. Returns True if valid."""
        for idx in elem_to_edges[x]:
            e = unique_edges[idx]
            if e.difference({x}).issubset(S_set):
                return False
        return True

    def greedy_add(S, candidates, rng):
        """Iteratively add elements from candidates to S if valid."""
        improved = True
        while improved:
            improved = False
            for x in list(candidates):
                if x in S:
                    candidates.remove(x)
                    continue
                if check_add(S, x):
                    S.add(x)
                    candidates.remove(x)
                    improved = True
        return S

    def local_search(S, rng):
        """Try to improve S via swaps and re-additions."""
        best_S = set(S)
        best_size = len(S)
        candidates = [x for x in range(1, N + 1) if x not in S]
        rng.shuffle(candidates)

        # First, do a full greedy pass
        greedy_add(S, candidates, rng)
        if len(S) > best_size:
            best_size = len(S)
            best_S = set(S)

        # Now try swaps to potentially enable more additions
        not_in_S = [x for x in range(1, N + 1) if x not in S]
        
        for _ in range(200):
            if time.time() > deadline:
                break
            if not not_in_S:
                break
                
            x = rng.choice(list(S))
            y = rng.choice(not_in_S)
            if y in S:
                continue

            # Try swap: remove x, add y
            S.remove(x)
            if check_add(S, y):
                S.add(y)
                # After swap, try to add more
                remaining = [z for z in not_in_S if z != y and z not in S]
                greedy_add(S, remaining, rng)
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = set(S)
                    not_in_S = [z for z in range(1, N + 1) if z not in S]
            else:
                # Swap invalid, revert
                S.add(x)
                continue

        return best_S

    best_set = set(range(N // 2 + 1, N + 1))
    best_size = len(best_set)

    rng = random.Random(42)

    # Multiple restarts with different strategies
    for restart in range(30):
        if time.time() > deadline:
            break

        if restart == 0:
            # Start with top half
            S = set(range(N // 2 + 1, N + 1))
        elif restart == 1:
            # Start with top 60%
            S = set(range(int(N * 0.4) + 1, N + 1))
        elif restart == 2:
            # Start with top 55%
            S = set(range(int(N * 0.45) + 1, N + 1))
        else:
            # Random subset of size ~65
            S = set(rng.sample(range(1, N + 1), 65))

        # Ensure S is valid by removing elements that cause violations
        # Simple approach: check and remove
        for x in list(S):
            S_temp = S - {x}
            if not is_valid(S_temp):
                # x is not the problem, keep it
                continue
            # Actually, we need to check if S is valid first
            pass

        # Make S valid: remove elements one by one until valid
        while not is_valid(S):
            # Find a violating edge
            for e in unique_edges:
                if e.issubset(S):
                    # Remove a random element from this edge
                    victim = rng.choice(list(e))
                    S.discard(victim)
                    break
            else:
                break

        # Now do local search
        S_improved = local_search(S, rng)
        if len(S_improved) > best_size:
            best_size = len(S_improved)
            best_set = S_improved

    # Final verification
    if not is_valid(best_set):
        best_set = set(range(N // 2 + 1, N + 1))
        best_size = len(best_set)

    return [int(x) for x in best_set]
