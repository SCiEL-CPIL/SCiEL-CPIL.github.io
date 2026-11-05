import time
import random
import heapq
from itertools import combinations

def construct():
    """
    Returns a dict {'n': n, 'edges': [[v1, ..., v5], ...]}
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    best_edges = None
    best_n = 9

    for n in [9, 10, 11]:
        if time.time() > deadline - 5.0:
            break

        all_edges = list(combinations(range(n), 5))
        num_edges = len(all_edges)
        num_masks = 1 << n

        # Precompute relationships
        mask_edges = [[] for _ in range(num_masks)]
        edge_masks = [[] for _ in range(num_edges)]

        for m in range(num_masks):
            for i, e in enumerate(all_edges):
                bit = (m >> e[0]) & 1
                mono = True
                for v in e[1:]:
                    if ((m >> v) & 1) != bit:
                        mono = False
                        break
                if mono:
                    mask_edges[m].append(i)
                    edge_masks[i].append(m)

        best_local = None

        for attempt in range(100):
            if time.time() > deadline - 5.0:
                break

            # Greedy Set Cover with dynamic counts and randomization
            uncovered = set(range(num_masks))
            cover_count = [len(edge_masks[i]) for i in range(num_edges)]
            selected = []

            while uncovered:
                if time.time() > deadline - 2.0:
                    break

                # Find top 5 edges with highest cover_count
                # Using heapq.nlargest is efficient enough for E~250
                top_edges = heapq.nlargest(5, range(num_edges), key=lambda i: cover_count[i])
                
                if not top_edges or cover_count[top_edges[0]] == 0:
                    break # No progress possible

                # Pick one randomly from top 5
                best_edge = random.choice(top_edges)
                selected.append(best_edge)

                # Update counts
                for m in edge_masks[best_edge]:
                    if m in uncovered:
                        uncovered.discard(m)
                        for e in mask_edges[m]:
                            cover_count[e] -= 1

                if not uncovered:
                    break

            if uncovered:
                continue

            # Prune
            current = list(selected)
            random.shuffle(current)
            i = 0
            while i < len(current):
                if time.time() > deadline - 1.0:
                    break
                # Try removing current[i]
                valid = True
                rem_set = set(current[:i] + current[i+1:])
                
                for m in range(num_masks):
                    hit = False
                    for e in mask_edges[m]:
                        if e in rem_set:
                            hit = True
                            break
                    if not hit:
                        valid = False
                        break
                
                if valid:
                    current = current[:i] + current[i+1:]
                else:
                    i += 1
            
            if best_local is None or len(current) < len(best_local):
                best_local = current

        if best_local is not None:
            if best_edges is None or len(best_local) < len(best_edges):
                best_edges = [all_edges[i] for i in best_local]
                best_n = n

    # Fallback
    if best_edges is None:
        n = 9
        best_edges = list(combinations(range(n), 5))
        best_n = n

    return {'n': best_n, 'edges': best_edges}
