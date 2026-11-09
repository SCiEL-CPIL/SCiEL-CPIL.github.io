import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    def time_left():
        return DEADLINE - (time.time() - start_time)

    def check_subsets_with_new(masks, n, new_v):
        """Check if adding vertex new_v creates any regular induced subgraph of size >= 6.
        Only checks subsets that include new_v."""
        for size in range(6, n + 1):
            # Subsets of size 'size' that include new_v
            # Choose size-1 other vertices from 0..new_v-1 and 0..n-1 excluding new_v
            others = [v for v in range(n) if v != new_v]
            for subset_others in combinations(others, size - 1):
                subset = tuple(sorted(subset_others + (new_v,)))
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = [(masks[v] & mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    return subset
        return None

    def greedy_construct(n_target, seed):
        """Build graph vertex by vertex, ensuring no regular induced subgraph."""
        rng = np.random.default_rng(seed)
        masks = [0] * n_target

        for v in range(n_target):
            if time_left() < 2.0:
                return None

            # Try all possible adjacency patterns for vertex v
            # To previous vertices 0..v-1
            num_prev = v
            # Try random orderings of possible edges
            possible_edges = list(range(num_prev))
            rng.shuffle(possible_edges)

            found = False
            # Try adding edges one at a time in random order
            # Actually, try all 2^v possibilities? No, too many for large v.
            # Instead, try random subsets of edges to previous vertices
            
            max_attempts = 200
            for attempt in range(max_attempts):
                # Choose a random subset of edges to previous vertices
                mask_v = 0
                for u in range(v):
                    if rng.random() < 0.5:
                        mask_v |= (1 << u)

                # Tentatively add vertex v with this adjacency
                masks[v] = mask_v
                for u in range(v):
                    if (mask_v >> u) & 1:
                        masks[u] |= (1 << v)
                    else:
                        masks[u] &= ~(1 << v)

                # Check if this creates a regular induced subgraph
                violation = check_subsets_with_new(masks, v + 1, v)
                if violation is None:
                    found = True
                    break
                else:
                    # Revert
                    for u in range(v):
                        if (mask_v >> u) & 1:
                            masks[u] &= ~(1 << v)

            if not found:
                return None

        # Final verification
        for size in range(6, n_target + 1):
            for subset in combinations(range(n_target), size):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                degs = [(masks[v] & mask).bit_count() for v in subset]
                if len(set(degs)) == 1:
                    return None

        return masks

    best_n = 0
    best_masks = None

    # Try to construct graphs of size 20, 21, 22
    for n_target in [20, 21, 22]:
        if time_left() < 10.0:
            break

        found = False
        for seed in range(100):
            if found or time_left() < 3.0:
                break
            masks = greedy_construct(n_target, seed)
            if masks is not None:
                best_n = n_target
                best_masks = masks
                found = True
                break

        if found:
            continue

    # Fallback
    if best_n == 0:
        for n_target in range(19, 5, -1):
            if time_left() < 3.0:
                break
            found = False
            for seed in range(50):
                if found or time_left() < 2.0:
                    break
                masks = greedy_construct(n_target, seed)
                if masks is not None:
                    best_n = n_target
                    best_masks = masks
                    found = True
                    break
            if found:
                break

    if best_masks is not None:
        n = len(best_masks)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (best_masks[i] >> j) & 1:
                    edges.append([i, j])
        return {'n': n, 'edges': edges}

    return {'n': 5, 'edges': []}
