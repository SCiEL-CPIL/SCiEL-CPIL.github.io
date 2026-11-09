"""Construct a dense C4-rainbow edge-coloured graph on 20 vertices."""
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill
    n = 20
    num_colours = 20

    def elapsed():
        return time.time() - start

    def time_ok():
        return elapsed() < deadline - 5.0

    def build_greedy(seed):
        rng = random.Random(seed)
        edge_color = {}  # (u,v) with u<v -> colour
        adj = {i: {} for i in range(n)}  # adj[u][v] = colour

        def add_edge(u, v, c):
            if u > v:
                u, v = v, u
            edge_color[(u, v)] = c
            adj[u][v] = c
            adj[v][u] = c

        def remove_edge(u, v):
            if u > v:
                u, v = v, u
            del edge_color[(u, v)]
            del adj[u][v]
            del adj[v][u]

        def is_valid_add(u, v, c):
            # Check if adding edge (u,v) with colour c creates a non-rainbow 4-cycle.
            nv = adj[v]
            nu = adj[u]
            
            for w, cw in nv.items():
                if w == u:
                    continue
                for x, cxu in nu.items():
                    if x == v or x == w:
                        continue
                    if x in adj[w]:
                        cwx = adj[w][x]
                        if len({c, cw, cwx, cxu}) != 4:
                            return False
            return True

        # Try to add all possible edges, choosing colour that works.
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        
        # Use a smarter ordering: prioritize edges that are "easier" to color
        # by trying vertices in random order but with some structure
        rng.shuffle(all_pairs)
        
        for (u, v) in all_pairs:
            if not time_ok():
                break
            # Try each colour, pick first that works
            cols = list(range(num_colours))
            rng.shuffle(cols)
            placed = False
            for c in cols:
                if is_valid_add(u, v, c):
                    add_edge(u, v, c)
                    placed = True
                    break
            # if no colour works, skip this edge
        return edge_color

    def local_improve(edge_color, max_iter=100):
        """Try to add missing edges or re-color existing ones to increase density."""
        if not edge_color:
            return edge_color
        
        adj = {i: {} for i in range(n)}
        for (u, v), c in edge_color.items():
            adj[u][v] = c
            adj[v][u] = c
        
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            if not time_ok():
                break
            
            # Get all missing edges
            all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            missing = [(u, v) for (u, v) in all_pairs if (u, v) not in edge_color]
            
            rng = random.Random(iter_count + 42)
            rng.shuffle(missing)
            
            for (u, v) in missing:
                if not time_ok():
                    break
                # Try each colour
                cols = list(range(num_colours))
                rng.shuffle(cols)
                for c in cols:
                    # Check validity
                    nv = adj[v]
                    nu = adj[u]
                    valid = True
                    for w, cw in nv.items():
                        if w == u:
                            continue
                        for x, cxu in nu.items():
                            if x == v or x == w:
                                continue
                            if x in adj[w]:
                                cwx = adj[w][x]
                                if len({c, cw, cwx, cxu}) != 4:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if valid:
                        edge_color[(u, v)] = c
                        adj[u][v] = c
                        adj[v][u] = c
                        improved = True
                        break
        
        return edge_color

    # Try multiple seeds with different strategies
    best = None
    best_count = -1
    
    # First pass: standard greedy with various seeds
    for seed in range(200):
        if not time_ok():
            break
        ec = build_greedy(seed)
        cnt = len(ec)
        if cnt > best_count:
            best_count = cnt
            best = ec
        if best_count >= 190:
            break

    # Second pass: local improvement on best result
    if best is not None and time_ok():
        improved = local_improve(best.copy())
        if len(improved) > best_count:
            best = improved
            best_count = len(best)

    # Third pass: try more seeds if time permits
    if time_ok():
        for seed in range(200, 500):
            if not time_ok():
                break
            ec = build_greedy(seed)
            cnt = len(ec)
            if cnt > best_count:
                best_count = cnt
                best = ec
            if best_count >= 190:
                break

    # Final improvement pass
    if best is not None and time_ok():
        improved = local_improve(best.copy())
        if len(improved) > best_count:
            best = improved
            best_count = len(best)

    if best is None:
        # fallback: star
        return [[0, v, 0] for v in range(1, n)]

    result = [[u, v, c] for (u, v), c in best.items()]
    return result
# EVOLVE-BLOCK-END
