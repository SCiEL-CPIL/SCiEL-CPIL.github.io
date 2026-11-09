"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    d = 7
    n = 1 << d  # 128
    total_edges = n * d // 2  # 448

    # Build edge list and index
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx

    # For each edge, precompute the set of edges that would form a C4 with it
    # Edge (x,y) along dimension i: for each other dim j, the opposite edge is (x^j, y^j)
    # A C4 is formed by edge (x,y) along dim i and edge (c,e) along dim j where
    # c = x^j, e = y^j, and edges (x,c) along dim j and (y,e) along dim j also exist.
    # Actually, a 2-face has 4 edges: two parallel edges in dim i and two in dim j.
    # Edge (x,y) is in dim i. The C4 it participates in with dim j has edges:
    #   (x, y) [dim i], (x^j, y^j) [dim i], (x, x^j) [dim j], (y, y^j) [dim j]
    # So edge (x,y) conflicts with edge (x^j, y^j) if both (x,x^j) and (y,y^j) are selected.
    # But for local search, we need: if we add edge (x,y), which existing edges create a C4?
    # If (x,y) is added, a C4 is formed if there exists j != i such that
    #   (x^j, y^j) is in selected, (x, x^j) is in selected, (y, y^j) is in selected.
    # So (x,y) conflicts with (x^j, y^j) given the other two edges are present.
    # For efficient checking, for each edge we can precompute the list of (other_edge, dim_i, dim_j)
    # such that together with two more edges they form a face.

    # Precompute for each edge: list of (parallel_edge_idx, edge1_idx, edge2_idx)
    # where adding this edge + parallel_edge + edge1 + edge2 forms a C4
    edge_conflicts = [[] for _ in range(total_edges)]

    for idx, (x, y) in enumerate(all_edges):
        # Determine dimension
        diff = x ^ y
        i = diff.bit_length() - 1
        for j in range(d):
            if j == i:
                continue
            xj = x ^ (1 << j)
            yj = y ^ (1 << j)
            # The parallel edge in dim i is (min(xj,yj), max(xj,yj))
            px, py = min(xj, yj), max(xj, yj)
            pe = (px, py)
            if pe not in edge_index:
                continue
            pidx = edge_index[pe]
            # Edge (x, xj) in dim j
            e1 = (min(x, xj), max(x, xj))
            e1idx = edge_index.get(e1)
            # Edge (y, yj) in dim j
            e2 = (min(y, yj), max(y, yj))
            e2idx = edge_index.get(e2)
            if e1idx is not None and e2idx is not None:
                edge_conflicts[idx].append((pidx, e1idx, e2idx))

    # Greedy initialization
    def greedy_init():
        selected = set()
        order = list(range(total_edges))
        random.shuffle(order)
        for idx in order:
            x, y = all_edges[idx]
            # Check if adding this edge creates a C4
            ok = True
            for (pidx, e1idx, e2idx) in edge_conflicts[idx]:
                if pidx in selected and e1idx in selected and e2idx in selected:
                    ok = False
                    break
            if ok:
                selected.add(idx)
        return selected

    # Local search: try to add removed edges or swap
    def local_search(selected, max_iters=20000):
        removed = [i for i in range(total_edges) if i not in selected]
        random.shuffle(removed)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            if time.time() > deadline - 5.0:
                break
            for idx in removed:
                if idx in selected:
                    continue
                # Check if we can add idx
                can_add = True
                for (pidx, e1idx, e2idx) in edge_conflicts[idx]:
                    if pidx in selected and e1idx in selected and e2idx in selected:
                        can_add = False
                        break
                if can_add:
                    selected.add(idx)
                    removed.remove(idx)
                    improved = True
        return selected

    # Best-first local search with multiple restarts
    best_selected = None
    best_count = 0

    # First greedy
    selected = greedy_init()
    selected = local_search(selected)
    best_selected = set(selected)
    best_count = len(selected)

    # Multiple restarts
    while time.time() < deadline - 10.0:
        selected = greedy_init()
        selected = local_search(selected, max_iters=5000)
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = set(selected)

        # Try 1-flip moves: remove one edge, try to add two
        if time.time() < deadline - 10.0:
            cur = set(best_selected)
            for _ in range(200):
                if time.time() > deadline - 10.0:
                    break
                # Randomly remove an edge
                rem = random.choice(list(cur))
                cur.discard(rem)
                # Try to add edges
                for idx in range(total_edges):
                    if idx in cur:
                        continue
                    can_add = True
                    for (pidx, e1idx, e2idx) in edge_conflicts[idx]:
                        if pidx in cur and e1idx in cur and e2idx in cur:
                            can_add = False
                            break
                    if can_add:
                        cur.add(idx)
                        if len(cur) > best_count:
                            best_count = len(cur)
                            best_selected = set(cur)

    # Final local search on best
    if time.time() < deadline - 5.0:
        best_selected = local_search(best_selected, max_iters=10000)
        best_count = len(best_selected)

    # Build output
    result = []
    for idx in best_selected:
        x, y = all_edges[idx]
        result.append([x, y])

    return result

# EVOLVE-BLOCK-END
