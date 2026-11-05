import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s limit

    n = 16
    target_edges = 65

    # Fast evaluator using bitsets for adjacency and precomputed degrees.
    # We store adjacency as Python ints (16-bit masks) for speed.
    # Edge set is a set of (u, v) with u < v.

    def build_adj_and_deg(edges):
        adj = [0] * n
        deg = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    def max_triangle_sum(edges):
        if len(edges) != target_edges:
            return 999
        adj, deg = build_adj_and_deg(edges)
        max_s = 0
        # Enumerate triangles by (u, v, w) with u < v < w using bitsets.
        for u in range(n):
            # neighbors of u with index > u
            nu = adj[u] & ~((1 << (u + 1)) - 1)
            # iterate over v in nu
            # We can iterate over set bits efficiently.
            v_bits = nu
            while v_bits:
                # lowest set bit
                lsb = v_bits & (-v_bits)
                v = lsb.bit_length() - 1
                v_bits ^= lsb
                # common neighbors of u and v with index > v
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                if not common:
                    continue
                # For each w in common, compute sum
                w_bits = common
                while w_bits:
                    lsb_w = w_bits & (-w_bits)
                    w = lsb_w.bit_length() - 1
                    w_bits ^= lsb_w
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_s:
                        max_s = s
                        # Early exit if we already know it's bad? 
                        # But we need exact max. We can break if max_s >= 26? 
                        # No, we want to minimize, so if we see 26 we might still find 25 elsewhere.
                        # However, if we are searching for < 26, we can early exit if we find 26? 
                        # No, we need to know if there's a better one.
                        # But for local search, if current is 26, we accept moves that keep it <=26 or improve.
                        # Actually, if max_s reaches a high value, we can stop evaluating this graph? 
                        # No, we need the exact max to compare.
                        # But if we are only interested in whether it's < current_best, we can early exit.
                        # Let's add an optional threshold.
        return max_s

    # Improved evaluator with threshold for early exit
    def max_triangle_sum_threshold(edges, threshold=999):
        if len(edges) != target_edges:
            return 999
        adj, deg = build_adj_and_deg(edges)
        max_s = 0
        for u in range(n):
            nu = adj[u] & ~((1 << (u + 1)) - 1)
            v_bits = nu
            while v_bits:
                lsb = v_bits & (-v_bits)
                v = lsb.bit_length() - 1
                v_bits ^= lsb
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                if not common:
                    continue
                w_bits = common
                while w_bits:
                    lsb_w = w_bits & (-w_bits)
                    w = lsb_w.bit_length() - 1
                    w_bits ^= lsb_w
                    s = deg[u] + deg[v] + deg[w]
                    if s > max_s:
                        max_s = s
                        if max_s >= threshold:
                            return max_s
        return max_s

    # Initial graph: K_{8,8} + one internal edge (0,1)
    # This is the baseline with score 26.
    base_edges = set()
    for u in range(8):
        for v in range(8, 16):
            base_edges.add((u, v))
    base_edges.add((0, 1))
    
    current_edges = set(base_edges)
    current_score = max_triangle_sum(current_edges)
    best_edges = set(current_edges)
    best_score = current_score

    # Precompute all possible cross edges and internal edges
    cross_edges = [(u, v) for u in range(8) for v in range(8, 16)]
    internal_A = [(u, v) for u in range(8) for v in range(u+1, 8)]
    internal_B = [(u, v) for u in range(8, 16) for v in range(u+1, 16)]
    
    # Local search
    # Moves:
    # 1. Swap a cross edge with an internal edge (keep 65 edges)
    # 2. Swap two internal edges (keep 65 edges)
    
    random.seed(42)
    no_improve_count = 0
    max_no_improve = 5000  # stop if no improvement for a while
    
    while time.time() < deadline and no_improve_count < max_no_improve:
        improved = False
        
        # Try random swaps
        # Type 1: Remove cross edge (c1, c2), add internal edge (i1, i2)
        # We need to ensure the internal edge is not already present and cross edge is present.
        # Also, the resulting graph must be simple (it will be, since we're swapping distinct types).
        
        # To be efficient, we can pick a random cross edge to remove and a random internal edge to add.
        # But we must ensure the internal edge is not already in the graph.
        
        # Let's pick a random cross edge from the current graph.
        current_cross = [e for e in current_edges if e[0] < 8 and e[1] >= 8]
        current_internal = [e for e in current_edges if (e[0] < 8 and e[1] < 8) or (e[0] >= 8 and e[1] >= 8)]
        
        if not current_cross or not current_internal:
            # If no internal edges, we can't swap internal for internal.
            # But we can swap cross for internal?
            # If no internal, we add one internal, remove one cross.
            pass
        
        # Strategy: Randomly choose a move type
        move_type = random.random()
        
        if move_type < 0.7:
            # Swap cross for internal
            if not current_cross:
                continue
            remove_edge = random.choice(current_cross)
            # Choose an internal edge not in current_edges
            # To avoid O(N) check, we can just pick random and check
            found = False
            for _ in range(100): # Try a few times
                if random.random() < 0.5:
                    add_edge = random.choice(internal_A)
                else:
                    add_edge = random.choice(internal_B)
                if add_edge not in current_edges:
                    found = True
                    break
            if not found:
                continue
            
            new_edges = set(current_edges)
            new_edges.discard(remove_edge)
            new_edges.add(add_edge)
            
            # Check if it's a valid graph (simple, 65 edges)
            if len(new_edges) != 65:
                continue
            
            # Evaluate with threshold best_score (if we find >= best_score, we don't care about exact max)
            # But if we want to accept equal scores to escape local minima, we need exact.
            # Let's use threshold = best_score + 1. If new_score < best_score, accept.
            # If new_score == best_score, maybe accept with some probability?
            # For now, only accept strict improvements.
            
            new_score = max_triangle_sum_threshold(new_edges, best_score)
            if new_score < best_score:
                current_edges = new_edges
                current_score = new_score
                best_edges = set(new_edges)
                best_score = new_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            # Swap two internal edges
            if len(current_internal) < 2:
                continue
            # Pick two distinct internal edges
            remove1, remove2 = random.sample(current_internal, 2)
            # Pick two distinct internal edges to add, not in current_edges
            add1 = None
            add2 = None
            for _ in range(100):
                if random.random() < 0.5:
                    e1 = random.choice(internal_A)
                else:
                    e1 = random.choice(internal_B)
                if e1 not in current_edges and e1 != remove1 and e1 != remove2:
                    add1 = e1
                    break
            if add1 is None:
                continue
            for _ in range(100):
                if random.random() < 0.5:
                    e2 = random.choice(internal_A)
                else:
                    e2 = random.choice(internal_B)
                if e2 not in current_edges and e2 != remove1 and e2 != remove2 and e2 != add1:
                    add2 = e2
                    break
            if add2 is None:
                continue
            
            new_edges = set(current_edges)
            new_edges.discard(remove1)
            new_edges.discard(remove2)
            new_edges.add(add1)
            new_edges.add(add2)
            
            if len(new_edges) != 65:
                continue
            
            new_score = max_triangle_sum_threshold(new_edges, best_score)
            if new_score < best_score:
                current_edges = new_edges
                current_score = new_score
                best_edges = set(new_edges)
                best_score = new_score
                improved = True
                no_improve_count = 0
            else:
                no_improve_count += 1

    # If we didn't find anything better than 26, return the baseline
    # But maybe we can try a few more deterministic moves?
    # For now, return best found.
    
    edges_list = sorted([list(e) for e in best_edges])
    return {'n': 16, 'edges': edges_list}
