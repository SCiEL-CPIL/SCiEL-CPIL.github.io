import time
import random
import numpy as np

def construct():
    N = 40
    deadline = time.time() + 115  # Safety margin (evaluator kills at 150s)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_edges_list = []
    best_count = 0
    
    def run_once(seed):
        nonlocal best_edges_list, best_count
        random.seed(seed)
        
        # Adjacency as bitmasks and sets
        adj_mask = [0] * N  # bitmask of neighbors
        adj_set = [set() for _ in range(N)]
        edges = []
        
        def add_edge(u, v):
            if (adj_mask[u] >> v) & 1:
                return False
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            adj_set[u].add(v)
            adj_set[v].add(u)
            edges.append((u, v))
            return True
        
        def remove_edge(u, v):
            if not ((adj_mask[u] >> v) & 1):
                return False
            adj_mask[u] &= ~(1 << v)
            adj_mask[v] &= ~(1 << u)
            adj_set[u].discard(v)
            adj_set[v].discard(u)
            try:
                edges.remove((u, v))
            except ValueError:
                edges.remove((v, u))
            return True
        
        def get_blocking_c6(u, v):
            """
            Returns a list of edges forming a C6 with (u,v), or None.
            Uses bitmask BFS to check if a path u-a-b-c-d-v exists.
            """
            # We need a simple path of length 5 from u to v (not using edge u-v directly)
            # Path: u -> n1 -> n2 -> n3 -> n4 -> v
            # So we need n4 to be reachable from u in exactly 4 steps (without using v or revisiting),
            # and n4 adjacent to v.
            
            # BFS layers from u, avoiding v
            # layers[d] = bitmask of nodes reachable in exactly d steps from u,
            #            not using v, and not revisiting nodes in earlier layers (simple path constraint is tricky)
            # For a quick check, we ignore the "no revisit" constraint (over-approximation).
            # If the over-approximation says no, then definitely no C6.
            # If it says yes, we do a targeted DFS to confirm.
            
            def neighbors_mask_excl(node, excl):
                m = adj_mask[node]
                m &= ~(1 << excl)  # exclude 'excl'
                return m
            
            layers = [0] * 5  # layers[0] to layers[4]
            layers[0] = 1 << u
            
            for d in range(1, 5):
                mask = 0
                prev = layers[d-1]
                m = prev
                while m:
                    low = m & (-m)
                    node = low.bit_length() - 1
                    m ^= low
                    mask |= neighbors_mask_excl(node, v)
                # Remove nodes in earlier layers to enforce simple path (approximate)
                # Actually, for C6 detection, we just need *some* path.
                # The bitmask BFS without simple path constraint might find a path that revisits nodes,
                # which wouldn't form a simple C6. But it's a good filter.
                # Let's keep it simple: just OR neighbors.
                layers[d] = mask
            
            # Check if any node in layers[4] is adjacent to v
            v_neighbors = adj_mask[v] & ~(1 << u)  # exclude u (start of path)
            if (layers[4] & v_neighbors) == 0:
                return None  # No C6 possible
            
            # Targeted DFS to find the actual simple path
            # State: (current_node, depth, path_so_far)
            # We want depth 4, and current_node adjacent to v.
            stack = [(u, 0, [u])]
            while stack:
                node, depth, path = stack.pop()
                if depth == 4:
                    if (adj_mask[node] >> v) & 1:
                        # Found path: path + [v]
                        c6_edges = []
                        for i in range(len(path) - 1):
                            c6_edges.append((path[i], path[i+1]))
                        c6_edges.append((node, v))
                        return c6_edges
                    continue
                # Explore neighbors
                nm = adj_mask[node]
                # Exclude v (we only go to v at the end)
                nm &= ~(1 << v)
                # Exclude nodes already in path (simple path)
                for p in path:
                    nm &= ~(1 << p)
                
                m = nm
                while m:
                    low = m & (-m)
                    nb = low.bit_length() - 1
                    m ^= low
                    stack.append((nb, depth + 1, path + [nb]))
            return None
        
        def is_c6_free(u, v):
            return get_blocking_c6(u, v) is None
        
        # --- Greedy phase with heuristic ranking ---
        def compute_priority():
            """
            Rank missing edges by a heuristic score.
            Prefer edges that connect high-degree nodes but have few common neighbors (to avoid C4s, which can lead to C6s).
            """
            missing = [(u, v) for u, v in all_pairs if not ((adj_mask[u] >> v) & 1)]
            if not missing:
                return []
            
            scores = []
            for u, v in missing:
                du = bin(adj_mask[u]).count('1')
                dv = bin(adj_mask[v]).count('1')
                common = bin(adj_mask[u] & adj_mask[v]).count('1')
                # Score: balance degree and common neighbors
                score = (du + dv) * 1.0 - common * 3.0
                score += random.random() * 0.5
                scores.append((score, u, v))
            
            scores.sort(reverse=True)
            return [(u, v) for _, u, v in scores]
        
        # Initial graph: random sparse graph or empty
        # Let's start with a few random edges to break symmetry
        for _ in range(20):
            u, v = random.sample(range(N), 2)
            if is_c6_free(u, v):
                add_edge(u, v)
        
        local_best = len(edges)
        local_best_adj = [set(s) for s in adj_set]
        
        # Local search loop
        while time.time() < deadline:
            # Get ranked missing edges
            ranked_missing = compute_priority()
            if not ranked_missing:
                break
            
            # Shuffle for diversity
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline:
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                
                if is_c6_free(u, v):
                    add_edge(u, v)
                    added_in_pass += 1
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = [set(s) for s in adj_set]
            
            if added_in_pass == 0:
                # Stuck: perturbation
                if ranked_missing:
                    target_u, target_v = random.choice(ranked_missing)
                    block_c6 = get_blocking_c6(target_u, target_v)
                    if block_c6:
                        # Remove 1-3 random edges from the blocking C6
                        num_removals = random.randint(1, 3)
                        for _ in range(num_removals):
                            if edges:
                                e_to_remove = random.choice(block_c6)
                                remove_edge(e_to_remove[0], e_to_remove[1])
                    elif edges:
                        e = random.choice(edges)
                        remove_edge(e[0], e[1])
        
        # Update global best
        if local_best > best_count:
            best_count = local_best
            best_edges_list = []
            for u in range(N):
                for v in local_best_adj[u]:
                    if u < v:
                        best_edges_list.append([u, v])
        
        return local_best
    
    # Multi-restart
    seeds = [42, 123, 7, 999, 31415, 271, 808, 555, 111, 222]
    for i, s in enumerate(seeds):
        if time.time() > deadline - 5:
            break
        run_once(s)
    
    return best_edges_list

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
