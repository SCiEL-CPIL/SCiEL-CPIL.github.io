import time
import random
from collections import defaultdict

def construct():
    n = 15
    start = time.time()
    deadline = start + 55.0  # Safety margin

    def build_adj_mask_deg(edge_list):
        adj_mask = [0] * n
        deg = [0] * n
        for u, v in edge_list:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
            deg[u] += 1
            deg[v] += 1
        return adj_mask, deg

    def is_valid_bitmask(adj_mask, deg):
        """Check validity using precomputed adjacency bitmasks."""
        deg_map = defaultdict(list)
        for i in range(n):
            deg_map[deg[i]].append(i)
        
        for d, verts in deg_map.items():
            if len(verts) < 2:
                continue
            for i in range(len(verts)):
                for j in range(i + 1, len(verts)):
                    s, t = verts[i], verts[j]
                    # For each u in adj[s], check if adj[u] & adj[t] has any vertex v != s, v != t
                    us_mask = adj_mask[s]
                    if us_mask == 0:
                        continue
                    # Iterate through set bits in us_mask
                    m = us_mask
                    while m:
                        u = (m & (-m)).bit_length() - 1
                        m &= m - 1
                        if u == t:
                            continue
                        common = adj_mask[u] & adj_mask[t]
                        # Remove s and t from consideration
                        common &= ~(1 << s)
                        common &= ~(1 << t)
                        if common:
                            return False
        return True

    def is_valid(edge_list):
        adj_mask, deg = build_adj_mask_deg(edge_list)
        return is_valid_bitmask(adj_mask, deg)

    # Base: complete bipartite K_{7,8} - always valid
    base_edges = [(i, j) for i in range(7) for j in range(7, 15)]
    best_edges = base_edges[:]
    best_count = len(base_edges)

    # All possible edges
    all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
    base_set = set(base_edges)
    remaining = [e for e in all_possible if e not in base_set]

    # Greedy add with multiple random orderings
    for attempt in range(500):
        if time.time() > deadline - 15:
            break
        current = base_edges[:]
        current_set = set(current)
        shuffled = remaining[:]
        random.shuffle(shuffled)
        for edge in shuffled:
            if edge in current_set:
                continue
            test = current + [edge]
            if is_valid(test):
                current = test
                current_set.add(edge)
        if len(current) > best_count:
            best_count = len(current)
            best_edges = current[:]

    # Local search: try to swap edges to get more total
    if time.time() < deadline - 8:
        for ls_attempt in range(100):
            if time.time() > deadline - 8:
                break
            cur = best_edges[:]
            cur_set = set(cur)
            improved = False
            
            # Try removing one edge and adding one or two others
            edges_to_remove = cur[:]
            random.shuffle(edges_to_remove)
            
            for rem_edge in edges_to_remove:
                if time.time() > deadline - 8:
                    break
                without = [e for e in cur if e != rem_edge]
                without_set = set(without)
                candidates = [e for e in all_possible if e not in without_set]
                random.shuffle(candidates)
                
                # Try adding just one edge (net same count, but may enable further additions)
                for add1 in candidates[:20]:
                    if time.time() > deadline - 8:
                        break
                    test1 = without + [add1]
                    if is_valid(test1):
                        # Check if this allows adding more edges
                        test1_set = set(test1)
                        rem1 = [e for e in all_possible if e not in test1_set]
                        random.shuffle(rem1)
                        for add2 in rem1[:15]:
                            test2 = test1 + [add2]
                            if is_valid(test2):
                                cur = test2
                                cur_set = set(cur)
                                improved = True
                                break
                        if improved:
                            break
                    # Also try adding two edges at once
                    for add2 in candidates[:15]:
                        if add2 == add1:
                            continue
                        if time.time() > deadline - 8:
                            break
                        test2 = without + [add1, add2]
                        if is_valid(test2):
                            cur = test2
                            cur_set = set(cur)
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                if len(cur) > best_count:
                    best_count = len(cur)
                    best_edges = cur[:]

    # Final polish: try to add individual edges in multiple passes
    if time.time() < deadline - 2:
        for polish_pass in range(5):
            if time.time() > deadline - 2:
                break
            cur = best_edges[:]
            cur_set = set(cur)
            rem = [e for e in all_possible if e not in cur_set]
            random.shuffle(rem)
            for edge in rem:
                if time.time() > deadline - 2:
                    break
                test = cur + [edge]
                if is_valid(test):
                    cur = test
                    cur_set.add(edge)
            if len(cur) > best_count:
                best_count = len(cur)
                best_edges = cur[:]

    out = [[u, v] for u, v in best_edges]
    return out
