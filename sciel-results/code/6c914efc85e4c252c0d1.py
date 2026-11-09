import time
import random
import sys
from collections import defaultdict

def construct():
    def instance(n, d, seed):
        rng = random.Random(seed)
        degree = [0] * n
        edges = set()
        for _ in range(4 * n * d):
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            e = (min(u, v), max(u, v))
            if e not in edges and degree[u] < d and degree[v] < d:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, adj):
        m = len(edges)
        if m == 0:
            return []
        
        max_node = max(max(u, v) for u, v in edges) + 1
        vert_to_edges = [[] for _ in range(max_node)]
        for i, (u, v) in enumerate(edges):
            vert_to_edges[u].append(i)
            vert_to_edges[v].append(i)
            
        conflict_masks = [0] * m
        
        # 1. Shared vertices
        for u in range(max_node):
            e_list = vert_to_edges[u]
            for i in range(len(e_list)):
                ei = e_list[i]
                for j in range(i + 1, len(e_list)):
                    ej = e_list[j]
                    conflict_masks[ei] |= (1 << ej)
                    conflict_masks[ej] |= (1 << ei)
        
        # 2. Induced matching violations (joined by edge)
        for i, (u, v) in enumerate(edges):
            neighbors_u = adj.get(u, set())
            neighbors_v = adj.get(v, set())
            for j in range(i + 1, m):
                if (conflict_masks[i] >> j) & 1:
                    continue
                w, x = edges[j]
                if w in neighbors_u or x in neighbors_u or w in neighbors_v or x in neighbors_v:
                    conflict_masks[i] |= (1 << j)
                    conflict_masks[j] |= (1 << i)
        
        return conflict_masks

    def greedy_coloring(m, conflict_masks, seed=0):
        """
        Fast greedy coloring using Welsh-Powell ordering.
        Returns list of colors.
        """
        if m == 0:
            return []
        
        degrees = [bin(conflict_masks[i]).count('1') for i in range(m)]
        order = list(range(m))
        order.sort(key=lambda x: -degrees[x])
        
        assign = [-1] * m
        color_masks = []
        
        for i in order:
            c = 0
            while c < len(color_masks):
                if (conflict_masks[i] & color_masks[c]) == 0:
                    break
                c += 1
            if c == len(color_masks):
                color_masks.append(0)
            color_masks[c] |= (1 << i)
            assign[i] = c
        
        return assign

    def local_search(assign, m, conflict_masks, k, rng, max_iters=5000, deadline=None):
        """
        Local search to reduce number of colors used.
        assign: list of color assignments (0-indexed)
        Returns improved assign if possible.
        """
        if m == 0:
            return assign
        
        # Build color class sets
        color_classes = defaultdict(set)
        for i in range(m):
            color_classes[assign[i]].add(i)
        
        current_k = len(color_classes)
        if current_k <= k:
            return assign
        
        best_assign = assign[:]
        best_k = current_k
        
        for it in range(max_iters):
            if deadline and time.time() > deadline:
                break
            
            # Pick a random color to try to eliminate
            colors = list(color_classes.keys())
            if len(colors) <= k:
                break
            target_color = rng.choice(colors)
            target_class = color_classes[target_color]
            
            if len(target_class) == 0:
                continue
            
            # Try to reassign each edge in target_color to another color
            success = True
            new_assign = best_assign[:]
            
            for e in list(target_class):
                # Find a valid color for e
                placed = False
                for c in range(len(colors)):
                    if c == target_color:
                        continue
                    # Check if e can be added to color c
                    if (conflict_masks[e] & 0) == 0:  # placeholder
                        pass
                    # Check against current color class
                    can_place = True
                    for other_e in color_classes.get(c, set()):
                        if (conflict_masks[e] >> other_e) & 1:
                            can_place = False
                            break
                    if can_place:
                        new_assign[e] = c
                        placed = True
                        break
                
                if not placed:
                    # Try to create a new color (but we want to reduce, not increase)
                    # Or swap: find an edge in another color that can swap
                    success = False
                    break
            
            if success:
                # Update best
                best_assign = new_assign
                # Rebuild color classes
                color_classes = defaultdict(set)
                for i in range(m):
                    color_classes[best_assign[i]].add(i)
                # Remove empty color classes
                color_classes = {c: s for c, s in color_classes.items() if s}
                best_k = len(color_classes)
                if best_k <= k:
                    break
        
        return best_assign

    def eliminate_colors(assign, m, conflict_masks, rng, deadline=None):
        """
        Try to eliminate colors one by one using local search.
        """
        if m == 0:
            return assign
        
        current_assign = assign[:]
        current_k = len(set(current_assign))
        
        while current_k > 1:
            if deadline and time.time() > deadline:
                break
            
            # Pick a color to eliminate (the one with fewest edges)
            color_counts = defaultdict(int)
            for c in current_assign:
                color_counts[c] += 1
            
            target_color = min(color_counts.keys(), key=lambda c: color_counts[c])
            
            # Try to reassign all edges of target_color to other colors
            success = True
            new_assign = current_assign[:]
            
            for e in range(m):
                if current_assign[e] != target_color:
                    continue
                
                # Find a valid color
                placed = False
                for c in range(current_k):
                    if c == target_color:
                        continue
                    # Check if e conflicts with any edge of color c
                    can_place = True
                    for other_e in range(m):
                        if current_assign[other_e] == c and (conflict_masks[e] >> other_e) & 1:
                            can_place = False
                            break
                    if can_place:
                        new_assign[e] = c
                        placed = True
                        break
                
                if not placed:
                    success = False
                    break
            
            if success:
                current_assign = new_assign
                current_k -= 1
            else:
                # Try local search to help
                improved = local_search(current_assign, m, conflict_masks, current_k - 1, rng, 
                                        max_iters=1000, deadline=deadline)
                if len(set(improved)) < current_k:
                    current_assign = improved
                    current_k = len(set(current_assign))
                else:
                    break
        
        return current_assign

    def solve_instance(n, d, seed, time_limit=30):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict_masks = build_conflict_bitmasks(edges, adj)
        
        # Get initial upper bound with greedy
        init_assign = greedy_coloring(m, conflict_masks, seed)
        current_k = max(init_assign) + 1 if init_assign else 0
        
        # Try multiple restarts with different orderings
        rng = random.Random(seed + 12345)
        best_assign = init_assign
        best_k = current_k
        
        deadline = time.time() + time_limit
        
        # Try different random orderings for greedy
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Shuffle the order for greedy
            degrees = [bin(conflict_masks[i]).count('1') for i in range(m)]
            order = list(range(m))
            # Add some randomness to the ordering
            for i in range(m - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
            order.sort(key=lambda x: -degrees[x])
            
            # Greedy with this order
            assign = [-1] * m
            color_masks = []
            for i in order:
                c = 0
                while c < len(color_masks):
                    if (conflict_masks[i] & color_masks[c]) == 0:
                        break
                    c += 1
                if c == len(color_masks):
                    color_masks.append(0)
                color_masks[c] |= (1 << i)
                assign[i] = c
            
            k = len(color_masks)
            if k < best_k:
                best_k = k
                best_assign = assign[:]
            
            # Try to eliminate colors
            if k <= best_k:
                improved = eliminate_colors(assign, m, conflict_masks, rng, deadline)
                improved_k = len(set(improved))
                if improved_k < best_k:
                    best_k = improved_k
                    best_assign = improved
        
        # Final local search refinement
        if time.time() < deadline - 2:
            final = local_search(best_assign, m, conflict_masks, best_k, rng, 
                                  max_iters=20000, deadline=deadline)
            final_k = len(set(final))
            if final_k < best_k:
                best_k = final_k
                best_assign = final
        
        return best_assign

    out = []
    deadline = time.time() + 115
    
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        remaining = deadline - time.time()
        if remaining < 10:
            # Fallback: Fast Greedy
            n, d, seed = spec
            edges = instance(n, d, seed)
            m = len(edges)
            if m == 0:
                out.append([])
                continue
            adj = {}
            for u, v in edges:
                adj.setdefault(u, set()).add(v)
                adj.setdefault(v, set()).add(u)
            conflict_masks = build_conflict_bitmasks(edges, adj)
            out.append(greedy_coloring(m, conflict_masks, seed))
        else:
            t_limit = min(remaining - 5, 35)
            out.append(solve_instance(*spec, time_limit=t_limit))
    
    return out

if __name__ == "__main__":
    # Quick test
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(set(r)) if r else 0} colors")
