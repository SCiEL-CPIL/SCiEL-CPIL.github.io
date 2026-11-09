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

    def greedy_independent_set(candidates, conflict_masks, deadline=None):
        """
        Greedy: repeatedly pick the vertex with minimum degree in the induced subgraph.
        Returns a list of vertex indices forming an independent set.
        """
        if not candidates:
            return []
        
        # Work with a copy of the candidate set as a bitmask
        remaining = 0
        for v in candidates:
            remaining |= (1 << v)
        
        result = []
        while remaining:
            if deadline and time.time() > deadline:
                break
            # Find vertex with minimum conflict-degree among remaining
            best_v = -1
            best_deg = 999999
            temp = remaining
            while temp:
                v = (temp & (-temp)).bit_length() - 1
                temp &= temp - 1
                # Count conflicts within remaining
                deg = (conflict_masks[v] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
            if best_v == -1:
                break
            result.append(best_v)
            # Remove best_v and all its neighbors from remaining
            remaining &= ~(conflict_masks[best_v] | (1 << best_v))
        
        return result

    def branch_and_bound_max_is(candidates, conflict_masks, deadline=None, best_size=0):
        """
        Branch and bound to find a maximum independent set among candidates.
        Uses greedy coloring of the subgraph as an upper bound.
        Returns (list_of_vertices, size).
        """
        if not candidates:
            return [], 0
        
        # Upper bound: greedy coloring of the induced subgraph on candidates
        # The chromatic number of the complement gives an upper bound on IS size
        # Actually, we use a simpler bound: the size of a greedy maximal independent set
        # plus the number of remaining vertices that could potentially be added
        # Better bound: use the fact that in any IS, we can have at most
        # floor(n / (min_degree + 1)) + 1 vertices... but let's use a practical bound.
        
        # Practical upper bound: greedy maximal IS size on the full candidate set
        greedy_is = greedy_independent_set(candidates, conflict_masks, deadline)
        ub = len(greedy_is)
        
        if ub <= best_size:
            return [], 0
        
        # If the greedy IS is already optimal (all vertices independent), return it
        # Check: is the greedy IS a maximal IS that covers all candidates?
        # Actually, if ub == len(candidates), all are independent
        if ub == len(candidates):
            return list(candidates), ub
        
        # Branch: pick a vertex with high degree to branch on
        # Choose vertex v, branch on include v or exclude v
        # Use degree in the induced subgraph
        best_v = -1
        best_deg = -1
        for v in candidates:
            deg = (conflict_masks[v] & 0)  # placeholder
            # Count conflicts within candidates
            mask_cands = 0
            for c in candidates:
                mask_cands |= (1 << c)
            deg = (conflict_masks[v] & mask_cands).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
        
        if best_v == -1:
            return list(candidates), len(candidates)
        
        # Build mask of candidates
        mask_cands = 0
        for c in candidates:
            mask_cands |= (1 << c)
        
        # Branch 1: include best_v
        # Remove best_v and its neighbors from candidates
        new_cands_include = [v for v in candidates 
                           if v != best_v and not ((conflict_masks[best_v] >> v) & 1)]
        
        if deadline and time.time() > deadline:
            return greedy_is, len(greedy_is)
        
        is_include, size_include = branch_and_bound_max_is(
            new_cands_include, conflict_masks, deadline, max(best_size, 0)
        )
        if is_include:
            is_include = [best_v] + is_include
        
        # Branch 2: exclude best_v
        new_cands_exclude = [v for v in candidates if v != best_v]
        
        if deadline and time.time() > deadline:
            if size_include > 0:
                return is_include, size_include
            return greedy_is, len(greedy_is)
        
        is_exclude, size_exclude = branch_and_bound_max_is(
            new_cands_exclude, conflict_masks, deadline, max(best_size, 0)
        )
        
        # Pick the better
        if size_include >= size_exclude:
            return is_include, size_include
        else:
            return is_exclude, size_exclude

    def fast_max_independent_set(candidates, conflict_masks, deadline=None):
        """
        Hybrid: try greedy first, then use B&B if time allows and improvement is possible.
        Returns list of vertex indices.
        """
        if not candidates:
            return []
        
        # Quick greedy
        greedy_is = greedy_independent_set(candidates, conflict_masks, deadline)
        greedy_size = len(greedy_is)
        
        # If greedy covers all, we're done
        if greedy_size == len(candidates):
            return greedy_is
        
        # Try B&B with a time limit
        if deadline and time.time() > deadline - 0.5:
            return greedy_is
        
        # For small candidate sets, use B&B
        if len(candidates) <= 25:
            sub_deadline = deadline if deadline else None
            # Pass a tighter deadline
            if sub_deadline:
                sub_deadline = min(sub_deadline, time.time() + 5)
            result, size = branch_and_bound_max_is(candidates, conflict_masks, sub_deadline, greedy_size)
            if size > greedy_size:
                return result
            return greedy_is
        
        # For larger sets, use a more sophisticated greedy
        # Multi-start greedy with different orderings
        rng = random.Random(hash(tuple(sorted(candidates))) % 10000)
        best_result = greedy_is
        best_size = greedy_size
        
        for restart in range(10):
            if deadline and time.time() > deadline:
                break
            
            # Randomized greedy: pick random vertex from those with min degree
            remaining = 0
            for v in candidates:
                remaining |= (1 << v)
            
            result = []
            while remaining:
                # Find all vertices with minimum degree
                min_deg = 999999
                min_verts = []
                temp = remaining
                while temp:
                    v = (temp & (-temp)).bit_length() - 1
                    temp &= temp - 1
                    deg = (conflict_masks[v] & remaining).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        min_verts = [v]
                    elif deg == min_deg:
                        min_verts.append(v)
                
                if not min_verts:
                    break
                
                # Pick randomly from min-degree vertices
                v = rng.choice(min_verts)
                result.append(v)
                remaining &= ~(conflict_masks[v] | (1 << v))
            
            if len(result) > best_size:
                best_size = len(result)
                best_result = result
        
        return best_result

    def extract_color_classes(m, conflict_masks, deadline=None):
        """
        Iteratively extract maximum independent sets from the conflict graph.
        Each IS becomes a color class.
        Returns: list of color classes (each a list of edge indices), and assignment array.
        """
        assign = [-1] * m
        remaining = set(range(m))
        color_idx = 0
        
        while remaining:
            if deadline and time.time() > deadline:
                # Fallback: assign remaining edges greedily
                for e in sorted(remaining):
                    assign[e] = color_idx
                    color_idx += 1
                break
            
            # Find maximum independent set among remaining
            is_vertices = fast_max_independent_set(list(remaining), conflict_masks, deadline)
            
            if not is_vertices:
                # Shouldn't happen, but safety
                for e in sorted(remaining):
                    assign[e] = color_idx
                    color_idx += 1
                break
            
            # Assign this color
            for v in is_vertices:
                assign[v] = color_idx
                remaining.discard(v)
            
            color_idx += 1
        
        return assign

    def refine_with_swaps(assign, m, conflict_masks, deadline=None):
        """
        Try to reduce colors by swapping edges between color classes.
        """
        if m == 0:
            return assign
        
        current_assign = assign[:]
        current_k = max(current_assign) + 1 if current_assign else 0
        
        if current_k <= 1:
            return current_assign
        
        # Build color classes
        color_classes = defaultdict(set)
        for i in range(m):
            color_classes[current_assign[i]].add(i)
        
        improved = True
        while improved:
            improved = False
            if deadline and time.time() > deadline:
                break
            
            # Try to eliminate the color with the fewest edges
            if len(color_classes) <= 1:
                break
            
            target_color = min(color_classes.keys(), key=lambda c: len(color_classes[c]))
            target_edges = list(color_classes[target_color])
            
            if not target_edges:
                del color_classes[target_color]
                improved = True
                continue
            
            # Try to reassign each edge in target_color to another color
            success = True
            new_assign = current_assign[:]
            temp_classes = {c: set(s) for c, s in color_classes.items()}
            
            for e in target_edges:
                placed = False
                for c in sorted(temp_classes.keys()):
                    if c == target_color:
                        continue
                    # Check if e can be added to color c
                    can_place = True
                    for other_e in temp_classes[c]:
                        if (conflict_masks[e] >> other_e) & 1:
                            can_place = False
                            break
                    if can_place:
                        new_assign[e] = c
                        temp_classes[c].add(e)
                        placed = True
                        break
                
                if not placed:
                    success = False
                    break
            
            if success:
                # Remove target color
                del temp_classes[target_color]
                current_assign = new_assign
                color_classes = temp_classes
                current_k -= 1
                improved = True
        
        return current_assign

    def solve_instance(n, d, seed, time_limit=35):
        edges = instance(n, d, seed)
        m = len(edges)
        if m == 0:
            return []
        
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        
        conflict_masks = build_conflict_bitmasks(edges, adj)
        
        deadline = time.time() + time_limit
        
        # Extract color classes using max independent set approach
        assign = extract_color_classes(m, conflict_masks, deadline)
        current_k = max(assign) + 1 if assign else 0
        
        # Refine with swaps
        if time.time() < deadline - 2:
            assign = refine_with_swaps(assign, m, conflict_masks, deadline)
        
        return assign

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
            # Simple greedy fallback
            assign = [-1] * m
            color_masks = []
            for i in range(m):
                c = 0
                while c < len(color_masks):
                    if (conflict_masks[i] & color_masks[c]) == 0:
                        break
                    c += 1
                if c == len(color_masks):
                    color_masks.append(0)
                color_masks[c] |= (1 << i)
                assign[i] = c
            out.append(assign)
        else:
            t_limit = min(remaining - 5, 40)
            out.append(solve_instance(*spec, time_limit=t_limit))
    
    return out

if __name__ == "__main__":
    # Quick test
    res = construct()
    for i, r in enumerate(res):
        print(f"Instance {i}: {len(set(r)) if r else 0} colors")
