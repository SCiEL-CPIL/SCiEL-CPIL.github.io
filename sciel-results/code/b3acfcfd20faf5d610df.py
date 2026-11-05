"""Strong edge colourings via greedy + local merging."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adj(edges, n):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def edge_conflicts(e, adj):
        """Return set of vertices that conflict with edge e (endpoints + neighbors)."""
        u, v = e
        near = {u, v}
        near.update(adj[u])
        near.update(adj[v])
        return near

    def is_valid_colouring(colour, edges, adj):
        """Check if colouring is valid: each colour class is an induced matching."""
        from collections import defaultdict
        classes = defaultdict(list)
        for e, c in colour.items():
            classes[c].append(e)
        for c, cls in classes.items():
            # Check no two edges share a vertex
            verts_used = set()
            for e in cls:
                if e[0] in verts_used or e[1] in verts_used:
                    return False
                verts_used.add(e[0])
                verts_used.add(e[1])
            # Check no two edges are joined by an edge
            for i in range(len(cls)):
                for j in range(i + 1, len(cls)):
                    u1, v1 = cls[i]
                    u2, v2 = cls[j]
                    if u2 in adj[u1] or v2 in adj[u1] or u1 in adj[u2] or v1 in adj[u2]:
                        return False
        return True

    def greedy_color(edges, adj, order=None):
        """Greedy colouring with given edge order."""
        if order is None:
            order = list(range(len(edges)))
        colour = {}
        # For each edge, track which colours are forbidden
        # A colour is forbidden for edge e if any edge of that colour conflicts with e
        for idx in order:
            e = edges[idx]
            near = edge_conflicts(e, adj)
            taken = set()
            for f, c in colour.items():
                fu, fv = f
                if fu in near or fv in near:
                    taken.add(c)
            c = 0
            while c in taken:
                c += 1
            colour[e] = c
        return colour

    def try_merge(colour, edges, adj, c1, c2):
        """Try to merge colour classes c1 and c2 into c1. Return new colour dict if successful."""
        # Collect all edges in classes c1 and c2
        cls1 = [e for e, c in colour.items() if c == c1]
        cls2 = [e for e, c in colour.items() if c == c2]
        # Check if merging is valid: no conflicts between any edge in cls1 and any in cls2
        # Also need to check that within cls1 ∪ cls2, it's still an induced matching
        # Since each is already an induced matching, we just need cross-checks
        for e1 in cls1:
            near1 = edge_conflicts(e1, adj)
            for e2 in cls2:
                if e2[0] in near1 or e2[1] in near1:
                    return None
        # Merge: reassign all edges from c2 to c1
        new_colour = dict(colour)
        for e in cls2:
            new_colour[e] = c1
        # Remove c2 from the colour set (it's now empty)
        # We don't need to explicitly remove; the colour dict just won't have c2
        return new_colour

    def try_merge(colour, edges, adj, c1, c2):
        """Try to merge colour classes c1 and c2 into c1. Return new colour dict if successful."""
        cls1 = [e for e, c in colour.items() if c == c1]
        cls2 = [e for e, c in colour.items() if c == c2]
        for e1 in cls1:
            near1 = edge_conflicts(e1, adj)
            for e2 in cls2:
                if e2[0] in near1 or e2[1] in near1:
                    return None
        new_colour = dict(colour)
        for e in cls2:
            new_colour[e] = c1
        return new_colour

    def local_improve(colour, edges, adj, max_iters=200):
        """Try to merge colour classes to reduce total colour count."""
        current = dict(colour)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            colours_used = sorted(set(current.values()))
            ncolours = len(colours_used)
            for i in range(ncolours):
                for j in range(i + 1, ncolours):
                    c1, c2 = colours_used[i], colours_used[j]
                    new_colour = try_merge(current, edges, adj, c1, c2)
                    if new_colour is not None:
                        current = new_colour
                        improved = True
                        break
                if improved:
                    break
        return current

    def swap_improve(colour, edges, adj, max_iters=100):
        """Swap-based local search: move blocking edges to other colors to enable merges."""
        current = dict(colour)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            colours_used = sorted(set(current.values()))
            ncolours = len(colours_used)
            for c_remove in colours_used:
                if c_remove not in set(current.values()):
                    continue
                cls = [e for e, c in current.items() if c == c_remove]
                if not cls:
                    continue
                reassignable = True
                assignments = {}
                for e in cls:
                    near = edge_conflicts(e, adj)
                    placed = False
                    for c_new in range(c_remove):
                        conflict = False
                        for f, c in current.items():
                            if c == c_new:
                                fu, fv = f
                                if fu in near or fv in near:
                                    conflict = True
                                    break
                        if not conflict:
                            assignments[e] = c_new
                            placed = True
                            break
                    if not placed:
                        reassignable = False
                        break
                if reassignable:
                    new_colour = dict(current)
                    for e, c_new in assignments.items():
                        new_colour[e] = c_new
                    current = new_colour
                    improved = True
                    break
        return current

    def polish(colour, edges, adj, max_rounds=50):
        """Final polish: repeatedly try to reduce colors via merge + swap."""
        current = dict(colour)
        for _ in range(max_rounds):
            before = len(set(current.values()))
            current = local_improve(current, edges, adj, max_iters=50)
            after_merge = len(set(current.values()))
            if after_merge < before:
                continue
            current = swap_improve(current, edges, adj, max_iters=50)
            after_swap = len(set(current.values()))
            if after_swap < after_merge:
                continue
            current = local_improve(current, edges, adj, max_iters=50)
            if len(set(current.values())) >= before:
                break
        return current

    def total_colours(colour):
        return len(set(colour.values()))

    results = []
    for spec in ((40, 4, 101), (60, 5, 102), (80, 6, 103)):
        n, d, seed = spec
        edges = instance(n, d, seed)
        adj = build_adj(edges, n)
        
        best_colour = None
        best_count = float('inf')
        
        # Try multiple random orderings
        num_trials = 40
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            rng = random.Random(seed * 1000 + trial)
            order = list(range(len(edges)))
            rng.shuffle(order)
            
            colour = greedy_color(edges, adj, order)
            colour = local_improve(colour, edges, adj, max_iters=100)
            count = total_colours(colour)
            
            if count < best_count:
                best_count = count
                best_colour = colour
        
        # Also try a deterministic ordering (by edge index)
        if time.time() < deadline:
            colour = greedy_color(edges, adj, order=list(range(len(edges))))
            colour = local_improve(colour, edges, adj, max_iters=100)
            count = total_colours(colour)
            if count < best_count:
                best_count = count
                best_colour = colour
        
        # Polish the best solution
        if best_colour is not None and time.time() < deadline:
            polished = polish(best_colour, edges, adj, max_rounds=30)
            polished_count = total_colours(polished)
            if polished_count < best_count:
                best_count = polished_count
                best_colour = polished
        
        if best_colour is None:
            # Fallback: simple greedy
            best_colour = greedy_color(edges, adj)
        
        results.append([best_colour[e] for e in edges])
    
    return results
# EVOLVE-BLOCK-END
