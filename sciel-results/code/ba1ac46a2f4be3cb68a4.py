"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
from random import Random

sys.setrecursionlimit(100000)

def construct():
    def instance(splits, seed, palette=5):
        rng = Random(seed)
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
        edges = {(min(a, b), max(a, b)) for f in faces for a, b in zip(f, f[1:] + f[:1])}
        n = 8
        for _ in range(splits):
            face = faces.pop(rng.randrange(len(faces)))
            inner = list(range(n, n + 4))
            n += 4
            for i in range(4):
                a, b = face[i], face[(i + 1) % 4]
                edges.add((min(a, inner[i]), max(a, inner[i])))
                edges.add((min(inner[i], inner[(i + 1) % 4]), max(inner[i], inner[(i + 1) % 4])))
                faces.append([a, b, inner[(i + 1) % 4], inner[i]])
            faces.append(inner)
        lists = [sorted(rng.sample(range(palette), 3)) for _ in range(n)]
        return n, sorted(edges), lists

    def build_adj(n, edges):
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_bipartition(n, adj):
        """Find the two parts of the bipartite graph using BFS."""
        color = [-1] * n
        part_a = []
        part_b = []
        for start in range(n):
            if color[start] == -1:
                color[start] = 0
                part_a.append(start)
                queue = [start]
                while queue:
                    v = queue.pop(0)
                    for u in adj[v]:
                        if color[u] == -1:
                            color[u] = 1 - color[v]
                            if color[u] == 0:
                                part_a.append(u)
                            else:
                                part_b.append(u)
                            queue.append(u)
        return part_a, part_b

    def solve_bipartite(n, adj, lists, deadline):
        """
        Solve list coloring on a bipartite graph by exploiting the bipartition.
        Uses arc-consistency pre-pass, then backtracking with MRV, LCV,
        forward checking, and bipartite-aware vertex ordering.
        """
        part_a, part_b = find_bipartition(n, adj)
        
        part_of = [0] * n
        for v in part_a:
            part_of[v] = 0
        for v in part_b:
            part_of[v] = 1
        
        adj_list = [list(adj[v]) for v in range(n)]
        
        # Arc-consistency pre-pass (AC-3 simplified for 3-color lists)
        domains = [set(l) for l in lists]
        queue = []
        for v in range(n):
            for u in adj_list[v]:
                if u > v:
                    queue.append((v, u))
                    queue.append((u, v))
        
        while queue:
            vi, vj = queue.pop()
            di, dj = domains[vi], domains[vj]
            if not di or not dj:
                continue
            # Check if any value in di has no support in dj
            to_remove = []
            for c in di:
                if not any(c != d for d in dj):
                    # c has no support: all of dj == c, but that means
                    # actually we need c to differ from some value in dj
                    # Support exists if there's a d in dj with d != c
                    # No support means all d in dj == c, i.e., dj == {c}
                    if dj == {c}:
                        to_remove.append(c)
            if to_remove:
                for c in to_remove:
                    di.discard(c)
                if not di:
                    return None  # wipeout
                for w in adj_list[vi]:
                    if w != vj:
                        queue.append((w, vi))
        
        # Build order: part_a vertices first, then part_b
        # Within each part, order by degree descending
        order_a = sorted(part_a, key=lambda v: -len(adj_list[v]))
        order_b = sorted(part_b, key=lambda v: -len(adj_list[v]))
        
        # Position in global order for tie-breaking
        pos_in_order = [0] * n
        for i, v in enumerate(order_a):
            pos_in_order[v] = i
        offset = len(order_a)
        for i, v in enumerate(order_b):
            pos_in_order[v] = offset + i
        
        colour = [None] * n
        uncoloured = set(range(n))
        uncoloured_list = list(range(n))
        in_uncoloured = [True] * n
        
        def remove_from_uncoloured(v):
            in_uncoloured[v] = False
            uncoloured.discard(v)
        
        def add_to_uncoloured(v):
            in_uncoloured[v] = True
            uncoloured.add(v)
        
        def get_mrv_vertex():
            """Find uncoloured vertex with minimum domain size, tie-break by part then degree."""
            best_v = -1
            best_count = 100
            best_pos = 10**9
            for v in uncoloured:
                cnt = len(domains[v])
                if cnt == 0:
                    return v
                if cnt < best_count or (cnt == best_count and pos_in_order[v] < best_pos):
                    best_count = cnt
                    best_pos = pos_in_order[v]
                    best_v = v
                    if cnt == 1:
                        break
            return best_v
        
        def lcv_order(v, c_set):
            """Order colors by least constraining value."""
            if len(c_set) <= 2:
                return list(c_set)
            def count_conflicts(c):
                count = 0
                for u in adj_list[v]:
                    if in_uncoloured[u] and c in domains[u]:
                        count += 1
                return count
            return sorted(c_set, key=count_conflicts)
        
        def forward_check(v, c):
            """
            Assign c to v, remove c from neighbors' domains.
            Returns (success, undo_list) where undo_list is [(neighbor, color), ...]
            """
            removed = []
            for u in adj_list[v]:
                if in_uncoloured[u] and c in domains[u]:
                    domains[u].discard(c)
                    removed.append((u, c))
                    if not domains[u]:
                        for uu, cc in removed:
                            domains[uu].add(cc)
                        return False, None
            return True, removed
        
        def backtrack():
            if time.time() > deadline:
                return None
            
            if not uncoloured:
                return True
            
            v = get_mrv_vertex()
            if v == -1:
                return True
            
            if not domains[v]:
                return False
            
            for c in lcv_order(v, domains[v]):
                remove_from_uncoloured(v)
                colour[v] = c
                
                success, removed = forward_check(v, c)
                if success:
                    result = backtrack()
                    if result is not None:
                        return True
                    for u, cc in removed:
                        domains[u].add(cc)
                add_to_uncoloured(v)
                colour[v] = None
            
            return False
        
        result = backtrack()
        if result:
            return colour
        return None

    def greedy_fallback(n, adj, lists):
        """Greedy coloring in order of decreasing degree."""
        colour = [None] * n
        order = sorted(range(n), key=lambda u: -len(adj[u]))
        for v in order:
            taken = set()
            for u in adj[v]:
                if colour[u] is not None:
                    taken.add(colour[u])
            free = [c for c in lists[v] if c not in taken]
            if free:
                colour[v] = free[0]
            else:
                colour[v] = lists[v][0]
        return colour

    def validate(colour, n, adj, lists):
        """Check if colouring is valid."""
        if colour is None:
            return False
        for v in range(n):
            if colour[v] is None:
                return False
            if colour[v] not in lists[v]:
                return False
            for u in adj[v]:
                if u > v and colour[u] == colour[v]:
                    return False
        return True

    start = time.time()
    deadline = start + 140.0  # 10s safety margin before 150s kill

    out = []
    for splits, seed in ((20, 701), (40, 702), (60, 703), (100, 704)):
        n, edges, lists = instance(splits, seed)
        adj = build_adj(n, edges)

        if time.time() > deadline:
            # Fallback: greedy for remaining
            result = greedy_fallback(n, adj, lists)
            out.append(result)
            continue

        # Try bipartite-aware CSP solver
        result = solve_bipartite(n, adj, lists, deadline)
        
        if result is not None and validate(result, n, adj, lists):
            out.append(result)
        else:
            # Fallback: greedy
            result = greedy_fallback(n, adj, lists)
            out.append(result)

    # Ensure all are Python ints in lists
    return [[int(c) for c in lst] for lst in out]

# EVOLVE-BLOCK-END
