"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import deque

def construct(k, g):
    deadline = time.time() + 115

    # ---------- LUW construction: guaranteed valid fallback (girth 16) ----------
    def luw_construct(k, g):
        q = k
        d = g - 5 if (g - 5) % 2 else g - 4
        if d < 3:
            d = 3
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
        else:
            add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
        rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
        for c in range(5, d + 1):
            r, c0 = (c - 5) % 4, c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        def neighbours(vertex):
            side, x = vertex
            for first in range(q):
                y = [None, first] + [0] * (d - 1)
                for c in range(2, d + 1):
                    a, b = rule[c]
                    y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
            if vertex[0] == 0:
                for other in neighbours(vertex):
                    edges.append((index[vertex], index[other]))
        return len(order), edges

    best_n, best_edges = luw_construct(k, g)

    # ---------- fast girth verifier (non-backtracking, certified) ----------
    def check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        max_depth = min_girth // 2 - 1
        for start in range(n):
            visited = set()
            queue = deque([(start, -1, 0)])
            visited.add((start, -1))
            while queue:
                node, parent, depth = queue.popleft()
                if depth == max_depth:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if (nb, node) in visited:
                        return False
                    visited.add((nb, node))
                    queue.append((nb, node, depth + 1))
        return True

    # ---------- direct backtracking construction (tree + non-tree edges) ----------
    def backtracking_construct(k, g):
        # BFS tree of a k-regular girth-g graph has level sizes:
        # 1, k, k(k-1), k(k-1)^2, ... for g/2 levels (0..g/2)
        half = g // 2
        level_sizes = [1] + [k * (k - 1) ** i for i in range(half)]
        total = sum(level_sizes)
        if total > 2000:
            return None  # too big to attempt

        # parent / child / level arrays
        parent = [-1] * total
        level = [0] * total
        children = [[] for _ in range(total)]
        # level 0 = vertex 0
        idx = 1
        for lv in range(1, half + 1):
            for _ in range(level_sizes[lv]):
                p = random.randrange(idx - level_sizes[lv - 1], idx)
                parent[idx] = p
                level[idx] = lv
                children[p].append(idx)
                idx += 1

        # adjacency: tree edges + candidate non-tree edges
        adj = [set() for _ in range(total)]
        for v in range(1, total):
            adj[v].add(parent[v])
            adj[parent[v]].add(v)

        # candidate non-tree edges: between vertices at same level (even cycle)
        # or level differing by 2 (odd cycle). Cycle length = 2*min_level + diff
        # where diff = 0 (same level) or 2 (level diff 2).
        # We need cycle length >= g, i.e. 2*min_level + diff >= g.
        # For even g=16: same level -> 2*lv >= 16 -> lv >= 8 (but max lv=8, so only lv=8)
        #                  diff 2 -> 2*lv + 2 >= 16 -> lv >= 7
        # We only need to close the tree into a regular graph: each vertex at level < half
        # needs (k-1) non-tree edges to vertices at level <= level (to keep girth).
        # Vertices at level == half need k non-tree edges (no children).
        # This is complex; instead, use a simpler approach:
        # For each vertex, we need degree k. Tree gives 1 (parent) + children_count.
        # Remaining = k - 1 - children_count for internal, k - children_count for leaves.
        # For k=3, half=8: internal levels 1..7 have 2 children, so remaining = 3-1-2 = 0.
        # Level 8 (leaves) have 0 children, remaining = 3.
        # Level 0 has 3 children, remaining = 3-3 = 0.
        # So only level 8 vertices need 3 non-tree edges each!
        # Level 8 size = 3*2^7 = 384. Each needs 3 edges to other level-8 vertices
        # (since level 8 is the max, and same-level edges give even cycles of length
        # 2*8 = 16, which is exactly g, so allowed).
        # So we need a 3-regular graph on the 384 level-8 vertices, with the constraint
        # that no edge creates a cycle shorter than 16. Since all are at level 8,
        # any edge between two level-8 vertices creates a cycle of length 2*8 = 16
        # (going up to LCA and back down), which is exactly g. So ANY 3-regular graph
        # on level-8 vertices works!

        # Find all level-8 vertices
        level8 = [v for v in range(total) if level[v] == half]
        n8 = len(level8)
        if n8 == 0 or n8 * k % 2 != 0:
            return None

        # Generate a random 3-regular graph on level8 vertices
        def random_3reg_on_vertices(verts, k):
            n = len(verts)
            if n * k % 2 != 0:
                return None
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
            for _ in range(20):
                random.shuffle(stubs)
                edges = []
                ok = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        ok = False
                        break
                    edges.append((verts[u], verts[v]))
                if not ok:
                    continue
                if len(set(edges)) != len(edges):
                    continue
                return edges
            return None

        lt_edges = random_3reg_on_vertices(level8, k)
        if lt_edges is None:
            return None

        # Build full edge list
        all_edges = []
        for v in range(1, total):
            all_edges.append((min(v, parent[v]), max(v, parent[v])))
        all_edges.extend(lt_edges)

        # Verify
        if check_girth(total, all_edges, g):
            return total, all_edges
        return None

    # Try backtracking construction with multiple attempts
    for attempt in range(30):
        if time.time() > deadline:
            break
        result = backtracking_construct(k, g)
        if result is not None:
            n, edges = result
            if n < best_n:
                best_n, best_edges = n, edges
                break  # found something better

    # If backtracking found nothing better, try random lifts as a secondary approach
    if best_n > 936:
        for m in range(100, 3000, 100):
            if time.time() > deadline:
                break
            n = 3 * m  # 3-regular on n vertices via lift
            if n % 2 != 0:
                continue
            for attempt in range(5):
                if time.time() > deadline:
                    break
                stubs = []
                for i in range(n):
                    stubs.extend([i] * 3)
                random.shuffle(stubs)
                edges = []
                ok = True
                for i in range(0, len(stubs), 2):
                    u, v = stubs[i], stubs[i + 1]
                    if u == v:
                        ok = False
                        break
                    edges.append((min(u, v), max(u, v)))
                if not ok or len(set(edges)) != len(edges):
                    continue
                if check_girth(n, edges, g):
                    if n < best_n:
                        best_n, best_edges = n, edges
                    break

    return best_n, best_edges
# EVOLVE-BLOCK-END
