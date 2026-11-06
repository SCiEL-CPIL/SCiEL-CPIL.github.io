import time
import random
import itertools

N = 20
C = 20
DEADLINE = 140.0

def _validate(edges):
    """Return True iff every 4-cycle is rainbow (4 distinct colours)."""
    adj = {}
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
    neigh = {i: [] for i in range(N)}
    for u, v, c in edges:
        neigh[u].append(v)
        neigh[v].append(u)
    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = [w for w in neigh[u] if (v, w) in adj and w != u]
            for x in common:
                for y in common:
                    if x >= y:
                        continue
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def construct():
    start = time.time()
    rng = random.Random(42)

    # Precompute all 4-cycles as (u, v, x, y) where cycle is u-x-v-y-u
    # u < v, x < y, x,y are common neighbors of u,v
    # Actually, let's enumerate all 4-cycles differently:
    # A 4-cycle is a set of 4 vertices {a,b,c,d} with edges forming a cycle.
    # For efficiency, enumerate by pairs of opposite vertices.
    # 4-cycle: a-b-c-d-a where a,c are opposite and b,d are opposite.
    # So for each pair (a,c) and each pair (b,d) where b,d are both adjacent to a and c,
    # and b != d, and edge b-d exists, we have a 4-cycle a-b-c-d.

    # Actually for the local search, let's use a different representation.
    # We'll maintain:
    #   color[u][v] = color of edge (u,v), or -1 if no edge
    #   adj[u] = set of neighbors of u

    # Precompute all potential 4-cycles: (a, b, c, d) where a-b-c-d-a is a cycle
    # a < c (opposite), b < d (opposite), b and d are common neighbors of a and c
    # We'll store them as tuples (a, b, c, d) representing edges (a,b),(b,c),(c,d),(d,a)

    all_4cycles = []
    for a in range(N):
        for c in range(a+1, N):
            # common neighbors of a and c
            common = [x for x in range(N) if x != a and x != c]
            for b in common:
                for d in common:
                    if b >= d:
                        continue
                    # cycle a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
                    all_4cycles.append((a, b, c, d))

    # We'll use a smarter approach: start with a known good structure.
    # Known construction: For n=20, use a "near-complete" graph with strategic removals.
    # Strategy: Start with complete graph, assign colors, then iteratively fix violations.

    # Better: use a structured initial coloring.
    # Color (u,v) = (u + v) % C for a base, then perturb.

    # Let's try a different structural approach:
    # Use a "blow-up" of a small valid configuration.
    # Or: start sparse, add edges greedily with smart color choice.

    # I'll use a hybrid: 
    # 1. Start with a moderate-density valid graph (greedy construction)
    # 2. Then do local search: try to add edges, recolor, remove edges

    # Phase 1: Greedy construction (fast)
    def greedy_construct(order_seed, time_budget):
        r = random.Random(order_seed)
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        r.shuffle(all_pairs)
        
        color = [[-1]*N for _ in range(N)]
        neigh = [set() for _ in range(N)]
        edges = []
        
        def can_add(u, v, c):
            # Check all 4-cycles that would be created by adding edge (u,v) with color c
            # Type 1: u-x-v-y-u where x,y are common neighbors of u,v (not using new edge in the "middle")
            # Actually the new edge (u,v) is part of the cycle.
            # Cycles containing new edge (u,v):
            #   u-v-x-y-u: need x in neigh[v], y in neigh[u], x!=y, x!=u, y!=v, edge(x,y) exists
            #   u-v-y-x-u: same as above with x,y swapped (covered)
            #   Also: cycles where (u,v) is a "diagonal"? No, in a 4-cycle each edge is a side.
            #   Wait, in 4-cycle a-b-c-d-a, the edges are (a,b),(b,c),(c,d),(d,a).
            #   If new edge is (u,v), it could be any of the 4 positions.
            #   Position 1: u-v is (a,b): cycle u-v-x-y-u, edges (u,v),(v,x),(x,y),(y,u)
            #   Position 2: u-v is (b,c): cycle x-u-v-y-x, edges (x,u),(u,v),(v,y),(y,x) -- same as position 1 with relabeling
            #   Actually all positions are equivalent by rotation. So we just need:
            #   For all x in neigh[v], y in neigh[u], x!=y, x!=u, y!=v, edge(x,y) exists:
            #     colors {(u,v),(v,x),(x,y),(y,u)} must be 4 distinct
            #   AND for all x,y in common neighbors of u and v (x<y, x,y != u,v):
            #     Wait, these cycles u-x-v-y-u do NOT contain edge (u,v). They already exist.
            #     Adding (u,v) doesn't affect them. So we only need to check cycles that USE the new edge.
            
            # Cycles using new edge (u,v):
            # u-v-x-y-u: x in neigh[v]\{u}, y in neigh[u]\{v}, x!=y, edge(x,y) exists
            for x in neigh[v]:
                if x == u:
                    continue
                for y in neigh[u]:
                    if y == v or y == x:
                        continue
                    if color[x][y] == -1:
                        continue
                    cols = {c, color[v][x], color[x][y], color[y][u]}
                    if len(cols) != 4:
                        return False
            return True
        
        for (u, v) in all_pairs:
            if time.time() > time_budget:
                break
            cols = list(range(C))
            r.shuffle(cols)
            for c in cols:
                if can_add(u, v, c):
                    color[u][v] = c
                    color[v][u] = c
                    neigh[u].add(v)
                    neigh[v].add(u)
                    edges.append([u, v, c])
                    break
        
        return color, neigh, edges

    # Phase 2: Local search to improve
    def local_search(color, neigh, edges, time_budget):
        r = random.Random(int(time.time()*1000) % 100000)
        
        def get_violations():
            """Return list of violating 4-cycles."""
            viols = []
            for (a, b, c, d) in all_4cycles:
                # Check if all 4 edges exist
                if color[a][b] == -1 or color[b][c] == -1 or color[c][d] == -1 or color[d][a] == -1:
                    continue
                cols = {color[a][b], color[b][c], color[c][d], color[d][a]}
                if len(cols) != 4:
                    viols.append((a, b, c, d))
            return viols
        
        def try_recolor(u, v, new_c):
            """Try to recolor edge (u,v) to new_c. Return True if valid."""
            old_c = color[u][v]
            if old_c == new_c:
                return True
            # Check all 4-cycles containing edge (u,v)
            # Cycles: u-v-x-y-u
            for x in neigh[v]:
                if x == u:
                    continue
                for y in neigh[u]:
                    if y == v or y == x:
                        continue
                    if color[x][y] == -1:
                        continue
                    cols = {new_c, color[v][x], color[x][y], color[y][u]}
                    if len(cols) != 4:
                        return False
            # Also cycles where (u,v) appears in other positions? 
            # No, (u,v) as an edge only appears in cycles u-v-x-y-u (and rotations which are same)
            return True
        
        def remove_edge(u, v):
            color[u][v] = -1
            color[v][u] = -1
            neigh[u].discard(v)
            neigh[v].discard(u)
            edges.remove([u, v, color[u][v]] if [u, v, color[u][v]] in edges else [v, u, color[v][u]])
        
        # Iterate: find violations, try to fix by recoloring or removing
        improved = True
        while improved and time.time() < time_budget:
            improved = False
            viols = get_violations()
            if not viols:
                break
            
            # Try to fix each violation
            for (a, b, c, d) in viols[:50]:  # limit per iteration
                if time.time() > time_budget:
                    break
                # Try recoloring one of the 4 edges
                edge_list = [(a,b), (b,c), (c,d), (d,a)]
                r.shuffle(edge_list)
                for (u, v) in edge_list:
                    fixed = False
                    for new_c in range(C):
                        if new_c == color[u][v]:
                            continue
                        if try_recolor(u, v, new_c):
                            color[u][v] = new_c
                            color[v][u] = new_c
                            # Update edges list
                            for i, e in enumerate(edges):
                                if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                                    edges[i][2] = new_c
                                    break
                            fixed = True
                            improved = True
                            break
                    if fixed:
                        break
                if not fixed:
                    # Try removing one edge
                    for (u, v) in edge_list:
                        old_c = color[u][v]
                        color[u][v] = -1
                        color[v][u] = -1
                        neigh[u].discard(v)
                        neigh[v].discard(u)
                        # Remove from edges
                        for i, e in enumerate(edges):
                            if (e[0] == u and e[1] == v) or (e[0] == v and e[1] == u):
                                del edges[i]
                                break
                        improved = True
                        break
        
        # Try to add more edges
        all_pairs = [(u, v) for u in range(N) for v in range(u + 1, N)]
        r.shuffle(all_pairs)
        for (u, v) in all_pairs:
            if time.time() > time_budget:
                break
            if color[u][v] != -1:
                continue
            cols = list(range(C))
            r.shuffle(cols)
            for c in cols:
                # Check if adding (u,v) with color c is valid
                valid = True
                for x in neigh[v]:
                    if x == u:
                        continue
                    for y in neigh[u]:
                        if y == v or y == x:
                            continue
                        if color[x][y] == -1:
                            continue
                        cols4 = {c, color[v][x], color[x][y], color[y][u]}
                        if len(cols4) != 4:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    color[u][v] = c
                    color[v][u] = c
                    neigh[u].add(v)
                    neigh[v].add(u)
                    edges.append([u, v, c])
                    break
        
        return edges

    # Run multiple attempts
    best_edges = None
    best_len = 0
    
    # Attempt 1: Greedy with different seeds
    for seed in range(20):
        if time.time() > DEADLINE:
            break
        budget = min(time.time() + 5.0, DEADLINE - 10.0)
        color, neigh, edges = greedy_construct(seed * 1000 + 42, budget)
        if len(edges) > best_len:
            best_len = len(edges)
            best_edges = edges[:]
    
    # Attempt 2: Local search on best so far
    if best_edges and time.time() < DEADLINE - 20:
        # Rebuild color/neigh from best_edges
        color = [[-1]*N for _ in range(N)]
        neigh = [set() for _ in range(N)]
        for u, v, c in best_edges:
            color[u][v] = c
            color[v][u] = c
            neigh[u].add(v)
            neigh[v].add(u)
        
        budget = min(time.time() + 30.0, DEADLINE - 5.0)
        improved_edges = local_search(color, neigh, best_edges[:], budget)
        if len(improved_edges) > best_len:
            best_len = len(improved_edges)
            best_edges = improved_edges
    
    # Attempt 3: More local search iterations
    if best_edges and time.time() < DEADLINE - 10:
        color = [[-1]*N for _ in range(N)]
        neigh = [set() for _ in range(N)]
        for u, v, c in best_edges:
            color[u][v] = c
            color[v][u] = c
            neigh[u].add(v)
            neigh[v].add(u)
        
        budget = min(time.time() + 20.0, DEADLINE - 2.0)
        improved_edges = local_search(color, neigh, best_edges[:], budget)
        if len(improved_edges) > best_len:
            best_len = len(improved_edges)
            best_edges = improved_edges
    
    if best_edges is None:
        return [[0, v, 0] for v in range(1, N)]
    return best_edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
