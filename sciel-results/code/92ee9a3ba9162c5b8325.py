"""Strong edge colouring via MWIS branch-and-bound on the edge-colour conflict graph.

Nodes: (edge, colour) for colour in 0..K-1.
Conflicts between nodes:
  - same edge
  - same colour and the two edges conflict (share a vertex or are joined by an edge)
Weight of a node: 1.0 if its edge is already assigned that colour in the seed, else 0.
Maximizing the independent set weight while forcing coverage of every edge (handled by
starting from a valid seed and only allowing moves that keep coverage, or by penalizing
missing coverage) is equivalent to finding the best strong edge colouring with <= K colours
that is "close" to the seed. We use a branch-and-bound that:
  1. Starts from a DSATUR greedy seed (valid colouring).
  2. Tries to reduce K by 1: re-solve MWIS with K-1 colours, forcing full coverage.
  3. Uses fractional upper bound: for remaining unassigned edges, the max additional
     weight we can get is bounded by the sum of the top (K_remaining) fractional weights
     in the independent set relaxation (approximated by greedy packing of available nodes).
"""
import time
from random import Random
from collections import defaultdict


def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # safety margin under 150 s kill

    def gen_instance(n, d, seed):
        rng = Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_adjacency(edges):
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def build_edge_conflict(edges, adj):
        """conflict[i] = set of j > i such that edges[i] and edges[j] conflict."""
        m = len(edges)
        conflict = [set() for _ in range(m)]
        for i in range(m):
            u, v = edges[i]
            # vertices "near" edge i: u, v, and all neighbours of u or v
            near = {u, v}
            for nb in adj[u]:
                near.add(nb)
            for nb in adj[v]:
                near.add(nb)
            for j in range(i + 1, m):
                a, b = edges[j]
                if a in near or b in near:
                    conflict[i].add(j)
                    conflict[j].add(i)
        return conflict

    def dsatur_greedy(edges, conflict, adj, K):
        """DSATUR-style greedy: colour edges one by one, choosing the edge with
        most saturated colour neighbourhood, then the least used compatible colour."""
        m = len(edges)
        colour = [-1] * m
        colour_count = [0] * K  # how many edges use each colour
        # saturation[i] = set of colours used by neighbours of edge i
        saturation = [set() for _ in range(m)]
        remaining = list(range(m))
        
        for _ in range(m):
            if not remaining:
                break
            # find edge with max saturation degree, tie-break by degree in conflict graph
            best_e = -1
            best_sat = -1
            best_deg = -1
            for e in remaining:
                sat = len(saturation[e])
                deg = len(conflict[e])
                if sat > best_sat or (sat == best_sat and deg > best_deg):
                    best_sat = sat
                    best_deg = deg
                    best_e = e
            # choose least used compatible colour
            used = saturation[best_e]
            chosen = None
            min_count = float('inf')
            for c in range(K):
                if c not in used and colour_count[c] < min_count:
                    min_count = colour_count[c]
                    chosen = c
            if chosen is None:
                # fallback: any unused colour
                for c in range(K):
                    if c not in used:
                        chosen = c
                        break
            if chosen is None:
                return None  # infeasible with K colours
            
            colour[best_e] = chosen
            colour_count[chosen] += 1
            remaining.remove(best_e)
            # update saturation of remaining edges
            for j in conflict[best_e]:
                if j in remaining:
                    saturation[j].add(chosen)
        return colour

    def is_valid_colouring(colour, conflict, edges):
        m = len(edges)
        for i in range(m):
            c = colour[i]
            if c < 0 or c >= len(set(colour)):
                return False
            for j in conflict[i]:
                if colour[j] == c:
                    return False
        return True

    def mwis_branch_and_bound(edges, conflict, K, seed_colour, deadline):
        """Try to find a valid strong edge colouring with exactly K colours
        (or fewer, by reindexing) starting from seed_colour as a guide.
        
        We formulate as: assign each edge a colour in 0..K-1 such that no two
        conflicting edges share a colour. We use branch-and-bound where:
        - Branch: pick an unassigned edge, try each colour.
        - Bound: for unassigned edges, the max number we can still assign is
          bounded by the size of a maximum independent set in the remaining
          conflict subgraph projected onto available colours. We approximate
          this by a greedy fractional bound: for each unassigned edge, the
          number of available colours = K - |used_by_conflictors|. The upper
          bound on total assigned is current_assigned + sum over unassigned
          of min(1, available_colours) but this is loose. 
          
        Better bound: we just do DFS with pruning when no colour is available.
        Since K is small (<=20) and m is moderate, we use a smarter ordering.
        """
        m = len(edges)
        if m == 0:
            return list(seed_colour) if seed_colour else []
        
        # Precompute conflict sets
        conf = [set() for _ in range(m)]
        for i in range(m):
            for j in conflict[i]:
                conf[i].add(j)
        
        # Sort edges by degree in conflict graph (most constrained first)
        order = sorted(range(m), key=lambda i: -len(conf[i]))
        # Map: position in order -> edge index
        pos_to_edge = order
        edge_to_pos = {e: p for p, e in enumerate(order)}
        
        # State: colour assignment for edges in order
        colour = [-1] * m
        # colour_used[c] = set of edge indices using colour c
        colour_used = [set() for _ in range(K)]
        
        best = None
        best_score = -1
        
        def bound(assigned_count, unassigned_edges):
            """Upper bound on total edges that can be coloured.
            Simple bound: assigned_count + len(unassigned_edges)
            Tighter: for each unassigned edge, check if at least one colour is available."""
            count = assigned_count
            for e in unassigned_edges:
                used_by_conf = set()
                for j in conf[e]:
                    if colour[j] >= 0:
                        used_by_conf.add(colour[j])
                if len(used_by_conf) < K:
                    count += 1
            return count
        
        def dfs(pos, assigned_count):
            nonlocal best, best_score
            if time.time() > deadline:
                return
            
            if pos == m:
                # All edges coloured
                score = assigned_count  # all m edges coloured
                if score > best_score:
                    best_score = score
                    best = list(colour)
                return
            
            e = pos_to_edge[pos]
            # Determine available colours for edge e
            used_by_conf = set()
            for j in conf[e]:
                if colour[j] >= 0:
                    used_by_conf.add(colour[j])
            
            available = [c for c in range(K) if c not in used_by_conf]
            if not available:
                return  # dead end
            
            # Guide: prefer the seed colour if available
            if seed_colour and seed_colour[e] in available:
                available.sort(key=lambda c: 0 if c == seed_colour[e] else 1)
            else:
                available.sort()
            
            for c in available:
                colour[e] = c
                colour_used[c].add(e)
                
                # Prune: check if remaining unassigned can still all be coloured
                # (lightweight check)
                dfs(pos + 1, assigned_count + 1)
                
                colour[e] = -1
                colour_used[c].remove(e)
        
        dfs(0, 0)
        
        if best is not None:
            return best
        return None

    def local_compress(colour, conflict, m, K, deadline, rng):
        """Try to reduce the number of colours by merging colour classes
        and moving edges."""
        colour = list(colour)
        # Reindex to 0..k-1
        present = sorted(set(colour))
        remap = {c: i for i, c in enumerate(present)}
        for i in range(m):
            colour[i] = remap[colour[i]]
        k = len(present)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Build colour classes
            classes = defaultdict(list)
            for i in range(m):
                classes[colour[i]].append(i)
            
            # Try to merge classes
            cols = sorted(classes.keys())
            merged = True
            while merged and time.time() < deadline:
                merged = False
                for c1 in cols:
                    if c1 not in classes:
                        continue
                    for c2 in cols:
                        if c2 <= c1 or c2 not in classes:
                            continue
                        # Check if all edges in class c2 are compatible with class c1
                        s1 = set(classes[c1])
                        ok = True
                        for e in classes[c2]:
                            if conf_set_overlap(e, s1, conflict):
                                ok = False
                                break
                        if ok:
                            for e in classes[c2]:
                                colour[e] = c1
                            del classes[c2]
                            merged = True
                            improved = True
                            break
                    if merged:
                        break
            
            # Single-edge moves
            if not improved and time.time() < deadline:
                cols = sorted(classes.keys())
                if len(cols) >= 2:
                    for _ in range(100):
                        e = rng.randrange(m)
                        c = colour[e]
                        others = [c2 for c2 in cols if c2 != c and c2 in classes]
                        if not others:
                            continue
                        target = rng.choice(others)
                        s_target = set(classes[target])
                        if not conf_set_overlap(e, s_target, conflict):
                            colour[e] = target
                            classes[c].remove(e)
                            classes[target].append(e)
                            improved = True
                            break
        
        # Reindex
        present = sorted(set(colour))
        remap = {c: i for i, c in enumerate(present)}
        for i in range(m):
            colour[i] = remap[colour[i]]
        return colour

    def conf_set_overlap(e, edge_set, conflict):
        """Check if edge e conflicts with any edge in edge_set."""
        for j in edge_set:
            if j in conflict[e]:
                return True
        return False

    results = []
    for idx, (n, d, seed) in enumerate(((40, 4, 101), (60, 5, 102), (80, 6, 103))):
        edges = gen_instance(n, d, seed)
        m = len(edges)
        if m == 0:
            results.append([])
            continue
        
        adj = build_adjacency(edges)
        conflict = build_edge_conflict(edges, adj)
        
        # Start with a generous K
        K_max = min(m, 5 * d * d // 4 + 2)
        
        # Get initial seed via DSATUR
        seed_colour = dsatur_greedy(edges, conflict, adj, K_max)
        if seed_colour is None:
            # Fallback: simple greedy
            seed_colour = [0] * m
            for i in range(m):
                used = set()
                for j in conflict[i]:
                    if j < i:
                        used.add(seed_colour[j])
                c = 0
                while c in used:
                    c += 1
                seed_colour[i] = c
            K_max = max(seed_colour) + 1
        
        current_k = max(seed_colour) + 1
        
        # Try to reduce K via MWIS B&B
        remaining_instances = 3 - idx
        time_per_instance = 140.0 / 3.0
        instance_deadline = t0 + time_per_instance * (idx + 1) - 5.0
        
        for K_try in range(current_k, 0, -1):
            if time.time() > instance_deadline:
                break
            if K_try < 1:
                break
            # Try to find a valid colouring with K_try colours
            result = mwis_branch_and_bound(edges, conflict, K_try, seed_colour, instance_deadline)
            if result is not None:
                # Validate
                if is_valid_colouring(result, conflict, edges):
                    # Compress
                    result = local_compress(result, conflict, m, K_try, instance_deadline, Random(seed + K_try))
                    seed_colour = result
                    current_k = max(result) + 1 if result else 0
                    break
        
        # Final local compression
        if seed_colour is not None:
            seed_colour = local_compress(seed_colour, conflict, m, current_k, instance_deadline, Random(seed + 777))
        
        results.append(seed_colour if seed_colour is not None else [])
    
    return results
