import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    def instance(n, k, p, seed):
        rng = random.Random(seed)
        edges = [(u, v) for u in range(k) for v in range(u + 1, k)]
        edges += [(u, v) for v in range(k, n) for u in range(k) if rng.random() < p]
        return sorted(edges)

    specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
    results = []
    
    for n, k, p, seed in specs:
        if time.time() > deadline:
            # Fallback to seed if timeout
            return [[list(e) for e in instance(*spec)] for spec in specs]
            
        edges = instance(n, k, p, seed)
        edge_set = set(edges)
        
        # Identify K and I
        K = list(range(k))
        I = list(range(k, n))
        
        # Adjacency for I-K edges
        N = {v: [] for v in I}
        for u in K:
            for v in I:
                if (u, v) in edge_set or (v, u) in edge_set:
                    N[v].append(u)
                    
        # Initialize mixed cliques
        # Each v in I will have a list of cliques.
        # Initially, one clique: {v} + N[v]
        mixed_cliques = {v: [[v] + sorted(N[v])] for v in I if N[v]}
        
        # Track which K-edges are covered by which mixed clique
        # edge -> set of v's
        edge_cover = defaultdict(set)
        
        def update_coverage(v, cliques):
            # Remove old coverage for v
            # This is inefficient to do globally. 
            # Better: Rebuild edge_cover from scratch? 
            # n is small (90). K is 30. Edges in K is 435.
            # Rebuilding is fast.
            pass

        # Let's just rebuild edge_cover after each modification
        def rebuild_edge_cover():
            nonlocal edge_cover
            edge_cover = defaultdict(set)
            for v in I:
                if v not in mixed_cliques: continue
                for clique in mixed_cliques[v]:
                    # clique contains v and some K vertices
                    K_part = [x for x in clique if x < k]
                    # All pairs in K_part are covered by this clique
                    for i in range(len(K_part)):
                        for j in range(i+1, len(K_part)):
                            u, w = K_part[i], K_part[j]
                            if u > w: u, w = w, u
                            edge_cover[(u, w)].add(v)
                            
        rebuild_edge_cover()
        
        # Resolve conflicts
        # We want to minimize the number of cliques.
        # Conflict: edge covered by >1 v.
        # Strategy: While conflicts exist, pick a v with many conflicts, 
        # and split its N[v] to resolve them.
        
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline: break
            
            # Find conflicts
            conflicts = {e: vs for e, vs in edge_cover.items() if len(vs) > 1}
            if not conflicts:
                break
                
            # Pick a v involved in the most conflicts
            v_conflict_count = defaultdict(int)
            for e, vs in conflicts.items():
                for v in vs:
                    v_conflict_count[v] += 1
            
            if not v_conflict_count: break
            
            v = max(v_conflict_count, key=v_conflict_count.get)
            
            # Get the current cliques for v
            # We assume v currently has one large clique or several.
            # We need to partition N[v] into subsets S_i such that 
            # no conflict edge is contained within any S_i.
            
            # Build conflict graph on N[v]
            # Nodes: N[v]
            # Edge (u, w) if (u, w) is a conflict AND v is in the conflict set for (u, w)
            # Actually, if (u, w) is a conflict, it means it's covered by v and some other v'.
            # We must separate u and w in v's partition.
            
            nodes = N[v]
            # Map node to index
            node_idx = {u: i for i, u in enumerate(nodes)}
            adj = {u: set() for u in nodes}
            
            for e, vs in conflicts.items():
                if v in vs:
                    u, w = e
                    if u in node_idx and w in node_idx:
                        adj[u].add(w)
                        adj[w].add(u)
                        
            # Greedy coloring of the conflict graph
            # Colors represent the partition sets S_i
            colors = {}
            color_count = 0
            # Order nodes by degree (descending) for better coloring
            sorted_nodes = sorted(nodes, key=lambda x: len(adj[x]), reverse=True)
            
            for u in sorted_nodes:
                # Find available color
                used_colors = {colors[w] for w in adj[u] if w in colors}
                c = 0
                while c in used_colors:
                    c += 1
                colors[u] = c
                color_count = max(color_count, c + 1)
                
            # Construct new cliques for v
            new_cliques = []
            for c in range(color_count):
                S_c = [u for u in nodes if colors[u] == c]
                if S_c:
                    new_cliques.append([v] + sorted(S_c))
            
            # Update mixed_cliques for v
            mixed_cliques[v] = new_cliques
            
            # Rebuild edge cover
            rebuild_edge_cover()
            
        # After resolving mixed conflicts, cover remaining K-edges
        # Remaining edges are those with len(vs) == 0 in edge_cover
        remaining_edges = [e for e, vs in edge_cover.items() if len(vs) == 0]
        
        # Cover remaining_edges with pure K-cliques
        # Greedy: Find maximal cliques in the subgraph induced by remaining_edges
        # Since it's a subgraph of K, a clique is a set of vertices with all pairs in remaining_edges.
        
        rem_set = set(remaining_edges)
        pure_cliques = []
        
        while rem_set:
            if time.time() > deadline: break
            # Pick a vertex with max degree in rem_set
            deg = defaultdict(int)
            for u, v in rem_set:
                deg[u] += 1
                deg[v] += 1
            
            if not deg: break
            
            start_v = max(deg, key=deg.get)
            
            # Expand to maximal clique
            clique = {start_v}
            # Greedy expansion
            changed = True
            while changed:
                changed = False
                candidates = [u for u in deg if u not in clique]
                for u in candidates:
                    # Check if u is connected to all in clique
                    is_clique = all((min(u, w), max(u, w)) in rem_set for w in clique)
                    if is_clique:
                        clique.add(u)
                        changed = True
                        # Remove edges incident to u that are now covered?
                        # No, we remove the whole clique's edges at the end.
                        
            # Convert clique to list
            clique_list = sorted(clique)
            pure_cliques.append(clique_list)
            
            # Remove edges in this clique from rem_set
            for i in range(len(clique_list)):
                for j in range(i+1, len(clique_list)):
                    e = (min(clique_list[i], clique_list[j]), max(clique_list[i], clique_list[j]))
                    rem_set.discard(e)
                    
        # Combine results
        all_cliques = []
        for v in I:
            if v in mixed_cliques:
                all_cliques.extend(mixed_cliques[v])
        all_cliques.extend(pure_cliques)
        
        # Validation and Type Conversion
        # Ensure all are lists of ints
        final_cliques = []
        for c in all_cliques:
            if len(c) < 2: continue # Should not happen, but safety
            # Check distinct
            if len(set(c)) != len(c): continue
            final_cliques.append([int(x) for x in c])
            
        results.append(final_cliques)
        
    return results
