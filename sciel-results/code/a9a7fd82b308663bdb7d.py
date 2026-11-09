import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    DEADLINE = 148.0

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose(edges, deadline):
        """
        Strategy: Decompose the graph into a maximum matching M and the remaining edges R.
        Gallai's Conjecture states that the path cover number is at most ceil(|M|/2) + |R|/2? 
        Actually, the bound is related to the number of odd components or similar.
        A constructive approach:
        1. Find a maximal matching M.
        2. The edges in M form disjoint paths of length 1 (which can be merged if they form a cycle, but they are disjoint).
        3. The remaining edges R form a subgraph.
        4. Any path in R can be extended by matching edges if endpoints match.
        
        More effective strategy:
        1. Start with a greedy path decomposition.
        2. Apply local search to minimize path count.
        
        Let's use a specific heuristic:
        1. Compute a 2-factor or near-2-factor? No, graphs are sparse.
        2. Use the fact that we can merge paths if they share endpoints.
        
        Alternative: 
        1. Find a maximum matching.
        2. Contract matching edges?
        
        Let's stick to a robust local search on the path cover.
        1. Initial: Greedy Warnsdorff.
        2. Local Search:
           a. Endpoint Merging (as in current).
           b. "Slide" operation: Move a vertex from one path to another to create merge opportunities.
           c. "Re-route" operation: Swap segments between paths.
        """
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Helper to check if a list of vertices is a valid simple path using only edges in 'remaining_edges'
        # We don't need to check edges if we maintain the set of used edges carefully.
        
        def greedy_init(seed_offset=0):
            rng = random.Random(seed_offset)
            remaining = set(edges)
            paths = []
            
            while remaining:
                # Pick a start node: highest degree in remaining graph
                deg = defaultdict(int)
                for u, v in remaining:
                    deg[u] += 1
                    deg[v] += 1
                if not deg: break
                # Sort by degree desc, then random tie-break
                candidates = sorted(deg.keys(), key=lambda x: (-deg[x], rng.random()))
                start = candidates[0]
                
                path = [start]
                pverts = {start}
                pedges = set()
                
                while True:
                    # Warnsdorff: pick neighbor with fewest available moves
                    best_next = None
                    best_score = float('inf')
                    # Shuffle neighbors for randomization
                    nbrs = list(adj[start])
                    rng.shuffle(nbrs)
                    for v in nbrs:
                        e = (min(start, v), max(start, v))
                        if e in remaining and v not in pverts:
                            # Count available moves from v
                            moves = 0
                            for w in adj[v]:
                                e2 = (min(v, w), max(v, w))
                                if e2 in remaining and w not in pverts and w != start:
                                    moves += 1
                            # Add noise to break ties
                            score = moves + rng.random() * 0.1
                            if score < best_score:
                                best_score = score
                                best_next = v
                    if best_next is None:
                        break
                    path.append(best_next)
                    pverts.add(best_next)
                    e = (min(start, best_next), max(start, best_next))
                    pedges.add(e)
                    start = best_next
                    
                if len(path) > 1:
                    paths.append(path)
                    for e in pedges:
                        remaining.discard(e)
                else:
                    # If no path found (should not happen if connected), take any edge
                    if remaining:
                        e = next(iter(remaining))
                        paths.append(list(e))
                        remaining.discard(e)
            return paths

        def try_merge(paths):
            # Try to merge any pair of paths that share an endpoint and are otherwise disjoint
            for i in range(len(paths)):
                for j in range(i+1, len(paths)):
                    p1 = paths[i]
                    p2 = paths[j]
                    if len(p1) < 2 or len(p2) < 2:
                        continue
                    set1 = set(p1)
                    set2 = set(p2)
                    common = set1 & set2
                    if len(common) != 1:
                        continue
                    shared = list(common)[0]
                    
                    # Check if shared is an endpoint in both
                    is_ep1 = (p1[0] == shared or p1[-1] == shared)
                    is_ep2 = (p2[0] == shared or p2[-1] == shared)
                    if is_ep1 and is_ep2:
                        # Determine merge order
                        # Case 1: p1 ends at shared, p2 starts at shared -> p1 + p2[1:]
                        if p1[-1] == shared and p2[0] == shared:
                            new_p = p1 + p2[1:]
                        # Case 2: p1 starts at shared, p2 ends at shared -> p2 + p1[1:]
                        elif p1[0] == shared and p2[-1] == shared:
                            new_p = p2 + p1[1:]
                        # Case 3: p1 starts at shared, p2 starts at shared -> p1[::-1] + p2[1:]
                        elif p1[0] == shared and p2[0] == shared:
                            new_p = p1[::-1] + p2[1:]
                        # Case 4: p1 ends at shared, p2 ends at shared -> p2 + p1[::-1][1:]
                        elif p1[-1] == shared and p2[-1] == shared:
                            new_p = p2 + p1[::-1][1:]
                        else:
                            continue
                            
                        if len(new_p) == len(set(new_p)):
                            # Valid merge
                            new_paths = [p for k, p in enumerate(paths) if k != i and k != j]
                            new_paths.append(new_p)
                            return new_paths
            return None

        def local_search(paths, deadline):
            # Repeatedly try to merge
            changed = True
            while changed and time.time() < deadline:
                changed = False
                new_paths = try_merge(paths)
                if new_paths is not None:
                    paths = new_paths
                    changed = True
            
            # Try "splitting and re-merging" to escape local optima?
            # Or try to move an internal vertex to an endpoint?
            # Simple move: Pick a path, split it into two, and see if one part can merge with another path.
            # This is complex. Let's try a simpler "2-opt" like move:
            # Reverse a segment? Doesn't help with endpoints.
            
            # Let's try a "swap" move:
            # Take an internal edge (u,v) from path P. Remove it. P becomes P1, P2.
            # If P1 or P2 can now merge with another path, do it.
            # This effectively allows "bending" the path.
            
            if time.time() > deadline - 2:
                return paths
                
            improved = True
            while improved and time.time() < deadline - 1:
                improved = False
                for i, p in enumerate(paths):
                    if len(p) < 3: continue
                    # Try splitting at each internal edge
                    for k in range(1, len(p)-1):
                        # Split p into p[0:k+1] and p[k:]
                        p1 = p[:k+1]
                        p2 = p[k:]
                        
                        # Check if p1 or p2 can merge with any other path
                        # We need to check if p1 (or p2) shares an endpoint with another path and is disjoint.
                        # This is expensive. Let's just try to see if removing this edge allows a merge.
                        # Actually, just try to merge p1 with others, or p2 with others.
                        
                        # To save time, only try a few splits
                        if k > 2 and k < len(p)-2:
                            continue
                            
                        # Try merging p1
                        for j in range(len(paths)):
                            if j == i: continue
                            q = paths[j]
                            # Check if p1 and q share an endpoint and are disjoint
                            set_p1 = set(p1)
                            set_q = set(q)
                            if len(p1) < 2 or len(q) < 2: continue
                            common = set_p1 & set_q
                            if len(common) == 1:
                                shared = list(common)[0]
                                is_ep1 = (p1[0] == shared or p1[-1] == shared)
                                is_ep2 = (q[0] == shared or q[-1] == shared)
                                if is_ep1 and is_ep2:
                                    # Form new path from p1 and q
                                    if p1[-1] == shared and q[0] == shared:
                                        new_p = p1 + q[1:]
                                    elif p1[0] == shared and q[-1] == shared:
                                        new_p = q + p1[1:]
                                    elif p1[0] == shared and q[0] == shared:
                                        new_p = p1[::-1] + q[1:]
                                    elif p1[-1] == shared and q[-1] == shared:
                                        new_p = q + p1[::-1][1:]
                                    else:
                                        continue
                                    if len(new_p) == len(set(new_p)):
                                        # Valid
                                        # New set of paths: p2, new_p, and all others except i, j
                                        new_paths = [paths[x] for x in range(len(paths)) if x != i and x != j]
                                        new_paths.append(p2)
                                        new_paths.append(new_p)
                                        # Check if this reduces total path count?
                                        # It keeps count same (1 split + 1 merge = net 0 change in count? 
                                        # Original: 2 paths (P, Q). New: 2 paths (P2, New).
                                        # But New is longer. This might enable future merges.
                                        # We should accept if it allows further merges.
                                        # For now, let's just accept and let the merge loop handle it.
                                        paths = new_paths
                                        improved = True
                                        break
                        if improved:
                            break
                    if improved:
                        break
                            
            return paths

        # Main loop
        best_paths = None
        best_count = float('inf')
        
        # Run multiple random initializations
        for attempt in range(20):
            if time.time() > deadline - 1:
                break
            
            paths = greedy_init(attempt)
            paths = local_search(paths, deadline)
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
                
            # If we hit the theoretical lower bound (e.g., based on odd degrees), we can stop
            # Lower bound: max(1, sum(deg(v) % 2) / 2)
            deg = defaultdict(int)
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            lb = sum(1 for d in deg.values() if d % 2 == 1) // 2
            if best_count <= lb:
                break

        if best_paths is None:
            best_paths = greedy_init(0)
            
        return best_paths

    result = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges, DEADLINE)
        result.append(paths)
        
    return result

# EVOLVE-BLOCK-END
