import time
import random

def construct():
    n = 15
    deadline = time.time() + 82  # Safety margin (90s limit)
    
    def get_degree(e_set):
        deg = [0] * n
        for (a, b) in e_set:
            deg[a] += 1
            deg[b] += 1
        return deg
    
    def is_valid(e_set):
        deg = get_degree(e_set)
        adj = [set() for _ in range(n)]
        for (a, b) in e_set:
            adj[a].add(b)
            adj[b].add(a)
        # Check all paths s-u-v-t
        for (u, v) in e_set:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True
    
    # Better initial construction: try to build a dense valid graph
    # Start with a structure that avoids equal-degree path endpoints
    # Use a "ladder" pattern with careful degree assignment
    
    edges = set()
    
    # Try multiple random initial seeds and pick the best
    best_edges = None
    best_count = 0
    
    # Strategy 1: Start with K_{7,8} bipartite and add internal edges
    e1 = set()
    for u in range(7):
        for v in range(7, 15):
            e1.add((u, v))
    if is_valid(e1):
        if len(e1) > best_count:
            best_edges = e1
            best_count = len(e1)
    
    # Strategy 2: Try adding edges within partitions to K_{7,8}
    e2 = set(e1)
    # Add some edges within {0..6} and within {7..14}
    for i in range(7):
        for j in range(i+1, 7):
            e2.add((i, j))
    for i in range(7):
        for j in range(i+1, 15):
            e2.add((i, j))
    if is_valid(e2):
        if len(e2) > best_count:
            best_edges = e2
            best_count = len(e2)
    
    # Strategy 3: Greedy construction with degree balancing
    e3 = set()
    # Add edges in order of increasing degree sum to balance degrees
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.seed(42)
    random.shuffle(all_pairs)
    for (i, j) in all_pairs:
        if time.time() > deadline - 60:
            break
        new_e = e3 | {(i, j)}
        if is_valid(new_e):
            e3 = new_e
    if len(e3) > best_count:
        best_edges = e3
        best_count = len(e3)
    
    # Use the best initial construction
    edges = best_edges if best_edges else e1
    
    # Phase 1: Greedy edge addition
    deg = get_degree(edges)
    all_possible = set()
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) not in edges and (j, i) not in edges:
                all_possible.add((i, j))
    
    # Sort by heuristic: prefer edges that keep degrees balanced
    candidates = sorted(all_possible, key=lambda e: -(deg[e[0]] + deg[e[1]]))
    
    improved = True
    while improved and time.time() < deadline - 30:
        improved = False
        for e in list(candidates):
            if time.time() > deadline - 30:
                break
            if e in edges:
                continue
            new_edges = edges | {e}
            if is_valid(new_edges):
                edges = new_edges
                deg = get_degree(edges)
                candidates = [c for c in candidates if c not in edges]
                improved = True
                break
        if improved:
            # Re-sort
            candidates.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
    
    # Phase 2: Local search with edge swaps (remove one, add another)
    # This can escape local optima
    current_count = len(edges)
    deg = get_degree(edges)
    
    # Precompute all possible edges
    all_edges_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Try removing an edge and adding a different one
    for _ in range(50):  # Limit iterations for time
        if time.time() > deadline - 15:
            break
        
        # Try random swaps
        made_swap = False
        for _ in range(100):
            if time.time() > deadline - 15:
                break
            
            # Pick a random edge to potentially remove
            edge_list = list(edges)
            if not edge_list:
                break
            remove_e = random.choice(edge_list)
            
            # Pick a random edge to potentially add (not already in edges)
            missing_edges = [e for e in all_edges_possible if e not in edges]
            if not missing_edges:
                break
            add_e = random.choice(missing_edges)
            
            # Try the swap
            new_edges = (edges - {remove_e}) | {add_e}
            if is_valid(new_edges):
                # Only accept if it doesn't reduce edge count (swap keeps count same)
                # Or if we're trying to add more, this won't help
                # For now, just use swaps to explore, then try to add
                edges = new_edges
                made_swap = True
                break
        
        # After swap, try to add edges again
        if made_swap or True:
            deg = get_degree(edges)
            missing_edges = [e for e in all_edges_possible if e not in edges]
            random.shuffle(missing_edges)
            for e in missing_edges[:20]:  # Try a few additions
                if time.time() > deadline - 15:
                    break
                new_edges = edges | {e}
                if is_valid(new_edges):
                    edges = new_edges
                    deg = get_degree(edges)
    
    # Final greedy pass
    deg = get_degree(edges)
    missing_edges = [(i, j) for i in range(n) for j in range(i+1, n) if (i, j) not in edges]
    missing_edges.sort(key=lambda e: -(deg[e[0]] + deg[e[1]]))
    for e in missing_edges:
        if time.time() > deadline - 5:
            break
        new_edges = edges | {e}
        if is_valid(new_edges):
            edges = new_edges
            deg = get_degree(edges)
    
    # Convert to list of lists
    result = []
    for (u, v) in edges:
        result.append([int(u), int(v)])
    
    return result
