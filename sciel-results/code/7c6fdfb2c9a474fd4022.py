import time
import random
import sys
from collections import defaultdict, deque

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_long_cycle(adj, n, rng):
    """
    Find a long simple cycle using randomized greedy DFS.
    """
    best_cycle = None
    best_len = 0
    
    # Try starting from high degree vertices
    starts = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    # Randomize order slightly
    rng.shuffle(starts)
    
    for start in starts[:min(n, 20)]:
        if not adj[start]:
            continue
            
        # Restart a few times for each start
        for _ in range(5):
            path = [start]
            visited = {start}
            current = start
            
            while True:
                neighbors = list(adj[current])
                unvisited = [v for v in neighbors if v not in visited]
                
                if not unvisited:
                    # Check if we can close
                    if start in adj[current] and len(path) >= 3:
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = path[:]
                    break
                
                # Heuristic: pick neighbor with most unvisited neighbors
                # But also consider closing if close to start
                scored = []
                for v in unvisited:
                    score = sum(1 for w in adj[v] if w not in visited)
                    # Bonus if v is start (to close)
                    if v == start:
                        score += 100
                    scored.append((score, v))
                
                scored.sort(reverse=True)
                # Pick best or random from top 3
                if rng.random() < 0.8:
                    next_v = scored[0][1]
                else:
                    next_v = rng.choice([s[1] for s in scored[:min(3, len(scored))]])
                
                path.append(next_v)
                visited.add(next_v)
                current = next_v
                
                # Optional: Early termination if path is very long
                if len(path) > n * 0.9:
                    break

    return best_cycle

def _decompose_greedy(edges, n, rng):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    pieces = []
    remaining = len(edges)
    
    while remaining > 0:
        cycle = _find_long_cycle(adj, n, rng)
        if cycle:
            m = len(cycle)
            for i in range(m):
                u, v = cycle[i], cycle[(i+1)%m]
                adj[u].discard(v)
                adj[v].discard(u)
            remaining -= m
            pieces.append(cycle)
        else:
            # No more cycles, add all remaining edges as singles
            for u in range(n):
                for v in list(adj[u]):
                    if u < v:
                        adj[u].discard(v)
                        adj[v].discard(u)
                        remaining -= 1
                        pieces.append([u, v])
                        
    return pieces

def _refine(pieces, n, rng):
    """
    Refine decomposition by absorbing single edges into cycles.
    """
    # Separate cycles and singles
    cycles = [p for p in pieces if len(p) >= 3]
    singles = [p for p in pieces if len(p) == 2]
    
    # If no cycles or no singles, return
    if not cycles or not singles:
        return pieces
        
    # Build adjacency for singles
    adj_single = defaultdict(set)
    for u, v in singles:
        adj_single[u].add(v)
        adj_single[v].add(u)
        
    improved = True
    while improved:
        improved = False
        
        # Try to absorb a path from singles into a cycle
        # We look for a path in adj_single between two vertices of the same cycle
        # that is longer than the arc in the cycle
        
        # To make this efficient, we can just check pairs of vertices in each cycle
        # and see if there's a path in adj_single
        
        new_cycles = []
        new_singles = list(singles)
        
        for c in cycles:
            # Map vertex to index in cycle
            # Try all pairs u, v in c
            # If path exists in adj_single, check length
            # This is O(|c|^2 * BFS)
            # |c| is small, so it's fine
            
            absorbed = False
            for i in range(len(c)):
                u = c[i]
                for j in range(i+1, len(c)):
                    v = c[j]
                    if u == v: continue
                    
                    # Find path from u to v in adj_single
                    # BFS
                    prev = {u: None}
                    q = deque([u])
                    found = False
                    while q:
                        cur = q.popleft()
                        if cur == v:
                            found = True
                            break
                        for nei in adj_single[cur]:
                            if nei not in prev:
                                prev[nei] = cur
                                q.append(nei)
                    
                    if found:
                        # Reconstruct path
                        path = []
                        cur = v
                        while cur is not None:
                            path.append(cur)
                            cur = prev[cur]
                        path.reverse()
                        path_len = len(path) - 1 # number of edges
                        
                        # Determine arc in cycle c from u to v
                        # There are two arcs. We want to replace the shorter one?
                        # Or the one that makes the new cycle valid?
                        # The new cycle will use the path and the *other* arc.
                        # So we should replace the arc that we are NOT using.
                        # Wait.
                        # Cycle c is u ... v ... u.
                        # Path P is u ... v.
                        # New cycle: P + arc(v to u in c).
                        # We remove the arc(u to v in c) from the cycle.
                        # So we want to remove the arc(u, v) and keep arc(v, u).
                        # The length of arc(u, v) is k1.
                        # The length of arc(v, u) is k2.
                        # k1 + k2 = len(c).
                        # We remove k1 edges from the cycle (they become singles).
                        # We add path_len edges to the cycle (they were singles).
                        # Change in cycle size: path_len - k1.
                        # Change in single count: -path_len + k1.
                        # Total piece count change:
                        # Before: 1 (cycle) + path_len (singles) + ...
                        # After: 1 (cycle) + k1 (singles) + ...
                        # We want to reduce total pieces.
                        # So we want k1 < path_len.
                        
                        # Find k1 (length of arc u->v in c)
                        # c is a list.
                        idx_u = c.index(u)
                        idx_v = c.index(v)
                        
                        # Arc 1: u -> ... -> v (forward)
                        len1 = (idx_v - idx_u) % len(c)
                        if len1 == 0: len1 = len(c)
                        
                        # Arc 2: v -> ... -> u (forward)
                        len2 = len(c) - len1
                        
                        # We replace Arc 1 with Path P.
                        # So we remove Arc 1 edges.
                        # Condition: path_len > len1
                        
                        if path_len > len1:
                            # Perform swap
                            # Remove Arc 1 edges from cycle, add to singles
                            # Remove Path edges from singles, add to cycle
                            
                            # Identify Arc 1 edges
                            arc_edges = []
                            for k in range(len1):
                                node1 = c[(idx_u + k) % len(c)]
                                node2 = c[(idx_u + k + 1) % len(c)]
                                arc_edges.append(tuple(sorted((node1, node2))))
                            
                            # Identify Path edges
                            path_edges = []
                            for k in range(len(path)-1):
                                node1 = path[k]
                                node2 = path[k+1]
                                path_edges.append(tuple(sorted((node1, node2))))
                            
                            # Update singles list
                            # Remove path_edges from new_singles
                            # Add arc_edges to new_singles
                            
                            # This is expensive to do with lists.
                            # Let's mark them.
                            
                            # Actually, let's just construct the new cycle
                            # New cycle: Path + Arc 2
                            # Path: u -> ... -> v
                            # Arc 2: v -> ... -> u
                            
                            new_cycle = list(path)
                            # Append Arc 2 vertices (excluding v and u to avoid dup? No, u is start of path, v is end)
                            # Path ends at v. Arc 2 starts at v.
                            # So we append vertices of Arc 2 starting from next of v up to u.
                            
                            # Arc 2 is from v to u in c.
                            # c index of v is idx_v.
                            # We want c[idx_v+1], ..., c[idx_u-1]
                            for k in range(len2 - 1):
                                new_cycle.append(c[(idx_v + 1 + k) % len(c)])
                            # Last vertex is u, which is path[0].
                            # So new_cycle is simple if path internal vertices not in c.
                            
                            # Check validity
                            if len(new_cycle) == len(set(new_cycle)):
                                # Valid
                                # Update data structures
                                # This is complex to do in-place with lists.
                                # Let's just return a flag and do it outside?
                                # Or rebuild the whole list.
                                
                                # For simplicity in this step, let's just record that we found an improvement
                                # and rebuild at the end of the while loop?
                                # No, we need to update adj_single and cycles list.
                                
                                # Let's do it properly.
                                # Remove path_edges from singles
                                new_singles = [e for e in new_singles if tuple(sorted(e)) not in path_edges]
                                # Add arc_edges to singles
                                new_singles.extend([list(e) for e in arc_edges])
                                
                                # Replace cycle c with new_cycle in cycles
                                # We are iterating over cycles, so we need to be careful.
                                # Let's just break and restart the loop?
                                # Or update a local variable.
                                
                                # Since we are modifying 'cycles' and 'new_singles', 
                                # and we are in a loop over 'cycles', 
                                # it's safer to just return and let the outer while loop handle it?
                                # No, _refine is called once.
                                
                                # Let's update the local structures and break out to restart the while loop.
                                improved = True
                                # Update global state for next iteration of while
                                # We need to update 'cycles' and 'singles'
                                # Let's just return the improved pieces immediately?
                                # No, we might find more.
                                
                                # Let's update the lists in place.
                                # This is tricky.
                                # Let's just break out of the for loop and restart the while loop.
                                # We need to store the new state.
                                
                                # To keep it simple and robust:
                                # If we find an improvement, we update the local variables 
                                # and break out to restart the while loop.
                                
                                # Update 'cycles'
                                # Find index of c in cycles
                                # Replace it
                                # Update 'singles'
                                
                                # Actually, let's just return the new pieces and let the caller 
                                # call _refine again? No, that's inefficient.
                                
                                # Let's just do the update and set improved=True and break.
                                # We need to update 'cycles' and 'singles' variables in the function scope.
                                
                                # Since we are in a nested loop, let's just use a flag.
                                # We'll rebuild the pieces list at the end of the while loop iteration.
                                
                                # For now, let's just assume we can update.
                                # We'll replace c in cycles with new_cycle.
                                # And update singles.
                                
                                # To avoid complexity, let's just return the new pieces 
                                # and rely on the while loop in _solve_one to call _refine again?
                                # No, _refine should do all refinements.
                                
                                # Let's just do the update and break.
                                # We'll store the new cycle and new singles in temporary vars 
                                # and assign them back before breaking.
                                
                                # This is getting messy.
                                # Simpler approach:
                                # Just return the pieces with the improvement applied 
                                # and let the outer loop in _solve_one call _refine again 
                                # if it wants? No.
                                
                                # Let's just implement the update carefully.
                                
                                # Update cycles list
                                cycles[cycles.index(c)] = new_cycle
                                # Update singles list
                                # We already have new_singles
                                singles = new_singles
                                
                                # Update adj_single
                                adj_single = defaultdict(set)
                                for u, v in singles:
                                    adj_single[u].add(v)
                                    adj_single[v].add(u)
                                    
                                break # Break inner for loop
                            
            if improved:
                break # Break outer for loop over cycles
        
        if not improved:
            break
            
    # Combine back
    return cycles + singles

def _solve_one(edges, n, deadline):
    best = None
    best_count = float('inf')
    
    start_time = time.time()
    attempts = 0
    max_attempts = 50 # Reduced attempts to allow time for refinement
    
    while time.time() < deadline - 1.0:
        seed = 12345 + attempts
        rng = random.Random(seed)
        pieces = _decompose_greedy(edges, n, rng)
        
        # Refine
        pieces = _refine(pieces, n, rng)
        
        cnt = len(pieces)
        if cnt < best_count:
            best_count = cnt
            best = pieces
            
        attempts += 1
        if attempts >= max_attempts:
            break
            
    return best if best else [[u, v] for u, v in edges]

def construct():
    start_time = time.time()
    total_budget = 145.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
