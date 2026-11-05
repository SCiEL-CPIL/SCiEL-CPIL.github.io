import time
import random
import numpy as np
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # Helper: Check if a specific edge addition is valid
    # adj: np array of colours, -1 for no edge
    # neighbors: list of lists of (neighbor, colour)
    def can_add_edge(adj, neighbors, u, v):
        # 1. Proper edge coloring: colour must differ from all incident edges at u and v
        forbidden = set()
        for w, c in neighbors[u]:
            if w != v:
                forbidden.add(c)
        for w, c in neighbors[v]:
            if w != u:
                forbidden.add(c)
        
        # 2. Rainbow 4-cycle constraint:
        # Any 4-cycle containing (u,v) must have 4 distinct colours.
        # Cycle structure: u - v - w - x - u
        # Edges: (u,v), (v,w), (w,x), (x,u)
        # We need to check if adding (u,v) with colour c creates a cycle where colours repeat.
        # Actually, we need to find a colour c for (u,v) such that for ALL 4-cycles 
        # containing (u,v), the set {c, c_vw, c_wx, c_xu} has size 4.
        
        # This implies:
        # a) c != c_vw, c != c_wx, c != c_xu for any such cycle.
        # b) c_vw, c_wx, c_xu must already be distinct (otherwise the cycle is invalid 
        #    regardless of c, but wait, if the cycle exists, it must be valid. 
        #    If the 3 edges already have a repeat, the graph is already invalid? 
        #    No, the 3 edges don't form a cycle. They form a path. 
        #    However, if c_vw == c_wx, then any c we pick will result in a repeat in the 4-cycle.
        #    So, if there exists a 4-cycle path u-v-w-x-u where the 3 existing edges 
        #    don't have distinct colours, we CANNOT add (u,v) at all.
        
        # Let's iterate over all potential 4-cycles involving (u,v).
        # Path: v - w - x - u
        # w must be neighbor of v
        # x must be neighbor of u
        # (w,x) must be an edge
        
        for w, c_vw in neighbors[v]:
            if w == u: continue
            for x, c_xu in neighbors[u]:
                if x == v or x == w: continue
                if adj[w, x] == -1: continue
                c_wx = adj[w, x]
                
                # Check if the 3 existing edges are distinct
                if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                    return None # Cannot add edge, would create non-rainbow cycle
                
                # The new colour c must be distinct from c_vw, c_wx, c_xu
                forbidden.add(c_vw)
                forbidden.add(c_wx)
                forbidden.add(c_xu)
                
        # Find a colour not in forbidden
        for c in range(num_colours):
            if c not in forbidden:
                return c
        return None

    def build_neighbors(adj):
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j] != -1:
                    c = adj[i, j]
                    neighbors[i].append((j, c))
                    neighbors[j].append((i, c))
        return neighbors

    def is_valid(adj):
        # Full verification
        # Precompute 4-cycles? For n=20, number of 4-cycles is manageable.
        # But for speed, we can just check all 4-subsets of vertices.
        # A 4-cycle on {a,b,c,d} is a permutation where edges exist.
        # There are 3 distinct 4-cycles on 4 vertices:
        # (a,b,c,d), (a,b,d,c), (a,c,b,d)
        
        # To be safe and fast, let's iterate all 4-combinations
        # and check the 3 possible cycle structures.
        for combo in combinations(range(n), 4):
            a, b, c, d = combo
            # Cycle 1: a-b-c-d-a
            if adj[a,b]!=-1 and adj[b,c]!=-1 and adj[c,d]!=-1 and adj[d,a]!=-1:
                if len({adj[a,b], adj[b,c], adj[c,d], adj[d,a]}) < 4:
                    return False
            # Cycle 2: a-b-d-c-a
            if adj[a,b]!=-1 and adj[b,d]!=-1 and adj[d,c]!=-1 and adj[c,a]!=-1:
                if len({adj[a,b], adj[b,d], adj[d,c], adj[c,a]}) < 4:
                    return False
            # Cycle 3: a-c-b-d-a
            if adj[a,c]!=-1 and adj[c,b]!=-1 and adj[b,d]!=-1 and adj[d,a]!=-1:
                if len({adj[a,c], adj[c,b], adj[b,d], adj[d,a]}) < 4:
                    return False
        return True

    # Strategy:
    # 1. Try to construct a dense graph using a structured approach (e.g. random dense start + repair)
    # 2. Use local search (add/remove) to maximize edges.
    
    best_adj = None
    best_edges = 0
    
    all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
    
    # Approach A: Random Dense Start + Repair
    # Start with a random subset of edges, assign random colours, then fix conflicts.
    # This is hard to fix.
    
    # Approach B: Greedy Construction with Randomized Order + Local Search
    # This is similar to current but we can improve the local search phase.
    
    # Approach C: "Simulated Annealing" on Edge Set
    # State: Set of edges and their colours.
    # Move: Add an edge, Remove an edge, Change a colour.
    # Objective: Maximize edges subject to validity.
    
    # Given the time limit, let's stick to a robust Greedy + Local Search but with better heuristics.
    
    # Heuristic: 
    # 1. Randomize edge order.
    # 2. Greedy add.
    # 3. Local Search: 
    #    - Try to add any missing edge.
    #    - If stuck, try removing an edge that is "blocking" many others, then re-add.
    #    - Specifically: Identify edges that appear in many "forbidden" sets.
    
    def local_search(adj, neighbors, time_limit):
        # Returns improved adj
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # Phase 1: Try to add all missing edges
            for u, v in all_edges:
                if adj[u, v] != -1:
                    continue
                c = can_add_edge(adj, neighbors, u, v)
                if c is not None:
                    adj[u, v] = c
                    adj[v, u] = c
                    neighbors[u].append((v, c))
                    neighbors[v].append((u, c))
                    improved = True
                    
            if not improved:
                # Phase 2: Try to remove an edge to unlock others
                # Score each edge by how many other edges it blocks.
                # Approximation: Count how many times it appears in "forbidden" checks?
                # Simpler: Randomly remove an edge, then try to add more.
                # Let's try a targeted removal.
                
                # Find an edge whose removal allows adding at least 2 new edges.
                # This is expensive. Let's just try random removals if stuck.
                
                # Pick a random existing edge
                existing_edges = [(u, v) for u, v in all_edges if adj[u, v] != -1]
                if not existing_edges:
                    break
                    
                # Try a few random removals
                for _ in range(10):
                    if time.time() > time_limit:
                        break
                    u, v = random.choice(existing_edges)
                    c_old = adj[u, v]
                    
                    # Remove
                    adj[u, v] = -1
                    adj[v, u] = -1
                    neighbors[u] = [(w, c) for w, c in neighbors[u] if w != v]
                    neighbors[v] = [(w, c) for w, c in neighbors[v] if w != u]
                    
                    # Try to add back u,v and others
                    # First, try to add u,v again? No, we removed it to unlock others.
                    # Try to add all missing edges
                    added_count = 0
                    for eu, ev in all_edges:
                        if adj[eu, ev] != -1:
                            continue
                        c = can_add_edge(adj, neighbors, eu, ev)
                        if c is not None:
                            adj[eu, ev] = c
                            adj[ev, eu] = c
                            neighbors[eu].append((ev, c))
                            neighbors[ev].append((eu, c))
                            added_count += 1
                            
                    # If we added more edges than we removed (1), we improved.
                    # Actually, we removed 1. If added_count > 1, net gain.
                    # If added_count == 1, it might be the same edge or a different one.
                    # If added_count == 0, we lost an edge.
                    
                    if added_count <= 1:
                        # Revert? Or keep if it's the same edge?
                        # If we removed (u,v) and added (u,v) back, it's a no-op.
                        # If we removed (u,v) and added (x,y), we traded one for another.
                        # We want to maximize total edges.
                        # If added_count < 1, we definitely got worse.
                        # If added_count == 1, we are neutral (unless we added a different edge, 
                        # in which case the graph structure changed, which might help future steps).
                        
                        # Let's accept if added_count >= 1 (neutral or better)
                        # But we need to ensure validity. can_add_edge ensures validity.
                        
                        # However, if we added a DIFFERENT edge, the count is same.
                        # This is a "move" in the search space.
                        
                        # If added_count == 0, we should revert to not lose progress?
                        # Actually, losing an edge might be necessary to gain more later.
                        # But simple hill climbing says: only accept if edge count increases.
                        # Let's accept if added_count > 1 (net gain) OR if added_count == 1 
                        # and the added edge is different (exploration).
                        
                        # For simplicity and safety, let's only accept if we strictly increased
                        # the number of edges, OR if we are in a "kick" phase.
                        
                        # Let's just track the max.
                        pass
                    
                    # Rebuild neighbors to ensure consistency after multiple adds
                    neighbors = build_neighbors(adj)
                    current_edges = int(np.sum(adj != -1)) // 2
                    
                    # If we didn't gain, we might want to revert to best known state?
                    # This local search is getting complex.
                    
                    # Let's simplify: 
                    # The "remove and re-add" loop is essentially a random walk.
                    # Let's break out if we don't improve quickly.
                    
        return adj

    # Main Loop: Multiple Random Restarts
    # Since n=20 is small, we can do many restarts.
    
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        # Randomize edge order
        edges = all_edges[:]
        random.shuffle(edges)
        
        adj = np.full((n, n), -1, dtype=int)
        neighbors = [[] for _ in range(n)]
        
        # Greedy Add
        for u, v in edges:
            c = can_add_edge(adj, neighbors, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
                neighbors[u].append((v, c))
                neighbors[v].append((u, c))
                
        # Second Pass (try to add skipped)
        neighbors = build_neighbors(adj)
        for u, v in edges:
            if adj[u, v] != -1:
                continue
            c = can_add_edge(adj, neighbors, u, v)
            if c is not None:
                adj[u, v] = c
                adj[v, u] = c
                neighbors[u].append((v, c))
                neighbors[v].append((u, c))
                
        # Local Search Phase (Short)
        # Try to add more
        neighbors = build_neighbors(adj)
        # Try removing edges to unlock
        # Simple strategy: Try to remove each edge once and see if we can add more.
        # This is O(E^2) per restart, which is 190^2 = 36100 ops. 
        # 5000 restarts * 36100 = 180M ops. Might be too slow in Python.
        
        # Let's limit local search to a few random removals.
        for _ in range(5):
            if time.time() > deadline:
                break
            # Pick a random edge to remove
            existing = [(u, v) for u, v in all_edges if adj[u, v] != -1]
            if not existing:
                break
            u, v = random.choice(existing)
            adj[u, v] = -1
            adj[v, u] = -1
            neighbors = build_neighbors(adj)
            
            # Try to add all missing
            for eu, ev in all_edges:
                if adj[eu, ev] != -1:
                    continue
                c = can_add_edge(adj, neighbors, eu, ev)
                if c is not None:
                    adj[eu, ev] = c
                    adj[ev, eu] = c
                    neighbors[eu].append((ev, c))
                    neighbors[ev].append((eu, c))
            
            neighbors = build_neighbors(adj)
            
        # Count edges
        edge_count = int(np.sum(adj != -1)) // 2
        
        if edge_count > best_edges:
            best_edges = edge_count
            best_adj = adj.copy()
            
            # Early exit if we hit a very high density?
            # Max edges is 190.
            # If we get > 100, it's good.
            
    # Final Verification
    if best_adj is None:
        # Fallback: Star graph
        adj = np.full((n, n), -1, dtype=int)
        for v in range(1, n):
            adj[0, v] = 0
            adj[v, 0] = 0
        best_adj = adj
        best_edges = 19
    else:
        if not is_valid(best_adj):
            # Fallback
            adj = np.full((n, n), -1, dtype=int)
            for v in range(1, n):
                adj[0, v] = 0
                adj[v, 0] = 0
            best_adj = adj
            best_edges = 19

    # Format output
    edges_list = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v] != -1:
                edges_list.append([u, v, int(best_adj[u, v])])
                
    return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
    if res:
        print(f"Sample: {res[0]}")
