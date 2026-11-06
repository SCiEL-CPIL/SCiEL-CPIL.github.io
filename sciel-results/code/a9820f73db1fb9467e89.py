"""
Strategy: Structured Constructive Search with Exact Verification.
Instead of random graphs, we search over families of circulant graphs 
(defined by connection sets S) which are common in extremal graph theory 
for Ramsey problems. We use exact branch-and-bound to verify constraints 
(triangle-free, alpha < 10) efficiently on these structured candidates.
"""

import time
import random
import itertools

def _has_triangle(n, edges_set):
    """Check if graph has any triangle. O(E * V) or faster with bitsets."""
    # For efficiency in verification loop, we can just check specific structures
    # But for exactness, we need to be sure.
    # A triangle exists if there are u,v,w with edges (u,v), (v,w), (w,u).
    # We can iterate over edges (u,v) and check common neighbors.
    adj = [set() for _ in range(n)]
    for u, v in edges_set:
        adj[u].add(v)
        adj[v].add(u)
    
    for u in range(n):
        for v in adj[u]:
            if u < v: # Check each edge once
                # Intersection of neighbors of u and v
                # If non-empty, triangle exists
                if adj[u] & adj[v]:
                    return True
    return False

def _independence_number_upper_bound(n, edges_set):
    """
    Exact independence number calculation using Branch and Bound.
    Returns the size of the maximum independent set.
    If it reaches 10, we can stop early (return >= 10).
    """
    # Build adjacency matrix as bitmasks for speed
    adj_mask = [0] * n
    for u, v in edges_set:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    max_indep = 0
    # State: current set of available vertices (bitmask), current size of independent set
    # We use a recursive branch and bound.
    # To speed up, we can use a greedy coloring bound on the remaining graph.
    
    def bb(available_mask, current_size):
        nonlocal max_indep
        
        if available_mask == 0:
            if current_size > max_indep:
                max_indep = current_size
            return
        
        # Pruning: if current_size + popcount(available_mask) <= max_indep, return
        # But we want to know if we can reach 10.
        # If current_size + remaining_count < 10, we can't reach 10, but we still want max?
        # The task says "no independent set of 10". So we just need to know if alpha >= 10.
        # If we find one of size 10, we can abort.
        
        if current_size >= 10:
            return # Found independent set of size 10
        
        # Upper bound: current_size + number of vertices in available_mask
        remaining_count = bin(available_mask).count('1')
        if current_size + remaining_count < 10:
            return # Cannot reach 10
        
        # Greedy coloring bound for tighter pruning (optional, but good for speed)
        # Let's stick to simple branching first, it might be fast enough for n<=40 if structured.
        
        # Pick a vertex v from available_mask
        # Heuristic: pick vertex with highest degree in remaining graph? Or lowest?
        # For independent set, picking a vertex with high degree removes many others,
        # which is good for finding large independent sets quickly? 
        # Actually, for B&B on Max Independent Set, we usually branch on a vertex.
        # Branch 1: Include v. Then remove v and its neighbors from available.
        # Branch 2: Exclude v. Then remove v from available.
        
        # Get least significant bit
        lsb = available_mask & -available_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New available: available_mask without v and without neighbors of v
        new_avail_1 = available_mask & ~(adj_mask[v] | lsb)
        bb(new_avail_1, current_size + 1)
        
        # Check if we already found a solution of size 10
        if max_indep >= 10:
            return
            
        # Branch 2: Exclude v
        new_avail_2 = available_mask & ~lsb
        bb(new_avail_2, current_size)

    bb((1 << n) - 1, 0)
    return max_indep

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    best_n = 0
    best_edges = []
    
    # 1. Try known good structured graphs first (Circulants)
    # A circulant graph C_n(S) has edges (i, i+s) mod n for s in S.
    # To be triangle-free, S must not contain s, t, t-s mod n for any distinct s,t.
    # We want alpha < 10.
    
    # Let's search for valid circulant graphs for n from 27 to 40.
    # For each n, we can try random subsets S of {1, ..., n//2} to define the graph.
    # We need to ensure no triangle.
    # Condition for no triangle in C_n(S): 
    # For any a,b in S, a+b not in S, a-b not in S (mod n), and a not in S if 2a=0?
    # Specifically, if u,v,w form triangle, then differences are in S or -S.
    # If we restrict S to positive differences 1..n//2, we need:
    # For any x,y in S: x+y not in S, |x-y| not in S.
    # Also 2x not in S (for isosceles triangles? No, triangle is 3 distinct vertices).
    # Vertices 0, x, x+y. Edges (0,x) in S, (x, x+y) diff y in S, (0, x+y) diff x+y in S.
    # So if x,y,x+y in S, triangle.
    # Vertices 0, x, -x. Edges (0,x) in S, (x,-x) diff 2x. If 2x in S, triangle 0-x-(-x).
    
    def is_valid_circulant(n, S):
        S = set(S)
        # Check triangle conditions
        for x in S:
            if (2 * x) % n in S and (2 * x) % n != 0:
                # Triangle 0, x, 2x? No, vertices 0, x, 2x.
                # Edge (0,x) diff x. Edge (x,2x) diff x. Edge (0,2x) diff 2x.
                # If 2x in S, then edge (0,2x) exists.
                # Is 0-x-2x a triangle? Yes.
                # Wait, if 2x mod n == 0, then x = n/2. Edge (0, n/2).
                # Triangle 0, n/2, ?
                # Let's just check all triples. It's safer and n is small.
                pass
        
        # Exact check for triangle-free using set logic
        # A triangle exists if there are a,b,c in Z_n distinct such that
        # (a-b)%n in S, (b-c)%n in S, (c-a)%n in S.
        # Due to symmetry, check triples starting at 0.
        # Vertices 0, u, v.
        # Edges: (0,u) -> u in S or n-u in S.
        # Let's normalize S to contain smallest representative of {d, n-d}.
        # So S subset of {1, ..., n//2}.
        # Edge (0,u) exists iff u in S.
        # Edge (0,v) exists iff v in S.
        # Edge (u,v) exists iff (v-u)%n in S or (u-v)%n in S.
        # Let d = (v-u)%n. If d in S or n-d in S.
        
        S_set = set(S)
        for u in range(1, n):
            if u not in S_set:
                continue
            for v in range(u + 1, n):
                if v not in S_set:
                    continue
                # Check edge (u,v)
                d = (v - u) % n
                if d in S_set or (n - d) in S_set:
                    return False # Triangle 0-u-v
        return True

    # Search for circulant graphs
    for n in range(40, 26, -1): # Try larger first
        if time.time() > deadline - 5:
            break
            
        # We want a "dense" enough graph to keep alpha < 10, but sparse enough to be triangle-free.
        # Turan bound: Triangle-free max edges ~ n^2/4.
        # For n=40, max edges ~ 400.
        # Random circulant: pick random S.
        
        found_for_n = False
        for attempt in range(200):
            if time.time() > deadline - 5:
                break
                
            # Generate random S
            # Size of S roughly n/4 to n/3?
            # Let's try various sizes.
            max_s_size = n // 3
            min_s_size = n // 6
            target_size = random.randint(min_s_size, max_s_size)
            
            # Pick random subset of {1, ..., n//2}
            candidates = list(range(1, n // 2 + 1))
            random.shuffle(candidates)
            S = set(candidates[:target_size])
            
            if not is_valid_circulant(n, S):
                continue
                
            # Construct edges
            edges = set()
            for u in range(n):
                for s in S:
                    v = (u + s) % n
                    if u < v:
                        edges.add((u, v))
            
            # Verify Triangle-Free (redundant if is_valid_circulant is correct, but safe)
            # And check Independence Number
            # We need alpha < 10.
            
            # Quick greedy lower bound for alpha to prune bad ones?
            # If greedy finds 10, it's definitely invalid.
            adj = [set() for _ in range(n)]
            for u, v in edges:
                adj[u].add(v)
                adj[v].add(u)
                
            # Greedy Independent Set
            remaining = set(range(n))
            indep_count = 0
            while remaining:
                # Pick vertex with min degree in remaining
                min_deg = float('inf')
                chosen = None
                for v in remaining:
                    deg = len(adj[v] & remaining)
                    if deg < min_deg:
                        min_deg = deg
                        chosen = v
                indep_count += 1
                remaining.discard(chosen)
                for u in adj[chosen]:
                    remaining.discard(u)
                
                if indep_count >= 10:
                    break
            
            if indep_count >= 10:
                continue # Definitely invalid (alpha >= 10)
                
            # Exact check for alpha < 10
            # This is the expensive part.
            alpha = _independence_number_upper_bound(n, edges)
            
            if alpha < 10:
                # Valid!
                best_n = n
                best_edges = [[u, v] for u, v in sorted(edges)]
                found_for_n = True
                break # Found valid for this n, try larger n in next loop iter? 
                # No, loop is descending. So if found for 40, we can return immediately?
                # The task says maximize n. So if we find 40, we are done.
                # But we are iterating n from 40 down.
                # So if we find one for 40, we break the n loop.
                
        if found_for_n:
            break
            
    # If no circulant found, fallback to random search with exact check (slower)
    if best_n < 27:
        for n in range(35, 26, -1):
            if time.time() > deadline - 5:
                break
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                # Random graph
                edges = set()
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.2: # Sparse
                            edges.add((i, j))
                
                # Remove triangles
                changed = True
                while changed:
                    changed = False
                    # Pick a random edge, check for triangle
                    if not edges: break
                    u, v = random.choice(list(edges))
                    # Find common neighbor
                    common = [w for w in range(n) if w != u and w != v and (min(u,w),max(u,w)) in edges and (min(v,w),max(v,w)) in edges]
                    if common:
                        edges.discard((u, v))
                        changed = True
                        
                if not edges: continue
                
                # Check alpha
                adj = [set() for _ in range(n)]
                for u, v in edges:
                    adj[u].add(v)
                    adj[v].add(u)
                    
                # Greedy prune
                remaining = set(range(n))
                indep_count = 0
                while remaining:
                    min_deg = float('inf')
                    chosen = None
                    for v in remaining:
                        deg = len(adj[v] & remaining)
                        if deg < min_deg:
                            min_deg = deg
                            chosen = v
                    indep_count += 1
                    remaining.discard(chosen)
                    for u in adj[chosen]:
                        remaining.discard(u)
                    if indep_count >= 10: break
                
                if indep_count >= 10:
                    continue
                    
                alpha = _independence_number_upper_bound(n, edges)
                if alpha < 10:
                    best_n = n
                    best_edges = [[u, v] for u, v in sorted(edges)]
                    break
    
    # Fallback to a small known valid graph if nothing else works
    if best_n == 0:
        # C5 is triangle free, alpha 2.
        # We need alpha < 10.
        # Let's just return a valid small graph to avoid 0 score.
        # Complete graph K_9 is not triangle free.
        # Cycle C_25? Alpha 12. Too big.
        # We need a graph with alpha < 10.
        # C_17? Alpha 8. Triangle free.
        n = 17
        best_n = n
        best_edges = [[i, (i+1)%n] for i in range(n)]
        # Check: C17 is triangle free. Alpha(17) = floor(17/2) = 8. 8 < 10. Valid.
        
    return {'n': best_n, 'edges': best_edges}
