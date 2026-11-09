"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _check_validity(added_edges):
    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    all_edges = set(base_edges)
    for e in added_edges:
        u, v = int(min(e[0], e[1])), int(max(e[0], e[1]))
        if (u, v) in all_edges:
            return False
        all_edges.add((u, v))
    
    adj = [set() for _ in range(n)]
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Triangle-free check: for each vertex, its neighbors must form an independent set
    for u in range(n):
        nbrs = list(adj[u])
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] in adj[nbrs[i]]:
                    return False
    
    # Connected check + diameter <= 4 via BFS from each node
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            node = q.popleft()
            if dist[node] >= 4:
                continue
            for nb in adj[node]:
                if dist[nb] == -1:
                    dist[nb] = dist[node] + 1
                    q.append(nb)
        if any(d == -1 for d in dist):
            return False
    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85

    n = 24
    even_verts = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    odd_verts = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    # Precompute valid hub-odd connections (not path-adjacent)
    hub_cands = {}
    for h in even_verts:
        hub_cands[h] = frozenset(o for o in odd_verts if abs(h - o) != 1)
    
    # Known good solution: 10 edges
    fallback = [[12, 1], [12, 3], [12, 5], [12, 7], [12, 9],
                [12, 15], [12, 17], [12, 19], [12, 21], [12, 23]]
    best = fallback
    best_len = 10
    
    # Try to find solutions with fewer than 10 edges
    # Strategy: for k edges, try all subsets of valid edges of size k
    # Valid edges: (even, odd) pairs where |even - odd| != 1
    
    all_valid_edges = []
    for h in even_verts:
        for o in hub_cands[h]:
            all_valid_edges.append((h, o))
    
    # For k=9: try to find a valid 9-edge solution
    # Use a smarter search: start from known good structures
    # Try single hub with 9 edges (skip one odd vertex that's close to hub via path)
    for h in even_verts:
        if time.time() > deadline - 3:
            break
        cands = sorted(hub_cands[h])
        # Try skipping each odd vertex
        for skip_idx in range(len(cands)):
            edges = [[h, o] for i, o in enumerate(cands) if i != skip_idx]
            if len(edges) < best_len and _check_validity(edges):
                best = edges
                best_len = len(edges)
    
    # Try two-hub with total 9 edges
    # Each hub connects to some subset of odd vertices
    for i in range(len(even_verts)):
        if time.time() > deadline - 3:
            break
        h1 = even_verts[i]
        for j in range(i + 1, len(even_verts)):
            h2 = even_verts[j]
            s1 = hub_cands[h1]
            s2 = hub_cands[h2]
            
            # Odd vertices not coverable by either
            if any(o not in s1 and o not in s2 for o in odd_verts):
                continue
            
            # We need to assign each odd vertex to h1, h2, or leave it (if close enough)
            # Try all assignments where total edges = 9
            # For each odd vertex, options: assign to h1, assign to h2, or skip
            # But skipping means it must be reachable within diameter 4 via path + other edges
            
            # Simplified: try assignments with exactly 9 edges
            # Count how many odd vertices each hub can cover
            # We need |assigned_to_h1| + |assigned_to_h2| = 9
            
            s1_list = sorted(s1)
            s2_list = sorted(s2)
            
            # Try: h1 gets a subset A of s1, h2 gets a subset B of s2, |A|+|B|=9
            # and all odd vertices are either in A, in B, or "covered" by path proximity
            # This is complex; use heuristic: try balanced splits
            
            for a_count in range(max(1, 9 - len(s2_list)), min(len(s1_list), 9)):
                b_count = 9 - a_count
                if b_count < 0 or b_count > len(s2_list):
                    continue
                # Pick which odd vertices go to h1 vs h2
                # Greedy: assign vertices that are far from both hubs to the closer one
                # For now, try a few specific patterns
                pass
    
    # Try a structured approach: "ring" or "cycle" augmentation
    # Add edges that create shortcuts at regular intervals
    # Pattern: add (i, i+3) for selected i values
    
    # Try adding edges at stride 3: (0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21)
    # That's 7 edges. Check validity.
    stride3_edges = [(0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21)]
    if len(stride3_edges) < best_len and _check_validity(stride3_edges):
        best = [list(e) for e in stride3_edges]
        best_len = len(stride3_edges)
    
    # Try stride 5: (0,5),(5,10),(10,15),(15,20),(20,23) - but (20,23): 20 even, 23 odd, |20-23|=3≠1 OK
    # Actually (0,5): even-odd, |0-5|=5≠1 OK. (5,10): odd-even OK. etc.
    stride5_edges = [(0,5),(5,10),(10,15),(15,20),(20,23)]
    if len(stride5_edges) < best_len and _check_validity(stride5_edges):
        best = [list(e) for e in stride5_edges]
        best_len = len(stride5_edges)
    
    # Try mixed: some stride-3 and some stride-5
    # (0,3),(3,8),(8,13),(13,18),(18,23) - 5 edges
    mixed1 = [(0,3),(3,8),(8,13),(13,18),(18,23)]
    if len(mixed1) < best_len and _check_validity(mixed1):
        best = [list(e) for e in mixed1]
        best_len = len(mixed1)
    
    # (0,5),(5,10),(10,15),(15,20),(20,23) checked above
    # (1,4),(4,9),(9,14),(14,19),(19,23) - 5 edges  
    mixed2 = [(1,4),(4,9),(9,14),(14,19),(19,23)]
    if len(mixed2) < best_len and _check_validity(mixed2):
        best = [list(e) for e in mixed2]
        best_len = len(mixed2)
    
    # (2,7),(7,12),(12,17),(17,22) - 4 edges
    mixed3 = [(2,7),(7,12),(12,17),(17,22)]
    if len(mixed3) < best_len and _check_validity(mixed3):
        best = [list(e) for e in mixed3]
        best_len = len(mixed3)
    
    # (0,4) invalid (both even). (0,3) valid.
    # Try: (0,3),(1,4),(2,5),(3,6),(4,7),(5,8),(6,9),(7,10),(8,11),(9,12),(10,13),(11,14),(12,15),(13,16),(14,17),(15,18),(16,19),(17,20),(18,21),(19,22),(20,23)
    # That's 21 edges - too many.
    
    # Try a "binary tree" like structure
    # Connect 0 to 4,8,12,16,20 (but 0-4 invalid, both even)
    # Connect 0 to 3,7,11,15,19,23 (even to odd, all valid if |diff|!=1)
    star0 = [(0,3),(0,7),(0,11),(0,15),(0,19),(0,23)]
    if len(star0) < best_len and _check_validity(star0):
        best = [list(e) for e in star0]
        best_len = len(star0)
    
    # Star from 12: (12,1),(12,3),(12,5),(12,7),(12,9),(12,15),(12,17),(12,19),(12,21),(12,23) = 10 edges (fallback)
    
    # Try: two stars
    # Star from 6: (6,1),(6,3),(6,5) - wait (6,5) is path edge. (6,1),(6,3),(6,9),(6,11),(6,13)
    # Star from 18: (18,13),(18,15),(18,17),(18,21),(18,23)
    two_stars = [(6,1),(6,3),(6,9),(6,11),(6,13),(18,13),(18,15),(18,17),(18,21),(18,23)]
    if len(two_stars) < best_len and _check_validity(two_stars):
        best = [list(e) for e in two_stars]
        best_len = len(two_stars)
    
    # Try: (2,5),(2,9),(2,13),(2,17),(2,21),(22,1),(22,3),(22,7),(22,11),(22,15),(22,19)
    # That's 11 edges - worse than 10
    
    # Try: (4,7),(4,11),(4,15),(4,19),(4,23),(20,1),(20,3),(20,5),(20,9),(20,13),(20,17)
    # 11 edges - worse
    
    # Try a "ladder" pattern: connect i to i+4 for odd i
    # (1,5),(3,7),(5,9),(7,11),(9,13),(11,15),(13,17),(15,19),(17,21),(19,23) - 10 edges
    ladder10 = [(1,5),(3,7),(5,9),(7,11),(9,13),(11,15),(13,17),(15,19),(17,21),(19,23)]
    if len(ladder10) < best_len and _check_validity(ladder10):
        best = [list(e) for e in ladder10]
        best_len = len(ladder10)
    
    # Try: (1,5),(5,9),(9,13),(13,17),(17,21) - 5 edges (odd to odd+4, but both odd - INVALID!)
    # Can't do odd-odd edges.
    
    # Try: (0,3),(4,7),(8,11),(12,15),(16,19),(20,23) - 6 edges
    stride6a = [(0,3),(4,7),(8,11),(12,15),(16,19),(20,23)]
    if len(stride6a) < best_len and _check_validity(stride6a):
        best = [list(e) for e in stride6a]
        best_len = len(stride6a)
    
    # Try: (1,4),(5,8),(9,12),(13,16),(17,20) - 5 edges
    stride6b = [(1,4),(5,8),(9,12),(13,16),(17,20)]
    if len(stride6b) < best_len and _check_validity(stride6b):
        best = [list(e) for e in stride6b]
        best_len = len(stride6b)
    
    # Combine: (0,3),(1,4),(4,7),(5,8),(8,11),(9,12),(12,15),(13,16),(16,19),(17,20),(20,23)
    # Too many. Let me try smaller combinations.
    
    # (0,3),(4,7),(8,11),(12,15),(16,19),(20,23) = 6 edges, checked above
    # (1,4),(5,8),(9,12),(13,16),(17,20) = 5 edges, checked above
    
    # Try combining both patterns: 
    # (0,3),(1,4),(4,7),(5,8),(8,11),(9,12),(12,15),(13,16),(16,19),(17,20),(20,23) = 11 edges
    # Too many.
    
    # Try: (0,3),(3,6) invalid (3 odd, 6 even, |3-6|=3 OK actually valid!)
    # Wait: (3,6): 3 is odd, 6 is even, |3-6|=3≠1, so valid.
    # (0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21),(21,24) - 24 doesn't exist
    # (0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21) = 7 edges
    chain7 = [(0,3),(3,6),(6,9),(9,12),(12,15),(15,18),(18,21)]
    if len(chain7) < best_len and _check_validity(chain7):
        best = [list(e) for e in chain7]
        best_len = len(chain7)
    
    # (1,4),(4,7),(7,10),(10,13),(13,16),(16,19),(19,22) = 7 edges
    chain7b = [(1,4),(4,7),(7,10),(10,13),(13,16),(16,19),(19,22)]
    if len(chain7b) < best_len and _check_validity(chain7b):
        best = [list(e) for e in chain7b]
        best_len = len(chain7b)
    
    # (2,5),(5,8),(8,11),(11,14),(14,17),(17,20),(20,23) = 7 edges
    chain7c = [(2,5),(5,8),(8,11),(11,14),(14,17),(17,20),(20,23)]
    if len(chain7c) < best_len and _check_validity(chain7c):
        best = [list(e) for e in chain7c]
        best_len = len(chain7c)
    
    # Try 6-edge solutions with larger jumps
    # (0,5),(5,10),(10,15),(15,20) = 4 edges
    big_jump4 = [(0,5),(5,10),(10,15),(15,20)]
    if len(big_jump4) < best_len and _check_validity(big_jump4):
        best = [list(e) for e in big_jump4]
        best_len = len(big_jump4)
    
    # (1,6),(6,11),(11,16),(16,21) = 4 edges
    big_jump4b = [(1,6),(6,11),(11,16),(16,21)]
    if len(big_jump4b) < best_len and _check_validity(big_jump4b):
        best = [list(e) for e in big_jump4b]
        best_len = len(big_jump4b)
    
    # (2,7),(7,12),(12,17),(17,22) = 4 edges, checked above as mixed3
    # (3,8),(8,13),(13,18),(18,23) = 4 edges
    big_jump4c = [(3,8),(8,13),(13,18),(18,23)]
    if len(big_jump4c) < best_len and _check_validity(big_jump4c):
        best = [list(e) for e in big_jump4c]
        best_len = len(big_jump4c)
    
    # Try 5-edge solutions
    # (0,5),(5,10),(10,15),(15,20),(20,23) = 5 edges, checked as stride5_edges
    # (1,6),(6,11),(11,16),(16,21),(21,23) - (21,23): both odd, INVALID
    # (1,6),(6,11),(11,16),(16,21),(16,19) - (16,19): even-odd, |16-19|=3 OK
    five_a = [(1,6),(6,11),(11,16),(16,21),(16,19)]
    if len(five_a) < best_len and _check_validity(five_a):
        best = [list(e) for e in five_a]
        best_len = len(five_a)
    
    # (0,5),(5,10),(10,15),(15,20),(0,3) = 5 edges
    five_b = [(0,5),(5,10),(10,15),(15,20),(0,3)]
    if len(five_b) < best_len and _check_validity(five_b):
        best = [list(e) for e in five_b]
        best_len = len(five_b)
    
    # (0,3),(3,8),(8,13),(13,18),(18,23) = 5 edges, checked as mixed1
    # (1,4),(4,9),(9,14),(14,19),(19,23) = 5 edges, checked as mixed2
    
    # Try 4-edge solutions
    # (0,6) invalid (both even). (0,7): even-odd, |0-7|=7 OK
    # (0,7),(7,14) invalid (both odd). (0,7),(7,12),(12,19),(19,23) - (19,23) both odd INVALID
    # (0,7),(7,12),(12,19),(12,23) - (12,23): even-odd OK
    four_a = [(0,7),(7,12),(12,19),(12,23)]
    if len(four_a) < best_len and _check_validity(four_a):
        best = [list(e) for e in four_a]
        best_len = len(four_a)
    
    # (0,5),(5,10),(10,15),(15,23) - (15,23): odd-odd INVALID
    # (0,5),(5,10),(10,15),(10,23) - (10,23): even-odd, |10-23|=13 OK
    four_b = [(0,5),(5,10),(10,15),(10,23)]
    if len(four_b) < best_len and _check_validity(four_b):
        best = [list(e) for e in four_b]
        best_len = len(four_b)
    
    # (0,3),(3,10),(10,17),(17,23) - (3,10): odd-even OK, (10,17): even-odd OK, (17,23): odd-odd INVALID
    # (0,3),(3,10),(10,17),(10,23)
    four_c = [(0,3),(3,10),(10,17),(10,23)]
    if len(four_c) < best_len and _check_validity(four_c):
        best = [list(e) for e in four_c]
        best_len = len(four_c)
    
    # (0,7),(7,14) invalid. (0,7),(7,12),(12,21),(21,23) invalid (both odd)
    # (0,7),(7,12),(12,21),(12,23)
    four_d = [(0,7),(7,12),(12,21),(12,23)]
    if len(four_d) < best_len and _check_validity(four_d):
        best = [list(e) for e in four_d]
        best_len = len(four_d)
    
    # Return best found
    if best_len < 10:
        return [list(e) if isinstance(e, tuple) else e for e in best]
    return fallback
# EVOLVE-BLOCK-END
