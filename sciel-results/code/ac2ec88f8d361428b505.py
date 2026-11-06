"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def _find_all_4cycles(n, adj):
    """Find all 4-cycles as frozensets of 4 edges. adj: dict (u,v) -> colour or None"""
    cycles = []
    # For each pair of vertices (a,c) that could be opposite in a 4-cycle,
    # find common neighbors b,d such that a-b-c-d-a forms a 4-cycle
    # A 4-cycle: a-b, b-c, c-d, d-a where a,b,c,d distinct
    # We can enumerate: for each unordered pair {a,c}, find common neighbors of a and c
    # that are not adjacent to each other... actually no, in a 4-cycle a-b-c-d-a,
    # a and c are NOT adjacent (they're opposite), b and d are NOT adjacent.
    
    # Better: enumerate all 4-tuples (a,b,c,d) with a<b, c<d, a!=c, etc.
    # A 4-cycle is determined by 4 distinct vertices and 2 perfect matchings of the K4
    # that form a cycle. For 4 vertices {w,x,y,z}, there are 3 possible 4-cycles:
    # w-x-y-z-w, w-x-z-y-w, w-y-x-z-w (up to rotation/reflection)
    
    # For each set of 4 vertices, check if the 4 edges of a C4 exist
    verts = list(range(n))
    for quad in itertools.combinations(verts, 4):
        a, b, c, d = quad
        # Three possible 4-cycles on these 4 vertices:
        # Cycle 1: a-b, b-c, c-d, d-a
        # Cycle 2: a-b, b-d, d-c, c-a
        # Cycle 3: a-c, c-b, b-d, d-a
        candidates = [
            [(a,b), (b,c), (c,d), (d,a)],
            [(a,b), (b,d), (d,c), (c,a)],
            [(a,c), (c,b), (b,d), (d,a)],
        ]
        for edges in candidates:
            # Check all 4 edges exist
            all_exist = True
            for (u,v) in edges:
                if (u,v) not in adj and (v,u) not in adj:
                    all_exist = False
                    break
            if all_exist:
                # This is a 4-cycle; store as frozenset of normalized edges
                norm_edges = frozenset(frozenset((u,v)) for (u,v) in edges)
                cycles.append(norm_edges)
    return cycles

def _edge_key(u, v):
    return (min(u,v), max(u,v))

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin
    
    n = 20
    num_colors = 20
    
    # Start with complete bipartite graph K_{10,10}
    # Left: 0..9, Right: 10..19
    left = list(range(10))
    right = list(range(10, 20))
    
    # Initialize edge colours randomly
    adj = {}  # (u,v) with u<v -> colour
    for u in left:
        for v in right:
            key = _edge_key(u, v)
            adj[key] = random.randint(0, num_colors - 1)
    
    best_adj = dict(adj)
    best_score = len(adj)
    
    # Precompute all possible 4-cycles in the complete bipartite graph
    # In K_{10,10}, a 4-cycle is: l1-r1-l2-r2-l1 where l1!=l2, r1!=r2
    # Edges: (l1,r1), (l1,r2), (l2,r1), (l2,r2)
    all_4cycles = []
    for l1, l2 in itertools.combinations(left, 2):
        for r1, r2 in itertools.combinations(right, 2):
            edges = [
                _edge_key(l1, r1),
                _edge_key(l1, r2),
                _edge_key(l2, r1),
                _edge_key(l2, r2),
            ]
            all_4cycles.append(frozenset(edges))
    
    def _count_violations(adj_dict):
        count = 0
        for cycle in all_4cycles:
            # Check if all 4 edges exist in adj_dict
            if all(e in adj_dict for e in cycle):
                colors = [adj_dict[e] for e in cycle]
                if len(set(colors)) < 4:
                    count += 1
        return count
    
    def _is_valid(adj_dict):
        return _count_violations(adj_dict) == 0
    
    # Simulated annealing
    current_adj = dict(adj)
    current_violations = _count_violations(current_adj)
    
    if current_violations == 0:
        best_adj = dict(current_adj)
        best_score = len(current_adj)
    
    temperature = 5.0
    cooling = 0.999
    
    while time.time() < deadline:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Recolour a random edge
            keys = list(current_adj.keys())
            if not keys:
                break
            key = random.choice(keys)
            old_color = current_adj[key]
            new_color = random.randint(0, num_colors - 1)
            if new_color == old_color:
                continue
            
            # Apply
            current_adj[key] = new_color
            new_violations = _count_violations(current_adj)
            
            delta = new_violations - current_violations
            if delta <= 0 or random.random() < pow(2.718, -delta / max(temperature, 0.01)):
                current_violations = new_violations
                if current_violations == 0:
                    score = len(current_adj)
                    if score > best_score:
                        best_score = score
                        best_adj = dict(current_adj)
            else:
                # Reject
                current_adj[key] = old_color
                
        elif move_type < 0.9:
            # Remove a random edge
            keys = list(current_adj.keys())
            if not keys:
                break
            key = random.choice(keys)
            old_color = current_adj[key]
            del current_adj[key]
            
            new_violations = _count_violations(current_adj)
            delta = new_violations - current_violations
            
            # Removing an edge can only decrease or maintain violations
            if delta <= 0 or random.random() < pow(2.718, -delta / max(temperature, 0.01)):
                current_violations = new_violations
            else:
                current_adj[key] = old_color
                
        else:
            # Add a random edge (if not present)
            # Pick random u from left, v from right not already connected
            u = random.choice(left)
            v = random.choice(right)
            key = _edge_key(u, v)
            if key in current_adj:
                continue
            new_color = random.randint(0, num_colors - 1)
            current_adj[key] = new_color
            
            new_violations = _count_violations(current_adj)
            delta = new_violations - current_violations
            
            if delta <= 0 or random.random() < pow(2.718, -delta / max(temperature, 0.01)):
                current_violations = new_violations
                if current_violations == 0:
                    score = len(current_adj)
                    if score > best_score:
                        best_score = score
                        best_adj = dict(current_adj)
            else:
                del current_adj[key]
        
        temperature *= cooling
    
    # Return best valid solution
    if _is_valid(best_adj):
        result = []
        for (u, v), c in best_adj.items():
            result.append([u, v, c])
        return result
    else:
        # Fallback: return a valid sparse solution (star)
        return [[0, v, 0] for v in range(1, 20)]
# EVOLVE-BLOCK-END
