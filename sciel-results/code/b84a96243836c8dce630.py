"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 18
    # Start with K6,6,6
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    edges = set()
    for i in range(3):
        for j in range(i+1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    edges.add((min(a,b), max(a,b)))
    
    def is_k4_free(edge_set, n=18):
        # Check if any 4 vertices form a K4
        for quad in itertools.combinations(range(n), 4):
            a, b, c, d = quad
            if ((a,b) in edge_set or (b,a) in edge_set and (a,b) in edge_set) or True:
                pass
            # Better: check all 6 pairs
            pairs = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
            if all((min(p[0],p[1]), max(p[0],p[1])) in edge_set for p in pairs):
                return False
        return True
    
    def has_triangle(edge_set, vertices):
        for i, u in enumerate(vertices):
            for v in vertices[i+1:]:
                if (min(u,v), max(u,v)) in edge_set:
                    for w in vertices[vertices.index(v)+1:]:
                        if (min(v,w), max(v,w)) in edge_set and (min(u,w), max(u,w)) in edge_set:
                            return True
        return False
    
    def max_triangle_free_size(edge_set, n=18, timeout=5.0):
        """Find largest subset of vertices inducing a triangle-free subgraph."""
        best = 0
        start = time.time()
        
        # Branch and bound
        def bt(remaining, current_size, current_set):
            nonlocal best
            if time.time() - start > timeout:
                return
            if current_size + len(remaining) <= best:
                return
            if not remaining:
                if current_size > best:
                    best = current_size
                return
            
            # Try adding the first vertex
            v = remaining[0]
            rest = remaining[1:]
            
            # Check if adding v creates a triangle with two in current_set
            can_add = True
            for i, u in enumerate(current_set):
                if (min(u,v), max(u,v)) in edge_set:
                    for w in current_set[i+1:]:
                        if (min(u,w), max(u,w)) in edge_set and (min(v,w), max(v,w)) in edge_set:
                            can_add = False
                            break
                if not can_add:
                    break
            
            if can_add:
                bt(rest, current_size + 1, current_set + [v])
            
            # Try not adding v
            bt(rest, current_size, current_set)
        
        bt(list(range(n)), 0, [])
        return best
    
    def score(edge_set):
        return max_triangle_free_size(edge_set)
    
    current_edges = set(edges)
    current_score = score(current_edges)
    best_edges = set(current_edges)
    best_score = current_score
    
    # Local search: try adding edges within parts, or removing edges
    random.seed(42)
    
    all_possible_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_possible_edges.add((i,j))
    
    # Edges within parts (not in K6,6,6)
    intra_part_edges = set()
    for part in parts:
        for i in range(len(part)):
            for j in range(i+1, len(part)):
                intra_part_edges.add((min(part[i],part[j]), max(part[i],part[j])))
    
    # Try adding intra-part edges one at a time
    intra_list = list(intra_part_edges)
    random.shuffle(intra_list)
    
    for e in intra_list:
        if time.time() > deadline:
            break
        new_edges = current_edges | {e}
        if is_k4_free(new_edges):
            s = score(new_edges)
            if s < current_score:
                current_edges = new_edges
                current_score = s
                if s < best_score:
                    best_score = s
                    best_edges = set(current_edges)
    
    # Try removing inter-part edges
    inter_part_edges = list(current_edges)
    random.shuffle(inter_part_edges)
    
    for e in inter_part_edges:
        if time.time() > deadline:
            break
        new_edges = current_edges - {e}
        # Removing edges can't create K4, so always K4-free
        s = score(new_edges)
        if s < current_score:
            current_edges = new_edges
            current_score = s
            if s < best_score:
                best_score = s
                best_edges = set(current_edges)
    
    # More iterations
    for iteration in range(50):
        if time.time() > deadline:
            break
        # Try random edge additions
        candidates = [e for e in all_possible_edges if e not in current_edges]
        random.shuffle(candidates)
        improved = False
        for e in candidates[:20]:
            if time.time() > deadline:
                break
            new_edges = current_edges | {e}
            if is_k4_free(new_edges):
                s = score(new_edges)
                if s < current_score:
                    current_edges = new_edges
                    current_score = s
                    improved = True
                    if s < best_score:
                        best_score = s
                        best_edges = set(current_edges)
                    break
        # Try random edge removals
        candidates = list(current_edges)
        random.shuffle(candidates)
        for e in candidates[:20]:
            if time.time() > deadline:
                break
            new_edges = current_edges - {e}
            s = score(new_edges)
            if s < current_score:
                current_edges = new_edges
                current_score = s
                improved = True
                if s < best_score:
                    best_score = s
                    best_edges = set(current_edges)
                break
    
    return [[u, v] for u, v in sorted(best_edges)]
# EVOLVE-BLOCK-END
