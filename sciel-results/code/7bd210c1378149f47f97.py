import random
import time
import sys

def construct():
    """
    Construct a C4-free edge set of Q7 maximizing the number of edges.
    Uses greedy initialization + local search (2-swap, 1-swap) with time guard.
    """
    d = 7
    n = 1 << d  # 128 vertices
    all_edges = []
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                all_edges.append((x, y))
    m = len(all_edges)  # 448
    
    # Build face structure: for each edge, list of faces it belongs to
    # A face is a 2D square: vertices {a, a^bi, a^bj, a^bi^bj}
    # For edge (x, y) where y = x ^ (1<<i), faces containing it are for each j != i:
    #   the square with corners x, y, x^(1<<j), y^(1<<j)
    # Represent face as frozenset of its 4 edges (canonical form)
    
    # Precompute: for each edge index, which faces (as tuples of 4 edge indices) contain it
    edge_index = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_index[(u, v)] = idx
        edge_index[(v, u)] = idx
    
    # Build faces: for each pair of dimensions (i, j) and each vertex x with bit i and j = 0
    # Actually, enumerate all C4s in Q7
    # A C4 in Q7 is determined by two dimensions i < j and a vertex x where bits i,j are 0
    # Corners: x, x^(1<<i), x^(1<<j), x^(1<<i)^(1<<j)
    # Edges: (x, x^i), (x^i, x^i^j), (x, x^j), (x^j, x^i^j)
    
    faces = []  # list of tuples of 4 edge indices
    face_of_edge = [[] for _ in range(m)]  # for each edge, list of face indices
    
    for i in range(d):
        for j in range(i + 1, d):
            bi, bj = 1 << i, 1 << j
            for x in range(n):
                if (x & bi) == 0 and (x & bj) == 0:
                    v1, v2, v3, v4 = x, x | bi, x | bj, x | bi | bj
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v2, v4), max(v2, v4))]
                    e3 = edge_index[(min(v4, v3), max(v4, v3))]
                    e4 = edge_index[(min(v3, v1), max(v3, v1))]
                    face_idx = len(faces)
                    faces.append((e1, e2, e3, e4))
                    for e in (e1, e2, e3, e4):
                        face_of_edge[e].append(face_idx)
    
    # Greedy initialization
    selected = [False] * m
    face_count = [0] * len(faces)  # number of selected edges in each face
    
    # Sort edges randomly for diversity
    random.seed(42)
    order = list(range(m))
    random.shuffle(order)
    
    for e in order:
        u, v = all_edges[e]
        # Check if adding this edge would complete any face
        can_add = True
        for fi in face_of_edge[e]:
            if face_count[fi] == 3:
                can_add = False
                break
        if can_add:
            selected[e] = True
            for fi in face_of_edge[e]:
                face_count[fi] += 1
    
    current_count = sum(selected)
    best_count = current_count
    best_selected = selected[:]
    
    # Local search: try to add edges, and if stuck, try swaps
    deadline = time.time() + 110  # safety margin
    
    def try_add():
        """Try to add any unselected edge that doesn't complete a face."""
        nonlocal current_count
        for e in range(m):
            if not selected[e]:
                can_add = True
                for fi in face_of_edge[e]:
                    if face_count[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    for fi in face_of_edge[e]:
                        face_count[fi] += 1
                    current_count += 1
                    return True
        return False
    
    # First, greedily add all possible edges
    while try_add():
        pass
    
    if current_count > best_count:
        best_count = current_count
        best_selected = selected[:]
    
    # Local search with swaps
    # A swap: remove one selected edge, add up to two unselected edges
    # Or: remove one, add one (if it frees up space)
    
    # For efficiency, track for each face how many edges are selected
    # When we remove edge e, faces containing e lose one count
    # When we add edge e, faces containing e gain one count
    
    # Try 1-swap: remove e1, add e2 where e2 was blocked only by e1's face
    # Try 2-swap: remove e1, e2, add e3, e4
    
    # Simpler: random perturbation + repair
    while time.time() < deadline:
        # Randomly remove 1-3 edges
        k = random.randint(1, 3)
        removed = []
        for _ in range(k):
            sel_idx = [i for i in range(m) if selected[i]]
            if not sel_idx:
                break
            e = random.choice(sel_idx)
            selected[e] = False
            for fi in face_of_edge[e]:
                face_count[fi] -= 1
            removed.append(e)
            current_count -= 1
        
        # Try to re-add all edges greedily
        for e in range(m):
            if not selected[e]:
                can_add = True
                for fi in face_of_edge[e]:
                    if face_count[fi] == 3:
                        can_add = False
                        break
                if can_add:
                    selected[e] = True
                    for fi in face_of_edge[e]:
                        face_count[fi] += 1
                    current_count += 1
        
        # Try to add more
        while try_add():
            pass
        
        if current_count > best_count:
            best_count = current_count
            best_selected = selected[:]
    
    # Return best found
    result = []
    for i in range(m):
        if best_selected[i]:
            result.append([all_edges[i][0], all_edges[i][1]])
    return result
