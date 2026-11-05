import time
import random
import itertools
from collections import defaultdict

N = 18
DEADLINE = 145.0  # safety margin before 150s kill

def _adj_to_edges(adj):
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j]:
                edges.append([i, j])
    return edges

def _edges_to_adj(edges):
    adj = [[False] * N for _ in range(N)]
    for u, v in edges:
        adj[u][v] = True
        adj[v][u] = True
    return adj

def _is_k4_free(adj):
    # A K4 exists if there is a triangle (i,j,k) with a common neighbor l connected to all three.
    # More efficiently: for each edge (i,j), check common neighbors; if two common neighbors are adjacent -> K4.
    for i in range(N):
        for j in range(i + 1, N):
            if not adj[i][j]:
                continue
            common = []
            for k in range(N):
                if k != i and k != j and adj[i][k] and adj[j][k]:
                    common.append(k)
            for a in range(len(common)):
                for b in range(a + 1, len(common)):
                    if adj[common[a]][common[b]]:
                        return False
    return True

def _max_tri_free_size(adj):
    # Find the largest subset of vertices inducing a triangle-free graph.
    # Use branch and bound: try to include as many vertices as possible.
    # A set S is triangle-free iff it contains no triangle.
    # We want the largest S with no triangle.
    # This is equivalent to: find the maximum independent set in the "triangle hypergraph"?
    # Actually, it's the largest subset with no triangle.
    # Use backtracking: order vertices, decide include/exclude, prune if current size + remaining < best.
    
    # Precompute: for each vertex, its neighbors
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i][j]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    # Triangle list: for quick checking
    triangles = []
    for i in range(N):
        for j in neighbors[i]:
            if j > i:
                for k in neighbors[j]:
                    if k > j and k in neighbors[i]:
                        triangles.append((i, j, k))
    
    # Convert triangles to sets for fast membership
    triangle_sets = [set(t) for t in triangles]
    
    best = [0]
    
    def backtrack(remaining, current_size, current_set):
        if not remaining:
            if current_size > best[0]:
                best[0] = current_size
            return
        # Prune: even if we take all remaining, can't beat best
        if current_size + len(remaining) <= best[0]:
            return
        
        # Try including the first vertex in remaining
        v = remaining[0]
        rest = remaining[1:]
        
        # Check if adding v creates a triangle with any two vertices in current_set
        can_add = True
        for t in triangle_sets:
            if v in t:
                # Check if the other two are in current_set
                others = [x for x in t if x != v]
                if others[0] in current_set and others[1] in current_set:
                    can_add = False
                    break
                # Also check if one is in current_set and the other is in rest (but we haven't decided on rest yet)
                # For now, we only check against current_set; rest vertices will be checked when they're considered
                # Actually, we need to be careful: a triangle could have vertices from current_set AND rest.
                # But since we're building incrementally, we only need to check against current_set.
                # When we later add a vertex from rest, we'll check against the then-current_set.
                # So this is correct.
        
        if can_add:
            backtrack(rest, current_size + 1, current_set | {v})
        
        # Try excluding v
        backtrack(rest, current_size, current_set)
    
    backtrack(list(range(N)), 0, set())
    return best[0]

def _score(adj):
    return _max_tri_free_size(adj)

def _local_search(adj, start_time, deadline):
    # Hill-climbing: flip edges to reduce the max triangle-free size while maintaining K4-free.
    current_adj = [row[:] for row in adj]
    current_score = _score(current_adj)
    best_adj = [row[:] for row in current_adj]
    best_score = current_score
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:
        improved = False
        # Try all single edge flips
        random.shuffle(list(range(N)))
        random.shuffle(list(range(N)))
        
        for i in range(N):
            for j in range(i + 1, N):
                if time.time() > deadline - 5:
                    break
                # Try flipping edge (i,j)
                old_val = current_adj[i][j]
                current_adj[i][j] = not old_val
                current_adj[j][i] = not old_val
                
                if _is_k4_free(current_adj):
                    new_score = _score(current_adj)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        if current_score < best_score:
                            best_score = current_score
                            best_adj = [row[:] for row in current_adj]
                    else:
                        # Revert
                        current_adj[i][j] = old_val
                        current_adj[j][i] = old_val
                else:
                    # Revert
                    current_adj[i][j] = old_val
                    current_adj[j][i] = old_val
            if time.time() > deadline - 5:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return best_adj, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with a random K4-free graph
    # Strategy: start with a random graph, then remove edges to make it K4-free
    best_adj = None
    best_score = 999
    
    # Try multiple random restarts
    random.seed(42)
    
    for restart in range(50):
        if time.time() > deadline - 10:
            break
        
        # Generate a random graph
        adj = [[False] * N for _ in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if random.random() < 0.5:
                    adj[i][j] = True
                    adj[j][i] = True
        
        # Remove edges to make it K4-free
        changed = True
        while changed and time.time() < deadline - 10:
            changed = False
            for i in range(N):
                for j in range(i + 1, N):
                    if not adj[i][j]:
                        continue
                    # Check if removing this edge is needed (i.e., is it part of a K4?)
                    # Actually, we need to check if the graph is K4-free; if not, remove an edge from a K4.
                    if not _is_k4_free(adj):
                        # Find a K4 and remove one of its edges
                        # For each triangle, check common neighbors
                        for i2 in range(N):
                            for j2 in range(i2 + 1, N):
                                if not adj[i2][j2]:
                                    continue
                                common = []
                                for k in range(N):
                                    if k != i2 and k != j2 and adj[i2][k] and adj[j2][k]:
                                        common.append(k)
                                for a in range(len(common)):
                                    for b in range(a + 1, len(common)):
                                        if adj[common[a]][common[b]]:
                                            # Found K4: i2, j2, common[a], common[b]
                                            # Remove edge (i2, j2)
                                            adj[i2][j2] = False
                                            adj[j2][i2] = False
                                            changed = True
                                            break
                                    if changed:
                                        break
                                if changed:
                                    break
                            if changed:
                                break
                        break
        
        if not _is_k4_free(adj):
            continue
        
        # Now do local search
        adj2, score = _local_search(adj, time.time(), deadline)
        if score < best_score:
            best_score = score
            best_adj = adj2
        
        if best_score <= 8:
            break
    
    if best_adj is None:
        # Fallback: complete tripartite K6,6,6
        parts = [range(0, 6), range(6, 12), range(12, 18)]
        return [[a, b] for i in range(3) for j in range(i + 1, 3) for a in parts[i] for b in parts[j]]
    
    return _adj_to_edges(best_adj)
