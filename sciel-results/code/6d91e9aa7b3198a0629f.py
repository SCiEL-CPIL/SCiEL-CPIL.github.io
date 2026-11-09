import numpy as np
import random
import time

N = 31
E = 91
DEADLINE_SECONDS = 140.0

def _count_c4(adj):
    """
    Count C4s in graph represented by adjacency matrix.
    C4 count = 1/2 * sum_{i<j} C(A2[i,j], 2)
    where A2 = adj @ adj gives number of length-2 paths (common neighbors).
    """
    A2 = adj @ adj
    i, j = np.triu_indices(N, k=1)
    common_counts = A2[i, j]
    terms = common_counts * (common_counts - 1) // 2
    total = np.sum(terms)
    return total // 2

def _build_polarity_graph():
    """
    Build a polarity graph on q^2+q+1 = 31 vertices.
    Vertices are points in PG(2, q) plus the line at infinity? 
    Actually, polarity graph construction:
    Vertices: (x, y) for x, y in GF(q) plus one extra vertex (infinity)?
    No, for q odd, polarity graph has q^2+q+1 vertices.
    
    Standard construction:
    Let V = GF(q) x GF(q) union {∞} ... wait, that's q^2+1.
    For q^2+q+1, we use projective plane points.
    
    Simpler: Use the standard polarity graph on q^2+q+1 vertices.
    Vertices are pairs (x, y) in GF(q)^2 plus one vertex at infinity? No.
    
    Let's use the construction from Bannai et al. or similar.
    Vertices: V = {0, 1, ..., q-1} x {0, 1, ..., q-1} plus one extra? No, q^2+q+1.
    
    Actually, the standard polarity graph on q^2+q+1 vertices is:
    Vertices are the points of the projective plane PG(2, q).
    Two vertices are adjacent if and only if the point represented by one lies on the line represented by the other (under the polarity).
    
    For q=5:
    Points in PG(2, 5): (x, y, z) with not all zero, modulo scaling.
    Total points = (5^3-1)/(5-1) = 124/4 = 31.
    
    Polarity: (x, y, z) is orthogonal to (x', y', z') if xx' + yy' + zz' = 0 in GF(5).
    (Since q=5 is odd, the form xx'+yy'+zz' is a non-degenerate symmetric bilinear form.)
    
    Vertices are points. Edge between P and Q if P is orthogonal to Q.
    Note: For q odd, a point P is NOT orthogonal to itself (since xx+yy+zz != 0 for non-zero P? 
    Actually, xx'+yy'+zz' with P=Q gives x^2+y^2+z^2. In GF(5), 1^2+1^2+1^2=3 != 0. 
    But there might be points where x^2+y^2+z^2=0? 
    If so, they are self-adjacent. But polarity graphs are usually simple graphs.
    Let's check: In GF(5), x^2+y^2+z^2=0 has solutions? 
    1+1+3=5=0? No, 3 is not a square. 
    Squares in GF(5): 0, 1, 4.
    1+1+4=6=1. 1+4+4=9=4. 4+4+4=12=2. 
    So x^2+y^2+z^2=0 only if x=y=z=0? 
    Wait, 1+4+4=9=4. 
    Actually, -1=4 is a square. So x^2+y^2=0 has solutions. 
    But x^2+y^2+z^2=0. 
    Let's just compute. If self-adjacent, we remove the loop.
    
    Let's enumerate all 31 points.
    Points: (1, y, z) for y,z in GF(5) -> 25 points.
    (0, 1, z) for z in GF(5) -> 5 points.
    (0, 0, 1) -> 1 point.
    Total 31.
    """
    q = 5
    # Generate all points in PG(2, q)
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: make first non-zero component 1
                if x != 0:
                    p = (1, y * pow(x, q - 2, q) % q, z * pow(x, q - 2, q) % q)
                elif y != 0:
                    p = (0, 1, z * pow(y, q - 2, q) % q)
                else:
                    p = (0, 0, 1)
                if p not in seen:
                    seen.add(p)
                    points.append(p)
    
    # Assign indices 0..30
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # Build adjacency
    adj = np.zeros((N, N), dtype=int)
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            p = points[i]
            q_pt = points[j]
            # Check orthogonality: p . q_pt = 0
            dot = (p[0] * q_pt[0] + p[1] * q_pt[1] + p[2] * q_pt[2]) % q
            if dot == 0:
                adj[i, j] = 1
                adj[j, i] = 1
                edges.append((i, j))
    
    # The polarity graph might have more or fewer than 91 edges.
    # For q=5, expected edges?
    # Each vertex has degree q+1=6? Or q? 
    # In a polarity graph, degree is q or q+1 depending on self-adjacency.
    # Total edges ~ 31 * 6 / 2 = 93 or 31 * 5 / 2 = 77.5.
    # We need exactly 91 edges.
    # If the polarity graph has, say, 93 edges, we can remove 2.
    # If it has 77, we need to add 14.
    
    # Let's check the actual number of edges.
    # For q=5, the number of edges in the polarity graph is (q^3+q^2+q+1)/2? No.
    # Let's just compute and adjust.
    
    current_edges = len(edges)
    
    # If we have too many, remove random edges.
    # If too few, add random non-edges.
    # This gives a good starting point for local search.
    
    all_possible_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    edge_set = set(edges)
    
    if current_edges > E:
        # Remove random edges
        to_remove = random.sample(edges, current_edges - E)
        for u, v in to_remove:
            edge_set.remove((u, v))
            adj[u, v] = 0
            adj[v, u] = 0
            edges.remove((u, v))
    elif current_edges < E:
        # Add random non-edges
        need = E - current_edges
        non_edges = [e for e in all_possible_edges if e not in edge_set]
        to_add = random.sample(non_edges, need)
        for u, v in to_add:
            edge_set.add((u, v))
            adj[u, v] = 1
            adj[v, u] = 1
            edges.append((u, v))
    
    return edges, adj, edge_set

def _delta_c4_swap(adj, adj2, remove_edge, add_edge):
    """
    Compute the change in C4 count when swapping remove_edge with add_edge.
    This is complex to do exactly in O(deg^2) for a swap.
    Given N=31, it might be faster to just recompute count_c4.
    Let's benchmark: count_c4 does a 31x31 matrix multiplication.
    31^3 = 29791 operations. Very fast.
    So we can just recompute.
    """
    pass

def construct():
    start_time = time.time()
    
    # Start with polarity graph
    edges, adj, edge_set = _build_polarity_graph()
    
    current_c4 = _count_c4(adj)
    best_c4 = current_c4
    best_edges = edges[:]
    
    all_possible_edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local search: random swaps
    # To speed up, we can precompute the list of non-edges and update it.
    # But with N=31, total edges = 465. Non-edges = 465 - 91 = 374.
    # We can just pick a random non-edge by rejection sampling.
    
    no_improve = 0
    max_no_improve = 200
    
    while time.time() - start_time < DEADLINE_SECONDS:
        # Pick a random edge to remove
        u1, v1 = random.choice(edges)
        
        # Pick a random non-edge to add
        u2, v2 = None
        for _ in range(50):
            u2, v2 = random.choice(all_possible_edges)
            if (u2, v2) not in edge_set:
                break
        if u2 is None:
            continue
        
        # Perform swap
        adj[u1, v1] = 0
        adj[v1, u1] = 0
        edge_set.remove((u1, v1))
        
        adj[u2, v2] = 1
        adj[v2, u2] = 1
        edge_set.add((u2, v2))
        
        # Update edges list
        # Finding the index is O(E). E=91. Fine.
        idx = edges.index((u1, v1))
        edges[idx] = (u2, v2)
        
        new_c4 = _count_c4(adj)
        
        if new_c4 < current_c4:
            current_c4 = new_c4
            no_improve = 0
            if current_c4 < best_c4:
                best_c4 = current_c4
                best_edges = edges[:]
                if best_c4 <= 2:
                    break
        else:
            # Revert
            adj[u2, v2] = 0
            adj[v2, u2] = 0
            edge_set.remove((u2, v2))
            
            adj[u1, v1] = 1
            adj[v1, u1] = 1
            edge_set.add((u1, v1))
            
            edges[idx] = (u1, v1)
            no_improve += 1
            
            if no_improve > max_no_improve:
                # Restart from best or random
                if random.random() < 0.5:
                    # Restart from best
                    edges = best_edges[:]
                    edge_set = set(edges)
                    adj = np.zeros((N, N), dtype=int)
                    for u, v in edges:
                        adj[u, v] = 1
                        adj[v, u] = 1
                    current_c4 = best_c4
                else:
                    # Random restart
                    edges, adj, edge_set = _build_polarity_graph()
                    current_c4 = _count_c4(adj)
                no_improve = 0
                max_no_improve = 200 + random.randint(0, 200)
        
        # Check time
        if time.time() - start_time > DEADLINE_SECONDS - 2:
            break
    
    return best_edges

if __name__ == "__main__":
    print(construct())
