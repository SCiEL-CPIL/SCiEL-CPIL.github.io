import time
import numpy as np
import random

def construct():
    n = 40
    deadline = 115.0
    start_time = time.time()

    # Helper to compute A^5 for a 40x40 int matrix
    def compute_A5(mat):
        A2 = mat @ mat
        A3 = A2 @ mat
        A4 = A3 @ mat
        A5 = A4 @ mat
        return A5

    def is_c6_free(A):
        # Exact check: no 6-cycle means no path of length 5 between adjacent vertices?
        # Actually, a 6-cycle u-v-w-x-y-z-u implies a path of length 5 from u to v (v-w-x-y-z-u? No, v-w-x-y-z is length 4 from v to z, then z-u is edge.
        # Path length 5 from u to v: u-a-b-c-d-v. If u-v is an edge, this forms a 6-cycle.
        # So check: for all edges (u,v), (A^5)[u,v] == 0.
        A5 = compute_A5(A)
        for u in range(n):
            for v in range(u+1, n):
                if A[u,v] == 1 and A5[u,v] > 0:
                    return False
        return True

    # Strategy: Construct a graph based on the affine plane of order 7 (AG(2,7)).
    # AG(2,7) has 49 points. We select 40 points.
    # In AG(2,7), each line has 7 points. Any two points define a unique line.
    # If we connect points on the same line, we get a graph.
    # However, the "line graph" of AG(2,7) might have C6s.
    # Let's analyze: 
    # Points are (x,y) in F_7^2.
    # Lines are y = mx + b or x = c.
    # If we connect all pairs on a line, the graph is a union of cliques K7? No, lines intersect.
    # This creates a dense graph.
    
    # Alternative Spectral/Combinatorial Heuristic:
    # The task suggests "spectral heuristic ... random regular graph ... projected gradient".
    # But the STAGNATION directive says "STRUCTURAL change".
    # A known extremal graph for C6 is often related to polarity graphs or finite geometry.
    # The polarity graph of the projective plane of order q has n = q^2 + q + 1 vertices.
    # For q=6, n=43. Close to 40.
    # The polarity graph (Erdos-Renyi graph) of PG(2,q) is C4-free and has ~ q^3/2 edges.
    # Wait, polarity graphs are C4-free, which implies C6-free? No. C4-free does not imply C6-free.
    # But C4-free graphs are sparse. ex(n, C4) ~ n^{3/2}. For n=40, 40^{1.5} ~ 253.
    # C6-free allows denser graphs? ex(n, C6) ~ n^{5/4} or similar? No, ex(n, C6) is larger.
    # Actually, ex(n, C6) is bounded by O(n^{1+1/5})? No.
    # Let's stick to the prompt's hint: "spectral norm ... maximize edge count while keeping largest eigenvalue low".
    # High edge count with low spectral radius is characteristic of expanders or random graphs, but we need to AVOID C6.
    # C6s are "long" cycles. Low spectral radius doesn't directly kill C6s.
    
    # Let's try a different structural approach: 
    # Random Regular Graph + Local Search with better initialization.
    # The current program starts with 8 K5s (disjoint cliques). This is very far from optimal.
    # A random 5-regular graph on 40 vertices has 100 edges.
    # If we start with a random 5-regular graph, we are already close to 105.
    # Then we can use the local search to add edges.
    
    # Step 1: Generate a random 5-regular graph on 40 vertices.
    # Step 2: Check if it is C6-free. If not, repair it.
    # Step 3: Greedy add edges.
    
    def generate_random_regular(n, k, seed):
        rng = random.Random(seed)
        # Use configuration model
        half_edges = n * k // 2
        ports = []
        for i in range(n):
            ports.extend([i] * k)
        rng.shuffle(ports)
        
        A = np.zeros((n, n), dtype=np.int64)
        for i in range(0, n * k, 2):
            u = ports[i]
            v = ports[i+1]
            if u == v:
                # Self loop, swap with another
                # Find another port
                for j in range(i+2, n*k, 2):
                    if ports[j] != u:
                        v = ports[j]
                        ports[j] = ports[i+1]
                        break
            if u != v and A[u,v] == 0:
                A[u,v] = 1
                A[v,u] = 1
            else:
                # Multi-edge or self-loop, try to fix by swapping
                # Simple fix: just ignore for now, degree might be slightly off
                pass
        
        # Check degrees
        deg = np.sum(A, axis=1)
        # If degrees are not exactly k, we might have issues. 
        # For a heuristic, approximate regularity is fine.
        return A

    # Let's try to construct a specific known good structure if possible, or use the random regular start.
    # The "best known" is 105 edges. 
    # A 5-regular graph has 100 edges. 
    # If we can start with a 5-regular graph that is C6-free, we have 100 edges.
    # Then we add 5 more.
    
    best_A = np.zeros((n, n), dtype=np.int64)
    best_count = 0
    
    # Try several random regular starts
    for seed in range(50):
        if time.time() > deadline: break
        A = generate_random_regular(n, 5, seed)
        count = np.sum(A) // 2
        
        # Check C6-free
        if not is_c6_free(A):
            continue
        
        if count > best_count:
            best_count = count
            best_A = A.copy()
            
    # If no valid 5-regular found, fall back to 4-regular or lower
    if best_count == 0:
        for seed in range(50):
            if time.time() > deadline: break
            A = generate_random_regular(n, 4, seed)
            count = np.sum(A) // 2
            if not is_c6_free(A):
                continue
            if count > best_count:
                best_count = count
                best_A = A.copy()
                
    # If still nothing, use the K5 init as fallback
    if best_count == 0:
        for k in range(8):
            for a in range(5):
                for b in range(a + 1, 5):
                    u = 5 * k + a
                    v = 5 * k + b
                    best_A[u, v] = 1
                    best_A[v, u] = 1
        best_count = 80

    # Now, local search to improve from this better starting point
    A = best_A.copy()
    A5 = compute_A5(A)
    
    while time.time() < deadline:
        improved = False
        
        # Greedy add
        non_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if A[u, v] == 0:
                    non_edges.append((u, v))
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() > deadline: break
            if A[u, v] == 1: continue
            if A5[u, v] == 0:
                A[u, v] = 1
                A[v, u] = 1
                A5 = compute_A5(A)
                improved = True
                current_count = np.sum(A) // 2
                if current_count > best_count:
                    best_count = current_count
                    best_A = A.copy()
        
        if not improved:
            # Try swaps
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if A[u, v] == 1:
                        edges.append((u, v))
            
            if not edges: break
            
            # Try 10 random swaps
            for _ in range(10):
                if time.time() > deadline: break
                rem_u, rem_v = random.choice(edges)
                A_old = A.copy()
                
                A[rem_u, rem_v] = 0
                A[rem_v, rem_u] = 0
                A5 = compute_A5(A)
                
                added = 0
                non_edges = []
                for u in range(n):
                    for v in range(u + 1, n):
                        if A[u, v] == 0:
                            non_edges.append((u, v))
                random.shuffle(non_edges)
                
                for u, v in non_edges:
                    if added >= 2: break
                    if time.time() > deadline: break
                    if A[u, v] == 1: continue
                    if A5[u, v] == 0:
                        A[u, v] = 1
                        A[v, u] = 1
                        A5 = compute_A5(A)
                        added += 1
                
                current_count = np.sum(A) // 2
                if current_count > best_count:
                    best_count = current_count
                    best_A = A.copy()
                    # Update edges list
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if A[u, v] == 1:
                                edges.append((u, v))
                else:
                    A = A_old

    edges_out = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges_out.append([u, v])
    return edges_out
