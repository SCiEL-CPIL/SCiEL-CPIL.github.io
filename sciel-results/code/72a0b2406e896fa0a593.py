import time
import numpy as np
import random

def construct():
    n = 40
    deadline = 118.0  # Safety margin from 120s limit
    start_time = time.time()
    
    # Helper to check if adding edge (u, v) creates a C6
    # A C6 is formed if there is a path of length 4 between u and v in the current graph.
    # This is equivalent to (A^4)[u, v] > 0.
    # We maintain A4 = A^4.
    def compute_A4(mat):
        # Efficient power for small matrix
        # A^2
        A2 = mat @ mat
        # A^4
        A4 = A2 @ A2
        return A4

    def is_c6_free(A):
        # Check if the graph is C6-free
        # This is useful for verification or initialization checks, but we mostly rely on incremental checks.
        A4 = compute_A4(A)
        # If any off-diagonal element of A4 is > 0, there is a C6?
        # Wait. (A^4)[u, v] > 0 means there is a walk of length 4 from u to v.
        # A walk of length 4 u-w1-w2-w3-v.
        # If u, w1, w2, w3, v are distinct, it's a C6 path (which closes to C6 if u-v is edge? No).
        # The condition for "Adding (u,v) creates a C6" is that there exists a path of length 4 between u and v.
        # If we are checking the whole graph, we need to ensure no C6 exists.
        # A C6 is a cycle of length 6.
        # If (A^5)[u, v] > 0 and A[u, v] == 1, does it imply C6?
        # A path of length 5 u-v1-v2-v3-v4-v. If all distinct, it's a C6.
        # If not distinct, it might be a shorter cycle or backtracking.
        # However, for sparse graphs, (A^4)[u, v] > 0 for u != v usually implies a C6 risk if we add the edge.
        # Specifically, if we maintain the invariant that the graph is C6-free, then adding (u,v) is safe iff (A^4)[u, v] == 0.
        # Why? If (A^4)[u, v] > 0, there is a walk u-x-y-z-v of length 4.
        # If the graph is C6-free, can this walk exist without being part of a C6?
        # If the walk is simple (distinct vertices), then u-x-y-z-v is a path of length 4.
        # Adding u-v creates a cycle of length 5? No, u-x-y-z-v-u is length 5.
        # Wait.
        # C6 is a cycle of length 6.
        # If we add edge (u,v), we create a C6 if there is a path of length 5 between u and v in the OLD graph.
        # Path u-w1-w2-w3-w4-v. Adding u-v makes u-w1-w2-w3-w4-v-u (length 6).
        # So the check is (A^5)[u, v] > 0.
        
        # Let's re-verify the power.
        # Cycle length k.
        # Adding edge (u,v) creates a cycle of length L if there is a path of length L-1 between u and v.
        # We want to avoid C6. So L=6. Path length 5.
        # So we need to check (A^5)[u, v] > 0.
        
        # The previous code used A^5.
        # My helper above computes A^4. I should compute A^5.
        pass

    def compute_A5(mat):
        A2 = mat @ mat
        A3 = A2 @ mat
        A5 = A3 @ A2
        return A5

    # Strategy:
    # 1. Spectral Initialization:
    #    Generate random vectors for vertices.
    #    Connect u, v if dot product > threshold.
    #    This creates a graph with specific geometric properties.
    #    For C6-free, we want high edge density but low "long paths".
    #    Random graphs G(n, p) have many C6s for p > n^-0.5.
    #    We want a graph that is "almost" random but with C6s removed or avoided.
    #    
    #    Alternative Initialization: 
    #    Start with a known C6-free structure, e.g., union of K5s or similar, but that's what the seed did.
    #    Let's try a "Greedy from Scratch" with a better order.
    #    Or "Simulated Annealing" style?
    
    # Let's try a different structural approach:
    # Construct the graph by adding edges in a deterministic order that prioritizes high-degree vertices early?
    # Or use a bipartite-like structure?
    # C6-free bipartite graphs have O(n^(4/3)) edges. 40^(4/3) ~ 350.
    # General graphs can have more.
    # ex(40, C6) is around 105-120?
    # Known bounds: ex(n, C6) <= (1/6) n^(4/3) + O(n). For n=40, 40^(4/3)/6 ~ 350/6 ~ 58?
    # Wait, the bound is for bipartite?
    # For general graphs, ex(n, C6) is O(n^(4/3)).
    # Actually, the best known upper bound for ex(n, C6) is (1/6) n^(4/3) + O(n).
    # 40^(1.333) = 40 * 40^(1/3) = 40 * 3.42 = 136.8.
    # 136.8 / 6 = 22.8? No, the constant is smaller?
    # Let's check the seed score. Seed gets 105 edges.
    # If the upper bound is ~100-120, then 105 is very close to optimal.
    
    # If 105 is close to optimal, we need a very efficient search to find 106+.
    # The current program uses random restarts and local swaps.
    
    # New Strategy: Deterministic Greedy with Backtracking (Limited Depth)
    # Or: Iterated Local Search (ILS) with a specific neighborhood.
    
    # Let's try ILS:
    # 1. Generate a random valid graph (using greedy add).
    # 2. Local Search: Try all possible single edge swaps (remove one, add one) to improve edge count.
    #    - Remove edge e.
    #    - Try to add any non-edge f such that adding f keeps it C6-free.
    #    - If we can add >1 edge than removed, accept.
    # 3. Perturb: Randomly remove k edges.
    # 4. Repeat Local Search.
    
    # The current program does a simplified version of this.
    # The bottleneck is checking C6-freeness.
    # (A^5)[u, v] computation is O(n^3) ~ 64k ops. Very fast.
    # We can afford to check many candidates.
    
    # Let's implement a more aggressive Local Search.
    # Instead of random swaps, we systematically try to "repair" the graph after removing an edge.
    
    A = np.zeros((n, n), dtype=np.int64)
    
    # Phase 1: Greedy Construction
    # Order edges by some heuristic? 
    # Random order is fine if we restart.
    # Let's do multiple restarts of Greedy + Local Search.
    
    best_edges = 0
    best_A = np.zeros((n, n), dtype=np.int64)
    
    def greedy_construct():
        A_init = np.zeros((n, n), dtype=np.int64)
        A5_init = compute_A5(A_init)
        edge_count = 0
        
        # Randomly shuffle all possible edges
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(all_edges)
        
        for u, v in all_edges:
            if time.time() > deadline:
                break
            if A5_init[u, v] == 0:
                A_init[u, v] = 1
                A_init[v, u] = 1
                edge_count += 1
                A5_init = compute_A5(A_init)
        
        return A_init, edge_count

    def local_search(A_ls, edge_count_ls):
        A5_ls = compute_A5(A_ls)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each edge
            edges_in_graph = np.argwhere(A_ls > 0)
            # Randomize order of removal to avoid bias
            random.shuffle(edges_in_graph)
            
            for u, v in edges_in_graph:
                if time.time() > deadline:
                    break
                if u > v:
                    u, v = v, u
                if A_ls[u, v] == 0:
                    continue # Already removed or invalid index in list?
                
                # Remove edge (u, v)
                A_ls[u, v] = 0
                A_ls[v, u] = 0
                edge_count_ls -= 1
                A5_ls = compute_A5(A_ls)
                
                # Try to add edges
                # We want to add at least 2 edges to be net +1, or 1 to be net 0 (but we need improvement to stop)
                # Actually, if we remove 1 and add 1, count is same. We want to escape local maxima.
                # So we should accept if count is same but structure is different?
                # For now, let's require strict improvement or allow neutral moves with probability?
                # Let's just try to add as many as possible.
                
                added = 0
                # Try all non-edges?
                # n=40, 780 edges.
                # Checking all is 780 * O(n^3) ~ 50M ops. Might be too slow per swap.
                # Let's try a subset or random sampling.
                
                # Sample random non-edges
                attempts = 0
                max_attempts = 200
                while added < 2 and attempts < max_attempts and time.time() < deadline:
                    attempts += 1
                    nu = random.randint(0, n - 1)
                    nv = random.randint(0, n - 1)
                    if nu == nv or A_ls[nu, nv] == 1:
                        continue
                    if A5_ls[nu, nv] == 0:
                        A_ls[nu, nv] = 1
                        A_ls[nv, nu] = 1
                        edge_count_ls += 1
                        added += 1
                        A5_ls = compute_A5(A_ls)
                
                # If we added enough to be better or equal, keep it?
                # If added >= 1, we are at least as good.
                # To ensure progress, we might need to accept neutral moves occasionally.
                # But for simplicity, let's only accept if we added > removed (i.e. added >= 2? No, removed 1, added 2 -> +1).
                # If added == 1, net 0.
                # If added == 0, net -1 (revert).
                
                if added < 1:
                    # Revert
                    A_ls[u, v] = 1
                    A_ls[v, u] = 1
                    edge_count_ls += 1
                    A5_ls = compute_A5(A_ls)
                else:
                    # Keep the change. Even if net 0, it changes the landscape.
                    # Mark as improved to continue the loop
                    if added >= 1:
                        improved = True
            
        return A_ls, edge_count_ls

    # Main Loop: Multiple Restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        A_curr, count_curr = greedy_construct()
        
        # Local Search
        A_curr, count_curr = local_search(A_curr, count_curr)
        
        if count_curr > best_edges:
            best_edges = count_curr
            best_A = A_curr.copy()
            
        # Early stop if we hit a theoretical upper bound?
        # We don't know it exactly, but if we get > 110, it's good.
        # Just keep going.

    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
    return edges

# Ensure function name matches
# The task says: Keep the function names and signatures of the seed program.
# Seed: def construct():
