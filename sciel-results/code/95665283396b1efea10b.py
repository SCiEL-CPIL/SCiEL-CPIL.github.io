import time
import random
import numpy as np

def construct():
    n = 40
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    # Helper: Check if adding edge (u, v) creates a C6 in the current graph.
    # A C6 is created iff there is a path of length 4 between u and v.
    # Path of length 4: u - a - b - c - v
    # This requires a in N(u), c in N(v), and (a, c) connected by a path of length 2 (a-b-c).
    # So we need to check if there exists a in N(u) and c in N(v) such that N(a) intersects N(c).
    # Equivalently, let S = N(u) ∩ N(v). If S is not empty, we can have u-a-...-v where a is in S?
    # No, u-a-b-c-v. a is neighbor of u, c is neighbor of v.
    # b is neighbor of a AND neighbor of c.
    # So we need to check if for any a in N(u) and c in N(v), N(a) ∩ N(c) is not empty.
    # This is O(deg(u) * deg(v) * n) which is too slow if done naively for every step.
    
    # Better approach for exact check:
    # We want to know if (A^4)[u][v] > 0?
    # As discussed, walks of length 4 might not be simple paths.
    # However, if the graph is C6-free, any walk of length 4 between u and v implies a C6?
    # If the walk is u-a-b-c-v.
    # If u,v,a,b,c are distinct, it's a C6.
    # If not distinct, e.g., u-a-u-b-v (impossible since simple graph, no loops, u!=a).
    # u-a-b-a-v? No, b-a is edge, a-v is edge. Walk u-a-b-a-v.
    # This implies edges (u,a), (a,b), (a,v).
    # Does this create a C6 when adding (u,v)?
    # The cycle would be u-v-a-b-a... no, simple cycle.
    # The walk u-a-b-a-v has repeated vertex a.
    # The simple path from u to v in this walk is u-a-v (length 2).
    # Adding (u,v) creates a triangle u-v-a-u. This is C3, not C6.
    # So A4 > 0 is NOT sufficient to reject if it corresponds to a non-simple walk.
    
    # However, for the purpose of *maximizing* edges, we can use a conservative check:
    # Reject if A4[u][v] > 0. This might reject some safe edges (false positives),
    # but it guarantees we never add an edge that creates a C6 (no false negatives).
    # Given the density, false positives might be low.
    # Alternatively, we can implement the exact check:
    # Count simple paths of length 4.
    # For n=40, we can precompute A2 (paths of length 2).
    # A path of length 4 u-a-b-c-v exists if there is a pair (a,c) such that:
    # a in N(u), c in N(v), and (a,c) is connected by a path of length 2 (a-b-c).
    # So we need to check if N(u) and N(v) contain a "2-step connected" pair.
    # Let S_u = N(u), S_v = N(v).
    # We need to check if there exists a in S_u, c in S_v such that A2[a][c] > 0.
    # Note: a and c must be distinct? If a=c, then u-a-v is a path of length 2.
    # If a=c, the walk is u-a-a-v? No, a-b-a.
    # If a=c, then we have u-a-v. Adding (u,v) creates a triangle. No C6.
    # So we should exclude a=c from the check?
    # If a=c, the "path of length 4" is u-a-b-a-v. This is not a simple path.
    # So we only care about a != c.
    
    # Efficient Exact Check:
    # Precompute A2 (int matrix).
    # For candidate (u,v):
    #   Create mask for N(u) and N(v).
    #   We need to check if (A2[N(u), N(v)]) has any non-zero entry where row != col?
    #   Actually, if A2[a][c] > 0, there is a path a-b-c.
    #   If a=c, A2[a][a] > 0 means there is a triangle involving a.
    #   If a=c, the walk is u-a-b-a-v. Not a simple C6.
    #   So we check if there exists a in N(u), c in N(v), a != c, such that A2[a][c] > 0.
    
    # Let's implement this using NumPy for speed.
    # A is 40x40 int8.
    # A2 = A @ A.
    
    def is_safe(u, v, A, A2):
        # Get neighbors of u and v as boolean masks or lists
        # Using lists is probably faster for n=40 than slicing NumPy arrays if done in Python loop?
        # Let's try NumPy slicing.
        # N_u = A[u, :] > 0
        # N_v = A[v, :] > 0
        # We want to check the submatrix A2[N_u, N_v]
        # But we must exclude a=c.
        # If we just check if any element in A2[N_u, N_v] is non-zero, it might be A2[a][a] (diagonal).
        # Diagonal of A2 corresponds to triangles.
        # If the only non-zero is on the diagonal, it's safe.
        # So we check if (A2[N_u, N_v] - np.diag(A2[N_u, N_v])) has any non-zero?
        # This is messy.
        
        # Alternative:
        # Iterate over neighbors of u.
        # For each a in N(u):
        #    Check if any neighbor c of v (c != a) has A2[a][c] > 0.
        # This is O(deg(u) * deg(v)). Max 40*40=1600 ops. Very fast.
        
        N_u = np.where(A[u, :] > 0)[0]
        N_v = np.where(A[v, :] > 0)[0]
        
        for a in N_u:
            if a == v: continue # u-v is the edge we are adding, so v is not in N(u) yet
            for c in N_v:
                if c == u: continue
                if a == c: continue
                if A2[a, c] > 0:
                    return False
        return True

    # Initialize with a random graph or empty?
    # Start with empty to ensure validity, then add edges greedily.
    # To improve density, we can start with a random sparse graph.
    
    best_edges = []
    best_count = 0
    
    # Run multiple restarts with different random seeds/orders
    for restart in range(20):
        if time.time() > deadline - 5:
            break
            
        # Randomize the order of edge additions?
        # Or start with a random base?
        # Let's start with empty and use a randomized greedy order.
        
        A = np.zeros((n, n), dtype=np.int8)
        A2 = A @ A
        
        current_edges = []
        
        # Pre-generate all pairs
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        # Greedy Add
        for u, v in all_pairs:
            if time.time() > deadline - 2:
                break
                
            # Quick check: if A[u][v] is already 1, skip (shouldn't happen in this loop)
            if A[u, v] == 1:
                continue
                
            if is_safe(u, v, A, A2):
                # Add edge
                A[u, v] = 1
                A[v, u] = 1
                current_edges.append([u, v])
                
                # Update A2
                # A2 = A @ A. Recomputing is 40x40 matmul. ~10-20 us.
                # 780 edges * 20us = 15ms per restart. 20 restarts = 300ms.
                # This is extremely fast. We can do many more restarts.
                A2 = A @ A
        
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
        # If we reached a high count, we might want to do local search?
        # But the directive says "different strategy". Greedy MIS is the strategy.
        # However, greedy can get stuck.
        # Let's add a second phase: Local Search (Swap) on the best found so far.
        # But wait, the prompt asks for a "MATERIALLY DIFFERENT" strategy.
        # The current program uses "Randomized Greedy with Backtracking/Restart... Swap".
        # My current plan is "Greedy MIS on Conflict Hypergraph".
        # Is that different enough?
        # The current program uses A4 > 0 as a proxy (conservative) and random swaps.
        # My program uses EXACT path check and greedy order.
        # To make it stronger, I will combine Greedy Construction with a "Conflict-Guided" Local Search.
        # Instead of random swaps, I will identify "conflicting" edges (those that prevent adding many others) and remove them.
        
        # Actually, let's just stick to the Greedy MIS with randomization.
        # The key improvement is the EXACT check which allows more edges to be added than the conservative A4 check.
        # And the greedy order (highest degree first? or random?)
        # Random order is standard for MIS.
        
        # Let's try to increase the number of restarts since it's fast.
        # We have 145s. 20 restarts take < 1s.
        # Let's do 1000 restarts?
        # 1000 * 0.05s = 50s. Feasible.
        
    # Refine the loop to use more restarts
    best_edges = []
    best_count = 0
    
    num_restarts = 0
    while time.time() < deadline - 5:
        num_restarts += 1
        A = np.zeros((n, n), dtype=np.int8)
        A2 = A @ A
        current_edges = []
        
        all_pairs = [(u, v) for u in range(n) for v in range(u+1, n)]
        random.shuffle(all_pairs)
        
        for u, v in all_pairs:
            if A[u, v] == 1:
                continue
            if is_safe(u, v, A, A2):
                A[u, v] = 1
                A[v, u] = 1
                current_edges.append([u, v])
                A2 = A @ A
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = list(current_edges)
            
        # Optional: Try to improve best_edges with a few random swaps?
        # If we have time, let's try a local search on the best solution.
        if num_restarts % 10 == 0 and time.time() < deadline - 10:
            # Local Search on best_edges
            A_ls = np.zeros((n, n), dtype=np.int8)
            for e in best_edges:
                A_ls[e[0], e[1]] = 1
                A_ls[e[1], e[0]] = 1
            A2_ls = A_ls @ A_ls
            
            for _ in range(1000):
                if time.time() > deadline - 5:
                    break
                # Pick random edge to remove
                if not best_edges:
                    break
                idx = random.randint(0, len(best_edges)-1)
                ru, rv = best_edges[idx]
                
                # Try to add a random non-edge
                cu = random.randint(0, n-1)
                cv = random.randint(0, n-1)
                if cu == cv or A_ls[cu, cv] == 1:
                    continue
                if cu > cv: cu, cv = cv, cu
                
                # Check if swap is valid:
                # 1. Remove (ru, rv)
                # 2. Add (cu, cv)
                # We need to check if the new graph is C6-free.
                # This is expensive.
                # Approximation: Check if (cu, cv) is safe in the graph WITHOUT (ru, rv).
                # And check if removing (ru, rv) doesn't break anything (it doesn't, it only removes edges).
                
                # Temporarily remove (ru, rv)
                A_ls[ru, rv] = 0
                A_ls[rv, ru] = 0
                A2_temp = A_ls @ A_ls
                
                if is_safe(cu, cv, A_ls, A2_temp):
                    # Swap successful
                    A_ls[cu, cv] = 1
                    A_ls[cv, cu] = 1
                    A2_ls = A2_temp # Already computed
                    best_edges[idx] = [cu, cv]
                    # Note: This maintains edge count. It doesn't increase it.
                    # To increase count, we need to ADD without removing.
                    # The greedy phase already did that.
                    # This swap is just to escape local optima for future adds?
                    # No, we don't have a future add phase here.
                    # So this swap is useless for increasing count.
                    pass
                else:
                    # Revert
                    A_ls[ru, rv] = 1
                    A_ls[rv, ru] = 1
                    A2_ls = A_ls @ A_ls # Recompute to be safe, or just keep old?
                    # Since we didn't change it, A2_ls is still valid for the original state.
                    # But we computed A2_temp. We should restore A2_ls.
                    # Actually, we didn't change A_ls permanently in the failed case.
                    # So A2_ls is still correct for the current best_edges.
                    
    return best_edges
