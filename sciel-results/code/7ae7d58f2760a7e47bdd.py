import numpy as np
import time
import random

def construct():
    n = 20
    # Initialize with the seed graph: 5 disjoint K4s
    A = np.zeros((n, n), dtype=int)
    for k in range(5):
        for a in range(4):
            for b in range(a + 1, 4):
                u = 4 * k + a
                v = 4 * k + b
                A[u, v] = 1
                A[v, u] = 1
    
    # Calculate common neighbors matrix K = A @ A
    # K[i, j] is number of common neighbors of i and j
    K = A @ A
    
    current_edges = int(np.sum(A)) // 2
    best_edges = current_edges
    best_A = A.copy()
    
    start_time = time.time()
    timeout = 110  # Leave 10s for safety
    
    # Helper to check if adding edge (u, v) is valid
    def can_add(u, v):
        # u and v are not connected
        # Check neighbors of u: for all x in N(u), K[x, v] must be <= 1
        # Because adding (u, v) makes u a common neighbor of x and v
        # So K[x, v] increases by 1. We need K[x, v] + 1 <= 2 => K[x, v] <= 1
        for x in range(n):
            if A[u, x] == 1 and x != v:
                if K[x, v] > 1:
                    return False
        # Check neighbors of v: for all y in N(v), K[u, y] must be <= 1
        for y in range(n):
            if A[v, y] == 1 and y != u:
                if K[u, y] > 1:
                    return False
        return True

    # Helper to add edge and update K
    def add_edge(u, v):
        nonlocal current_edges
        A[u, v] = 1
        A[v, u] = 1
        # Update K
        # K[x, v] += 1 for x in N(u)
        for x in range(n):
            if A[u, x] == 1 and x != v:
                K[x, v] += 1
                K[v, x] += 1
        # K[u, y] += 1 for y in N(v)
        for y in range(n):
            if A[v, y] == 1 and y != u:
                K[u, y] += 1
                K[y, u] += 1
        # Note: K[u, v] itself does not change because u and v are not common neighbors of themselves in this context?
        # Wait. K[u, v] is number of z such that u-z and v-z.
        # Adding (u, v) doesn't change z's connections. So K[u, v] is unchanged.
        # However, the loop above:
        # If x=u? A[u, u]=0.
        # If x=v? A[u, v]=1 now, but we skipped x==v.
        # So K[u, v] is not touched. Correct.
        current_edges += 1

    # Helper to remove edge and update K
    def remove_edge(u, v):
        nonlocal current_edges
        A[u, v] = 0
        A[v, u] = 0
        # Update K
        for x in range(n):
            if A[u, x] == 1 and x != v:
                K[x, v] -= 1
                K[v, x] -= 1
        for y in range(n):
            if A[v, y] == 1 and y != u:
                K[u, y] -= 1
                K[y, u] -= 1
        current_edges -= 1

    # Main Loop
    while time.time() - start_time < timeout:
        # Phase 1: Greedy Add
        changed = True
        while changed:
            changed = False
            # Shuffle order to avoid bias
            pairs = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 0]
            random.shuffle(pairs)
            for u, v in pairs:
                if can_add(u, v):
                    add_edge(u, v)
                    changed = True
                    break # Restart to re-evaluate with new state
        
        # Phase 2: Update Best
        if current_edges > best_edges:
            best_edges = current_edges
            best_A = A.copy()
            
        # Phase 3: Perturbation
        # If stuck or to explore, remove some random edges
        # We always perturb to try to find a better configuration
        # Number of edges to remove: 1 to 5
        num_remove = random.randint(1, 5)
        edges_list = [(u, v) for u in range(n) for v in range(u + 1, n) if A[u, v] == 1]
        if len(edges_list) < num_remove:
            num_remove = len(edges_list)
        
        if num_remove > 0:
            random.shuffle(edges_list)
            for u, v in edges_list[:num_remove]:
                remove_edge(u, v)
                
        # If we removed too many, maybe restart from best?
        # If current_edges is much lower than best, restart might be faster
        if current_edges < best_edges - 10:
            A = best_A.copy()
            K = A @ A
            current_edges = best_edges

    # Extract edges from best_A
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_A[u, v] == 1:
                edges.append([u, v])
                
    return edges
