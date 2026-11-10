import time
import random
import numpy as np

def construct():
    """
    Constructs a C6-free graph on 40 vertices maximizing edges.
    Uses spectral gap heuristic to guide edge insertion.
    """
    N = 40
    deadline = time.time() + 115  # Safety margin
    
    # Helper to check if adding edge (u, v) creates a C6
    # Uses iterative DFS with bitmask for visited nodes
    def can_add(adj_list, adj_set, u, v):
        # We look for a simple path u-a-b-c-d-v of length 5.
        # This is equivalent to finding a node d at distance 4 from u
        # such that d is adjacent to v, and the path u..d is simple and avoids v.
        
        # Stack: (node, depth, mask)
        # mask: integer where bit i is 1 if node i is in the current path
        start_mask = 1 << u
        stack = [(u, 0, start_mask)]
        
        while stack:
            node, depth, mask = stack.pop()
            
            if depth == 4:
                # Check if v is a neighbor of node and v is not in the path
                if v in adj_set[node] and not (mask & (1 << v)):
                    return False # C6 exists
                continue
            
            for nb in adj_list[node]:
                if nb == v:
                    continue
                if mask & (1 << nb):
                    continue
                stack.append((nb, depth + 1, mask | (1 << nb)))
        
        return True
    
    # Build adjacency matrix from adj_list
    def build_adj_matrix(adj_list):
        A = np.zeros((N, N), dtype=np.float64)
        for u in range(N):
            for v in adj_list[u]:
                A[u, v] = 1.0
        return A
    
    # Compute spectral gap (lambda_1 - lambda_2)
    def spectral_gap(adj_list):
        A = build_adj_matrix(adj_list)
        # Use numpy's eigh for symmetric matrices
        eigenvalues = np.linalg.eigvalsh(A)
        # eigenvalues are in ascending order
        lambda_1 = eigenvalues[-1]  # largest
        lambda_2 = eigenvalues[-2]  # second largest
        return lambda_1 - lambda_2

    # Initial Graph: 8 disjoint K5s
    adj_list = [[] for _ in range(N)]
    adj_set = [set() for _ in range(N)]
    edges = []
    
    for k in range(8):
        base = 5 * k
        for a in range(5):
            for b in range(a + 1, 5):
                u, v = base + a, base + b
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append([u, v])
                
    best_edges = len(edges)
    best_adj_list = [lst[:] for lst in adj_list]
    best_adj_set = [set(s) for s in adj_set]
    
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Local Search with spectral gap guidance
    while time.time() < deadline:
        # Create list of missing edges
        missing = []
        for u, v in all_pairs:
            if v not in adj_set[u]:
                missing.append((u, v))
                
        if not missing:
            break
        
        # Use spectral gap to prioritize candidate edges
        # Compute current spectral gap
        current_gap = spectral_gap(adj_list)
        
        # Score each missing edge by potential spectral gap improvement
        # We'll use a cheaper proxy: prefer edges between vertices with high degree
        # and that don't create short cycles (checked via can_add)
        
        # For efficiency, we'll do a two-phase approach:
        # Phase 1: Quick filter with can_add
        # Phase 2: Among viable edges, prioritize by degree difference (proxy for spectral gap)
        
        viable_edges = []
        for u, v in missing:
            if time.time() > deadline:
                break
            if can_add(adj_list, adj_set, u, v):
                # Score by degree difference (higher degree vertices first)
                score = abs(len(adj_list[u]) - len(adj_list[v]))
                viable_edges.append((score, u, v))
        
        if not viable_edges:
            # Stuck. Perform a kick
            if len(edges) > 0:
                # Remove 1 to 3 random edges
                kick_size = random.randint(1, 3)
                for _ in range(kick_size):
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    # Remove from structures
                    if v in adj_set[u]:
                        adj_set[u].remove(v)
                        adj_set[v].remove(u)
                        adj_list[u].remove(v)
                        adj_list[v].remove(u)
                    edges.pop(idx)
            continue
        
        # Sort by score (descending) and add edges
        viable_edges.sort(key=lambda x: -x[0])
        
        added_in_pass = 0
        for score, u, v in viable_edges:
            if time.time() > deadline:
                break
            if v in adj_set[u]:
                continue
            # Re-check can_add since graph may have changed
            if can_add(adj_list, adj_set, u, v):
                # Add edge
                adj_list[u].append(v)
                adj_list[v].append(u)
                adj_set[u].add(v)
                adj_set[v].add(u)
                edges.append([u, v])
                added_in_pass += 1
                if len(edges) > best_edges:
                    best_edges = len(edges)
                    best_adj_list = [lst[:] for lst in adj_list]
                    best_adj_set = [set(s) for s in adj_set]
        
        if added_in_pass == 0:
            # Stuck. Perform a kick
            if len(edges) > 0:
                # Remove 1 to 3 random edges
                kick_size = random.randint(1, 3)
                for _ in range(kick_size):
                    if not edges:
                        break
                    idx = random.randint(0, len(edges) - 1)
                    u, v = edges[idx]
                    # Remove from structures
                    if v in adj_set[u]:
                        adj_set[u].remove(v)
                        adj_set[v].remove(u)
                        adj_list[u].remove(v)
                        adj_list[v].remove(u)
                    edges.pop(idx)

    # Return the best graph found
    final_edges = []
    for u in range(N):
        for v in best_adj_list[u]:
            if u < v:
                final_edges.append([u, v])
    return final_edges

if __name__ == "__main__":
    e = construct()
    print(f"Edges: {len(e)}")
