"""Spectral-aware C4 minimization for 31 vertices, 91 edges."""
import numpy as np
from collections import defaultdict
import random
import time

def construct():
    n = 31
    m = 91
    
    start_time = time.time()
    timeout = 110  # leave 10s margin
    
    # Initialize with a structured graph: start with a regular-ish graph
    # then use spectral-aware local search
    
    best_edges = None
    best_c4 = float('inf')
    best_adj = None
    
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize: create a graph with approximately uniform degree
        # 91 edges on 31 vertices: average degree = 2*91/31 ≈ 5.87
        # Try to make it as regular as possible
        
        edges = set()
        # Start with a cycle (31 edges)
        for i in range(n):
            edges.add((i, (i+1) % n))
        
        # Add remaining edges randomly but avoid creating too many C4s
        while len(edges) < m:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u != v:
                edge = (min(u,v), max(u,v))
                if edge not in edges:
                    edges.add(edge)
        
        # Build adjacency matrix
        A = np.zeros((n, n), dtype=np.int32)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        
        # Compute initial C4 count
        # C4 count = (1/8) * sum of (A^2)^2 diagonal entries? 
        # Actually: number of C4s = (1/8) * trace((A^2)^2) - (1/2) * sum of degrees
        # Or more simply: for each pair of vertices, count common neighbors
        # C4 = sum_{u<v} C(common_neighbors(u,v), 2)
        
        A2 = A @ A
        # A2[i,j] = number of common neighbors of i and j
        c4_count = 0
        for i in range(n):
            for j in range(i+1, n):
                c = A2[i, j]
                if c >= 2:
                    c4_count += c * (c - 1) // 2
        
        # Spectral analysis: compute eigenvalues
        eigenvalues = np.linalg.eigvalsh(A)
        eigenvalues = np.sort(eigenvalues)[::-1]
        
        # The sum of squares of eigenvalues relates to C4 count
        # trace(A^4) = sum lambda_i^4
        # trace(A^4) counts closed walks of length 4
        # Each C4 contributes 8 closed walks of length 4 (4 starting vertices × 2 directions)
        # Plus triangles and other structures
        
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = set(edges)
            best_adj = A.copy()
        
        # Spectral-aware local search
        improved = True
        iteration = 0
        while improved and (time.time() - start_time) < timeout:
            improved = False
            iteration += 1
            
            # Recompute A2 for efficiency
            A2 = A @ A
            
            # Try all edge swaps: remove (a,b), add (c,d)
            # Use spectral guidance: prefer swaps that increase spectral gap
            # or reduce the sum of squared eigenvalues (which correlates with C4)
            
            edge_list = list(edges)
            random.shuffle(edge_list)
            
            # Precompute non-edges for quick lookup
            non_edges = set()
            for i in range(n):
                for j in range(i+1, n):
                    if (i, j) not in edges:
                        non_edges.add((i, j))
            
            for (a, b) in edge_list:
                if (time.time() - start_time) > timeout:
                    break
                if not non_edges:
                    break
                
                # Compute delta C4 for removing (a,b)
                # Removing edge (a,b) changes A2 for pairs involving a or b
                # For pair (x,y): A2[x,y] changes if x or y is a or b and the other is adjacent to the removed vertex
                # Specifically:
                # - For x in N(b)\{a}: A2[b,x] decreases by 1 (a was common neighbor)
                # - For y in N(a)\{b}: A2[a,y] decreases by 1 (b was common neighbor)
                
                delta_c4_rem = 0
                # For each x in N(b)\{a}: C(A2[b,x],2) -> C(A2[b,x]-1,2), change = -(A2[b,x]-1)
                for x in range(n):
                    if x != a and A[b, x] == 1:
                        old_c = A2[b, x]
                        if old_c >= 1:
                            delta_c4_rem -= (old_c - 1)
                # For each y in N(a)\{b}: C(A2[a,y],2) -> C(A2[a,y]-1,2), change = -(A2[a,y]-1)
                for y in range(n):
                    if y != b and A[a, y] == 1:
                        old_c = A2[a, y]
                        if old_c >= 1:
                            delta_c4_rem -= (old_c - 1)
                
                # Try random non-edges for adding
                found_swap = False
                # Sample up to 50 non-edges
                non_edge_list = list(non_edges)
                random.shuffle(non_edge_list)
                
                for (c, d) in non_edge_list[:50]:
                    if c == a and d == b:
                        continue
                    if c == b and d == a:
                        continue
                    
                    # Compute delta C4 for adding (c,d)
                    delta_c4_add = 0
                    # For each x in N(d)\{c}: A2[c,x] increases by 1, change = A2[c,x]
                    for x in range(n):
                        if x != c and A[d, x] == 1:
                            delta_c4_add += A2[c, x]
                    # For each y in N(c)\{d}: A2[d,y] increases by 1, change = A2[d,y]
                    for y in range(n):
                        if y != d and A[c, y] == 1:
                            delta_c4_add += A2[d, y]
                    
                    total_delta = delta_c4_rem + delta_c4_add
                    
                    if total_delta < 0:
                        # Perform the swap
                        edges.remove((a, b))
                        edges.add((c, d))
                        non_edges.remove((c, d))
                        non_edges.add((a, b))
                        
                        # Update A
                        A[a, b] = 0
                        A[b, a] = 0
                        A[c, d] = 1
                        A[d, c] = 1
                        
                        c4_count += total_delta
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    continue
            
            # If no improvement found in this pass, try a different strategy
            if not improved and (time.time() - start_time) < timeout:
                # Try to find a single edge addition/removal pair that helps
                # Or restart with a different initialization
                pass
        
        if c4_count < best_c4:
            best_c4 = c4_count
            best_edges = set(edges)
            best_adj = A.copy()
        
        if (time.time() - start_time) > timeout:
            break
    
    # Convert best_edges to list of lists
    result = [[u, v] for u, v in sorted(best_edges)]
    return result
