import time
import random
import itertools
import numpy as np

def _has_k222_count_np(A, n):
    """
    Counts the number of K2,2,2 subgraphs in the graph defined by adjacency matrix A.
    A is an n x n numpy array of 0s and 1s.
    """
    count = 0
    # Iterate over pairs (c, d) which form one part of the K2,2,2
    # c < d
    for c in range(n - 1):
        for d in range(c + 1, n):
            if A[c, d] == 0:
                continue
            
            # Common neighbors of c and d
            # These are vertices v such that A[c,v]=1 and A[d,v]=1
            common = A[:, c] & A[:, d]
            # Exclude c and d themselves from common if they are connected to each other?
            # In a simple graph, A[c,c]=0, A[d,d]=0.
            # common[c] is A[c,c] & A[d,c] = 0 & 1 = 0.
            # common[d] is A[c,d] & A[d,d] = 1 & 0 = 0.
            # So c and d are automatically excluded from common.
            
            # We need to find a K2,2 in the subgraph induced by 'common' vertices.
            # The vertices of the K2,2 are {a, b, e, f} all in common.
            # Edges required: a-e, a-f, b-e, b-f.
            # Note that a and b are in common, so A[c,a]=1, A[d,a]=1, etc.
            # The edges a-e, etc. must exist in the original graph.
            
            # Let M be the submatrix of A induced by the vertices in common.
            # We want to count the number of K2,2 subgraphs in M.
            # A K2,2 is a 4-cycle C4.
            # Number of C4s in M can be computed using matrix multiplication?
            # Or just iterate over pairs (a, b) in common.
            
            # For each pair (a, b) in common:
            #   neighbors_in_common = (M[:, a] & M[:, b])
            #   if count(neighbors_in_common) >= 2, then we have a K2,2.
            #   Specifically, if k neighbors, we have C(k, 2) K2,2s involving (a,b) and those neighbors.
            
            # To be efficient, we can precompute the list of common vertices.
            common_verts = np.where(common)[0]
            m_size = len(common_verts)
            
            if m_size < 4:
                continue
                
            # Extract submatrix
            # This is a bit expensive if done naively, but n is small (29).
            # m_size is at most 27.
            # We can avoid explicit submatrix extraction by indexing A directly.
            
            for i in range(m_size - 1):
                a = common_verts[i]
                for j in range(i + 1, m_size):
                    b = common_verts[j]
                    # Check if a and b are connected in the original graph?
                    # No, a and b are just vertices in the "middle" parts.
                    # The K2,2 structure is:
                    # Part 1: {c, d}
                    # Part 2: {a, b}
                    # Part 3: {e, f}
                    # Edges:
                    # c-a, c-b, c-e, c-f
                    # d-a, d-b, d-e, d-f
                    # a-e, a-f, b-e, b-f
                    # Note: No edges required between c,d (they are connected),
                    #       between a,b (not required),
                    #       between e,f (not required).
                    
                    # So we need a, b in common (connected to c, d).
                    # And e, f in common (connected to c, d).
                    # And a, b, e, f must form a K2,2 in the original graph.
                    # i.e., a-e, a-f, b-e, b-f must be edges.
                    
                    # This is equivalent to:
                    # In the subgraph induced by common, do a and b have at least 2 common neighbors?
                    # Let common_neighbors_of_ab = (A[:, a] & A[:, b]) & common
                    # If count(common_neighbors_of_ab) >= 2, then yes.
                    
                    # Optimization:
                    # We can compute A[:, a] & A[:, b] once? No, a and b vary.
                    
                    # Let's try to vectorize the inner loop slightly.
                    # For a fixed a, we can compute common_neighbors_of_ab for all b > a.
                    # But simple loops might be faster in Python for small n due to overhead.
                    
                    # Let's stick to loops for now, but use bitsets if possible?
                    # Or just numpy operations on small arrays.
                    
                    # A[:, a] is a 1D array of size n.
                    # A[:, b] is a 1D array of size n.
                    # & is vectorized.
                    # & common is vectorized.
                    # sum() is vectorized.
                    
                    # This is O(n) per pair (a,b).
                    # Total O(m_size^2 * n).
                    # m_size <= 27, n=29.
                    # 27^2 * 29 ~ 20k operations per (c,d) pair.
                    # Number of (c,d) pairs is ~400.
                    # Total ~8 million operations.
                    # In NumPy, this should be fast enough (milliseconds).
                    
                    neighbors = (A[:, a] & A[:, b]) & common
                    # Count bits
                    # np.sum(neighbors)
                    # If >= 2, we have a violation.
                    # We can just check if sum >= 2.
                    # To be even faster, we can check if there are at least 2 ones.
                    # np.any(neighbors) is not enough, we need at least 2.
                    # We can use np.count_nonzero(neighbors)
                    
                    # Actually, we just need to know if count >= 2.
                    # We can find the first two indices?
                    # Or just sum.
                    
                    if np.sum(neighbors) >= 2:
                        count += 1
                        
    return count

def _is_valid_fast_np(A, n):
    return _has_k222_count_np(A, n) == 0

def _build_adj_np(colors, n):
    A = np.zeros((n, n), dtype=np.int8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            if colors[idx] == 1:
                A[i, j] = 1
                A[j, i] = 1
            idx += 1
    return A

def _get_color_list_np(A, n):
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(int(A[i, j]))
    return colors

def _local_search_np(colors, n, max_iters=10000, time_limit=50):
    start_time = time.time()
    A = _build_adj_np(colors, n)
    current_cost = _has_k222_count_np(A, n)
    
    if current_cost == 0:
        return colors, True
        
    # Precompute edge indices mapping for faster flip?
    # Or just iterate idx and update A[i,j] and A[j,i].
    # We need to map idx to (i,j).
    # idx = i*(2n-i-1)//2 + (j-i-1)
    # Inverse:
    # k = i
    # while True:
    #   next_idx = (k+1)*(2n-k-2)//2
    #   if idx < next_idx:
    #      i = k
    #      break
    #   k += 1
    # j = i + 1 + (idx - i*(2n-i-1)//2)
    
    # This is O(n) per step.
    # Since we are doing O(n^4) cost check, O(n) index finding is negligible.
    
    for _ in range(max_iters):
        if time.time() - start_time > time_limit:
            break
            
        # Pick a random edge to flip
        idx = random.randint(0, len(colors) - 1)
        
        # Find i, j
        i = 0
        base = 0
        while True:
            next_base = (i + 1) * (2 * n - i - 2) // 2
            if idx < next_base:
                break
            i += 1
            base = next_base
        j = i + 1 + (idx - base)
        
        # Flip edge (i, j)
        old_val = A[i, j]
        new_val = 1 - old_val
        A[i, j] = new_val
        A[j, i] = new_val
        colors[idx] = new_val
        
        new_cost = _has_k222_count_np(A, n)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if current_cost == 0:
                return colors, True
        else:
            # Revert
            A[i, j] = old_val
            A[j, i] = old_val
            colors[idx] = old_val

    return colors, False

def _paley(n, p):
    """Paley coloring using prime p >= n."""
    if p < n:
        return None
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append(0 if (j - i) % p in squares else 1)
    return colors

def construct():
    deadline = time.time() + 140
    best_n = 0
    best_colors = None
    
    # Target n=29
    target_n = 29
    p = 29 # 29 is prime
    
    # 1. Try Paley(29)
    colors = _paley(target_n, p)
    if colors:
        A = _build_adj_np(colors, target_n)
        if _is_valid_fast_np(A, target_n):
            return {'n': target_n, 'colors': colors}
        
    # 2. Try Local Search from Paley
    colors_copy = colors[:]
    _, success = _local_search_np(colors_copy, target_n, max_iters=5000, time_limit=60)
    if success:
        return {'n': target_n, 'colors': colors_copy}
        
    # 3. Random Restarts
    for attempt in range(50):
        if time.time() > deadline:
            break
        colors = [random.randint(0, 1) for _ in range(target_n * (target_n - 1) // 2)]
        _, success = _local_search_np(colors, target_n, max_iters=2000, time_limit=10)
        if success:
            return {'n': target_n, 'colors': colors}
            
    # Fallback: Return best found or smaller valid n
    # If we couldn't find 29, try 28, etc.
    for n in range(28, 10, -1):
        if time.time() > deadline:
            break
        p_candidate = None
        for pp in [17, 19, 23, 29, 31]:
            if pp >= n:
                p_candidate = pp
                break
        if not p_candidate:
            continue
        colors = _paley(n, p_candidate)
        if colors:
            A = _build_adj_np(colors, n)
            if _is_valid_fast_np(A, n):
                return {'n': n, 'colors': colors}
                
    # Absolute fallback
    return {'n': 1, 'colors': [0]}
