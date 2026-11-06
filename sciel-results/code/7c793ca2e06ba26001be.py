import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    def get_idx(i, j, n):
        return i * (2 * n - i - 1) // 2 + (j - i - 1)

    def build_adj_from_colors(colors, n):
        """Build adjacency matrix (n x n) for each color from flat color list."""
        adj = np.zeros((5, n, n), dtype=np.int8)
        for i in range(n):
            for j in range(i + 1, n):
                idx = get_idx(i, j, n)
                c = colors[idx]
                adj[c, i, j] = 1
                adj[c, j, i] = 1
        return adj

    def count_violations(adj, n):
        """Count C4 violations: for each color, sum over pairs (u,w) of max(0, common_neighbors - 1)."""
        total = 0
        for c in range(5):
            A = adj[c]
            # For each pair (u,w), common neighbors = sum of A[u,:] * A[w,:]
            # We need max(0, common - 1) for all pairs u < w
            common = A @ A.T  # n x n matrix, common[i,j] = |N_i ∩ N_j|
            # Only count upper triangle (i < j), exclude diagonal
            np.fill_diagonal(common, 0)
            # For i < j: violation contribution is max(0, common[i,j] - 1)
            upper = np.triu(common, k=1)
            total += int(np.sum(np.maximum(upper - 1, 0)))
        return total

    def solve_vectorized(n, max_iters, start_temp, cooling):
        """Vectorized simulated annealing using NumPy."""
        num_edges = n * (n - 1) // 2
        colors = np.array([random.randint(0, 4) for _ in range(num_edges)], dtype=np.int8)
        
        # Precompute edge endpoints
        edge_i = np.array([i for i in range(n) for j in range(i+1, n)], dtype=np.int32)
        edge_j = np.array([j for i in range(n) for j in range(i+1, n)], dtype=np.int32)
        
        adj = build_adj_from_colors(colors, n)
        current_violations = count_violations(adj, n)
        
        if current_violations == 0:
            return True, colors.tolist()

        best_violations = current_violations
        best_colors = colors.copy()
        temp = start_temp

        it = 0
        while it < max_iters:
            if time.time() > deadline - 3.0:
                break
                
            if current_violations == 0:
                return True, colors.tolist()

            # Pick random edge
            idx = random.randint(0, num_edges - 1)
            i = int(edge_i[idx])
            j = int(edge_j[idx])
            
            old_color = int(colors[idx])
            new_color = random.randint(0, 4)
            if new_color == old_color:
                it += 1
                continue

            # Compute delta efficiently using adjacency matrices
            # When we change edge (i,j) from old_color to new_color:
            # - For old_color: remove i-j edge
            # - For new_color: add i-j edge
            
            delta = 0
            
            # Effect on old_color: removing edge (i,j)
            # For each vertex w != i,j:
            #   common(i,w) in old_color decreases by 1 if j was neighbor of w in old_color
            #   common(j,w) in old_color decreases by 1 if i was neighbor of w in old_color
            A_old = adj[old_color]
            # Neighbors of i in old_color (excluding j): mask_i
            # Neighbors of j in old_color (excluding i): mask_j
            # For pair (i,w): old_common = A_old[i,w] (which is 1 if w is neighbor of i)
            #   Actually, common neighbors of i and w = sum of A_old[i,:] * A_old[w,:]
            # This is getting complex. Let's use a simpler but still vectorized approach.
            
            # Alternative: compute the change in violations more directly
            # Violation for pair (u,v) in color c = max(0, |N_u^c ∩ N_v^c| - 1)
            # When we add edge (i,j) to color c:
            #   For pair (i,w): common neighbors gain j if w~j in c, so common increases by 1 if A[c,w,j]=1
            #   For pair (j,w): common neighbors gain i if w~i in c, so common increases by 1 if A[c,w,i]=1
            #   For pair (u,v) where u,v != i,j: no change to their common neighbors
            #   But pairs (i,j) itself: they now share no common neighbors initially... 
            #   Actually (i,j) as a pair: their common neighbors are those w where A[i,w]=A[j,w]=1
            
            # Let me use a cleaner incremental approach:
            # Removing edge (i,j) from old_color:
            #   For each w: 
            #     pair (i,w): common_{i,w} decreases by 1 iff A_old[w,j] == 1
            #     pair (j,w): common_{j,w} decreases by 1 iff A_old[w,i] == 1
            # Adding edge (i,j) to new_color:
            #   For each w:
            #     pair (i,w): common_{i,w} increases by 1 iff A_new[w,j] == 1
            #     pair (j,w): common_{j,w} increases by 1 iff A_new[w,i] == 1
            
            A_old = adj[old_color]
            A_new = adj[new_color]
            
            # Vectors: which w have j as neighbor in old_color?
            # A_old[:, j] gives for each w whether w~j in old_color
            # But we exclude w=i and w=j
            mask_old_j = A_old[:, j].copy()  # w where w~j
            mask_old_i = A_old[:, i].copy()  # w where w~i
            mask_new_j = A_new[:, j].copy()
            mask_new_i = A_new[:, i].copy()
            
            # Exclude i and j themselves
            mask_old_j[i] = 0; mask_old_j[j] = 0
            mask_old_i[i] = 0; mask_old_i[j] = 0
            mask_new_j[i] = 0; mask_new_j[j] = 0
            mask_new_i[i] = 0; mask_new_i[j] = 0
            
            # For removal from old_color:
            # Pair (i,w): old_common = A_old[i,:] @ A_old[w,:].bit_count()
            #   new_common = old_common - mask_old_j[w] (decrease by 1 if w~j)
            #   delta_violation = max(0, new_common-1) - max(0, old_common-1)
            #   = -1 if old_common >= 2 and new_common >= 1 (i.e., old_common >= 2)
            #   = +1 if old_common == 1 and new_common == 0 (i.e., old_common == 1 and mask_old_j[w]==1)
            #   More precisely: if old_common >= 2: delta = -1 (since new_common = old_common-1 >= 1, still violation if old_common-1 >= 2, i.e., old_common >= 3: delta=0; if old_common==2: new_common=1, delta=-1)
            #   Let me just compute it:
            #   old_c = common count, new_c = old_c - mask
            #   delta = max(0,new_c-1) - max(0,old_c-1)
            #   If old_c >= 2: max(0,old_c-1) = old_c-1
            #     If new_c >= 2: delta = (new_c-1)-(old_c-1) = new_c-old_c = -mask
            #     If new_c < 2: delta = 0-(old_c-1) = -(old_c-1)
            #   If old_c < 2: max(0,old_c-1)=0
            #     If new_c >= 2: impossible since new_c <= old_c
            #     If new_c < 2: delta = 0
            
            # This is getting complicated. Let me just recompute violations for affected pairs.
            
            # Affected pairs when changing edge (i,j):
            # Pairs involving i or j: (i,w) and (j,w) for all w != i,j
            # That's 2*(n-2) pairs per color, times 2 colors = 4*(n-2) pairs
            # For n=26, that's 96 pairs. Much better than all ~325 pairs.
            
            # For old_color, removing (i,j):
            for w in range(n):
                if w == i or w == j:
                    continue
                # Pair (i, w)
                old_common = int(np.sum(A_old[i, :] * A_old[w, :]))
                # After removal, j is no longer a neighbor of i in old_color
                # So common decreases by 1 if j was a common neighbor, i.e., if A_old[w,j]==1
                new_common = old_common - (1 if A_old[w, j] == 1 else 0)
                delta += max(0, new_common - 1) - max(0, old_common - 1)
                
                # Pair (j, w)
                old_common = int(np.sum(A_old[j, :] * A_old[w, :]))
                new_common = old_common - (1 if A_old[w, i] == 1 else 0)
                delta += max(0, new_common - 1) - max(0, old_common - 1)
            
            # For new_color, adding (i,j):
            for w in range(n):
                if w == i or w == j:
                    continue
                # Pair (i, w)
                old_common = int(np.sum(A_new[i, :] * A_new[w, :]))
                new_common = old_common + (1 if A_new[w, j] == 1 else 0)
                delta += max(0, new_common - 1) - max(0, old_common - 1)
                
                # Pair (j, w)
                old_common = int(np.sum(A_new[j, :] * A_new[w, :]))
                new_common = old_common + (1 if A_new[w, i] == 1 else 0)
                delta += max(0, new_common - 1) - max(0, old_common - 1)
            
            new_violations = current_violations + delta
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                # Accept
                colors[idx] = new_color
                adj[old_color, i, j] = 0
                adj[old_color, j, i] = 0
                adj[new_color, i, j] = 1
                adj[new_color, j, i] = 1
                current_violations = new_violations
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors.copy()
            else:
                # Reject (no state change needed since we didn't modify adj yet)
                pass
            
            temp *= cooling
            it += 1

        if current_violations == 0:
            return True, colors.tolist()
        elif best_violations == 0:
            return True, best_colors.tolist()
        else:
            return False, best_colors.tolist() if best_violations < 50 else None

    # Try to find valid coloring for increasing n
    best_n = 11
    best_colors = [0] * (11 * 10 // 2)
    
    # Try n=26 first with multiple restarts
    for target_n in [26, 24, 22, 20]:
        if time.time() > deadline - 15.0:
            break
        
        success = False
        for restart in range(15):
            if time.time() > deadline - 5.0:
                break
            
            max_iters = 50000
            start_temp = 2.0
            cooling = 0.99995
            
            res_success, res_colors = solve_vectorized(target_n, max_iters, start_temp, cooling)
            
            if res_success:
                best_n = target_n
                best_colors = res_colors
                success = True
                break
        
        if success:
            continue
        else:
            if target_n <= 22:
                break

    return {'n': best_n, 'colors': best_colors}
