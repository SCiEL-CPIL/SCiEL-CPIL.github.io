import numpy as np
import random
import time
from itertools import combinations

def construct():
    n = 16
    target_edges = 65
    
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Precompute all possible edges
    all_edges_list = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_possible_edges = len(all_edges_list)  # 120
    
    # Precompute all triplets for vectorized scoring
    triplets_list = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triplets_list.append((i, j, k))
    triplets_arr = np.array(triplets_list, dtype=np.int32)
    ti, tj, tk = triplets_arr[:, 0], triplets_arr[:, 1], triplets_arr[:, 2]
    
    def get_score(A_mat):
        """Vectorized score computation."""
        deg = A_mat.sum(axis=1).astype(np.int32)
        is_tri = A_mat[ti, tj] & A_mat[tj, tk] & A_mat[ti, tk]
        deg_sums = deg[ti] + deg[tj] + deg[tk]
        deg_sums = np.where(is_tri, deg_sums, 0)
        return int(deg_sums.max())
    
    def make_seed(a, b, seed_val):
        """Create K_{a,b} plus (target_edges - a*b) edges within the b-part."""
        rng = random.Random(seed_val)
        A = np.zeros((n, n), dtype=np.uint8)
        
        # Bipartite edges: part A = {0..a-1}, part B = {a..a+b-1}
        for u in range(a):
            for v in range(a, a + b):
                A[u, v] = 1
                A[v, u] = 1
        
        # Add edges within part B
        needed = target_edges - a * b
        b_part = list(range(a, a + b))
        internal_edges = list(combinations(b_part, 2))
        rng.shuffle(internal_edges)
        for e in internal_edges[:needed]:
            u, v = e
            A[u, v] = 1
            A[v, u] = 1
            
        return A
    
    def systematic_local_search(A_mat, max_iter=50000):
        """Systematic full-swap local search: try ALL edge swaps from current state."""
        current_A = A_mat.copy()
        current_score = get_score(current_A)
        best_A = current_A.copy()
        best_score = current_score
        
        # Track edges as sets
        current_edge_set = set()
        current_nonedge_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if current_A[i, j]:
                    current_edge_set.add((i, j))
                else:
                    current_nonedge_set.add((i, j))
        
        # For systematic search, we need to iterate through all possible swaps
        # To make this efficient, we'll do rounds where we try each edge removal
        # with a subset of non-edge additions
        
        edges_list = sorted(current_edge_set)
        nonedges_list = sorted(current_nonedge_set)
        
        # We'll do multiple passes, each trying a random subset of swaps
        # to avoid O(65*55) = 3575 evaluations per pass being too slow
        # But we'll be smarter: focus on swaps that might help
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Pick a random edge to remove
            u, v = random.choice(edges_list)
            if (u, v) not in current_edge_set:
                continue
                
            # Try several random non-edges to add
            for _ in range(5):  # Try 5 random non-edges per edge
                if time.time() > deadline:
                    break
                    
                x, y = random.choice(nonedges_list)
                if (x, y) in current_edge_set:
                    continue
                    
                # Apply move
                current_A[u, v] = 0
                current_A[v, u] = 0
                current_A[x, y] = 1
                current_A[y, x] = 1
                current_edge_set.discard((u, v))
                current_edge_set.add((x, y))
                current_nonedge_set.discard((x, y))
                current_nonedge_set.add((u, v))
                
                new_score = get_score(current_A)
                
                if new_score < current_score:
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_A = current_A.copy()
                        edges_list = sorted(current_edge_set)
                        nonedges_list = sorted(current_nonedge_set)
                    # Keep the improvement and continue from here
                    break
                else:
                    # Revert
                    current_A[u, v] = 1
                    current_A[v, u] = 1
                    current_A[x, y] = 0
                    current_A[y, x] = 0
                    current_edge_set.discard((x, y))
                    current_edge_set.add((u, v))
                    current_nonedge_set.discard((u, v))
                    current_nonedge_set.add((x, y))
        
        return best_A, best_score
    
    def simulated_annealing(A_mat, max_iter=100000):
        """Simulated annealing for edge swaps."""
        current_A = A_mat.copy()
        current_score = get_score(current_A)
        best_A = current_A.copy()
        best_score = current_score
        
        T = 2.0
        cooling = 0.9998
        
        current_edge_set = set()
        for i in range(n):
            for j in range(i + 1, n):
                if current_A[i, j]:
                    current_edge_set.add((i, j))
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
                
            # Pick random edge to remove
            u, v = random.choice(all_edges_list)
            if (u, v) not in current_edge_set:
                continue
                
            # Pick random non-edge to add
            x, y = random.choice(all_edges_list)
            if (x, y) in current_edge_set:
                continue
            if (u, v) == (x, y):
                continue
                
            # Apply move
            current_A[u, v] = 0
            current_A[v, u] = 0
            current_A[x, y] = 1
            current_A[y, x] = 1
            current_edge_set.discard((u, v))
            current_edge_set.add((x, y))
            
            new_score = get_score(current_A)
            
            # Acceptance
            if new_score < current_score or (new_score > current_score and 
                    random.random() < np.exp(-(new_score - current_score) / T)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_A = current_A.copy()
            else:
                # Revert
                current_A[u, v] = 1
                current_A[v, u] = 1
                current_A[x, y] = 0
                current_A[y, x] = 0
                current_edge_set.discard((x, y))
                current_edge_set.add((u, v))
                
            T *= cooling
            
        return best_A, best_score
    
    # Try multiple seed configurations and search strategies
    best_A_global = None
    best_score_global = float('inf')
    
    # Different bipartite splits
    splits = [(6, 10), (5, 11), (7, 9)]
    
    for a, b in splits:
        if a * b > target_edges:
            continue
        needed = target_edges - a * b
        if needed < 0 or needed > b * (b - 1) // 2:
            continue
            
        for trial in range(3):
            if time.time() > deadline:
                break
                
            seed_val = 42 + trial * 100 + a
            A_seed = make_seed(a, b, seed_val)
            
            # First do systematic local search
            A_ls, score_ls = systematic_local_search(A_seed, max_iter=20000)
            
            if score_ls < best_score_global:
                best_score_global = score_ls
                best_A_global = A_ls.copy()
            
            # Then do simulated annealing from the best found
            if time.time() < deadline - 10:
                A_sa, score_sa = simulated_annealing(A_ls, max_iter=50000)
                if score_sa < best_score_global:
                    best_score_global = score_sa
                    best_A_global = A_sa.copy()
                    
            if best_score_global <= 23:
                break
        
        if best_score_global <= 23:
            break
    
    # Final polish: run SA from the best found
    if best_A_global is not None and time.time() < deadline - 5:
        A_final, score_final = simulated_annealing(best_A_global, max_iter=50000)
        if score_final < best_score_global:
            best_score_global = score_final
            best_A_global = A_final.copy()
    
    # Reconstruct edges list from best_A
    if best_A_global is None:
        # Fallback
        best_A_global = make_seed(6, 10, 42)
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_A_global[i, j]:
                edges.append([i, j])
                
    return {'n': n, 'edges': edges}
