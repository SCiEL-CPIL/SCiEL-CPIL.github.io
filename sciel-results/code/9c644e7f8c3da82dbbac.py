import time
import random
from itertools import combinations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin for 90s limit
    
    n = 14
    m = 35
    all_edges = list(combinations(range(n), 2))
    num_all_edges = len(all_edges)
    
    best_result = None
    best_score = float('inf')
    
    # --- Phase 1: Generate diverse candidate value sets ---
    candidate_value_sets = []
    
    # Powers of 2
    candidate_value_sets.append([2**i for i in range(14)])
    
    # Small composites with overlapping factors
    candidate_value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36])
    candidate_value_sets.append([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 32])
    candidate_value_sets.append([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84])
    candidate_value_sets.append([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72])
    candidate_value_sets.append([4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30])
    candidate_value_sets.append([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    candidate_value_sets.append([3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42])
    candidate_value_sets.append([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])
    
    # Geometric-like sequences
    for ratio_num, ratio_den in [(3, 2), (4, 3), (5, 4)]:
        vals = []
        v = 1
        for i in range(n):
            vals.append(v)
            v = v * ratio_num // ratio_den + 1
            if v > 10000:
                break
        if len(vals) == n and len(set(vals)) == n and all(1 <= x <= 10000 for x in vals):
            candidate_value_sets.append(vals)
    
    # Random composites
    random.seed(42)
    for trial in range(30):
        pool = [x for x in range(1, 300) if sum(1 for d in range(1, int(x**0.5)+1) if x % d == 0) >= 4]
        if len(pool) >= n:
            vals = random.sample(pool, n)
            if len(set(vals)) == n:
                candidate_value_sets.append(sorted(vals))
    
    # --- Phase 2: Dense subgraph extraction via collision-weighted spectral method ---
    def evaluate_edge_set(selected_indices, edge_sums, edge_prods):
        """Compute max(distinct sums, distinct products) for a set of edge indices."""
        sums_set = set()
        prods_set = set()
        for idx in selected_indices:
            sums_set.add(edge_sums[idx])
            prods_set.add(edge_prods[idx])
        return max(len(sums_set), len(prods_set))
    
    def build_collision_weight_matrix(edge_sums, edge_prods):
        """
        Build a weight matrix W where W[i,j] reflects how much edges i and j
        share sum/product values. Higher weight = more collision = better to have
        both in the selected set.
        """
        W = np.zeros((num_all_edges, num_all_edges), dtype=np.float64)
        
        # For each pair of edges, compute collision weight
        for i in range(num_all_edges):
            for j in range(i+1, num_all_edges):
                w = 0.0
                # Sum collision: if edge i and j have the same sum, that's a collision
                if edge_sums[i] == edge_sums[j]:
                    w += 2.0
                # Product collision
                if edge_prods[i] == edge_prods[j]:
                    w += 2.0
                # Partial collision: same sum OR same product contributes
                if edge_sums[i] == edge_sums[j] or edge_prods[i] == edge_prods[j]:
                    w += 1.0
                
                W[i, j] = w
                W[j, i] = w
        
        return W
    
    def spectral_subgraph_extraction(W, m):
        """
        Use spectral clustering on the collision-weighted graph to find
        m edges that form a dense subgraph (high internal collision weight).
        """
        # Compute degree-weighted Laplacian or just use the weight matrix directly
        # We want to find m vertices (edges) that maximize internal weight
        # This is a densest subgraph problem, which is NP-hard
        # Approximate via: rank edges by their total collision weight with others
        # Then iteratively add the edge with highest marginal gain
        
        # Method 1: Greedy with collision-aware scoring
        selected = []
        selected_set = set()
        
        # Precompute total collision weight for each edge
        total_weight = W.sum(axis=1)
        
        for _ in range(m):
            best_idx = -1
            best_gain = -1
            
            for idx in range(num_all_edges):
                if idx in selected_set:
                    continue
                
                # Gain = collision weight with already-selected edges
                gain = sum(W[idx, s] for s in selected_set)
                
                # Small bonus for high total weight (diversity)
                gain += 0.01 * total_weight[idx]
                
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            selected_set.add(best_idx)
        
        return selected
    
    def local_search_improve(selected, edge_sums, edge_prods, W, max_iter=20000):
        """Improve the selected edge set via swap moves."""
        selected_set = set(selected)
        current_score = evaluate_edge_set(selected, edge_sums, edge_prods)
        
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            if T < T_min:
                break
            
            # Pick random edge to remove
            idx_out = random.choice(list(selected_set))
            
            # Pick random edge to add
            idx_in = random.randint(0, num_all_edges - 1)
            while idx_in in selected_set:
                idx_in = random.randint(0, num_all_edges - 1)
            
            # Compute new score
            new_selected = selected_set - {idx_out} | {idx_in}
            new_score = evaluate_edge_set(new_selected, edge_sums, edge_prods)
            
            delta = new_score - current_score
            
            accept = False
            if delta <= 0:
                accept = True
            else:
                if random.random() < pow(2.718281828, -delta / T):
                    accept = True
            
            if accept:
                selected_set = new_selected
                current_score = new_score
            
            T *= cooling
        
        return list(selected_set), current_score
    
    for values in candidate_value_sets:
        if time.time() > deadline - 5:
            break
        
        if len(values) != n or len(set(values)) != n:
            continue
        if any(v < 1 or v > 10000 for v in values):
            continue
        
        edge_sums = [values[u] + values[v] for u, v in all_edges]
        edge_prods = [values[u] * values[v] for u, v in all_edges]
        
        # Build collision weight matrix
        W = build_collision_weight_matrix(edge_sums, edge_prods)
        
        # Spectral/dense subgraph extraction
        selected = spectral_subgraph_extraction(W, m)
        
        # Local search improvement
        selected, current_score = local_search_improve(selected, edge_sums, edge_prods, W, max_iter=30000)
        
        if current_score < best_score:
            best_score = current_score
            best_result = {
                'values': list(values),
                'edges': [[all_edges[i][0], all_edges[i][1]] for i in selected]
            }
    
    if best_result is None:
        values = list(range(1, n + 1))
        best_result = {
            'values': values,
            'edges': [list(e) for e in all_edges[:m]]
        }
    
    # Ensure correct types
    best_result['values'] = [int(v) for v in best_result['values']]
    best_result['edges'] = [[int(u), int(v)] for u, v in best_result['edges']]
    
    return best_result
