import numpy as np
import time
from itertools import combinations, product
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    D = 11
    indices = list(range(D))
    
    # Generate all weight-3 supports (combinations of 3 indices)
    supports = list(combinations(indices, 3))
    n_supports = len(supports)
    
    # Build conflict graph: two supports conflict if they share 2+ indices
    # For weight-3 vectors, intersection size >= 2 means they share at least 2 positions
    # We want maximum independent set (no conflicts)
    
    # Precompute intersection sizes between all pairs of supports
    # Two supports (a,b,c) and (d,e,f) conflict if |intersection| >= 2
    
    conflict = np.zeros((n_supports, n_supports), dtype=bool)
    
    for i in range(n_supports):
        for j in range(i+1, n_supports):
            inter = set(supports[i]) & set(supports[j])
            if len(inter) >= 2:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Now we need to find maximum independent set in this conflict graph
    # Use spectral initialization + local search (flip heuristics)
    
    # Build adjacency matrix for the conflict graph (1 = conflict)
    A = conflict.astype(np.float64)
    
    # Spectral initialization: use eigenvector of Laplacian to get initial solution
    # L = D - A where D is degree matrix
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    
    # Get smallest eigenvector (corresponds to smallest eigenvalue, which is 0 for connected graph)
    # Actually, for maximum independent set, we want to minimize conflicts
    # Let's use the Fiedler vector (second smallest eigenvector) or just random init
    
    # Simpler: use random initialization with local search
    # State: x[i] = 1 if support i is selected, 0 otherwise
    # Objective: maximize sum(x) - penalty for conflicts
    
    best_x = None
    best_score = -1
    
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Random initialization: select supports with probability p
        # Start with a greedy seed
        x = np.zeros(n_supports, dtype=np.int8)
        
        # Greedy initialization: pick supports in random order, add if no conflict with already selected
        order = np.random.permutation(n_supports)
        selected_set = set()
        for i in order:
            # Check if i conflicts with any selected support
            conflicts_with_selected = conflict[i, list(selected_set)]
            if not conflicts_with_selected.any():
                x[i] = 1
                selected_set.add(i)
        
        current_score = x.sum()
        
        # Local search: flip heuristics
        # For each support, consider flipping (add or remove)
        # Gain from adding i: 1 - (number of conflicts with selected)
        # Gain from removing i: -1 + (number of conflicts with selected)
        
        improved = True
        max_iterations = 1000
        iteration = 0
        
        while improved and iteration < max_iterations:
            if time.time() > deadline - 5:
                break
            
            improved = False
            iteration += 1
            
            # Compute gain for each possible flip
            # For unselected i: gain = 1 - conflict[i, selected].sum()
            # For selected i: gain = -1 + conflict[i, selected_without_i].sum()
            
            # Vectorized computation
            # selected_mask = x == 1
            # For each i, count conflicts with currently selected (excluding i itself)
            
            # conflict_with_selected[i] = sum of conflict[i, j] for j where x[j]=1 and j != i
            conflict_with_selected = (A @ x).astype(np.int32)
            # Subtract self-conflict (which is 0 since no self-loops)
            # Actually A[i,i] = 0, so this is fine
            
            # For unselected i: gain = 1 - conflict_with_selected[i]
            # For selected i: gain = -1 + conflict_with_selected[i] - 1 (because i is in selected, so conflict_with_selected[i] includes conflicts with other selected, not with itself)
            # Wait: conflict_with_selected[i] = sum_j A[i,j]*x[j]. If x[i]=1, this doesn't include i itself (A[i,i]=0).
            # So for selected i: removing i changes score by -1 (lose i) + (conflicts removed)
            # Conflicts removed = number of selected j that conflict with i = conflict_with_selected[i]
            # So gain from removing i = -1 + conflict_with_selected[i]
            
            # For unselected i: adding i changes score by +1 (gain i) - (conflicts added)
            # Conflicts added = number of selected j that conflict with i = conflict_with_selected[i]
            # So gain from adding i = 1 - conflict_with_selected[i]
            
            gain_add = 1 - conflict_with_selected
            gain_remove = -1 + conflict_with_selected
            
            # Best flip: max of gain_add (for unselected) and gain_remove (for selected)
            # Only consider flips with positive gain
            
            # Find best candidate
            best_gain = -1
            best_i = -1
            
            # For unselected: only consider gain_add
            unselected_mask = (x == 0)
            # For selected: only consider gain_remove
            selected_mask = (x == 1)
            
            # Combine: create a single gain array where invalid flips have -inf
            gain = np.full(n_supports, -np.inf)
            gain[unselected_mask] = gain_add[unselected_mask]
            gain[selected_mask] = gain_remove[selected_mask]
            
            # Find max gain
            if gain.max() > 0:
                best_i = int(np.argmax(gain))
                best_gain = gain[best_i]
                
                # Apply flip
                if x[best_i] == 0:
                    x[best_i] = 1
                else:
                    x[best_i] = 0
                
                current_score = x.sum()
                improved = True
        
        # Update best
        if current_score > best_score:
            best_score = current_score
            best_x = x.copy()
        
        # If we've found a large enough set, we can stop early
        if best_score >= 150:
            break
    
    # If local search didn't work well, try a different approach:
    # Use simulated annealing or tabu search
    
    if best_score < 100:
        # Try tabu search
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            x = np.zeros(n_supports, dtype=np.int8)
            selected_set = set()
            order = np.random.permutation(n_supports)
            for i in order:
                conflicts_with_selected = conflict[i, list(selected_set)]
                if not conflicts_with_selected.any():
                    x[i] = 1
                    selected_set.add(i)
            
            current_score = x.sum()
            best_local_score = current_score
            best_local_x = x.copy()
            
            # Tabu list
            tabu_list = {}
            tabu_tenure = 20
            max_iterations = 500
            
            for iteration in range(max_iterations):
                if time.time() > deadline - 5:
                    break
                
                # Compute gains
                conflict_with_selected = (A @ x).astype(np.int32)
                gain_add = 1 - conflict_with_selected
                gain_remove = -1 + conflict_with_selected
                
                unselected_mask = (x == 0)
                selected_mask = (x == 1)
                
                gain = np.full(n_supports, -np.inf)
                gain[unselected_mask] = gain_add[unselected_mask]
                gain[selected_mask] = gain_remove[selected_mask]
                
                # Apply tabu penalty
                for i, tabu_until in list(tabu_list.items()):
                    if iteration < tabu_until:
                        gain[i] -= 100  # Large penalty for tabu moves
                
                # Find best move
                best_i = int(np.argmax(gain))
                best_gain = gain[best_i]
                
                if best_gain <= -100 and iteration > 10:
                    # All moves are tabu or bad, break
                    break
                
                # Apply move
                if x[best_i] == 0:
                    x[best_i] = 1
                else:
                    x[best_i] = 0
                
                # Add to tabu list
                tabu_list[best_i] = iteration + tabu_tenure
                # Remove expired tabu entries
                tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
                
                current_score = x.sum()
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_x = x.copy()
            
            if best_local_score > best_score:
                best_score = best_local_score
                best_x = best_local_x
    
    # Extract selected supports
    selected_indices = np.where(best_x == 1)[0]
    
    if len(selected_indices) == 0:
        # Fallback: return a small valid set
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
        return np.array(vectors, dtype=np.int8)
    
    # For each selected support, generate all 8 sign combinations
    # But we need to ensure that the full vectors (with signs) satisfy the distance constraint
    # For weight-3 vectors, two vectors with the same support but different signs may conflict
    # Let's just pick one sign pattern per support for simplicity, or try all and keep compatible ones
    
    # Actually, let's be more careful: for each support, we can pick any of the 8 sign patterns
    # But two vectors with the same support will have dot product = sum of products of signs
    # If they have the same support, dot = +/-1 +/-1 +/-1, which can be -3,-1,1,3
    # For weight-3 vs weight-3, we need dot <= 1
    # So if two vectors have the same support, they must have dot <= 1, which means they can't be identical or negatives
    
    # Simplest: for each selected support, pick the all-+1 sign pattern
    # This gives dot product between any two vectors with different supports = intersection_size (since all +1)
    # We already ensured intersection <= 1, so dot <= 1. Good.
    
    vectors = []
    for idx in selected_indices:
        support = supports[idx]
        v = np.zeros(D, dtype=np.int8)
        for pos in support:
            v[pos] = 1
        vectors.append(v)
    
    if not vectors:
        # Fallback
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                v = np.zeros(D, dtype=np.int8)
                v[i] = 1
                v[j] = 1
                vectors.append(v)
    
    return np.array(vectors, dtype=np.int8)
