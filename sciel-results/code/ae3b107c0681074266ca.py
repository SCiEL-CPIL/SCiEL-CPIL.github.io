"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    # 1. Generate Candidate Vectors
    # We focus on vectors with squared norms 1, 2, 3, 4.
    # The condition for two vectors u, v to be compatible is:
    # ||u - v||^2 >= max(||u||^2, ||v||^2)
    # ||u||^2 + ||v||^2 - 2<u,v> >= max(||u||^2, ||v||^2)
    # If ||u||^2 <= ||v||^2:
    # ||u||^2 + ||v||^2 - 2<u,v> >= ||v||^2  =>  ||u||^2 >= 2<u,v>
    # So we need 2<u,v> <= min(||u||^2, ||v||^2).
    
    candidates = []
    
    # Norm 1 (squared norm 1): 22 vectors (e_i, -e_i)
    for i in range(D):
        v = np.zeros(D, dtype=np.int64); v[i] = 1; candidates.append(v)
        v[i] = -1; candidates.append(v)
        
    # Norm 2 (squared norm 2): 220 vectors (e_i +/- e_j)
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si; v[j] = sj
                    candidates.append(v)
                    
    # Norm 3 (squared norm 3): 1980 vectors (e_i +/- e_j +/- e_k)
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for si in [1, -1]:
                    for sj in [1, -1]:
                        for sk in [1, -1]:
                            v = np.zeros(D, dtype=np.int64)
                            v[i] = si; v[j] = sj; v[k] = sk
                            candidates.append(v)
                            
    # Norm 4 (squared norm 4):
    # Type 1: e_i +/- e_j +/- e_k +/- e_l (4 non-zero entries of 1/-1)
    # C(11,4) * 16 = 330 * 16 = 5280
    for i in range(D):
        for j in range(i+1, D):
            for k in range(j+1, D):
                for l in range(k+1, D):
                    for si in [1, -1]:
                        for sj in [1, -1]:
                            for sk in [1, -1]:
                                for sl in [1, -1]:
                                    v = np.zeros(D, dtype=np.int64)
                                    v[i] = si; v[j] = sj; v[k] = sk; v[l] = sl
                                    candidates.append(v)
                                    
    # Type 2: 2*e_i (1 non-zero entry of 2/-2)
    # 11 * 2 = 22
    for i in range(D):
        for si in [1, -1]:
            v = np.zeros(D, dtype=np.int64)
            v[i] = si * 2
            candidates.append(v)
            
    # Type 3: 2*e_i +/- 2*e_j (2 non-zero entries of 2/-2)
    # C(11,2) * 4 = 55 * 4 = 220
    for i in range(D):
        for j in range(i+1, D):
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = si * 2; v[j] = sj * 2
                    candidates.append(v)

    candidates = np.array(candidates)
    n_cand = len(candidates)
    norms_sq = np.sum(candidates * candidates, axis=1)
    
    # 2. Build Conflict Graph
    # Conflict[i, j] = True if vectors i and j CANNOT be in the same set.
    # Condition for conflict: 2 <u, v> > min(||u||^2, ||v||^2)
    
    # To save memory and time, we can compute the conflict matrix in blocks or use bitsets.
    # n_cand is roughly 22 + 220 + 1980 + 5522 = 7744.
    # A 7744 x 7744 boolean matrix is ~60 MB. This is acceptable.
    
    # Compute dot products
    # dots = candidates @ candidates.T
    # This is 7744 x 7744 matrix of ints.
    # 7744^2 * 8 bytes = ~480 MB. This might be tight but usually OK.
    # Let's do it in chunks to be safe if memory is an issue, but 480MB is usually fine on modern systems.
    # However, creating the full matrix might take a few seconds.
    
    # Alternative: Iterate and build adjacency list?
    # For Branch and Bound, we need fast access to neighbors.
    # Let's build the conflict matrix as a boolean array.
    
    # Optimization: We only need to check i < j.
    # But for B&B, having the full matrix or a fast neighbor lookup is key.
    
    # Let's compute the conflict matrix.
    # To avoid 480MB peak if possible, we can compute row by row and store as bitsets or just a list of lists of ints.
    # Given n=7744, a list of sets of neighbors is also good.
    
    conflict_graph = [set() for _ in range(n_cand)]
    
    # Vectorized conflict detection in blocks to manage memory
    block_size = 1000
    for start in range(0, n_cand, block_size):
        end = min(start + block_size, n_cand)
        block_vecs = candidates[start:end]
        block_norms = norms_sq[start:end]
        
        # Compute dots for this block against all candidates
        # (block_size, D) @ (D, n_cand) -> (block_size, n_cand)
        dots = block_vecs @ candidates.T
        
        # Limits: min(block_norms[i], norms_sq[j])
        # We need 2 * dots[i, j] > limits[i, j]
        
        # Vectorized comparison
        # limits = np.minimum(block_norms[:, None], norms_sq[None, :])
        # conflicts = 2 * dots > limits
        
        # This creates a (block_size, n_cand) boolean matrix.
        # 1000 * 7744 * 1 byte = 7.7 MB. Very fast.
        
        limits = np.minimum(block_norms[:, None], norms_sq[None, :])
        conflicts = 2 * dots > limits
        
        # For each row i in the block, add conflicts to the graph
        for i in range(end - start):
            global_idx = start + i
            # Get indices where conflict is True
            # Exclude self (though 2*||u||^2 > ||u||^2 is true, so self is conflict, but we don't add self to neighbor set usually, or we handle it in B&B)
            # In MIS, we don't select a node and its neighbors. Self-conflict is irrelevant for selection logic if we assume distinct nodes.
            # The condition for u, u is 2||u||^2 > ||u||^2 which is true. So u conflicts with itself.
            # But we select a set of DISTINCT vectors. So we just need to ensure we don't pick two different vectors that conflict.
            
            neighbor_indices = np.where(conflicts[i])[0]
            # Filter out self?
            # If we pick u, we cannot pick u again. The set logic handles distinctness.
            # The graph edge (u, v) means u and v cannot both be in the set.
            for nb in neighbor_indices:
                if nb != global_idx:
                    conflict_graph[global_idx].add(nb)
                    # Since we process all pairs, we will add v to u and u to v.
                    # To save time, we can just add to one and rely on symmetry, but sets are fine.
                    # Actually, the loop above adds nb to global_idx.
                    # It does NOT add global_idx to nb.
                    # We need symmetry.
                    # Let's just add both.
                    conflict_graph[nb].add(global_idx)

    # 3. Branch and Bound for Maximum Independent Set
    # We want the largest set of vertices with no edges between them.
    # This is equivalent to Maximum Clique in the complement graph.
    # Or just run MIS directly.
    
    # State:
    # P: set of candidate vertices (available to add)
    # X: set of selected vertices (current independent set)
    # We use a recursive backtracking search.
    
    # To speed up B&B:
    # 1. Order vertices by degree (or heuristic).
    # 2. Upper bound: |P| + |X|. If <= best, prune.
    # 3. Better upper bound: Greedy coloring of P. The number of colors is an upper bound on the size of the clique in the complement (which is the independent set in original).
    #    Wait, MIS in G is Max Clique in G_bar.
    #    Coloring G_bar gives an upper bound on Max Clique.
    #    So we need to color the complement of the induced subgraph on P.
    #    This is expensive.
    
    # Alternative: Use a simple degree-based bound or just |P| if coloring is too slow.
    # Given n ~ 7700, |P| bound might be weak.
    
    # Let's try a simpler approach first: Greedy MIS + Local Search (Swaps) but with a smarter initialization and more aggressive swaps.
    # The previous program did this but maybe inefficiently.
    
    # Let's stick to B&B but with a small candidate set?
    # No, the candidate set is 7700. B&B on 7700 nodes is impossible without strong bounds.
    
    # Let's reconsider the "Inspiration 2" and "3".
    # ILP is a strong candidate.
    # Let's try ILP using scipy.optimize.milp.
    
    # Variables: x_i in {0, 1} for each candidate i.
    # Maximize sum(x_i)
    # Subject to: x_i + x_j <= 1 for all (i, j) in conflict_graph.
    
    # This has ~7700 variables and ~ (7700 * avg_degree / 2) constraints.
    # Avg degree?
    # Let's estimate.
    # For norm 1 vector e_i, it conflicts with e_j (j!=i) if 2<e_i, e_j> > 1 => 0 > 1 (False).
    # It conflicts with -e_i if 2<-e_i, e_i> > 1 => -2 > 1 (False).
    # It conflicts with e_j + e_k?
    # <e_i, e_j+e_k> = 0 or 1.
    # If 1, 2*1 = 2 > 1 (True). Conflict.
    # So e_i conflicts with many norm 2 vectors.
    
    # The number of constraints could be huge (millions).
    # MILP solvers might struggle or take too long.
    
    # Let's go back to a refined Greedy + Local Search, but optimized.
    # The previous program's local search was O(N^2) per iteration potentially.
    
    # New Strategy:
    # 1. Generate candidates.
    # 2. Build conflict graph (as above).
    # 3. Use a "Maximal Independent Set" heuristic with restarts.
    # 4. Improve with "1-swap" and "2-swap" moves using the precomputed conflict graph for O(1) or O(degree) checks.
    
    # Precompute neighbor sets for fast conflict checking.
    # conflict_graph is a list of sets.
    
    def get_conflicts(v_idx):
        return conflict_graph[v_idx]
    
    best_set = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Heuristic: Sort candidates by degree (ascending) to pick "easy" ones first?
    # Or descending?
    # Low degree vertices are less restrictive.
    degrees = [len(conflict_graph[i]) for i in range(n_cand)]
    sorted_indices = np.argsort(degades) # Ascending degree
    
    # Multiple restarts with different orderings
    for restart in range(20):
        if time.time() > deadline - 20:
            break
            
        # Random permutation or sorted?
        # Mix of sorted and random.
        if restart % 2 == 0:
            order = sorted_indices
        else:
            order = rng.permutation(n_cand)
            
        curr_set = set()
        
        # Greedy Add
        for idx in order:
            # Check if idx conflicts with any in curr_set
            # Optimization: If idx is not in conflict_graph of any in curr_set...
            # We can check: is there any v in curr_set such that idx in conflict_graph[v]?
            # This is O(|curr_set|).
            # Or: precompute for each idx, the list of candidates it conflicts with.
            # Then check intersection with curr_set.
            
            # Fast check:
            # conflicts_of_idx = conflict_graph[idx]
            # if not conflicts_of_idx.intersection(curr_set):
            #     curr_set.add(idx)
            
            # Set intersection is fast in Python for small sets.
            # curr_set size might grow to ~1000-2000.
            # conflicts_of_idx size might be large.
            # Intersection of two sets of size M and N is O(min(M, N)).
            
            if not conflict_graph[idx].intersection(curr_set):
                curr_set.add(idx)
                
        if len(curr_set) > best_size:
            best_size = len(curr_set)
            best_set = curr_set.copy()
            
        # Local Search: 1-Swap
        # Try to replace one element in curr_set with one outside to increase size or maintain size but improve "slack"?
        # Since we maximize size, we only swap if we can add a new one by removing an old one?
        # No, 1-swap keeps size same.
        # We need 2-swap (remove 1, add 2) or just try to add more?
        # If greedy is maximal, we can't add any.
        # So we must swap.
        # Swap: Remove u, Add v. Size same.
        # This doesn't increase size directly.
        # We need to find a sequence of swaps that leads to a state where we CAN add a new vertex.
        
        # Alternative: Simulated Annealing or Tabu Search on the set of vertices.
        
        # Let's try a simple "Kick" strategy:
        # Randomly remove k vertices, then greedily add back as many as possible, including new ones.
        
        if time.time() < deadline - 5:
            for kick_iter in range(50):
                if time.time() > deadline - 5:
                    break
                    
                # Remove k random vertices
                k = rng.integers(1, 5)
                to_remove = rng.choice(list(curr_set), size=min(k, len(curr_set)), replace=False)
                for v in to_remove:
                    curr_set.remove(v)
                    
                # Greedily add back
                # Try all candidates not in curr_set
                # This is expensive if done naively.
                # We can try to add the removed ones first, then others.
                
                # Just do a full greedy pass on the remaining + all candidates?
                # That's O(N^2).
                
                # Better: Try to add candidates that were removed or are "close" to the set.
                # Let's just do a limited greedy add.
                
                # Create a list of candidates to try
                # Prioritize: 1. Removed vertices. 2. Random others.
                
                candidates_to_try = list(to_remove)
                # Add some random others
                others = [i for i in range(n_cand) if i not in curr_set and i not in to_remove]
                if others:
                    sample_size = min(50, len(others))
                    candidates_to_try.extend(rng.choice(others, size=sample_size, replace=False))
                    
                for idx in candidates_to_try:
                    if idx in curr_set:
                        continue
                    if not conflict_graph[idx].intersection(curr_set):
                        curr_set.add(idx)
                        
                if len(curr_set) > best_size:
                    best_size = len(curr_set)
                    best_set = curr_set.copy()
                    
    # Final check
    if not best_set:
        return np.array([np.ones(D, dtype=np.int64)], dtype=np.int64)
        
    final_vecs = candidates[np.array(list(best_set))]
    return final_vecs

# EVOLVE-BLOCK-END
