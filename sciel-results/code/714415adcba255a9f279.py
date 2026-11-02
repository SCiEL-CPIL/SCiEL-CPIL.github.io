"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time
from collections import defaultdict

def _generate_candidates(weights):
    """Generate all integer vectors with the specified weights."""
    candidates = []
    for w in weights:
        if w > D:
            continue
        for indices in itertools.combinations(range(D), w):
            for signs in itertools.product([1, -1], repeat=w):
                v = [0] * D
                for idx, sign in zip(indices, signs):
                    v[idx] = sign
                candidates.append(tuple(v))
    return candidates

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    # Generate candidates with weight 3 and 4
    # Weight 3: norm^2 = 3
    # Weight 4: norm^2 = 4
    # Condition: dist^2(u, v) >= max(norm^2(u), norm^2(v))
    # dist^2 = norm^2(u) + norm^2(v) - 2 dot(u, v)
    # If both weight 3: 3+3-2d >= 3 => 3 >= 2d => d <= 1.5 => d <= 1
    # If both weight 4: 4+4-2d >= 4 => 4 >= 2d => d <= 2
    # If w3 and w4: 3+4-2d >= 4 => 3 >= 2d => d <= 1.5 => d <= 1
    
    candidates = _generate_candidates([3, 4])
    n_cand = len(candidates)
    
    # Precompute norms squared
    norms_sq = np.array([np.sum(np.array(v)**2) for v in candidates])
    
    # Build conflict graph
    # Conflict if NOT compatible.
    # We want to select a subset with NO conflicts.
    # So, edge exists if they CONFLICT.
    # We want Max Independent Set.
    
    # Efficiently compute conflicts
    # Convert candidates to numpy array for vectorized dot products
    cand_arr = np.array(candidates, dtype=np.int64)
    
    # We will build an adjacency list for the conflict graph.
    # adj[i] is a set of indices j that conflict with i.
    adj = [set() for _ in range(n_cand)]
    
    # To speed up, we can group by norm.
    # But n_cand is roughly:
    # w=3: C(11,3)*2^3 = 165 * 8 = 1320
    # w=4: C(11,4)*2^4 = 330 * 16 = 5280
    # Total ~ 6600 candidates.
    # O(N^2) dot products is ~ 44M ops. This is fast in numpy.
    
    # Compute all pairwise dots
    # This might be memory heavy if N is large, but 6600x6600 is ~43M floats, ~350MB. 
    # Might be too much for some environments, let's do it in batches or check memory.
    # Actually, 6600x6600 int64 is huge. Let's do it in chunks or just use the loop if it's fast enough?
    # No, loop is too slow in Python.
    # Let's use matrix multiplication in chunks.
    
    # Alternative: Since we only care about d > threshold, and vectors are sparse,
    # maybe we can optimize. But let's try chunked matrix mult first.
    
    chunk_size = 500
    for i_start in range(0, n_cand, chunk_size):
        i_end = min(i_start + chunk_size, n_cand)
        # Dots of candidates[i_start:i_end] with ALL candidates
        dots = cand_arr[i_start:i_end] @ cand_arr.T
        
        # Determine conflicts for these rows
        # For each i in [i_start, i_end):
        for i_rel in range(i_end - i_start):
            i = i_start + i_rel
            n_i = norms_sq[i]
            
            # For all j:
            # dist_sq = n_i + norms_sq[j] - 2*dots[i_rel, j]
            # Condition for NO conflict: dist_sq >= max(n_i, norms_sq[j])
            # Condition for CONFLICT: dist_sq < max(n_i, norms_sq[j])
            
            # We only need j > i to avoid double adding and self-conflict
            # But we can just compute all and filter.
            
            # Vectorized check
            # max_norms = np.maximum(n_i, norms_sq)
            # dist_sq = n_i + norms_sq - 2 * dots[i_rel]
            # conflicts = dist_sq < max_norms
            
            # We want to add j to adj[i] if conflict.
            # And i to adj[j].
            
            # Let's just compute the boolean mask for conflicts
            # Note: j=i is always conflict (dist 0 < norm), but we ignore self.
            
            max_norms = np.maximum(n_i, norms_sq)
            dist_sq = n_i + norms_sq - 2 * dots[i_rel]
            is_conflict = dist_sq < max_norms
            
            # Get indices of conflicts
            conflict_indices = np.where(is_conflict)[0]
            
            for j in conflict_indices:
                if i != j:
                    adj[i].add(j)
                    adj[j].add(i)
        
        if time.time() > deadline - 60:
            break # Safety break, though graph construction should be fast

    # Now we have the conflict graph.
    # We want to find the Maximum Independent Set (MIS).
    # This is equivalent to Maximum Clique in the complement graph.
    # However, the conflict graph is sparse? 
    # Let's check density.
    # Most pairs are compatible. So conflict graph is SPARSE.
    # MIS in sparse graph is easier? Not necessarily, but branch and bound works well.
    # Actually, MIS is NP-hard. 
    # But we can use a greedy + local search approach on the graph, or a proper B&B.
    # Given N=6600, B&B might be too slow if not pruned well.
    # Let's use a heuristic: Greedy coloring bound + Branch and Bound.
    
    # Standard B&B for MIS:
    # 1. Color the graph (greedy) to get an upper bound on clique size (in complement) or independent set size.
    #    Wait, coloring gives upper bound on CLIQUE in the graph. 
    #    For MIS, we work on the COMPLEMENT graph? No.
    #    MIS in G = Max Clique in G_bar.
    #    Coloring G_bar gives upper bound on Clique in G_bar.
    #    Coloring G (conflict graph) gives... ?
    #    Let's stick to MIS on G.
    #    Upper bound: Size of maximum independent set in a graph is bounded by the number of colors needed to color the graph? No, that's for cliques.
    #    Actually, a simple upper bound for MIS is N. 
    #    A better bound: If we can partition the graph into k cliques, then MIS size <= k. (Dual of coloring).
    #    Or, use the greedy algorithm to find a good lower bound, then use B&B.
    
    # Let's try a simpler approach: 
    # 1. Greedy MIS to get a good initial solution.
    # 2. Local search (swap moves) to improve it.
    # 3. If time permits, a limited-depth B&B or simulated annealing on the graph.
    
    # Greedy MIS:
    # Sort nodes by degree (ascending).
    # Pick node with lowest degree, add to set, remove it and its neighbors.
    
    degrees = [len(adj[i]) for i in range(n_cand)]
    order = np.argsort(degades) # Wait, degrees is list, use np.argsort(degrees)
    
    # Actually, let's just use the vectors directly for local search, it's often faster.
    # But the graph structure is already built.
    
    # Let's use a heuristic based on the graph:
    # 1. Start with empty set.
    # 2. Repeatedly pick a node with minimum degree in the remaining graph.
    #    This is O(N^2) if done naively, but N=6600 is small enough? 
    #    6600^2 = 43M. Python loops are slow. 
    #    Let's try a faster greedy: 
    #    Randomize order, pick first available.
    
    best_set = set()
    best_size = 0
    rng = np.random.default_rng(42)
    
    # Multiple random restarts of greedy
    for restart in range(50):
        if time.time() > deadline - 20:
            break
            
        # Random permutation of nodes
        perm = rng.permutation(n_cand)
        current_set = set()
        blocked = set() # Nodes that cannot be added because they conflict with current_set
        
        for node in perm:
            if node in blocked:
                continue
            # Check if node is compatible with all in current_set
            # Since we maintain 'blocked', if node is not blocked, it is compatible?
            # No, 'blocked' only contains neighbors of nodes in current_set.
            # If node is not in blocked, it has no neighbors in current_set.
            # So it is compatible.
            current_set.add(node)
            # Add all neighbors of node to blocked
            blocked.update(adj[node])
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()
            
    # Local Search: Try to swap nodes to improve set size?
    # Or just try to add more? Greedy usually gets stuck.
    # Let's try a "repair" phase: 
    # Remove a random node from best_set, see if we can add more?
    # This is complex.
    
    # Alternative: Use the graph for a simple simulated annealing.
    # State: Set of nodes.
    # Move: Add a node, Remove a node, Swap.
    # Objective: Size of set - Penalty for conflicts (should be 0).
    # Since we must maintain independence, we can only move to valid states.
    
    # Let's try a simple hill climbing on the graph.
    # Start from best_set.
    # Try to add a node. If can't, try swapping one node in set with two outside?
    # This is getting complicated.
    
    # Given the time, the greedy with restarts is the most robust.
    # The previous program did this but with vector checks. 
    # Using the precomputed graph is faster.
    
    # Let's try one more improvement: 
    # The greedy "pick min degree" is often better than random.
    # Let's implement a fast "Min-Degree" greedy.
    
    # To do this efficiently, we need to update degrees.
    # This is O(N^2) per restart. 50 restarts * 43M = 2B ops. Too slow in Python.
    # So random restarts are better for Python.
    
    # Let's increase restarts if time permits.
    # The graph construction took some time.
    # Let's see how much time is left.
    
    # If we have time, do more restarts.
    while time.time() < deadline - 5:
        perm = rng.permutation(n_cand)
        current_set = set()
        blocked = set()
        
        for node in perm:
            if node in blocked:
                continue
            current_set.add(node)
            blocked.update(adj[node])
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    # Convert best_set indices back to vectors
    result_vectors = [candidates[i] for i in best_set]
    
    if not result_vectors:
        # Fallback
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(tuple(v))
        return np.array(vectors, dtype=np.int64)
    
    return np.array(result_vectors, dtype=np.int64)

# EVOLVE-BLOCK-END
