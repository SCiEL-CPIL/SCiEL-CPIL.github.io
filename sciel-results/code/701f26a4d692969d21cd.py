import numpy as np
import itertools
import time
import random

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # 1. Generate Initial Pool of Weight-4 Vectors
    # We use weight-4 vectors (norm 2) as the base building blocks.
    # The constraint is: for all distinct u, v in set, ||u - v|| >= max_norm = 2.
    # This implies ||u - v||^2 >= 4.
    # ||u||^2 + ||v||^2 - 2 u.v >= 4 => 4 + 4 - 2 u.v >= 4 => u.v <= 2.
    # So we need a set of weight-4 vectors with pairwise dot product <= 2.
    
    candidates = []
    indices = range(D)
    for comb in itertools.combinations(indices, 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            candidates.append(v)
            
    candidates = np.array(candidates, dtype=np.int8)
    n_cands = len(candidates)
    
    if time.time() > deadline - 10:
        return candidates[:1] # Fallback

    # 2. Precompute Conflict Matrix
    # conflict[i, j] = True if candidates[i] and candidates[j] cannot coexist.
    # Condition: dot > 2.
    dots = candidates @ candidates.T
    conflict = (dots > 2).astype(np.int8)
    np.fill_diagonal(conflict, 0)
    
    # 3. Local Search Strategy
    # We maintain a set of selected indices.
    # We try to add a new vector from the pool that doesn't conflict with any current member.
    # If stuck, we try to "swap" a current member with a different vector to potentially
    # allow more additions later, or simply restart with a different initial seed.
    # 
    # A more effective local search for packing problems is often "Greedy + Backtracking" 
    # or "Simulated Annealing" style. However, given the discrete structure, 
    # a robust Multi-Start Greedy with a "Lookahead" or "Swap" heuristic is often best.
    
    # Let's implement a specialized Local Search:
    # 1. Start with an empty set.
    # 2. Greedily add vectors that are compatible.
    # 3. If no more can be added, try to improve by removing one vector and 
    #    trying to add two or more that were previously blocked by it? 
    #    Actually, a simpler and often effective approach for MIS in these graphs is 
    #    **Branch and Bound** (Tomita algorithm) if the graph isn't too large, 
    #    but 5280 nodes is too big for standard B&B without heavy pruning in 150s.
    
    # Alternative Structural Change: **Genetic Algorithm / Evolutionary Search**
    # Represent a solution as a bitmask of size n_cands.
    # Fitness = size of the valid subset.
    # But checking validity of a full bitmask is expensive.
    
    # Let's stick to a highly optimized **Multi-Start Greedy with Adaptive Ordering** 
    # but add a **Local Improvement** step: 
    # After a greedy pass, iterate through the selected set. For each selected vector, 
    # check if removing it allows adding a net gain of > 0 vectors.
    
    best_set = []
    best_size = 0
    
    # Precompute neighbor lists for faster conflict checking?
    # conflict is 5280x5280. 
    # neighbors[i] = list of indices j where conflict[i,j]==1
    
    # To speed up the "can I add this?" check, we maintain a boolean array `blocked`.
    # When we add index i, we block all neighbors of i.
    
    def greedy_solve(order):
        """
        Runs a greedy pass given a specific order of candidate indices.
        Returns the list of selected indices.
        """
        selected = []
        # blocked[j] is True if j conflicts with any already selected vector
        blocked = np.zeros(n_cands, dtype=bool)
        
        for idx in order:
            if blocked[idx]:
                continue
            
            # Select idx
            selected.append(idx)
            
            # Block idx and all its neighbors
            # conflict[idx, :] is a row of the conflict matrix
            blocked[conflict[idx, :] == 1] = True
            blocked[idx] = True
            
            # Early exit if we've picked everyone? (Unlikely)
            # Time check inside loop? No, too frequent.
            
        return selected

    def local_improve(selected_indices):
        """
        Attempts to improve a set by removing one element and re-greeding the rest?
        Or simpler: Try to add any unselected, unblocked vector? (Greedy already did this).
        
        Let's try a "1-swap" heuristic:
        For each selected vector v, remove it. Then try to greedily add vectors 
        that were blocked by v, to see if we can get a larger set.
        This is expensive. 
        """
        pass # Skipping complex local improve for now, focusing on diverse restarts

    # Number of restarts depends on time.
    # Each greedy pass is O(N^2) worst case if we iterate through all N and check blocked.
    # N = 5280. N^2 ~ 27M ops. NumPy vectorization makes the blocking step fast.
    # The loop is Python-level, so 5280 iterations per restart.
    # 5280 * 200 restarts = 1M Python loop iterations. Should be fast (< 1s).
    
    # Let's do more restarts with varied orderings.
    
    num_restarts = 500 # We have time budget. Let's use it.
    
    # Pre-generate random permutations? No, generate on the fly.
    
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        if r == 0:
            # Ascending degree (most promising for MIS)
            degrees = np.sum(conflict, axis=1)
            order = np.argsort(degrees)
        elif r == 1:
            # Descending degree
            degrees = np.sum(conflict, axis=1)
            order = np.argsort(degrees)[::-1]
        elif r == 2:
            # Random
            order = np.random.permutation(n_cands)
        else:
            # Perturbed degree
            degrees = np.sum(conflict, axis=1)
            noise = np.random.rand(n_cands) * (np.max(degrees) - np.min(degrees) + 1)
            order = np.argsort(degrees + noise)
            
        current_set = greedy_solve(order)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
            # If we found a very good one, maybe do a quick local search?
            # For now, just record it.

    # Final check: ensure the best set is valid.
    # The greedy algorithm guarantees validity by construction (blocking neighbors).
    # But let's double check the final array.
    
    if not best_set:
        return candidates[:1]
        
    result_vectors = candidates[best_set]
    
    # Verify constraint: min pairwise dist >= max norm
    # max norm is 2 (sqrt(4)). dist^2 >= 4.
    # We enforced dot <= 2.
    # dist^2 = 4 + 4 - 2*dot = 8 - 2*dot.
    # If dot <= 2, dist^2 >= 8 - 4 = 4. dist >= 2.
    # So the logic holds.
    
    return result_vectors

# EVOLVE-BLOCK-END
