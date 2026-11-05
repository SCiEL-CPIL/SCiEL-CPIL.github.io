import time
import random
from itertools import combinations
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s buffer
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute pair sets for each triple
    triple_pairs = []
    for t in all_triples:
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        triple_pairs.append(pairs)
        
    # Precompute masks for each triple for fast independence check
    triple_masks = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)

    # Build conflict graph structure for MILP
    all_pairs = [(i, j) for i in range(16) for j in range(i+1, 16)]
    pair_to_idx = {p: idx for idx, p in enumerate(all_pairs)}
    n_pairs = len(all_pairs)
    
    # A[i, j] = 1 if pair j is in triple i
    A = np.zeros((n_pairs, n_triples), dtype=np.int8)
    for i, t in enumerate(all_triples):
        for p in triple_pairs[i]:
            A[pair_to_idx[p], i] = 1
            
    # Objective: maximize sum(x)
    c = -np.ones(n_triples)
    
    # Constraints: A @ x <= 1
    constraints = [LinearConstraint(A, 0, 1)]
    
    # Bounds: 0 <= x[i] <= 1
    bounds = Bounds(0, 1)
    
    # Integrate variables
    integrality = np.ones(n_triples)
    
    # Solve MILP
    best_alpha = 16
    best_triples = []
    curr_triples = []
    curr_alpha = 16
    
    def get_alpha(triple_list):
        if not triple_list:
            return 16
        
        # Convert to masks
        masks = [triple_masks[all_triples.index(t)] if t in all_triples else 0 for t in triple_list]
        # Optimization: precompute index mapping for faster lookup if needed, 
        # but all_triples.index is O(N). Let's use a dict.
        # Actually, let's just store the masks directly in the list during search.
        # For now, this function takes a list of tuples.
        
        # To speed up, we can check sizes from 16 down.
        # But we can also just check all subsets if we are smart.
        # Let's check k from 16 down to 1.
        # If we find an independent set of size k, return k.
        
        # Precompute subset masks for each k? No, generate on fly.
        
        # Faster approach: 
        # A set S is independent if no triple is contained in S.
        # We want max |S| such that S is independent.
        
        # We can iterate k from 16 down to 1.
        for k in range(16, 0, -1):
            for subset in combinations(points, k):
                mask = 0
                for p in subset:
                    mask |= (1 << p)
                
                is_ind = True
                for m in masks:
                    if (m & mask) == m:
                        is_ind = False
                        break
                if is_ind:
                    return k
        return 0

    # Faster alpha check using bitsets and precomputed subset lists?
    # Let's precompute all subset masks and their sizes.
    # There are 2^16 = 65536 subsets.
    # subset_masks[size][index] = mask
    # This might be memory heavy but fast.
    
    # Let's stick to the combination generation but optimize the inner loop.
    
    # Initial Solution via MILP
    try:
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)
        if res.status == 0 and res.x is not None:
            selected_indices = [i for i in range(n_triples) if res.x[i] > 0.5]
            curr_triples = [all_triples[i] for i in selected_indices]
            curr_alpha = get_alpha(curr_triples)
            best_triples = curr_triples
            best_alpha = curr_alpha
    except Exception:
        pass
    
    # Fallback if MILP fails or returns empty
    if not curr_triples:
        # Simple greedy seed
        curr_triples = []
        used_pairs = set()
        indices = list(range(n_triples))
        random.shuffle(indices)
        for idx in indices:
            t = all_triples[idx]
            pairs = triple_pairs[idx]
            if not any(p in used_pairs for p in pairs):
                curr_triples.append(t)
                for p in pairs:
                    used_pairs.add(p)
        curr_alpha = get_alpha(curr_triples)
        best_triples = curr_triples
        best_alpha = curr_alpha

    # Local Search / Simulated Annealing
    # State: current set of triple indices
    curr_indices = [all_triples.index(t) for t in curr_triples]
    
    # Precompute compatibility: for each triple i, which other triples j are compatible?
    # compatible[i] = set of j such that triple i and j share at most 1 point.
    # Since n_triples is 560, this is 560*560 = 313,600 checks. Feasible.
    compatible = [set() for _ in range(n_triples)]
    for i in range(n_triples):
        for j in range(i+1, n_triples):
            # Check if they share a pair
            # If they share a pair, they are NOT compatible (in the context of a linear system, 
            # wait. The constraint is "any two triples share at most one point".
            # So if they share 2 points, they conflict.
            # If they share 1 or 0 points, they are compatible.
            t_i = all_triples[i]
            t_j = all_triples[j]
            # Check intersection size
            if len(set(t_i) & set(t_j)) <= 1:
                compatible[i].add(j)
                compatible[j].add(i)
            # else: they share 2 points, so conflict.

    # Helper to get a random compatible triple not in current set
    def get_random_compatible(excluded_indices):
        # excluded_indices is a set of indices currently in the system
        # We want to pick a random index from compatible[any_current] or just from all?
        # If we are adding a triple to a system T, it must be compatible with ALL t in T.
        # So we need the intersection of compatible[t] for all t in T.
        # This is expensive if T is large.
        
        # Alternative: Pick a random candidate from all_triples, check if compatible with all in T.
        # Since T is ~30-40 triples, checking 30-40 pairs is fast.
        while True:
            cand = random.randint(0, n_triples - 1)
            if cand in excluded_indices:
                continue
            # Check compatibility with all in excluded_indices
            ok = True
            for idx in excluded_indices:
                if cand not in compatible[idx]:
                    ok = False
                    break
            if ok:
                return cand

    # Simulated Annealing Loop
    temperature = 1.0
    cooling_rate = 0.99
    
    # We track indices for speed
    curr_set = set(curr_indices)
    
    for step in range(20000):
        if time.time() > deadline:
            break
            
        # 1. Generate Neighbor
        # Remove a random triple
        if not curr_set:
            break
        t_remove_idx = random.choice(list(curr_set))
        curr_set.remove(t_remove_idx)
        
        # Add a random compatible triple
        t_add_idx = get_random_compatible(curr_set)
        if t_add_idx is None:
            # Could not add any triple? Restore and break or try again
            curr_set.add(t_remove_idx)
            continue
            
        curr_set.add(t_add_idx)
        
        # 2. Evaluate
        new_triples = [all_triples[i] for i in curr_set]
        new_alpha = get_alpha(new_triples)
        
        # 3. Acceptance
        delta_alpha = new_alpha - curr_alpha
        if delta_alpha < 0:
            # Better
            curr_alpha = new_alpha
            if new_alpha < best_alpha:
                best_alpha = new_alpha
                best_triples = new_triples
        else:
            # Worse or equal
            # Accept with probability exp(-delta / T)
            # Note: We want to minimize alpha, so delta > 0 is bad.
            if delta_alpha == 0:
                # Equal alpha, might help escape local minima? 
                # Keep it with some probability
                if random.random() < 0.5:
                    curr_alpha = new_alpha
            else:
                prob = np.exp(-delta_alpha / temperature)
                if random.random() < prob:
                    curr_alpha = new_alpha
        
        temperature *= cooling_rate
        if temperature < 0.01:
            temperature = 0.01

    return best_triples
