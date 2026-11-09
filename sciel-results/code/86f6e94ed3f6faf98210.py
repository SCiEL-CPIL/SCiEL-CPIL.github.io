import random
import time
import itertools
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s buffer for safety
    
    n = 16
    best_alpha = n
    best_triples = []
    
    # Precomputation
    all_triples = list(itertools.combinations(range(n), 3))
    num_triples = len(all_triples)
    
    # Map triples to indices
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute pair masks for each triple to check linearity (pair disjointness)
    # Pair (i,j) maps to an index 0..119
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            pair_to_idx[(i, j)] = idx
            idx += 1
            
    triple_pair_masks = []
    for a, b, c in all_triples:
        mask = 0
        mask |= (1 << pair_to_idx[(a, b)])
        mask |= (1 << pair_to_idx[(a, c)])
        mask |= (1 << pair_to_idx[(b, c)])
        triple_pair_masks.append(mask)
        
    # Precompute subset_triples: for each mask (0..2^16-1), list of triple indices contained in it
    # This allows fast independence check: is_independent(mask, H) iff no triple in subset_triples[mask] is in H
    subset_triples = [[] for _ in range(1 << n)]
    for i, (a, b, c) in enumerate(all_triples):
        t_mask = (1 << a) | (1 << b) | (1 << c)
        # The triple i is contained in any superset of t_mask
        # Iterate over all supersets of t_mask
        # Supersets of t_mask are formed by adding any subset of the remaining bits
        remaining_bits_mask = ((1 << n) - 1) ^ t_mask
        # Iterate submasks of remaining_bits_mask
        sub = remaining_bits_mask
        while True:
            superset_mask = t_mask | sub
            subset_triples[superset_mask].append(i)
            if sub == 0:
                break
            sub = (sub - 1) & remaining_bits_mask
            
    # Precompute lists of masks by popcount to iterate efficiently
    masks_by_size = {k: [] for k in range(n + 1)}
    for mask in range(1 << n):
        pc = mask.bit_count()
        masks_by_size[pc].append(mask)
        
    # Sort masks in descending order of popcount is not needed if we iterate k from n down to 1
    # But for a given k, we just iterate the list.
    
    def compute_alpha(H_indices):
        """
        Compute the independence number of the hypergraph defined by H_indices (set of triple indices).
        """
        if not H_indices:
            return n
            
        # Check from largest size down
        for k in range(n, 0, -1):
            for mask in masks_by_size[k]:
                # Check if mask is independent
                # Independent means no triple in H_indices is fully contained in mask
                # i.e., intersection of H_indices and subset_triples[mask] is empty
                is_independent = True
                for t_idx in subset_triples[mask]:
                    if t_idx in H_indices:
                        is_independent = False
                        break
                if is_independent:
                    return k
        return 0

    def try_random_greedy():
        """Generate a maximal packing of triples."""
        triples = []
        H_indices = set()
        used_pairs_mask = 0
        
        candidates = list(range(num_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            pmask = triple_pair_masks[idx]
            if (used_pairs_mask & pmask) == 0:
                H_indices.add(idx)
                used_pairs_mask |= pmask
                triples.append(all_triples[idx])
                
        return H_indices, triples

    def local_search_sa(H_indices, triples, time_limit):
        """
        Simulated Annealing local search.
        H_indices: set of triple indices
        triples: list of triple tuples (for return)
        """
        current_alpha = compute_alpha(H_indices)
        current_triples = triples[:]
        
        # SA Parameters
        T = 1.0
        cooling_rate = 0.999
        min_T = 0.01
        last_improvement_time = time.time()
        
        # Precompute list of triple indices for random choice
        all_indices = list(range(num_triples))
        
        while time.time() < time_limit:
            if time.time() - last_improvement_time > 3.0:
                break # Stuck, break to restart
                
            # Pick a random triple to remove
            if not H_indices:
                break
            rem_idx = random.choice(list(H_indices))
            
            # Temporarily remove
            H_indices.remove(rem_idx)
            
            # Pick a random triple to add
            # We need it to be compatible with the current H_indices
            # To speed up, we can try a few random ones
            added_idx = None
            for _ in range(20): # Try 20 random candidates
                add_idx = random.choice(all_indices)
                if add_idx in H_indices:
                    continue
                # Check compatibility
                # We need to check if add_idx conflicts with any triple in H_indices
                # Since we removed rem_idx, we just check against current H_indices
                # Conflict means sharing a pair.
                # We can use the pair masks.
                # But we don't have a global "used pairs" mask easily updated in SA without overhead.
                # However, we can just check pairwise against H_indices.
                # H_indices size is ~40-50.
                # 20 candidates * 50 checks = 1000 checks.
                # Each check is (triple_pair_masks[add_idx] & triple_pair_masks[other_idx]) != 0
                # This is fast.
                
                compatible = True
                for other_idx in H_indices:
                    if triple_pair_masks[add_idx] & triple_pair_masks[other_idx]:
                        compatible = False
                        break
                
                if compatible:
                    added_idx = add_idx
                    break
            
            if added_idx is None:
                # Couldn't find a valid add, revert removal
                H_indices.add(rem_idx)
                continue
            
            # Apply move
            H_indices.add(added_idx)
            
            new_alpha = compute_alpha(H_indices)
            delta = new_alpha - current_alpha
            
            # Acceptance criterion
            if delta <= 0 or (T > min_T and random.random() < math.exp(-delta / T)):
                # Accept
                current_alpha = new_alpha
                if current_alpha < best_alpha:
                    best_alpha = current_alpha
                    best_triples = [all_triples[i] for i in H_indices]
                    last_improvement_time = time.time()
            else:
                # Revert
                H_indices.remove(added_idx)
                H_indices.add(rem_idx)
                
            T *= cooling_rate
            if T < min_T:
                T = min_T
                
        return H_indices, [all_triples[i] for i in H_indices]

    # Main Loop
    trials = 0
    while time.time() < deadline:
        trials += 1
        H_indices, triples = try_random_greedy()
        
        # Calculate alpha
        alpha = compute_alpha(H_indices)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_triples = triples[:]
            if best_alpha <= 7:
                break
                
        # Run SA for a short duration
        sa_time_limit = min(5.0, deadline - time.time() - 2.0)
        if sa_time_limit > 1.0:
            H_indices, triples = local_search_sa(H_indices, triples, sa_time_limit)
            alpha = compute_alpha(H_indices)
            if alpha < best_alpha:
                best_alpha = alpha
                best_triples = triples[:]
                if best_alpha <= 7:
                    break

    return best_triples
# EVOLVE-BLOCK-END
