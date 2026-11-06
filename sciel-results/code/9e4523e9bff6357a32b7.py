import time
import itertools
import random
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # Map pair -> triples containing it
    pair_to_triples = defaultdict(list)
    for t in all_triples:
        for p in triple_pairs[t]:
            pair_to_triples[p].append(t)
            
    # Precompute bitmask for each triple
    triple_masks = {}
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks[t] = mask
    
    # Precompute subset masks for all 2^16 subsets
    subset_masks = np.zeros(1 << 16, dtype=np.int32)
    for s in range(1, 1 << 16):
        # Use Brian Kernighan's bit trick
        lsb = s & (-s)
        idx = lsb.bit_length() - 1
        subset_masks[s] = subset_masks[s ^ lsb] | (1 << idx)
    
    # Precompute popcount for all subsets
    popcounts = np.zeros(1 << 16, dtype=np.int8)
    for s in range(1, 1 << 16):
        lsb = s & (-s)
        popcounts[s] = popcounts[s ^ lsb] + 1

    def get_alpha_and_witness(triples):
        """
        Compute the independence number and the witness set using NumPy.
        A subset S is independent if no triple is fully contained in S.
        """
        if not triples:
            return 16, set(points)
            
        # Build a matrix: for each triple, its bitmask
        t_masks = np.array([triple_masks[t] for t in triples], dtype=np.int32)
        
        # For each subset, check if any triple is contained
        # triple_mask is contained in subset_mask if (subset_mask & triple_mask) == triple_mask
        # Equivalently: subset_mask has all bits of triple_mask
        
        # Check from largest k down
        for k in range(16, 0, -1):
            # Find all subsets of size k
            candidates = np.where(popcounts == k)[0]
            if len(candidates) == 0:
                continue
            
            # For each candidate subset, check if any triple is contained
            # subset_masks[candidates] has shape (n_candidates,)
            # t_masks has shape (n_triples,)
            # We need: for each subset, does there exist a triple with (subset & triple) == triple?
            
            # Vectorized check: 
            # For each triple, the subset must have all bits of the triple
            # (subset_masks[candidates][:, None] & t_masks[None, :]) == t_masks[None, :]
            # But this creates a huge matrix. Instead, check per triple.
            
            # More efficient: for each triple, find which candidates contain it
            # A candidate contains triple t if (candidate_mask & t_mask) == t_mask
            # i.e., candidate_mask has all bits set that t_mask has
            # i.e., (candidate_mask | t_mask) == candidate_mask
            
            independent_found = False
            for t_mask in t_masks:
                # Check which candidates contain this triple
                # candidate contains triple if (candidate & t_mask) == t_mask
                # i.e., candidate has all bits of t_mask
                contains = (subset_masks[candidates] & t_mask) == t_mask
                if not np.all(contains):
                    # Some candidates don't contain this triple - they might be independent
                    # But we need NO triple to be contained
                    # So a candidate is independent if for ALL triples, it doesn't contain them
                    # Let's just check each candidate individually if the set is small enough
                    pass
            
            # Alternative approach: for each candidate, check all triples
            # If n_candidates * n_triples is manageable
            for idx in candidates:
                smask = subset_masks[idx]
                is_independent = True
                for t_mask in t_masks:
                    if (smask & t_mask) == t_mask:
                        is_independent = False
                        break
                if is_independent:
                    # Reconstruct the subset from the mask
                    subset_points = [p for p in range(16) if smask & (1 << p)]
                    return k, set(subset_points)
        
        return 1, set(points[:1])

    # Strategy:
    # 1. Generate random greedy packings.
    # 2. Local search with guidance from the witness set.
    
    best_alpha = 16
    best_system = []
    
    # Initialize with random greedy packings - more restarts due to fast alpha check
    n_restarts = 50
    for r in range(n_restarts):
        if time.time() - start_time > deadline:
            break
        random.shuffle(all_triples)
        current = []
        used_pairs = set()
        for t in all_triples:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        
        alpha, _ = get_alpha_and_witness(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]
            
        if best_alpha <= 5:
            break
    
    current_system = best_system[:]
    current_alpha = best_alpha
    current_witness = set()
    
    # Update witness for current best
    if current_system:
        current_alpha, current_witness = get_alpha_and_witness(current_system)
    else:
        current_witness = set(points)

    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])

    def weighted_choice(candidates, current_witness):
        """
        Choose a candidate with probability proportional to its intersection with the witness set.
        """
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        
        # Compute weights based on intersection with witness set
        witness_set = current_witness
        weights = []
        for t in candidates:
            inter = 0
            for p in t:
                if p in witness_set:
                    inter += 1
            weights.append(inter + 1)  # +1 to avoid zero weight
        
        # Weighted random choice
        total = sum(weights)
        r = random.random() * total
        cumulative = 0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return candidates[i]
        return candidates[-1]

    while time.time() - start_time < deadline:
        move_type = random.choice(['add', 'swap', 'swap_witness'])
        
        if move_type == 'add':
            used_pairs = set()
            for t in current_system:
                used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs):
                    candidates.append(t)
            
            if candidates:
                new_t = weighted_choice(candidates, current_witness)
                if new_t is None:
                    continue
                new_system = current_system + [new_t]
                new_alpha, new_witness = get_alpha_and_witness(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < 0.5:
                        current_system = new_system
                        current_witness = new_witness

        elif move_type == 'swap':
            if not current_system:
                continue
                
            idx = random.randint(0, len(current_system) - 1)
            t_remove = current_system[idx]
            
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs) and t != t_remove:
                    candidates.append(t)
            
            if candidates:
                new_t = weighted_choice(candidates, current_witness)
                if new_t is None:
                    continue
                    
                new_system = current_system[:]
                new_system[idx] = new_t
                new_alpha, new_witness = get_alpha_and_witness(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < 0.5:
                        current_system = new_system
                        current_witness = new_witness

        elif move_type == 'swap_witness':
            # Remove a triple that covers a witness point, and try to replace it
            # with a triple that covers different witness points
            if not current_system or not current_witness:
                continue
            
            # Find triples that cover witness points
            witness_covering_triples = []
            for i, t in enumerate(current_system):
                t_set = set(t)
                if t_set & current_witness:
                    witness_covering_triples.append(i)
            
            if not witness_covering_triples:
                continue
            
            idx = random.choice(witness_covering_triples)
            t_remove = current_system[idx]
            
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])
            
            # Find candidates that cover witness points NOT covered by t_remove
            t_remove_set = set(t_remove)
            uncovered_witness = current_witness - t_remove_set
            
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs) and t != t_remove:
                    # Prefer triples that cover uncovered witness points
                    t_set = set(t)
                    inter_uncovered = len(t_set & uncovered_witness)
                    if inter_uncovered > 0:
                        candidates.append((t, inter_uncovered))
            
            if candidates:
                # Sort by coverage of uncovered witness points (descending)
                candidates.sort(key=lambda x: x[1], reverse=True)
                # Pick from top candidates
                top_k = min(10, len(candidates))
                new_t = random.choice(candidates[:top_k])[0]
                
                new_system = current_system[:]
                new_system[idx] = new_t
                new_alpha, new_witness = get_alpha_and_witness(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < 0.7:
                        current_system = new_system
                        current_witness = new_witness

        if best_alpha <= 5:
            break
    
    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(f"Number of triples: {len(res)}")
    print(f"First few: {res[:5]}")
