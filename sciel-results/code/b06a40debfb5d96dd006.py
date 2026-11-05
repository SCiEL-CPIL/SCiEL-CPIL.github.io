import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    triple_pairs = []
    triple_masks = []
    for t in all_triples:
        pairs = [tuple(sorted((t[0], t[1]))), tuple(sorted((t[0], t[2]))), tuple(sorted((t[1], t[2])))]
        triple_pairs.append(pairs)
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        triple_masks.append(mask)
        
    # Precompute subset masks and their sizes
    # We only need to check specific sizes
    # Let's just use combinations in the check function
    
    def has_independent_set(indices, k):
        # Check if there is an independent set of size k
        # indices: list of triple indices
        if not indices:
            return True
        masks = [triple_masks[i] for i in indices]
        
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
                return True
        return False

    def get_alpha(indices, hint_alpha=None):
        # If hint_alpha is provided, we can start from there
        # If we added a triple, alpha decreases or stays. Start from hint_alpha - 1?
        # If we removed, alpha increases. Start from hint_alpha + 1?
        
        if hint_alpha is None:
            for k in range(16, 0, -1):
                if has_independent_set(indices, k):
                    return k
            return 0
        else:
            # We know the alpha changed slightly
            # We don't know direction from here, but caller can pass range
            # Let's just do full search for safety, but it's slow.
            # Better: caller updates alpha incrementally.
            pass

    # Initial greedy
    def greedy():
        indices = list(range(n_triples))
        random.shuffle(indices)
        current_indices = []
        pair_count = {}
        
        for idx in indices:
            pairs = triple_pairs[idx]
            conflict = False
            for p in pairs:
                if pair_count.get(p, 0) > 0:
                    conflict = True
                    break
            if not conflict:
                current_indices.append(idx)
                for p in pairs:
                    pair_count[p] = pair_count.get(p, 0) + 1
        return current_indices, pair_count

    best_indices, best_pair_count = greedy()
    best_alpha = 16
    # Calculate initial alpha
    for k in range(16, 0, -1):
        if has_independent_set(best_indices, k):
            best_alpha = k
            break

    T = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        new_indices = best_indices[:]
        new_pair_count = best_pair_count.copy()
        added_idx = -1
        removed_idx = -1
        
        move_type = random.random()
        if move_type < 0.3:
            # Add
            for _ in range(10):
                idx = random.randint(0, n_triples-1)
                pairs = triple_pairs[idx]
                conflict = False
                for p in pairs:
                    if new_pair_count.get(p, 0) > 0:
                        conflict = True
                        break
                if not conflict:
                    new_indices.append(idx)
                    for p in pairs:
                        new_pair_count[p] = new_pair_count.get(p, 0) + 1
                    added_idx = idx
                    break
        elif move_type < 0.6:
            # Remove
            if new_indices:
                pos = random.randint(0, len(new_indices)-1)
                idx = new_indices[pos]
                new_indices.pop(pos)
                pairs = triple_pairs[idx]
                for p in pairs:
                    new_pair_count[p] -= 1
                    if new_pair_count[p] == 0:
                        del new_pair_count[p]
                removed_idx = idx
        else:
            # Swap
            if new_indices:
                pos = random.randint(0, len(new_indices)-1)
                idx_out = new_indices[pos]
                new_indices.pop(pos)
                pairs_out = triple_pairs[idx_out]
                for p in pairs_out:
                    new_pair_count[p] -= 1
                    if new_pair_count[p] == 0:
                        del new_pair_count[p]
                removed_idx = idx_out
                
                for _ in range(10):
                    idx_in = random.randint(0, n_triples-1)
                    pairs_in = triple_pairs[idx_in]
                    conflict = False
                    for p in pairs_in:
                        if new_pair_count.get(p, 0) > 0:
                            conflict = True
                            break
                    if not conflict:
                        new_indices.append(idx_in)
                        for p in pairs_in:
                            new_pair_count[p] = new_pair_count.get(p, 0) + 1
                        added_idx = idx_in
                        break

        # Update Alpha
        new_alpha = best_alpha
        
        if added_idx != -1:
            # Alpha can only decrease or stay
            # Check if best_alpha is still independent
            if has_independent_set(new_indices, best_alpha):
                new_alpha = best_alpha
            else:
                # Check best_alpha - 1
                new_alpha = best_alpha - 1
                while new_alpha > 0 and not has_independent_set(new_indices, new_alpha):
                    new_alpha -= 1
        elif removed_idx != -1:
            # Alpha can only increase or stay
            # Check if best_alpha + 1 is independent
            new_alpha = best_alpha
            # We need to check up to 16
            # But likely it increases by 1
            # Check best_alpha + 1
            if best_alpha < 16 and has_independent_set(new_indices, best_alpha + 1):
                new_alpha = best_alpha + 1
                # Check if it increases further?
                # Unlikely to jump by more than 1 in one step?
                # But possible.
                while new_alpha < 16 and has_independent_set(new_indices, new_alpha + 1):
                    new_alpha += 1
        else:
            # No change? (Should not happen if moves are valid)
            new_alpha = best_alpha

        if new_alpha < best_alpha:
            best_alpha = new_alpha
            best_indices = new_indices
            best_pair_count = new_pair_count
        else:
            if random.random() < pow(2.71828, -(new_alpha - best_alpha) / T):
                best_indices = new_indices
                best_pair_count = new_pair_count
                # Note: best_alpha might be higher now if we accepted a worse move
                # But we only accept if it's "close".
                # If we accept a worse move, best_alpha becomes new_alpha.
                best_alpha = new_alpha

        T *= cooling
        if T < 0.01:
            T = 0.01

    result = [all_triples[i] for i in best_indices]
    return result
