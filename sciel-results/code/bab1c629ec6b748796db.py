import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    points = list(range(16))
    all_triples = list(combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute masks for each triple
    triple_masks = [0] * n_triples
    for i, t in enumerate(all_triples):
        mask = 0
        for v in t:
            mask |= (1 << v)
        triple_masks[i] = mask
        
    # Precompute pairs for each triple to check linearity
    # We need to quickly check if a triple conflicts with current set
    # We can maintain a set of used pairs.
    # There are C(16,2) = 120 pairs.
    
    def calc_alpha_and_witness(triple_indices):
        """
        Calculate independence number and return a witness set (list of vertices).
        """
        if not triple_indices:
            return 16, list(range(16))
            
        n = 16
        active_masks = [triple_masks[i] for i in triple_indices]
        
        # Sort vertices by degree
        deg = [0] * n
        for mask in active_masks:
            for v in range(n):
                if mask & (1 << v):
                    deg[v] += 1
                    
        order = sorted(range(n), key=lambda v: -deg[v])
        
        best_alpha = [0]
        best_witness = [[]]
        
        # Branch and bound
        def search(idx, current_mask, current_size, current_vertices):
            if time.time() > deadline - 2:
                return
            
            # Pruning: if max possible size (current + remaining) <= best, return
            if current_size + (n - idx) <= best_alpha[0]:
                return
                
            if current_size > best_alpha[0]:
                best_alpha[0] = current_size
                best_witness[0] = list(current_vertices)
                
            if idx == n:
                return
                
            v = order[idx]
            
            # Option 1: Exclude v
            search(idx + 1, current_mask, current_size, current_vertices)
            
            # Option 2: Include v
            new_mask = current_mask | (1 << v)
            
            valid = True
            for am in active_masks:
                if am & new_mask == am:
                    valid = False
                    break
                    
            if valid:
                current_vertices.append(v)
                search(idx + 1, new_mask, current_size + 1, current_vertices)
                current_vertices.pop()
                
        search(0, 0, 0, [])
        return best_alpha[0], best_witness[0]

    def is_valid_addition(triple_idx, used_pairs):
        t = all_triples[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            if p in used_pairs:
                return False
        return True

    def add_triple(triple_idx, current_triples, current_triple_set, used_pairs):
        t = all_triples[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            used_pairs.add(p)
        current_triples.append(triple_idx)
        current_triple_set.add(triple_idx)

    def remove_triple(triple_idx, current_triples, current_triple_set, used_pairs):
        t = all_triples[triple_idx]
        pairs = [(t[0], t[1]), (t[0], t[2]), (t[1], t[2])]
        for p in pairs:
            used_pairs.discard(p)
        current_triple_set.discard(triple_idx)
        # Remove from list
        for i, idx in enumerate(current_triples):
            if idx == triple_idx:
                current_triples.pop(i)
                break

    best_alpha = 17
    best_solution = None
    
    num_restarts = 50
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        random.seed(42 + restart)
        
        used_pairs = set()
        current_triples = []
        current_triple_set = set()
        
        # Greedy initialization
        indices = list(range(n_triples))
        random.shuffle(indices)
        
        for t_idx in indices:
            if is_valid_addition(t_idx, used_pairs):
                add_triple(t_idx, current_triples, current_triple_set, used_pairs)
                
        alpha, witness = calc_alpha_and_witness(current_triples)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_solution = list(current_triples)
            if best_alpha <= 4:
                break
            
        # Local Search with Witness Breaking
        steps = 0
        max_steps = 200
        while steps < max_steps:
            if time.time() > deadline - 5:
                break
            steps += 1
            
            # Recalculate alpha and witness periodically or if changed
            if steps % 5 == 0 or steps == 1:
                alpha, witness = calc_alpha_and_witness(current_triples)
                if alpha < best_alpha:
                    best_alpha = alpha
                    best_solution = list(current_triples)
                    if best_alpha <= 4:
                        break
                        
            if best_alpha <= 4:
                break
                
            # Strategy 1: Try to break the witness
            # Find triples contained in witness
            witness_mask = 0
            for v in witness:
                witness_mask |= (1 << v)
                
            candidates = []
            for t_idx in range(n_triples):
                if t_idx in current_triple_set:
                    continue
                if triple_masks[t_idx] & witness_mask == triple_masks[t_idx]:
                    # Triple is subset of witness
                    if is_valid_addition(t_idx, used_pairs):
                        candidates.append(t_idx)
                        
            if candidates:
                # Pick a random candidate
                cand = random.choice(candidates)
                add_triple(cand, current_triples, current_triple_set, used_pairs)
                # This move is always "good" or neutral in terms of breaking this specific witness
                # But might increase alpha elsewhere? Unlikely to increase max alpha, might decrease.
            else:
                # Strategy 2: Random Swap or Move
                # Try to remove a random triple and add a random valid one
                if current_triples:
                    rem_idx = random.choice(current_triples)
                    remove_triple(rem_idx, current_triples, current_triple_set, used_pairs)
                    
                    # Try to add a random triple
                    add_idx = random.randint(0, n_triples - 1)
                    if is_valid_addition(add_idx, used_pairs):
                        add_triple(add_idx, current_triples, current_triple_set, used_pairs)
                    else:
                        # Restore
                        add_triple(rem_idx, current_triples, current_triple_set, used_pairs)
                else:
                    # System is empty, add random
                    add_idx = random.randint(0, n_triples - 1)
                    if is_valid_addition(add_idx, used_pairs):
                        add_triple(add_idx, current_triples, current_triple_set, used_pairs)
                        
            # Occasionally, try a "K-move" to escape local minima
            if steps % 20 == 0 and current_triples:
                # Remove 2 random triples
                to_remove = random.sample(current_triples, min(2, len(current_triples)))
                for idx in to_remove:
                    remove_triple(idx, current_triples, current_triple_set, used_pairs)
                
                # Try to fill with random valid triples
                for _ in range(10):
                    add_idx = random.randint(0, n_triples - 1)
                    if is_valid_addition(add_idx, used_pairs):
                        add_triple(add_idx, current_triples, current_triple_set, used_pairs)

    if best_solution is None:
        return [[0, 1, 2]]
        
    return [list(all_triples[i]) for i in best_solution]
