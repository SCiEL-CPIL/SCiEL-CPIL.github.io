import time
import itertools
import random
from collections import defaultdict

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

    # Precompute all subset masks for fast independent set checking?
    # There are 2^16 = 65536 subsets.
    # We can precompute a list of masks for each size k.
    subset_masks_by_size = {k: [] for k in range(17)}
    for mask in range(1, 1 << 16):
        k = bin(mask).count('1')
        subset_masks_by_size[k].append(mask)

    def get_alpha_and_witness(triples):
        """
        Compute the independence number and the witness set.
        Uses precomputed subset masks.
        """
        if not triples:
            return 16, (1 << 16) - 1
            
        t_masks = [triple_masks[t] for t in triples]
        
        # Check from largest k down
        for k in range(16, 0, -1):
            for mask in subset_masks_by_size[k]:
                is_independent = True
                for t_mask in t_masks:
                    if (mask & t_mask) == t_mask:
                        is_independent = False
                        break
                if is_independent:
                    return k, mask
        return 1, 1 # Fallback

    best_alpha = 16
    best_system = []
    
    # Initialize with random greedy packings
    # We can try more restarts if time permits
    num_restarts = 50 
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 5:
            break
            
        random.shuffle(all_triples)
        current = []
        used_pairs = set()
        for t in all_triples:
            if all(p not in used_pairs for p in triple_pairs[t]):
                current.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        
        alpha, witness = get_alpha_and_witness(current)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = current[:]
            if best_alpha <= 6: # Good lower bound
                break
    
    current_system = best_system[:]
    current_alpha = best_alpha
    current_witness = (1 << 16) - 1
    if current_system:
        current_alpha, current_witness = get_alpha_and_witness(current_system)

    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])

    # Local Search
    # We will use a simulated annealing-like approach
    # Accept moves that improve alpha.
    # Accept moves that keep alpha same with probability p (decaying).
    
    temp = 1.0
    cooling = 0.999
    
    while time.time() - start_time < deadline:
        move_type = random.choice(['swap', 'swap', 'add']) # Swaps are more important for maximal packings
        
        if move_type == 'add':
            used_pairs = set()
            for t in current_system:
                used_pairs.update(triple_pairs[t])
            
            # Find candidates that intersect witness
            # To speed up, just pick a random candidate from all valid ones
            # But bias towards those hitting the witness
            candidates = []
            for t in all_triples:
                if can_add(t, used_pairs):
                    # Calculate intersection with witness
                    inter = len([p for p in t if (current_witness >> p) & 1])
                    if inter > 0:
                        candidates.append((inter, t))
            
            if candidates:
                # Pick one with high intersection
                candidates.sort(key=lambda x: x[0], reverse=True)
                # Take top 5 random
                top_k = min(5, len(candidates))
                chosen = random.choice(candidates[:top_k])[1]
                
                new_system = current_system + [chosen]
                new_alpha, new_witness = get_alpha_and_witness(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < temp:
                        current_system = new_system
                        current_witness = new_witness

        elif move_type == 'swap':
            if not current_system:
                continue
                
            idx = random.randint(0, len(current_system) - 1)
            t_remove = current_system[idx]
            
            # Remove it temporarily to find candidates
            used_pairs = set()
            for i, t in enumerate(current_system):
                if i != idx:
                    used_pairs.update(triple_pairs[t])
            
            candidates = []
            for t in all_triples:
                if t == t_remove:
                    continue
                if can_add(t, used_pairs):
                    # Bias towards triples that hit the witness
                    inter = len([p for p in t if (current_witness >> p) & 1])
                    # Also consider if removing t_remove helps?
                    # Hard to know without checking.
                    candidates.append((inter, t))
            
            if candidates:
                # Sort by intersection
                candidates.sort(key=lambda x: x[0], reverse=True)
                top_k = min(10, len(candidates))
                chosen = random.choice(candidates[:top_k])[1]
                
                new_system = current_system[:]
                new_system[idx] = chosen
                new_alpha, new_witness = get_alpha_and_witness(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    current_witness = new_witness
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
                elif new_alpha == current_alpha:
                    if random.random() < temp:
                        current_system = new_system
                        current_witness = new_witness
                
        temp *= cooling
        if best_alpha <= 6:
            break

    return [list(t) for t in best_system]

if __name__ == "__main__":
    res = construct()
    print(res)
