import time
import itertools
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit

    points = list(range(16))
    full_mask = (1 << 16) - 1
    all_triples = list(itertools.combinations(points, 3))
    n_triples = len(all_triples)
    
    # Precompute for each triple: its bitmask and its 3 pairs
    triple_info = []
    for t in all_triples:
        mask = (1 << t[0]) | (1 << t[1]) | (1 << t[2])
        pairs = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
        triple_info.append((t, mask, pairs))
    
    # Map pair -> list of triple indices containing that pair
    pair_to_triple_indices = defaultdict(list)
    for i, (t, mask, pairs) in enumerate(triple_info):
        for p in pairs:
            pair_to_triple_indices[p].append(i)
    
    def compute_alpha(triple_indices):
        """Compute independence number by checking all subsets from largest to smallest."""
        if not triple_indices:
            return 16
        
        masks = [triple_info[i][1] for i in triple_indices]
        n = len(masks)
        
        # Check subsets from size 16 down to 1
        for s_size in range(16, 0, -1):
            for subset in itertools.combinations(points, s_size):
                smask = 0
                for p in subset:
                    smask |= (1 << p)
                # Check if any triple is fully contained
                contained = False
                for m in masks:
                    if (smask & m) == m:
                        contained = True
                        break
                if not contained:
                    return s_size
        return 1

    def compute_alpha_fast(triple_indices):
        """Faster alpha computation using bitmask precomputation for small systems."""
        if not triple_indices:
            return 16
        
        masks = [triple_info[i][1] for i in triple_indices]
        n = len(masks)
        
        # For small n, check all subsets
        # For larger n, use a smarter approach
        if n <= 30:
            for s_size in range(16, 0, -1):
                for subset in itertools.combinations(points, s_size):
                    smask = 0
                    for p in subset:
                        smask |= (1 << p)
                    contained = False
                    for m in masks:
                        if (smask & m) == m:
                            contained = True
                            break
                    if not contained:
                        return s_size
            return 1
        else:
            # For large systems, use a heuristic: try to find large independent sets
            # by checking random subsets
            for s_size in range(16, 0, -1):
                # Try a few random subsets of this size
                attempts = min(100, 10 ** 6 // (1 << s_size))
                for _ in range(attempts):
                    subset = random.sample(points, s_size)
                    smask = 0
                    for p in subset:
                        smask |= (1 << p)
                    contained = False
                    for m in masks:
                        if (smask & m) == m:
                            contained = True
                            break
                    if not contained:
                        return s_size
            # If we didn't find one, do a full check for small sizes
            for s_size in range(8, 0, -1):
                for subset in itertools.combinations(points, s_size):
                    smask = 0
                    for p in subset:
                        smask |= (1 << p)
                    contained = False
                    for m in masks:
                        if (smask & m) == m:
                            contained = True
                            break
                    if not contained:
                        return s_size
            return 1

    # Greedy construction with different random orderings
    def greedy_construct(order):
        system = []
        used_pairs = set()
        for idx in order:
            t, mask, pairs = triple_info[idx]
            if all(p not in used_pairs for p in pairs):
                system.append(idx)
                for p in pairs:
                    used_pairs.add(p)
        return system

    # Initialize with several greedy constructions
    best_system = []
    best_alpha = 16
    
    n_init = 50
    for i in range(n_init):
        if time.time() - start_time > deadline - 60:
            break
        order = list(range(n_triples))
        random.shuffle(order)
        system = greedy_construct(order)
        alpha = compute_alpha_fast(system)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = system[:]
    
    # Simulated annealing
    current_system = best_system[:]
    current_alpha = best_alpha
    
    # Precompute for faster move evaluation
    def get_used_pairs(system):
        used = set()
        for idx in system:
            used.update(triple_info[idx][2])
        return used

    def can_add(system, triple_idx, used_pairs):
        _, _, pairs = triple_info[triple_idx]
        return all(p not in used_pairs for p in pairs)

    temperature = 1.0
    cooling = 0.9995
    min_temp = 0.01
    
    step_count = 0
    while time.time() - start_time < deadline:
        step_count += 1
        
        if temperature < min_temp:
            temperature = min_temp
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.4 and len(current_system) < 100:
            # Add a triple
            used_pairs = get_used_pairs(current_system)
            candidates = []
            for idx in range(n_triples):
                if can_add(current_system, idx, used_pairs):
                    candidates.append(idx)
            
            if candidates:
                new_idx = random.choice(candidates)
                new_system = current_system + [new_idx]
                new_alpha = compute_alpha_fast(new_system)
                
                delta = current_alpha - new_alpha  # positive means improvement
                if delta >= 0 or random.random() < (0.5 ** (delta / max(temperature, 0.01))):
                    current_system = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
        
        elif move_type < 0.8 and current_system:
            # Swap a triple
            remove_idx = random.randint(0, len(current_system) - 1)
            remove_triple = current_system[remove_idx]
            
            # Build used pairs excluding the one being removed
            used_pairs = set()
            for i, idx in enumerate(current_system):
                if i != remove_idx:
                    used_pairs.update(triple_info[idx][2])
            
            candidates = []
            for idx in range(n_triples):
                if idx != remove_triple and can_add(current_system, idx, used_pairs):
                    candidates.append(idx)
            
            if candidates:
                new_idx = random.choice(candidates)
                new_system = current_system[:]
                new_system[remove_idx] = new_idx
                new_alpha = compute_alpha_fast(new_system)
                
                delta = current_alpha - new_alpha
                if delta >= 0 or random.random() < (0.5 ** (delta / max(temperature, 0.01))):
                    current_system = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
        
        else:
            # Remove a triple
            if current_system:
                remove_idx = random.randint(0, len(current_system) - 1)
                new_system = current_system[:]
                new_system.pop(remove_idx)
                new_alpha = compute_alpha_fast(new_system)
                
                delta = current_alpha - new_alpha
                if delta >= 0 or random.random() < (0.5 ** (delta / max(temperature, 0.01))):
                    current_system = new_system
                    current_alpha = new_alpha
                    if current_alpha < best_alpha:
                        best_alpha = current_alpha
                        best_system = current_system[:]
        
        temperature *= cooling
        
        if best_alpha <= 5:
            break
    
    # Final verification and return
    result = [list(triple_info[idx][0]) for idx in best_system]
    
    # Verify linearity
    used_pairs = set()
    for t in result:
        pairs = (t[0], t[1]), (t[0], t[2]), (t[1], t[2])
        for p in pairs:
            if p in used_pairs:
                # This shouldn't happen, but if it does, remove the triple
                result.remove(t)
                break
            used_pairs.add(p)
    
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
