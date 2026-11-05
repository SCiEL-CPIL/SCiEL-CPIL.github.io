import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, aim for 140s
    deadline = start_time + 140
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    num_triples = len(all_triples)
    
    # Precompute index for O(1) lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # A config is defined by 3 disjoint pairs.
    # We need to map each triple to the configs it participates in.
    # And for each config, list the 8 triples.
    
    configs = []
    config_id_map = {}
    triples_in_config = [] # List of lists of 8 triple indices
    
    points = range(N)
    # Iterate over all ways to choose 6 points and partition them into 3 pairs
    # C(16,6) * 15 = 120120 configs. This is fast to precompute.
    
    for pts in itertools.combinations(points, 6):
        p0, p1, p2, p3, p4, p5 = pts
        pairs = [
            [(p0, p1), (p2, p3), (p4, p5)],
            [(p0, p1), (p2, p4), (p3, p5)],
            [(p0, p1), (p2, p5), (p3, p4)],
            [(p0, p2), (p1, p3), (p4, p5)],
            [(p0, p2), (p1, p4), (p3, p5)],
            [(p0, p2), (p1, p5), (p3, p4)],
            [(p0, p3), (p1, p2), (p4, p5)],
            [(p0, p3), (p1, p4), (p2, p5)],
            [(p0, p3), (p1, p5), (p2, p4)],
            [(p0, p4), (p1, p2), (p3, p5)],
            [(p0, p4), (p1, p3), (p2, p5)],
            [(p0, p4), (p1, p5), (p2, p3)],
            [(p0, p5), (p1, p2), (p3, p4)],
            [(p0, p5), (p1, p3), (p2, p4)],
            [(p0, p5), (p1, p4), (p2, p3)]
        ]
        for pair_set in pairs:
            norm_pairs = tuple(sorted(tuple(sorted(p)) for p in pair_set))
            if norm_pairs not in config_id_map:
                cid = len(configs)
                config_id_map[norm_pairs] = cid
                configs.append(norm_pairs)
                
                (a,b), (c,d), (e,f) = norm_pairs
                tris = []
                for x in (a,b):
                    for y in (c,d):
                        for z in (e,f):
                            t = tuple(sorted((x,y,z)))
                            tris.append(triple_to_idx[t])
                triples_in_config.append(tris)
            else:
                pass
                
    num_configs = len(configs)
    
    # Reverse map: triple_index -> list of config_ids
    triple_to_configs = [[] for _ in range(num_triples)]
    for cid, tris in enumerate(triples_in_config):
        for t_idx in tris:
            triple_to_configs[t_idx].append(cid)
            
    # State
    present = np.zeros(num_triples, dtype=bool)
    config_counts = np.zeros(num_configs, dtype=np.int8)
    
    def add_triple(t_idx):
        present[t_idx] = True
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] += 1
            if config_counts[cid] == 8:
                return False
        return True
        
    def remove_triple(t_idx):
        present[t_idx] = False
        for cid in triple_to_configs[t_idx]:
            config_counts[cid] -= 1
        return True

    def is_valid_addition(t_idx):
        if present[t_idx]:
            return False
        for cid in triple_to_configs[t_idx]:
            if config_counts[cid] == 7:
                return False
        return True

    def greedy_add(current_set_indices, time_limit):
        """
        Attempts to add as many triples as possible from the current state.
        Returns the new set of indices and the count.
        """
        # Reset state based on current_set_indices
        present[:] = False
        config_counts[:] = 0
        for idx in current_set_indices:
            add_triple(idx)
            
        count = len(current_set_indices)
        # Create a list of missing triples
        missing = [i for i in range(num_triples) if not present[i]]
        
        # Greedy add loop
        # To make it faster, we can shuffle and add in one pass, then repeat?
        # Or just iterate until no more can be added.
        # Given the size (560 triples), a full scan is cheap.
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(missing)
            for t_idx in missing:
                if is_valid_addition(t_idx):
                    add_triple(t_idx)
                    current_set_indices.append(t_idx)
                    count += 1
                    improved = True
                    # Remove from missing list? 
                    # It's easier to just rebuild missing or mark it.
                    # Since we are modifying current_set_indices, we can just filter at the end if needed, 
                    # but for efficiency in the loop, let's just continue.
                    # Actually, if we add it, it's no longer missing.
                    # We can't easily remove from 'missing' while iterating.
                    # Let's just re-create missing if we added many?
                    # Or just check 'present' in the loop.
                    # The loop above shuffles 'missing' which contains indices that WERE missing.
                    # If we add one, subsequent checks on that index in the same pass will see present=True (handled by is_valid_addition).
                    # But we will keep checking it unnecessarily.
                    # For 560 items, this is fine.
                    
        return current_set_indices, count

    def perturb(current_set_indices, k):
        """
        Randomly remove k triples from the set.
        """
        if len(current_set_indices) < k:
            return current_set_indices[:]
        
        # Copy to avoid modifying the original directly if we need to revert
        # But ILS usually works on a working copy.
        perturbed = current_set_indices[:]
        
        # Remove k random elements
        # Efficient removal:
        for _ in range(k):
            if not perturbed:
                break
            idx_to_remove = random.randrange(len(perturbed))
            perturbed.pop(idx_to_remove)
            
        return perturbed

    # Main ILS Loop
    best_count = 0
    best_set = []
    
    # Initial solution: Random Greedy
    # Start with empty set, greedy add
    initial_set, initial_count = greedy_add([], deadline - 10)
    best_count = initial_count
    best_set = initial_set[:]
    
    # Simulated Annealing parameters
    T = 10.0 # Initial temperature
    cooling_rate = 0.999
    min_T = 0.1
    
    current_set = best_set[:]
    current_count = best_count
    
    while time.time() < deadline - 5:
        # Perturb
        # The perturbation size can vary. 
        # Removing 1-5 triples is a good start.
        k = random.randint(1, 5)
        perturbed_set = perturb(current_set, k)
        
        # Local Search (Greedy Add)
        # Set a tight time limit for this phase, e.g., 0.1s
        phase_deadline = time.time() + 0.1
        if time.time() > phase_deadline:
            phase_deadline = deadline - 2
            
        new_set, new_count = greedy_add(perturbed_set, phase_deadline)
        
        # Acceptance Criteria
        delta = new_count - current_count
        
        if delta > 0:
            # Improvement
            current_set = new_set
            current_count = new_count
            
            if current_count > best_count:
                best_count = current_count
                best_set = current_set[:]
                
        else:
            # Non-improvement
            # Accept with probability exp(-delta / T)
            if T > min_T:
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    current_set = new_set
                    current_count = new_count
            
        T *= cooling_rate
        
        # If we hit a local optimum and T is low, maybe kick it harder?
        # The random k in perturb handles some of this.
        
    # Convert indices back to triples
    result = [all_triples[i] for i in best_set]
    
    return result

