import time
import random
import itertools
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    n = 16
    points = list(range(n))
    
    # Generate all possible triples
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)  # 560
    
    # Precompute pair indices for each triple
    # There are C(16,2) = 120 pairs
    pair_to_idx = {}
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            pair_to_idx[(i,j)] = idx
            idx += 1
    num_pairs = idx  # 120
    
    # For each triple, store the 3 pair indices it covers
    triple_pairs = []
    for t in all_triples:
        a, b, c = t
        triple_pairs.append([pair_to_idx[(a,b)], pair_to_idx[(a,c)], pair_to_idx[(b,c)]])
    
    # Precompute bitmasks for triples
    triple_masks = []
    for t in all_triples:
        mask = 0
        for p in t:
            mask |= (1 << p)
        triple_masks.append(mask)
    
    # Fast independence number check using bit operations
    # We'll use a clever approach: for a given system, check if there exists an independent set of size k
    # by iterating over all C(16,k) subsets
    
    def has_independent_set_of_size(system_set, k):
        """Check if there's a subset of size k with no triple fully contained in it."""
        # system_set is a set of triple indices
        # Precompute which triples are in the system as a set for O(1) lookup
        if k <= 0:
            return True
        if k > n:
            return False
        
        # For small k, enumerate all subsets of size k
        # C(16,8) = 12870, C(16,9) = 11440, C(16,10) = 8008
        # This is feasible for k up to ~10
        
        for subset in itertools.combinations(points, k):
            subset_mask = 0
            for p in subset:
                subset_mask |= (1 << p)
            
            # Check if any triple in the system is fully contained in this subset
            contained = False
            for t_idx in system_set:
                if (triple_masks[t_idx] & ~subset_mask) == 0:
                    # Triple is fully contained in subset
                    contained = True
                    break
            if not contained:
                return True
        return False
    
    def calc_alpha_fast(system_indices):
        """Calculate independence number using binary search + fast check."""
        system_set = set(system_indices)
        
        # Binary search for the largest k such that there exists an independent set of size k
        lo, hi = 1, n
        best = 1
        
        # First check if alpha >= 8 (common threshold)
        # Actually, let's just do a direct search from high to low
        for k in range(n, 0, -1):
            if has_independent_set_of_size(system_set, k):
                return k
        return 1
    
    # Greedy construction with randomization
    def greedy_construct(random_seed=None):
        if random_seed is not None:
            rng = random.Random(random_seed)
        else:
            rng = random.Random()
        
        indices = list(range(num_triples))
        rng.shuffle(indices)
        
        selected = []
        used_pairs = [False] * num_pairs
        
        for t_idx in indices:
            pairs = triple_pairs[t_idx]
            conflict = False
            for p in pairs:
                if used_pairs[p]:
                    conflict = True
                    break
            if not conflict:
                selected.append(t_idx)
                for p in pairs:
                    used_pairs[p] = True
        
        return selected
    
    # Local search: swap triples to improve alpha
    def local_search(initial_selection, time_limit=5.0):
        """Improve the selection by swapping triples."""
        start_ls = time.time()
        current = list(initial_selection)
        current_set = set(current)
        
        # Build used pairs
        used_pairs_count = [0] * num_pairs
        for t_idx in current:
            for p in triple_pairs[t_idx]:
                used_pairs_count[p] += 1
        
        # Compute initial alpha
        current_alpha = calc_alpha_fast(current)
        
        improved = True
        no_improve_count = 0
        
        while improved and (time.time() - start_ls) < time_limit:
            improved = False
            no_improve_count = 0
            
            # Try removing one triple and adding another
            # Or swapping one triple for another
            
            # Strategy: Try to find a triple that, when added (after removing a conflicting one), reduces alpha
            # Simpler: random restart within local search - try random swaps
            
            num_swaps = 100
            for _ in range(num_swaps):
                if (time.time() - start_ls) > time_limit:
                    break
                
                # Pick a random selected triple to potentially remove
                if not current:
                    break
                remove_idx = random.randrange(len(current))
                remove_t = current[remove_idx]
                
                # Find a candidate triple to add
                # It must not conflict with the remaining triples
                candidate_found = False
                for attempt in range(50):
                    add_t = random.randrange(num_triples)
                    if add_t in current_set:
                        continue
                    
                    # Check if add_t conflicts with any triple in current except remove_t
                    pairs_add = triple_pairs[add_t]
                    conflict = False
                    for t_idx in current:
                        if t_idx == remove_t:
                            continue
                        for p in triple_pairs[t_idx]:
                            if p in pairs_add:
                                conflict = True
                                break
                        if conflict:
                            break
                    if not conflict:
                        # Try the swap
                        new_current = [t for i, t in enumerate(current) if i != remove_idx]
                        new_current.append(add_t)
                        
                        # Quick alpha check
                        new_alpha = calc_alpha_fast(new_current)
                        
                        if new_alpha < current_alpha:
                            current = new_current
                            current_set = set(current)
                            current_alpha = new_alpha
                            improved = True
                            break
                        elif new_alpha == current_alpha and random.random() < 0.1:
                            # Accept with small probability to escape local minima
                            current = new_current
                            current_set = set(current)
                            improved = True
                            break
                    candidate_found = True
                    break
                if not candidate_found:
                    continue
        
        return current, current_alpha
    
    # Main search loop
    best_system = None
    best_alpha = n + 1
    
    # First, do a few greedy constructions and evaluate
    num_initial = 20
    for i in range(num_initial):
        if time.time() > deadline - 30.0:
            break
        
        selection = greedy_construct(random_seed=i)
        alpha = calc_alpha_fast(selection)
        
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = selection[:]
            
            if best_alpha <= 6:
                break
    
    # If we haven't found a good solution, try local search from multiple starts
    if best_alpha > 6:
        num_ls_starts = 50
        for i in range(num_ls_starts):
            if time.time() > deadline - 10.0:
                break
            
            # Create a diverse starting point
            if i < num_initial and best_system is not None:
                start = list(best_system)
            else:
                start = greedy_construct(random_seed=100 + i)
            
            # Run local search
            improved, alpha = local_search(start, time_limit=3.0)
            
            if alpha < best_alpha:
                best_alpha = alpha
                best_system = improved[:]
                
                if best_alpha <= 6:
                    break
    
    # Final polish: try more aggressive local search from the best solution
    if best_system is not None and best_alpha > 5 and time.time() < deadline - 5.0:
        improved, alpha = local_search(best_system, time_limit=10.0)
        if alpha < best_alpha:
            best_alpha = alpha
            best_system = improved[:]
    
    # Convert indices back to triples
    if best_system is None:
        # Fallback: simple partition
        return [[3*i, 3*i+1, 3*i+2] for i in range(5)]
    
    return [list(all_triples[t]) for t in best_system]

if __name__ == "__main__":
    result = construct()
    print(f"Found {len(result)} triples")
