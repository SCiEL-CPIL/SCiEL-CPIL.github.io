import time
import random
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    N = 16
    points = list(range(N))
    all_triples = list(itertools.combinations(points, 3))
    
    # Precompute pairs for each triple
    triple_pairs = {t: set(itertools.combinations(t, 2)) for t in all_triples}
    
    # Precompute triple index for fast lookup
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    # Helper to compute alpha using bitset logic
    # We represent a set of points as a bitmask (16 bits).
    # For a given set of triples, we want to find the largest subset of points
    # that contains no triple.
    # This is equivalent to finding the largest independent set in a 3-uniform hypergraph.
    
    # Optimization: 
    # Instead of checking all subsets, we can use a recursive backtracking with pruning,
    # or simply iterate k from 16 down to 3 and check if an independent set of size k exists.
    # C(16, 8) is ~12k, C(16, 9) is ~11k. This is very fast.
    
    def has_independent_set(triples, k):
        # Check if there is an independent set of size k
        # We iterate all combinations of size k
        # For each combination, check if any triple is a subset of it.
        # To speed this up, we can precompute a set of "blocked" masks?
        # Or just check against the list of triples.
        
        # Optimization: If k < 3, return True (always possible)
        if k < 3:
            return True
            
        # Optimization: If the number of triples is 0, return True
        if not triples:
            return True
            
        # Heuristic: If k is large, it's likely to contain a triple if the system is dense.
        # If k is small, it's likely to be independent if the system is sparse.
        
        # We can use a simple filter:
        # A set S is independent if for all t in triples, not (t subset S).
        
        # Let's use a bitset representation for the triples for faster subset check?
        # Or just set intersection.
        
        # Since we call this many times, let's optimize the inner loop.
        # We can precompute the triples as a list of sets?
        # triple_sets = [set(t) for t in triples]
        
        # Better: Represent each triple as a bitmask.
        # Then t subset S is equivalent to (t_mask & S_mask) == t_mask.
        
        # Let's precompute triple masks for the current set of triples?
        # No, the set of triples changes.
        
        # Let's just do the combination check.
        for subset in itertools.combinations(points, k):
            subset_set = set(subset)
            is_ind = True
            for t in triples:
                if t[0] in subset_set and t[1] in subset_set and t[2] in subset_set:
                    is_ind = False
                    break
            if is_ind:
                return True
        return False

    def get_alpha(triples):
        # Find the largest k such that has_independent_set(triples, k) is True
        # We know alpha >= 2.
        # Check from high to low? Or low to high?
        # If we check from high (16) down, we stop early if alpha is large.
        # If we check from low (3) up, we stop early if alpha is small.
        # We want to MINIMIZE alpha, so we expect it to be small.
        # So checking 3, 4, 5... is better for early termination on "good" systems.
        
        # However, if alpha is 16, we check 3..16 (14 checks).
        # If alpha is 3, we check 3 (1 check).
        
        # Let's check from 3 upwards.
        # But wait, if has_independent_set(k) is True, it doesn't guarantee k+1 is True?
        # Actually, if there is an independent set of size k, there is one of size k-1.
        # So the property "has independent set of size k" is monotonic decreasing with k?
        # No. If there is an independent set of size k, there is one of size k-1.
        # So if has_ind(k) is True, has_ind(k-1) is True.
        # This means the set of k for which has_ind(k) is True is {2, 3, ..., alpha}.
        # So we can binary search? Or just iterate up.
        
        # Since we want to find alpha, and we suspect it's small, let's iterate up.
        # But if it's large, we do many checks.
        # C(16, 8) is small. Let's just iterate 3 to 16.
        
        # Wait, if has_ind(8) is False, then has_ind(9) is False.
        # So we can stop as soon as we find a k where has_ind(k) is False.
        # The previous k is alpha.
        
        max_k = 2
        for k in range(3, 17):
            if has_independent_set(triples, k):
                max_k = k
            else:
                break
        return max_k

    # Strategy:
    # 1. Generate a good initial packing.
    # 2. Use Local Search (Hill Climbing / Simulated Annealing) to minimize alpha.
    
    # Initial Packing:
    # Greedy with randomization.
    # To get a denser packing, we can try to maximize the number of triples first,
    # or just pick random valid triples.
    
    def generate_initial_system():
        system = []
        used_pairs = set()
        available = all_triples[:]
        random.shuffle(available)
        
        for t in available:
            if all(p not in used_pairs for p in triple_pairs[t]):
                system.append(t)
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        return system

    best_system = []
    best_alpha = 16
    
    # Phase 1: Multiple restarts to find a good starting point
    # We will run several restarts, keeping the best alpha found.
    
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() - start_time > deadline - 60: # Leave time for local search
            break
            
        current_system = generate_initial_system()
        current_alpha = get_alpha(current_system)
        
        if current_alpha < best_alpha:
            best_alpha = current_alpha
            best_system = current_system[:]
            
        if best_alpha <= 4: # Theoretical lower bound for STS-like structures on 16? 
                           # Actually, alpha=3 is possible? 
                           # If alpha=3, every 4-set has a triple.
                           # This is a very strong condition.
            break

    # Phase 2: Local Search
    # Move: Remove a triple, add a different valid triple.
    # Objective: Minimize alpha.
    
    current_system = best_system[:]
    current_alpha = best_alpha
    
    # Precompute for speed?
    # We need to check if a triple can be added given used_pairs.
    
    def can_add(triple, used_pairs):
        return all(p not in used_pairs for p in triple_pairs[triple])

    # We will perform a simple hill climbing with random restarts within the search.
    # Or just continuous hill climbing until timeout.
    
    no_improvement_count = 0
    max_no_improvement = 500
    
    while time.time() - start_time < deadline:
        if not current_system:
            break
            
        # Pick a random triple to remove
        idx = random.randint(0, len(current_system) - 1)
        t_remove = current_system[idx]
        
        # Calculate used_pairs after removal
        used_pairs = set()
        for t in current_system:
            if t is not t_remove: # Note: tuples are immutable, identity works here if we replace by index
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        # Actually, simpler:
        used_pairs = set()
        for i, t in enumerate(current_system):
            if i != idx:
                for p in triple_pairs[t]:
                    used_pairs.add(p)
        
        # Find candidates to add
        # Candidates are triples not in current_system (except the removed one? No, we want a *different* one usually, or same? 
        # If we add the same one, no change. Let's pick a different one.)
        # We can just pick a random triple from all_triples and check if it's valid.
        # To be more efficient, we could look at triples that share a pair with t_remove?
        # Not necessarily.
        
        # Let's try a few random replacements
        improved = False
        for _ in range(10): # Try 10 candidates
            t_new = random.choice(all_triples)
            
            # Ensure t_new is not the same as t_remove (to force a change)
            # Actually, if t_new == t_remove, alpha doesn't change.
            # If t_new is already in current_system (and not t_remove), it will conflict?
            # No, we check used_pairs. If t_new is in system, its pairs are in used_pairs.
            # So can_add will return False unless t_new is t_remove.
            # So we naturally avoid duplicates.
            
            if can_add(t_new, used_pairs):
                # Apply move
                new_system = current_system[:]
                new_system[idx] = t_new
                
                # Calculate new alpha
                new_alpha = get_alpha(new_system)
                
                if new_alpha < current_alpha:
                    current_system = new_system
                    current_alpha = new_alpha
                    improved = True
                    no_improvement_count = 0
                    break
                elif new_alpha == current_alpha:
                    # Accept equal to explore?
                    # Let's accept with some probability to escape local minima?
                    # Or just accept to move around.
                    # If we always accept equal, we might cycle.
                    # Let's accept equal with probability 0.5?
                    # Or just accept to allow further moves.
                    if random.random() < 0.5:
                        current_system = new_system
                        # current_alpha stays same
                        # We don't break, we keep looking for improvement in this iteration?
                        # Or accept and restart loop?
                        # Let's accept and break to restart selection.
                        no_improvement_count += 1
                        break
                else:
                    # Worse, reject
                    no_improvement_count += 1
        
        if current_alpha <= 3:
            break
            
        if no_improvement_count > max_no_improvement:
            # Restart from a new random point or perturb current
            # Let's do a random restart
            current_system = generate_initial_system()
            current_alpha = get_alpha(current_system)
            no_improvement_count = 0
            
            if current_alpha < best_alpha:
                best_alpha = current_alpha
                best_system = current_system[:]
            else:
                # Keep the best found so far?
                # If current is worse than best, maybe we should go back to best?
                # Or continue from current?
                # Let's go back to best to ensure we don't degrade.
                current_system = best_system[:]
                current_alpha = best_alpha

    # Ensure we return the best system found
    if best_alpha < current_alpha:
        return [list(t) for t in best_system]
    else:
        return [list(t) for t in current_system]

# Run construct to test
if __name__ == "__main__":
    res = construct()
    print(res)
    # Calculate alpha for verification
    def calc_alpha(triples):
        points = list(range(16))
        def has_ind(k):
            for subset in itertools.combinations(points, k):
                subset_set = set(subset)
                is_ind = True
                for t in triples:
                    if t[0] in subset_set and t[1] in subset_set and t[2] in subset_set:
                        is_ind = False
                        break
                if is_ind:
                    return True
            return False
        max_k = 2
        for k in range(3, 17):
            if has_ind(k):
                max_k = k
            else:
                break
        return max_k
    print("Alpha:", calc_alpha(res))
