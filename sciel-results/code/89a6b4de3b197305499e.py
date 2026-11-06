import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    n = 16
    # Generate all possible triples
    all_triples = list(combinations(range(n), 3))
    # Map triple to index for fast lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute conflicts: for each triple, which other triples form a K(2,2,2) with it?
    # A K(2,2,2) is defined by 3 disjoint pairs {a,a'}, {b,b'}, {c,c'}.
    # The 8 transversal triples are all combinations of picking one from each pair.
    # If we have a triple T = {a, b, c}, it is part of a K(2,2,2) if there exist 
    # distinct a', b', c' such that {a,a'}, {b,b'}, {c,c'} are disjoint pairs 
    # and the other 7 transversal triples are also present.
    
    # Actually, checking "if adding T creates a K(2,2,2)" is easier by checking:
    # Does T complete any K(2,2,2) that is otherwise already fully present?
    # A K(2,2,2) consists of 8 triples. If 7 are present, and we add the 8th, we create a forbidden structure.
    
    # Let's precompute for each triple T, the list of "potential partners" that would form a K(2,2,2) with T.
    # This is still complex. A more direct approach for local search:
    # Maintain a set of present triples.
    # To check if adding T is valid:
    # 1. Identify all K(2,2,2)s that contain T.
    # 2. For each such K, check if the other 7 triples are already present.
    #    If yes, adding T is invalid.
    
    # How to find K(2,2,2)s containing T={a,b,c}?
    # A K(2,2,2) containing {a,b,c} must have pairs {a, a'}, {b, b'}, {c, c'} where a', b', c' are distinct 
    # and not in {a,b,c}.
    # The other 7 triples are:
    # {a', b, c}, {a, b', c}, {a, b, c'},
    # {a', b', c}, {a', b, c'}, {a, b', c'},
    # {a', b', c'}
    
    # Precompute for each triple T, the list of sets of 7 other triples that would form a K(2,2,2) with T.
    # This list might be large.
    # Number of choices for a': 13 (from 1..15 excluding a, but must be distinct from b,c too? No, just distinct from a in the pair sense, but the set of points must be 6 distinct points).
    # The 6 points of a K(2,2,2) are distinct. So a', b', c' must be distinct from each other and from a, b, c.
    # So for T={a,b,c}, a' can be any of the 13 points not in T.
    # b' can be any of the 12 points not in T and not a'.
    # c' can be any of the 11 points not in T and not a' or b'.
    # Total K(2,2,2)s containing a specific T: 13 * 12 * 11 = 1716.
    # This is manageable.
    
    # For each triple T, store a list of tuples (idx1, idx2, ..., idx7) representing the indices of the other 7 triples.
    # If all 7 indices are present in the current solution, T cannot be added.
    
    conflict_map = {}
    for T in all_triples:
        a, b, c = T
        others = [x for x in range(n) if x not in (a, b, c)]
        conflicts = []
        # We need to pick a', b', c' from others such that they are distinct.
        # The 8 triples are:
        # T = (a, b, c)
        # t1 = (a', b, c)
        # t2 = (a, b', c)
        # t3 = (a, b, c')
        # t4 = (a', b', c)
        # t5 = (a', b, c')
        # t6 = (a, b', c')
        # t7 = (a', b', c')
        
        for a_prime in others:
            rem1 = [x for x in others if x != a_prime]
            for b_prime in rem1:
                rem2 = [x for x in rem1 if x != b_prime]
                for c_prime in rem2:
                    # Construct the 7 other triples
                    t1 = (min(a_prime, b), min(b, c), max(a_prime, b, c)) if False else tuple(sorted((a_prime, b, c)))
                    t2 = tuple(sorted((a, b_prime, c)))
                    t3 = tuple(sorted((a, b, c_prime)))
                    t4 = tuple(sorted((a_prime, b_prime, c)))
                    t5 = tuple(sorted((a_prime, b, c_prime)))
                    t6 = tuple(sorted((a, b_prime, c_prime)))
                    t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                    
                    # Get indices
                    idxs = [triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], 
                            triple_to_idx[t4], triple_to_idx[t5], triple_to_idx[t6], 
                            triple_to_idx[t7]]
                    conflicts.append(tuple(idxs))
        conflict_map[T] = conflicts
        
    # Initialize solution: Start with the seed (star at 0) or random?
    # Seed: All triples with 0.
    # Count: 15 * 14 / 2 = 105.
    # Let's try to improve from there.
    
    def get_seed_solution():
        sol = set()
        for a in range(1, 16):
            for b in range(a + 1, 16):
                sol.add((0, a, b))
        return sol
    
    current_sol = get_seed_solution()
    current_size = len(current_sol)
    best_sol = set(current_sol)
    best_size = current_size
    
    # Local Search: Try to add triples.
    # If we can't add any more, try to remove one and add two? Or just random restarts?
    # Given the time limit, a simple "greedy add" until stuck, then random restarts with different initial sets might work.
    # But the seed is already quite dense.
    
    # Let's do a greedy fill from the seed.
    def try_add_greedy(sol):
        added = True
        while added:
            added = False
            # Shuffle to avoid bias
            random.shuffle(all_triples)
            for T in all_triples:
                if T in sol:
                    continue
                # Check if T can be added
                conflicts = conflict_map[T]
                valid = True
                for conf_set in conflicts:
                    # Check if all 7 are in sol
                    # Using a set of indices for fast lookup?
                    # sol is a set of tuples. 
                    # It's faster to keep a set of indices for the solution.
                    pass
        return sol
    
    # Better data structure: Set of indices for current solution.
    current_idx_set = set()
    for T in current_sol:
        current_idx_set.add(triple_to_idx[T])
        
    best_idx_set = set(current_idx_set)
    best_size = len(current_idx_set)
    
    def is_valid_add(idx):
        T = all_triples[idx]
        conflicts = conflict_map[T]
        for conf_set in conflicts:
            # Check if all indices in conf_set are in current_idx_set
            # Since conf_set is a tuple of 7 indices
            if all(i in current_idx_set for i in conf_set):
                return False
        return True
    
    # Greedy fill
    while True:
        improved = False
        random.shuffle(all_triples)
        for idx in range(num_triples):
            if idx in current_idx_set:
                continue
            if is_valid_add(idx):
                current_idx_set.add(idx)
                improved = True
                if len(current_idx_set) > best_size:
                    best_size = len(current_idx_set)
                    best_idx_set = set(current_idx_set)
                    if time.time() > deadline - 5:
                        break
        if not improved or time.time() > deadline - 5:
            break
            
    # If we are still within time, do some random restarts or local swaps
    # Since greedy fill from seed is likely to reach a local optimum, 
    # maybe start with a random valid set and fill?
    # Or try to remove a triple and see if we can add 2?
    
    # Let's try a few random restarts with random initial sets if time permits.
    # But generating a random valid set is hard.
    # Alternative: Randomly remove a triple from current solution, try to add different ones.
    
    # Given the constraints and the strong seed, the greedy fill is the most likely to succeed quickly.
    # Let's just ensure we return the best found.
    
    # Convert best_idx_set back to list of triples
    result = [list(all_triples[i]) for i in best_idx_set]
    
    return result

# Keep the function name and signature exactly as requested
# The function above is named construct and returns a list of lists.

# Wait, the prompt asks for "COMPLETE program in a single python code block".
# I should put everything inside construct or define helpers outside if needed, 
# but since it's a single function interface, I'll define helpers inside or as module level.
# Module level is fine.

# Let's refine the timing and logic to be robust.

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    # Precompute conflict_map
    # This takes some time, but is done once.
    conflict_map = {}
    for T in all_triples:
        a, b, c = T
        others = [x for x in range(n) if x not in (a, b, c)]
        conflicts = []
        for a_prime in others:
            rem1 = [x for x in others if x != a_prime]
            for b_prime in rem1:
                rem2 = [x for x in rem1 if x != b_prime]
                for c_prime in rem2:
                    t1 = tuple(sorted((a_prime, b, c)))
                    t2 = tuple(sorted((a, b_prime, c)))
                    t3 = tuple(sorted((a, b, c_prime)))
                    t4 = tuple(sorted((a_prime, b_prime, c)))
                    t5 = tuple(sorted((a_prime, b, c_prime)))
                    t6 = tuple(sorted((a, b_prime, c_prime)))
                    t7 = tuple(sorted((a_prime, b_prime, c_prime)))
                    
                    idxs = (triple_to_idx[t1], triple_to_idx[t2], triple_to_idx[t3], 
                            triple_to_idx[t4], triple_to_idx[t5], triple_to_idx[t6], 
                            triple_to_idx[t7])
                    conflicts.append(idxs)
        conflict_map[T] = conflicts
        
    # Seed solution: Star at 0
    current_idx_set = set()
    for a in range(1, 16):
        for b in range(a + 1, 16):
            T = (0, a, b)
            current_idx_set.add(triple_to_idx[T])
            
    best_idx_set = set(current_idx_set)
    best_size = len(current_idx_set)
    
    # Helper to check validity
    def is_valid_add(idx):
        T = all_triples[idx]
        conflicts = conflict_map[T]
        for conf_set in conflicts:
            if all(i in current_idx_set for i in conf_set):
                return False
        return True
        
    # Greedy Fill
    # We iterate through all triples multiple times, adding any that are valid.
    # To avoid O(N^2) overhead in the inner loop, we just loop.
    # N = 560. 560 * 1716 checks is ~1M ops per pass. 
    # We can do many passes.
    
    while time.time() < deadline - 5:
        improved = False
        # Randomize order to break symmetries
        random.shuffle(all_triples)
        for idx in range(num_triples):
            if idx in current_idx_set:
                continue
            if is_valid_add(idx):
                current_idx_set.add(idx)
                improved = True
                if len(current_idx_set) > best_size:
                    best_size = len(current_idx_set)
                    best_idx_set = set(current_idx_set)
        if not improved:
            # If no improvement in a full pass, try a few random restarts or swaps?
            # For now, break out of greedy loop.
            break
            
    # If we have time, do some local search: remove one, try to add two.
    # This is complex to implement correctly and fast. 
    # Given the seed is strong, greedy fill might be sufficient to beat 105.
    # Let's just return the best found.
    
    result = [list(all_triples[i]) for i in best_idx_set]
    return result
