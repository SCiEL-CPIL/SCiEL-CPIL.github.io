import itertools
import time
import numpy as np
from numpy.random import default_rng

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin for 150s limit

    points = list(range(16))
    n_points = 16

    # Precompute all triples
    all_triples = list(itertools.combinations(points, 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    n_triples = len(all_triples)

    # Precompute transversals: each is a tuple of 8 triple indices
    transversals = []
    transversal_to_triples = []
    
    # We need to map each triple to the transversals it is part of
    # And each transversal to the triples it contains
    
    # Optimization: Use a set of frozensets for matchings to avoid duplicates
    # But there are only 16 points, C(16,6) = 8008 subsets.
    # For each subset of 6 points, there are 15 perfect matchings.
    # Total transversals = 8008 * 15 = 120120.
    
    # To speed up precomputation, we can iterate over all 6-subsets.
    
    triple_to_transversals = [[] for _ in range(n_triples)]
    
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        # Generate all 15 perfect matchings on 6 points
        # We can generate these by fixing the partner of the first element
        # and recursing, or just iterating permutations and deduplicating.
        # Deduplication via set of frozensets of pairs is fast enough for 8008 * 720 / 8?
        # Actually, 6! = 720 permutations. 720/8 = 90 distinct matchings? No, 15.
        # Let's just generate the 15 matchings explicitly for speed.
        
        # Matchings on indices 0..5
        # (0,1), (2,3), (4,5)
        # (0,2), (1,3), (4,5)
        # ...
        
        matchings = set()
        for perm in itertools.permutations(range(6)):
            m = frozenset([
                (sp[perm[0]], sp[perm[1]]),
                (sp[perm[2]], sp[perm[3]]),
                (sp[perm[4]], sp[perm[5]])
            ])
            matchings.add(m)
            
        for m in matchings:
            pairs = list(m)
            # Sort pairs for consistent ordering, then sort by first element
            pairs.sort()
            pairs.sort(key=lambda x: x[0])
            
            (a, b), (c, d), (e, f) = pairs
            
            t_indices = []
            for x in (a, b):
                for y in (c, d):
                    for z in (e, f):
                        t = tuple(sorted((x, y, z)))
                        t_indices.append(triple_to_idx[t])
            
            t_indices = tuple(sorted(t_indices))
            transversals.append(t_indices)
            transversal_to_triples.append(t_indices)
            
            for t_idx in t_indices:
                triple_to_transversals[t_idx].append(len(transversals) - 1)

    n_transversals = len(transversals)
    
    # Precompute bitmask for each transversal (8 bits set)
    # And for each triple, the list of transversal indices it touches.
    # We will use a boolean array for the current state for simplicity in updates,
    # but we can use numpy for vectorized checks if needed.
    
    # Let's use a standard list of booleans for the state, as n_triples is 560.
    # Updating counts is O(1) per transversal affected.
    
    # Initial state: All triples present.
    current_state = [True] * n_triples
    
    # Count how many triples in each transversal are present
    conflict_count = [0] * n_transversals
    for t_idx in range(n_triples):
        for trans_idx in triple_to_transversals[t_idx]:
            conflict_count[trans_idx] += 1
            
    def is_valid():
        # Check if any transversal has count 8
        # Optimized: keep track of max count or just check the array
        # Since we update incrementally, we can just check if any is 8.
        # But checking all 120k every time is slow.
        # Better: maintain a count of "bad" transversals (count == 8).
        # Let's refactor to maintain `num_conflicts`.
        pass

    # Refactor state management to be faster
    # State: list of bools
    # Conflict Count: list of ints
    # Num Conflicts: int (number of transversals with count 8)
    
    current_state = [True] * n_triples
    conflict_count = [0] * n_transversals
    num_conflicts = 0
    
    for t_idx in range(n_triples):
        for trans_idx in triple_to_transversals[t_idx]:
            conflict_count[trans_idx] += 1
            if conflict_count[trans_idx] == 8:
                num_conflicts += 1
                
    def add_triple(t_idx):
        nonlocal num_conflicts
        if current_state[t_idx]:
            return
        current_state[t_idx] = True
        for trans_idx in triple_to_transversals[t_idx]:
            conflict_count[trans_idx] += 1
            if conflict_count[trans_idx] == 8:
                num_conflicts += 1
            elif conflict_count[trans_idx] == 7:
                # No change in num_conflicts yet, just got closer
                pass
                
    def remove_triple(t_idx):
        nonlocal num_conflicts
        if not current_state[t_idx]:
            return
        current_state[t_idx] = False
        for trans_idx in triple_to_transversals[t_idx]:
            old_count = conflict_count[trans_idx]
            conflict_count[trans_idx] -= 1
            if old_count == 8:
                num_conflicts -= 1
            elif old_count == 7:
                # Now 6, no change in num_conflicts
                pass

    # Phase 1: Get a valid state by removing triples
    # We want to minimize removals.
    # Greedy: Find a conflicting transversal, remove a triple from it that is in the most other conflicting transversals.
    
    def find_best_removal():
        if num_conflicts == 0:
            return None
            
        # Find a conflicting transversal
        # We can just pick the first one, or a random one, or one with "high degree" of conflict.
        # For simplicity, let's find one.
        # To speed up, we can maintain a list of conflicting transversal indices.
        # But for now, let's just scan if it's not too slow.
        # 120k scan is ~0.01s in Python? Maybe.
        
        # Better: Maintain a set of conflicting transversals.
        # Let's just scan for the first one for now, and hope it's fast enough.
        # If not, we'll optimize.
        
        for trans_idx in range(n_transversals):
            if conflict_count[trans_idx] == 8:
                # Pick a triple from this transversal to remove
                # Score: how many other conflicting transversals it is in
                best_t = -1
                best_score = -1
                for t_idx in transversals[trans_idx]:
                    score = 0
                    for tj in triple_to_transversals[t_idx]:
                        if conflict_count[tj] == 8:
                            score += 1
                    if score > best_score:
                        best_score = score
                        best_t = t_idx
                return best_t
                
        return None

    while time.time() < deadline - 60:
        if num_conflicts == 0:
            break
        t_to_remove = find_best_removal()
        if t_to_remove is None:
            break
        remove_triple(t_to_remove)

    # Phase 2: Greedy Addition
    # Try to add back any triple that was removed and can be added without creating a conflict.
    # We need to know which triples were removed.
    # Actually, we can just iterate all triples not in current_state.
    
    removed_triples = [i for i in range(n_triples) if not current_state[i]]
    # Shuffle for randomization
    rng = default_rng(42)
    rng.shuffle(removed_triples)
    
    for t_idx in removed_triples:
        if time.time() > deadline - 30:
            break
        # Check if adding t_idx creates a conflict
        can_add = True
        for trans_idx in triple_to_transversals[t_idx]:
            if conflict_count[trans_idx] == 7:
                can_add = False
                break
        if can_add:
            add_triple(t_idx)

    # Phase 3: Local Search / Simulated Annealing
    # We have a valid state. We want to maximize the number of triples.
    # Move: Swap a present triple with an absent triple?
    # Or: Remove a present triple, add two absent triples?
    # Simplest: Try to add any absent triple. If conflict, remove one from the conflict, try to add others.
    
    best_count = sum(current_state)
    best_state = current_state.copy()
    best_conflict = conflict_count.copy()
    best_num_conflicts = num_conflicts
    
    # To allow backtracking, we need to save state.
    # Since n_triples is 560, copying the list is fast.
    
    def save_state():
        return (current_state.copy(), conflict_count.copy(), num_conflicts)
        
    def restore_state(state):
        nonlocal current_state, conflict_count, num_conflicts
        current_state, conflict_count, num_conflicts = state

    # Iterated Local Search
    for iteration in range(1000):
        if time.time() > deadline - 10:
            break
            
        # Perturbation: Remove k random present triples
        present = [i for i in range(n_triples) if current_state[i]]
        if len(present) < 5:
            break
            
        k = rng.integers(1, 4)
        to_remove = rng.choice(present, size=min(k, len(present)), replace=False)
        
        old_state = save_state()
        
        for t_idx in to_remove:
            remove_triple(t_idx)
            
        # Greedy Repair/Add
        # Try to add back the removed ones first? Or any absent one?
        # Let's try to add any absent one that fits.
        absent = [i for i in range(n_triples) if not current_state[i]]
        rng.shuffle(absent)
        
        for t_idx in absent:
            # Check if time is tight
            if time.time() > deadline - 10:
                break
            can_add = True
            for trans_idx in triple_to_transversals[t_idx]:
                if conflict_count[trans_idx] == 7:
                    can_add = False
                    break
            if can_add:
                add_triple(t_idx)
                
        new_count = sum(current_state)
        if new_count > best_count:
            best_count = new_count
            best_state = current_state.copy()
            best_conflict = conflict_count.copy()
            best_num_conflicts = num_conflicts
        else:
            # Revert
            restore_state(old_state)

    # Final sweep: Ensure no valid additions are missed
    # This is important because the local search might leave "holes" that can be filled.
    for t_idx in range(n_triples):
        if not best_state[t_idx]:
            # Check if we can add it to best_state
            # We need to simulate adding it to best_state's conflict counts
            # But we don't have the conflict counts for best_state easily unless we restore.
            # Let's just run a final greedy pass on the best_state found.
            pass
            
    # Let's restore the best state and do a final greedy add pass
    restore_state((best_state, best_conflict, best_num_conflicts))
    
    # Final Greedy Add
    # Iterate all absent triples
    absent = [i for i in range(n_triples) if not current_state[i]]
    rng.shuffle(absent)
    for t_idx in absent:
        can_add = True
        for trans_idx in triple_to_transversals[t_idx]:
            if conflict_count[trans_idx] == 7:
                can_add = False
                break
        if can_add:
            add_triple(t_idx)
            
    result = [list(all_triples[i]) for i in range(n_triples) if current_state[i]]
    
    # Final Verification (Safety)
    # Check if result is valid
    # Build a set of triples
    res_set = set(tuple(t) for t in result)
    
    # Check all transversals
    for trans in transversals:
        count = 0
        for t_idx in trans:
            if all_triples[t_idx] in res_set:
                count += 1
        if count == 8:
            # Invalid! Remove one triple from this transversal
            # This should not happen if our logic is correct, but it's a safety net.
            # Find a triple in this transversal that is in the fewest other transversals?
            # Or just the first one.
            for t_idx in trans:
                t_tuple = all_triples[t_idx]
                if t_tuple in res_set:
                    res_set.remove(t_tuple)
                    break
            # Re-verify? No, just break out and hope it's fixed.
            # Actually, we should rebuild the list.
            pass
            
    if res_set != set(tuple(t) for t in result):
        result = [list(t) for t in res_set]
        
    return result
