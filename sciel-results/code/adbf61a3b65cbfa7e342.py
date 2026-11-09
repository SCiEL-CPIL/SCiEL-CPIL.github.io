import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate all 5-subsets and 6-subsets
    points = range(12)
    all_5sets = list(combinations(points, 5))
    all_6sets = list(combinations(points, 6))
    
    # Map 5-subset to list of 6-subsets containing it
    blocks_containing_5set = {s: [] for s in all_5sets}
    # Map 6-subset to list of 5-subsets it contains
    block_5sets = {}
    
    for b in all_6sets:
        subsets_5 = list(combinations(b, 5))
        block_5sets[b] = subsets_5
        for s in subsets_5:
            blocks_containing_5set[s].append(b)
            
    total_5sets = len(all_5sets)
    covered = set()
    chosen = []
    
    # Heuristic: Precompute intersection sizes or just use set operations
    # To speed up compatibility check, we can maintain a set of "blocked" 5-sets?
    # No, the constraint is on the blocks themselves.
    # Two blocks are incompatible if they share a 5-subset.
    # So if we have chosen blocks C1, C2, ..., Ck, a new block B is compatible
    # if B does not share any 5-subset with any Ci.
    # This is equivalent to: for all Ci, |B & Ci| <= 4.
    
    def is_compatible(b, chosen_blocks):
        for cb in chosen_blocks:
            if len(set(b) & set(cb)) > 4:
                return False
        return True
        
    # To make is_compatible faster, we can precompute the set of 5-subsets covered by chosen blocks?
    # No, that's not sufficient. Two blocks can be incompatible without sharing a covered 5-subset?
    # Wait. The condition is "no 5-subset lies in two blocks".
    # This means for any pair of distinct blocks B1, B2 in the final set, they must not share a 5-subset.
    # So yes, B must be compatible with ALL previously chosen blocks.
    
    # Optimization:
    # Instead of checking against all chosen blocks, we can check if any 5-subset of B is in the "forbidden" set?
    # No, the "forbidden" set is not just the covered 5-subsets.
    # If B1 and B2 share a 5-subset S, then S is in B1 and S is in B2.
    # If we have already chosen B1, then any 5-subset of B1 is "covered".
    # If B2 contains any 5-subset of B1, then B2 is incompatible.
    # So, B is compatible with B1 iff B contains NO 5-subset of B1.
    # This is equivalent to |B & B1| <= 4.
    
    # So the check `is_compatible` is correct.
    # To speed it up, we can precompute the 5-subsets of each chosen block?
    # We already have `block_5sets`.
    # B is incompatible with CB if any 5-subset of B is in `block_5sets[CB]`.
    # Since `block_5sets[CB]` has 6 elements, we can check:
    # `any(s in block_5sets[CB] for s in block_5sets[B])`
    # This is 6*6 = 36 set lookups.
    
    def is_compatible_fast(b, chosen_blocks):
        b_5sets = block_5sets[b]
        for cb in chosen_blocks:
            cb_5sets = block_5sets[cb]
            # Check if any 5-subset of b is in cb_5sets
            # Since they are tuples, we can use set intersection or simple loop
            # Convert to set for O(1) lookup? 
            # Precompute sets for block_5sets?
            pass
            
    # Let's precompute sets for faster intersection
    block_5sets_set = {b: set(block_5sets[b]) for b in all_6sets}
    
    def is_compatible_v2(b, chosen_blocks):
        b_5sets = block_5sets_set[b]
        for cb in chosen_blocks:
            cb_5sets = block_5sets_set[cb]
            if b_5sets & cb_5sets:
                return False
        return True

    # CSP Solver
    def solve(covered, chosen):
        if time.time() > deadline:
            return None
            
        if len(covered) == total_5sets:
            return chosen[:]
            
        # Find the uncovered 5-subset with the fewest compatible candidates
        best_s = None
        best_candidates = None
        min_count = float('inf')
        
        for s in all_5sets:
            if s in covered:
                continue
            candidates = []
            for b in blocks_containing_5set[s]:
                if is_compatible_v2(b, chosen):
                    candidates.append(b)
            if len(candidates) < min_count:
                min_count = len(candidates)
                best_s = s
                best_candidates = candidates
                if min_count == 0:
                    return None
                if min_count == 1:
                    break # Forced choice, no need to check others
            
        if best_s is None:
            return None # Should not happen if covered < total
            
        # Heuristic: Order candidates by degree or just random?
        # Randomization helps escape local minima, but this is exact search.
        # Let's shuffle to vary the path
        random.shuffle(best_candidates)
        
        for b in best_candidates:
            new_covered = covered.union(block_5sets_set[b])
            result = solve(new_covered, chosen + [b])
            if result is not None:
                return result
        return None

    # Run solver
    # To handle potential recursion depth or time issues, we can limit restarts
    # But the CSP should be tight enough.
    
    # Initial covered is empty
    result = solve(set(), [])
    
    if result is None:
        # Fallback: Greedy
        covered = set()
        blocks = []
        # Shuffle blocks to avoid lex bias
        shuffled_blocks = all_6sets[:]
        random.shuffle(shuffled_blocks)
        for b in shuffled_blocks:
            if is_compatible_v2(b, blocks):
                blocks.append(b)
                covered.update(block_5sets_set[b])
        result = blocks
        
    return [list(b) for b in result]
