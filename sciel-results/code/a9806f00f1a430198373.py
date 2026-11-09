# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s kill

    n = 256
    max_size = 64

    def is_sidon(s):
        """Check if s is a Sidon set (all pair sums including a+a are unique)."""
        sums = [a + b for a, b in combinations_with_replacement(s, 2)]
        return len(sums) == len(set(sums))

    def get_diffs(s):
        """Get the set of nonzero differences (positive and negative)."""
        diffs = set()
        for a, b in combinations(s, 2):
            diffs.add(b - a)
            diffs.add(a - b)
        return diffs

    def greedy_build(order, diff_sets):
        """
        Build two Sidon sets by processing elements in 'order'.
        diff_sets[0] and diff_sets[1] track differences used by each set.
        Returns (A, B).
        """
        sets = [[], []]
        used_diffs = [set(), set()]
        
        for x in order:
            for side in (0, 1):
                if len(sets[side]) >= max_size:
                    continue
                candidate = sets[side] + [x]
                # Check Sidon condition
                if not is_sidon(candidate):
                    continue
                # Check difference disjointness
                new_diffs = get_diffs(candidate) - used_diffs[side]
                # Actually, we need to check that new_diffs don't conflict with the OTHER set's diffs
                # And that candidate's diffs don't conflict with the other set's existing diffs
                other_diffs = used_diffs[1 - side]
                if new_diffs & other_diffs:
                    continue
                # Also need to ensure that within the same set, the new element doesn't create 
                # differences that conflict with... wait, the Sidon condition handles internal uniqueness.
                # The difference condition is: (diffs(A) ∪ diffs(B)) has no overlap between A's diffs and B's diffs.
                # So we just need: get_diffs(candidate) & used_diffs[1-side] == empty set.
                # But we also need to make sure that the new differences don't conflict with future additions? No, that's handled when future additions happen.
                # However, we must ensure that the *entire* difference set of the candidate is disjoint from the other set's difference set.
                # Since used_diffs[side] is the current diff set of sets[side], and candidate = sets[side] + [x],
                # get_diffs(candidate) = used_diffs[side] ∪ {x - a, a - x for a in sets[side]}.
                # So we just need to check that these new differences don't intersect with used_diffs[1-side].
                new_diffs_only = {x - a for a in sets[side]} | {a - x for a in sets[side]}
                if new_diffs_only & used_diffs[1 - side]:
                    continue
                
                # Also, we must ensure that the existing differences of this set don't conflict with the other set?
                # That was already ensured when we added them. So only the new differences matter for cross-conflict.
                # But wait, what if the other set is empty? Then used_diffs[1-side] is empty, so no conflict.
                
                # One more check: the new differences must also not conflict with the *existing* differences of the other set?
                # Yes, that's exactly what `new_diffs_only & used_diffs[1 - side]` checks.
                
                sets[side] = candidate
                used_diffs[side] = get_diffs(candidate)
                break
        
        return sets[0], sets[1]

    def local_search(A, B, time_limit):
        """
        Try to improve A and B by swapping elements or adding/removing.
        """
        A = list(A)
        B = list(B)
        best_score = len(A) + len(B)
        best_A, best_B = A[:], B[:]
        
        end_time = time.time() + time_limit
        no_improve_time = 0
        last_improve = time.time()
        
        while time.time() < end_time:
            improved = False
            # Try adding a new element to A or B
            for side in (0, 1):
                S = A if side == 0 else B
                if len(S) >= max_size:
                    continue
                other = B if side == 0 else A
                # Try to add a random unused element
                used_elements = set(A) | set(B)
                candidates = [x for x in range(1, n + 1) if x not in used_elements]
                random.shuffle(candidates)
                for x in candidates[:50]:  # Try up to 50 random candidates
                    candidate = S + [x]
                    if not is_sidon(candidate):
                        continue
                    new_diffs = get_diffs(candidate)
                    other_diffs = get_diffs(other)
                    if new_diffs & other_diffs:
                        continue
                    # Success
                    if side == 0:
                        A = candidate
                    else:
                        B = candidate
                    best_score = len(A) + len(B)
                    best_A, best_B = A[:], B[:]
                    improved = True
                    break
                if improved:
                    break
            
            if not improved:
                # Try swapping elements
                # This is more complex, so we'll skip for now or do simple removals
                # Try removing an element from A and adding to B, or vice versa
                pass
            
            if not improved:
                no_improve_time += 1
                if no_improve_time > 50:
                    break
            else:
                no_improve_time = 0
                last_improve = time.time()
            
            if time.time() - last_improve > 10:
                break
        
        return best_A, best_B

    # Main search loop
    best_A, best_B = [], []
    best_score = 0
    
    # First, do a greedy build with natural order
    A, B = greedy_build(range(1, n + 1), None)
    best_A, best_B = A, B
    best_score = len(A) + len(B)
    
    # Try multiple random orders
    while time.time() < deadline:
        order = list(range(1, n + 1))
        random.shuffle(order)
        A, B = greedy_build(order, None)
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A, best_B = A[:], B[:]
        
        # If we have a good solution, try local search
        if score >= best_score - 2 and time.time() < deadline - 10:
            A_ls, B_ls = local_search(A, B, min(5, deadline - time.time()))
            ls_score = len(A_ls) + len(B_ls)
            if ls_score > best_score:
                best_score = ls_score
                best_A, best_B = A_ls[:], B_ls[:]
        
        if time.time() > deadline - 5:
            break
    
    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2] if 2 not in best_A else [3]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
