import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140
    random.seed(12345)

    n = 16
    all_triples = list(combinations(range(n), 3))
    T = len(all_triples)  # 560
    triple_idx = {t: i for i, t in enumerate(all_triples)}

    # Generate constraints
    # For each set of 6 points, there are 15 pairings.
    # Precompute the 15 pairings of (0,1,2,3,4,5)
    # We'll just hardcode or generate them once.
    
    # Generation of 15 matchings for 6 nodes
    def gen_matchings():
        # Nodes 0,1,2,3,4,5
        # Pair 0 with 1,2,3,4,5
        # If 0-1, remaining 2,3,4,5 have 3 matchings:
        #   (2,3)(4,5), (2,4)(3,5), (2,5)(3,4)
        # If 0-2, remaining 1,3,4,5 have 3 matchings:
        #   (1,3)(4,5), (1,4)(3,5), (1,5)(3,4)
        # ...
        # Total 5 * 3 = 15.
        
        # Let's just list them to be safe and fast.
        m = [
            [(0,1),(2,3),(4,5)],
            [(0,1),(2,4),(3,5)],
            [(0,1),(2,5),(3,4)],
            [(0,2),(1,3),(4,5)],
            [(0,2),(1,4),(3,5)],
            [(0,2),(1,5),(3,4)],
            [(0,3),(1,2),(4,5)],
            [(0,3),(1,4),(2,5)],
            [(0,3),(1,5),(2,4)],
            [(0,4),(1,2),(3,5)],
            [(0,4),(1,3),(2,5)],
            [(0,4),(1,5),(2,3)],
            [(0,5),(1,2),(3,4)],
            [(0,5),(1,3),(2,4)],
            [(0,5),(1,4),(2,3)]
        ]
        return m

    matchings = gen_matchings()
    
    constraints = []
    # Map from constraint tuple of 8 triple indices to list of triple indices
    # We need to avoid duplicate constraints? 
    # Different sets of 6 points yield different constraints.
    # Different pairings of the same 6 points yield different sets of 8 triples.
    # So all 120,120 are distinct? 
    # Yes, the set of 8 transversal triples uniquely identifies the 3 pairs (and thus the 6 points and the pairing).
    
    for pts in combinations(range(n), 6):
        # Map local indices 0..5 to global points
        # pts is sorted tuple of 6 distinct ints
        for m in matchings:
            # m is list of 3 pairs of local indices
            # We need to form the 8 transversal triples
            # The 3 pairs are (pts[m[0][0]], pts[m[0][1]]), etc.
            p1 = (pts[m[0][0]], pts[m[0][1]])
            p2 = (pts[m[1][0]], pts[m[1][1]])
            p3 = (pts[m[2][0]], pts[m[2][1]])
            
            tris = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        tris.append(tuple(sorted((a, b, c))))
            
            # Convert to indices
            c_idx = tuple(sorted(triple_idx[t] for t in tris))
            constraints.append(c_idx)

    C = len(constraints)
    
    # Precompute triple_to_constraints
    triple_to_constraints = [[] for _ in range(T)]
    for ci, c in enumerate(constraints):
        for ti in c:
            triple_to_constraints[ti].append(ci)

    # State
    present = [True] * T
    viol = [8] * C
    
    # violated_set: set of constraint indices where viol[ci] == 8
    violated_set = set(range(C))
    
    # at_7_set: set of constraint indices where viol[ci] == 7
    at_7_set = set()
    
    # Greedy Removal
    # We want to remove triples to clear violated_set.
    # To be efficient, we can just pick any violated constraint, and remove a triple from it.
    # Which triple? The one that appears in the most violated constraints (greedy heuristic).
    
    # Precompute initial "weight" for each triple: number of violated constraints it's in.
    # Initially all constraints are violated, so weight[ti] = len(triple_to_constraints[ti]).
    # We can just use len(triple_to_constraints[ti]) as a static heuristic for removal priority?
    # Or dynamic. Dynamic is better.
    
    # Let's use a simple heuristic: remove the triple from the violated constraint that has the highest degree (static).
    # Or just random?
    # Let's try: Pick a random violated constraint. Pick a random triple from it. Remove it.
    # This is fast.
    
    # Actually, let's try to remove the triple that is in the MOST violated constraints.
    # We can maintain `viol_count_per_triple`.
    # Initially, viol_count_per_triple[ti] = len(triple_to_constraints[ti]).
    
    viol_count_per_triple = [len(triple_to_constraints[ti]) for ti in range(T)]
    
    while violated_set:
        if time.time() > deadline - 5:
            break
            
        # Pick a violated constraint
        # To make it faster, we can just pick the first one from the set
        # But to be smart, we should pick a constraint that has a triple with high viol_count.
        # Let's just pick a random violated constraint to avoid bias.
        # Or, pick the constraint with the smallest number of "options"?
        # Let's stick to simple: pick random violated constraint.
        ci = random.choice(list(violated_set))
        
        # Find best triple in this constraint to remove
        # Best = highest viol_count_per_triple
        best_t = -1
        max_vc = -1
        for ti in constraints[ci]:
            if present[ti]: # Should be true
                if viol_count_per_triple[ti] > max_vc:
                    max_vc = viol_count_per_triple[ti]
                    best_t = ti
                    
        # Remove best_t
        present[best_t] = False
        # Update viol_count_per_triple for best_t? No, it's removed.
        
        # Update constraints
        for c_idx in triple_to_constraints[best_t]:
            viol[c_idx] -= 1
            # Update viol_count_per_triple for other triples in this constraint?
            # No, viol_count_per_triple is the count of *violated* constraints containing the triple.
            # If c_idx was violated (in violated_set) and is no longer violated (viol < 8),
            # then for all other triples in c_idx, their viol_count_per_triple decreases by 1.
            
            if viol[c_idx] == 7:
                # It was 8, now 7.
                violated_set.remove(c_idx)
                at_7_set.add(c_idx)
                
                # Decrease viol_count_per_triple for all other present triples in this constraint
                for ti in constraints[c_idx]:
                    if present[ti] and ti != best_t:
                        viol_count_per_triple[ti] -= 1
            elif viol[c_idx] < 7:
                # It was 8, now < 7. (Impossible in single step, but safe)
                violated_set.remove(c_idx)
                # No at_7 update needed if it skips 7?
                # Wait, if it goes 8 -> 7, we handle above.
                # If it goes 8 -> 6? No, we remove one triple at a time.
                # So it goes 8 -> 7.
                pass

    # Greedy Add-back
    # We want to add triples back.
    # A triple ti can be added if no constraint containing it is in at_7_set.
    # (If a constraint is in at_7_set, adding ti makes it 8, which is a violation).
    
    # We can iterate through all absent triples.
    # To be efficient, we can try to add them in an order.
    # Triples with fewer constraints are easier to add?
    # Or just random order?
    
    # Let's do multiple passes.
    # Pass 1: Try to add all possible triples.
    # We need to update at_7_set when we add a triple.
    # If we add ti, and some constraint c containing ti was in at_7_set, it becomes 8 (violated).
    # So c moves from at_7_set to violated_set.
    
    # This might create new violations!
    # So simple add-back might fail if we add a triple that blocks another.
    # But we check `if not any(c in at_7_set for c in triple_to_constraints[ti])`.
    # If we add ti, and it causes c to become 8, that's bad.
    # So we should NOT add ti if it causes any constraint to become violated.
    # Condition: For all c in triple_to_constraints[ti], viol[c] < 7.
    # i.e., c is NOT in at_7_set and c is NOT in violated_set (should be empty).
    
    # So condition: `not any(c in at_7_set for c in triple_to_constraints[ti])`.
    
    # Let's try adding triples.
    # We'll shuffle the absent triples and try to add them.
    
    absent_triples = [i for i in range(T) if not present[i]]
    random.shuffle(absent_triples)
    
    for ti in absent_triples:
        if time.time() > deadline - 10:
            break
        # Check if can add
        can_add = True
        for c_idx in triple_to_constraints[ti]:
            if c_idx in at_7_set:
                can_add = False
                break
        
        if can_add:
            present[ti] = True
            # Update structures
            for c_idx in triple_to_constraints[ti]:
                viol[c_idx] += 1
                if viol[c_idx] == 8:
                    # New violation!
                    # This shouldn't happen if we checked at_7_set correctly?
                    # If viol[c_idx] was 7, it's in at_7_set, so we wouldn't add ti.
                    # So viol[c_idx] must have been < 7.
                    # So it becomes <= 7.
                    pass
                elif viol[c_idx] == 7:
                    # It was 6, now 7.
                    # Add to at_7_set
                    at_7_set.add(c_idx)

    # Local Search: Swaps
    # Try to swap present t_out with absent t_in.
    # Gain: 0 (count stays same).
    # But it might allow more adds later?
    # Or maybe we can do "break and add" where we remove 1 and add 2?
    # Let's stick to swaps first.
    
    best_count = sum(present)
    
    # We can try to improve by swapping.
    # A swap is valid if:
    # 1. Remove t_out.
    # 2. Add t_in.
    # 3. No violations created.
    
    # When we remove t_out, some constraints drop from 7 to 6 (leaving at_7_set).
    # When we add t_in, some constraints rise.
    # We need to ensure that after the swap, no constraint is at 8.
    
    # Let's implement a function to check swap validity.
    
    def try_swap(t_out, t_in):
        # Check if swap is valid
        # Temporarily remove t_out
        # We need to know which constraints are affected.
        
        # Constraints containing t_out
        cons_out = triple_to_constraints[t_out]
        # Constraints containing t_in
        cons_in = triple_to_constraints[t_in]
        
        # We need to check if adding t_in creates a violation,
        # given that t_out is removed.
        
        # A constraint c is violated if viol[c] + (1 if t_in in c else 0) - (1 if t_out in c else 0) == 8.
        # We know current viol[c].
        
        # For c in cons_in:
        #   new_viol = viol[c] - (1 if c in cons_out_set else 0) + 1
        #   If new_viol == 8, invalid.
        
        # To be fast, we can precompute a set of cons_out.
        cons_out_set = set(cons_out)
        
        for c in cons_in:
            delta = 1
            if c in cons_out_set:
                delta -= 1
            new_viol = viol[c] + delta
            if new_viol == 8:
                return False
        
        return True

    # Perform random swaps
    present_list = [i for i in range(T) if present[i]]
    absent_list = [i for i in range(T) if not present[i]]
    
    # We can do a limited number of swaps
    # Since the time limit is tight, we do a few hundred swaps.
    
    for _ in range(500):
        if time.time() > deadline - 5:
            break
        if not present_list or not absent_list:
            break
            
        t_out = random.choice(present_list)
        t_in = random.choice(absent_list)
        
        if try_swap(t_out, t_in):
            # Commit swap
            present[t_out] = False
            present[t_in] = True
            
            # Update structures
            # Remove t_out
            for c in triple_to_constraints[t_out]:
                viol[c] -= 1
                if viol[c] == 7:
                    at_7_set.discard(c) # If it was 8, it's not in at_7. If it was 7, it goes to 6.
                    # Wait, if viol[c] was 8, it was in violated_set.
                    # But violated_set should be empty after greedy phase?
                    # Yes, we cleared it.
                    # So viol[c] was <= 7.
                    # If viol[c] was 7, it is in at_7_set. Now it is 6. Remove from at_7_set.
                    # If viol[c] was 6, it is not in at_7_set. Now it is 5.
                    pass
                # If viol[c] becomes 7? No, it decreases.
                
            # Add t_in
            for c in triple_to_constraints[t_in]:
                viol[c] += 1
                if viol[c] == 7:
                    at_7_set.add(c)
                # If viol[c] becomes 8? try_swap prevented that.
            
            # Update lists
            present_list.remove(t_out)
            absent_list.remove(t_in)
            present_list.append(t_in)
            absent_list.append(t_out)
            
            # After swap, we might be able to add more triples.
            # Let's do a quick add-back pass?
            # Maybe too slow. Let's just continue swapping.
            
            # Actually, we can try to add back any triple that became addable.
            # But that's complex. Let's just rely on the next swap or a final add pass.

    # Final Add-back pass
    # Try to add any remaining absent triples.
    # We need to refresh at_7_set? 
    # Our updates kept it consistent.
    
    absent_list = [i for i in range(T) if not present[i]]
    random.shuffle(absent_list)
    
    for ti in absent_list:
        if time.time() > deadline - 2:
            break
        can_add = True
        for c_idx in triple_to_constraints[ti]:
            if c_idx in at_7_set:
                can_add = False
                break
        
        if can_add:
            present[ti] = True
            for c_idx in triple_to_constraints[ti]:
                viol[c_idx] += 1
                if viol[c_idx] == 7:
                    at_7_set.add(c_idx)

    result = [list(all_triples[i]) for i in range(T) if present[i]]
    return result
