"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    
    # Enumerate all K(2,2,2)s and build incidence structure
    triple_to_k222s = {t: [] for t in all_triples}
    k222_to_triples = []  # list of lists of 8 triple indices
    
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    for six_points in combinations(range(n), 6):
        a, b, c, d, e, f = six_points
        matchings = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for m in matchings:
            (p1, p2), (q1, q2), (r1, r2) = m
            triples_in_k222 = [
                tuple(sorted((p1, q1, r1))),
                tuple(sorted((p1, q1, r2))),
                tuple(sorted((p1, q2, r1))),
                tuple(sorted((p1, q2, r2))),
                tuple(sorted((p2, q1, r1))),
                tuple(sorted((p2, q1, r2))),
                tuple(sorted((p2, q2, r1))),
                tuple(sorted((p2, q2, r2))),
            ]
            if len(set(triples_in_k222)) == 8:
                k222_idx = len(k222_to_triples)
                triple_indices = [triple_index[t] for t in triples_in_k222]
                k222_to_triples.append(triple_indices)
                for ti in triple_indices:
                    triple_to_k222s[all_triples[ti]].append(k222_idx)
    
    num_k222s = len(k222_to_triples)
    
    # Precompute for each triple, the list of K(2,2,2) indices it belongs to
    triple_k222_list = [triple_to_k222s[all_triples[i]] for i in range(num_triples)]
    
    # Greedy initial solution with multiple restarts
    best_count = 0
    best_selected = set()
    
    # Use more restarts with varied strategies
    num_restarts = 60
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        random.seed(restart * 7919 + 12345)
        
        # Vary the greedy strategy: sometimes prioritize triples with fewer constraints
        if restart % 3 == 0:
            shuffled = list(range(num_triples))
            random.shuffle(shuffled)
        elif restart % 3 == 1:
            # Sort by number of K(2,2,2) memberships (fewer first = less restrictive)
            shuffled = sorted(range(num_triples), key=lambda i: len(triple_k222_list[i]))
            random.shuffle(shuffled)
        else:
            # Sort by number of K(2,2,2) memberships (more first = more restrictive)
            shuffled = sorted(range(num_triples), key=lambda i: -len(triple_k222_list[i]))
            random.shuffle(shuffled)
        
        k222_count = [0] * num_k222s
        selected = set()
        
        for ti in shuffled:
            can_add = True
            for kid in triple_k222_list[ti]:
                if k222_count[kid] >= 7:
                    can_add = False
                    break
            if can_add:
                selected.add(ti)
                for kid in triple_k222_list[ti]:
                    k222_count[kid] += 1
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = set(selected)
    
    # Local improvement: aggressive add loop with randomization
    search_deadline = deadline - 5
    improved_any = True
    while improved_any and time.time() < search_deadline:
        improved_any = False
        
        # Recompute k222 counts
        k222_count = [0] * num_k222s
        for ti in best_selected:
            for kid in triple_k222_list[ti]:
                k222_count[kid] += 1
        
        # Try adding unselected triples in random order
        unselected = [i for i in range(num_triples) if i not in best_selected]
        random.shuffle(unselected)
        
        for ti in unselected:
            if time.time() > search_deadline:
                break
            can_add = True
            for kid in triple_k222_list[ti]:
                if k222_count[kid] >= 7:
                    can_add = False
                    break
            if can_add:
                best_selected.add(ti)
                best_count += 1
                for kid in triple_k222_list[ti]:
                    k222_count[kid] += 1
                improved_any = True
        
        # If no add was possible, try swaps
        if not improved_any and best_count > 0:
            # Try 1-for-1 swaps: remove one selected, add one unselected that was blocked
            selected_list = list(best_selected)
            random.shuffle(selected_list)
            unselected = [i for i in range(num_triples) if i not in best_selected]
            random.shuffle(unselected)
            
            swap_found = False
            num_swap_attempts = min(100, len(selected_list))
            
            for si in range(num_swap_attempts):
                if time.time() > search_deadline:
                    break
                
                rem = selected_list[si % len(selected_list)]
                
                # Temporarily remove
                best_selected.discard(rem)
                for kid in triple_k222_list[rem]:
                    k222_count[kid] -= 1
                
                # Try to add one unselected triple that was previously blocked
                added = None
                for add in unselected:
                    can_add = True
                    for kid in triple_k222_list[add]:
                        if k222_count[kid] >= 7:
                            can_add = False
                            break
                    if can_add:
                        added = add
                        break
                
                if added is not None:
                    # We removed one and added one - count is same, but this may unlock more adds
                    best_selected.add(added)
                    for kid in triple_k222_list[added]:
                        k222_count[kid] += 1
                    # Remove from unselected list
                    unselected.remove(added)
                    # Mark as improved to trigger another add pass
                    improved_any = True
                    swap_found = True
                    break
                else:
                    # Revert: put rem back
                    best_selected.add(rem)
                    for kid in triple_k222_list[rem]:
                        k222_count[kid] += 1
            
            if not swap_found:
                # Try 1-for-2 swaps: remove one, add two
                for si in range(min(50, len(selected_list))):
                    if time.time() > search_deadline:
                        break
                    
                    rem = selected_list[si % len(selected_list)]
                    
                    # Temporarily remove
                    best_selected.discard(rem)
                    for kid in triple_k222_list[rem]:
                        k222_count[kid] -= 1
                    
                    # Try to add two unselected triples
                    added_list = []
                    for add in unselected:
                        if len(added_list) >= 2:
                            break
                        can_add = True
                        for kid in triple_k222_list[add]:
                            if k222_count[kid] >= 7:
                                can_add = False
                                break
                        if can_add:
                            best_selected.add(add)
                            for kid in triple_k222_list[add]:
                                k222_count[kid] += 1
                            added_list.append(add)
                            unselected.remove(add)
                    
                    if len(added_list) >= 2:
                        best_count += len(added_list) - 1
                        improved_any = True
                        swap_found = True
                        break
                    else:
                        # Revert
                        best_selected.add(rem)
                        for kid in triple_k222_list[rem]:
                            k222_count[kid] += 1
                        # Remove any added
                        for add in added_list:
                            best_selected.discard(add)
                            for kid in triple_k222_list[add]:
                                k222_count[kid] -= 1
                            if add in unselected:
                                pass
                            else:
                                unselected.append(add)
            
            if not swap_found:
                # No improvement possible, try a fresh random restart if time allows
                if time.time() < deadline - 15:
                    random.seed(int(time.time() * 1000000) % (2**31))
                    shuffled = list(range(num_triples))
                    random.shuffle(shuffled)
                    
                    k222_count2 = [0] * num_k222s
                    selected2 = set()
                    
                    for ti in shuffled:
                        can_add = True
                        for kid in triple_k222_list[ti]:
                            if k222_count2[kid] >= 7:
                                can_add = False
                                break
                        if can_add:
                            selected2.add(ti)
                            for kid in triple_k222_list[ti]:
                                k222_count2[kid] += 1
                    
                    if len(selected2) > best_count:
                        best_count = len(selected2)
                        best_selected = selected2
                        improved_any = True
                else:
                    break

    final_selected = best_selected
    if not final_selected:
        final_selected = set(range(num_triples))
    
    result = [list(all_triples[i]) for i in sorted(final_selected)]
    return result
# EVOLVE-BLOCK-END
