import itertools
import random

def construct():
    vertices = list(range(16))
    all_triples = list(itertools.combinations(vertices, 3))
    
    # Precompute conflicts
    # conflicts[t] is a list of frozensets of 7 triples (the other 7 in the K(2,2,2))
    conflicts = {}
    for t in all_triples:
        u, v, w = t
        remaining = [x for x in vertices if x not in t]
        conf_list = []
        for combo in itertools.combinations(remaining, 3):
            x, y, z = combo
            for perm in itertools.permutations(combo):
                px, py, pz = perm
                others = [
                    frozenset((u, v, pz)),
                    frozenset((u, py, w)),
                    frozenset((u, py, pz)),
                    frozenset((px, v, w)),
                    frozenset((px, v, pz)),
                    frozenset((px, py, w)),
                    frozenset((px, py, pz))
                ]
                others_tuples = [tuple(sorted(o)) for o in others]
                conf_list.append(frozenset(others_tuples))
        conflicts[t] = conf_list

    best_size = 0
    best_set = set()
    
    # Run multiple restarts with different weight initializations or tie-breakers
    for restart in range(50):
        available = set(all_triples)
        current_set = set()
        weights = {t: 1.0 for t in all_triples}
        
        while available:
            # Select triple with max weight
            # Tie-breaker: random or fixed
            # To add diversity, we can shuffle weights slightly or use random tie-break
            max_weight = -1
            best_t = None
            candidates = []
            for t in available:
                if weights[t] > max_weight:
                    max_weight = weights[t]
                    best_t = t
                    candidates = [t]
                elif weights[t] == max_weight:
                    candidates.append(t)
            
            if not candidates:
                break
                
            # Random pick from top candidates to ensure diversity between restarts
            t = random.choice(candidates)
            
            # Check safety
            safe = True
            for cs in conflicts[t]:
                if cs.issubset(current_set):
                    safe = False
                    break
            
            if safe:
                current_set.add(t)
                available.discard(t)
                
                # Update weights of available neighbors
                # For each conflict set of t, if it has members in current_set (including t? no, t is not in cs),
                # it's "hot".
                # Actually, t is NOT in cs. cs contains the other 7.
                # If we add t, we don't complete any cs by itself.
                # But we might complete cs if we add one more from cs?
                # No, cs has 7 elements. If all 7 are in current_set, adding t creates the 8th.
                # So if cs.issubset(current_set), it was already unsafe.
                # Since it was safe, no cs is a subset of current_set.
                
                # Penalty logic:
                # If t is added, any triple u that is in a conflict set with t
                # is "more constrained" because the number of "slots" left for that conflict set is reduced?
                # Let's penalize u if u is in a conflict set C where C has many elements in current_set.
                for cs in conflicts[t]:
                    # Count how many elements of cs are in current_set
                    # Note: t is not in cs.
                    # If k elements of cs are in current_set, then we need (7-k) more from cs to complete it.
                    # The triples in cs are "dangerous" if k is high.
                    # Let's penalize all u in cs.
                    for u in cs:
                        if u in available:
                            # Simple penalty
                            weights[u] *= 0.95
            else:
                # If not safe, we cannot add t.
                # Remove t from available.
                available.discard(t)
                # Optionally penalize neighbors of t?
                # Since t is blocked, it doesn't affect others directly in this step.
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set.copy()

    return [list(t) for t in best_set]
