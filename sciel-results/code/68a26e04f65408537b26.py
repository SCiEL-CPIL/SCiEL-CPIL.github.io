import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n = 16
    best_system = None
    best_count = 0
    
    # Precompute all triples
    all_triples = list(itertools.combinations(range(n), 3))
    
    # For each triple, precompute the set of "witness" configurations
    # A witness for triple t=(a,b,c) is a set of 7 other triples that, together with t,
    # form the 8 transversals of 3 disjoint pairs.
    # 
    # For triple (a,b,c), we need to find 3 disjoint pairs (p1,q1),(p2,q2),(p3,q3)
    # such that {a,b,c} picks one from each pair.
    # 
    # The key insight: for a given triple (a,b,c), the 3 pairs must be:
    # - One pair contains a, another contains b, another contains c
    # - The "opposite" elements x,y,z are chosen from the remaining 13 points
    # - Pairs are (a,x), (b,y), (c,z) [or some permutation of which element goes to which pair]
    #
    # For efficiency, we precompute for each triple the list of "blocking sets"
    # A blocking set is a frozenset of 7 triples. If all 7 are in the system, 
    # then adding the triple would create a K(2,2,2).
    
    # Actually, let's use a different approach for speed:
    # Maintain a counter for each triple: how many of its "potential completions" are satisfied.
    # 
    # For triple t = (a,b,c), define:
    #   For each way to assign a,b,c to 3 pair-positions and choose opposites x,y,z:
    #     The 7 other transversals must all be present.
    #   Let's call each such (x,y,z,assignment) a "witness pattern".
    #   For each witness pattern, we track how many of the 7 required triples are in the system.
    #   When this count reaches 7, the triple is blocked.
    
    # This is complex. Let me try a simpler but effective approach:
    # Use a randomized greedy with local search / simulated annealing.
    
    def check_violation(system_set, triple):
        """Check if adding triple to system_set creates a forbidden K(2,2,2)."""
        a, b, c = triple
        abc = {a, b, c}
        
        # For each choice of x, y, z (distinct from a,b,c and each other)
        # Check if (a,x),(b,y),(c,z) form 3 disjoint pairs with all 8 transversals present
        for x in range(n):
            if x in abc:
                continue
            for y in range(n):
                if y in abc or y == x:
                    continue
                for z in range(n):
                    if z in abc or z == x or z == y:
                        continue
                    
                    # Check 7 other transversals
                    if (tuple(sorted((a, b, z))) in system_set and
                        tuple(sorted((a, y, c))) in system_set and
                        tuple(sorted((a, y, z))) in system_set and
                        tuple(sorted((x, b, c))) in system_set and
                        tuple(sorted((x, b, z))) in system_set and
                        tuple(sorted((x, y, c))) in system_set and
                        tuple(sorted((x, y, z))) in system_set):
                        return True
        return False
    
    def try_greedy_with_restarts(seed, max_restarts=20):
        rng = random.Random(seed)
        best_local = []
        best_local_count = 0
        
        for restart in range(max_restarts):
            if time.time() > deadline - 10:
                break
            
            all_triples_shuffled = list(all_triples)
            rng.shuffle(all_triples_shuffled)
            
            system = set()
            system_list = []
            
            for t in all_triples_shuffled:
                if time.time() > deadline - 10:
                    break
                if t in system:
                    continue
                if not check_violation(system, t):
                    system.add(t)
                    system_list.append(list(t))
            
            if len(system_list) > best_local_count:
                best_local_count = len(system_list)
                best_local = system_list
        
        return best_local
    
    # Try multiple seeds
    num_seeds = 30
    for seed in range(num_seeds):
        if time.time() > deadline - 15:
            break
        result = try_greedy_with_restarts(seed)
        if len(result) > best_count:
            best_count = len(result)
            best_system = result
    
    # Try a structured approach: start with a known good structure and add more
    # The seed program gives C(15,2) = 105 triples (all triples containing 0)
    # Let's try to build from there and add more triples
    
    def try_structured_build():
        # Start with all triples containing point 0
        system = set()
        for a in range(1, n):
            for b in range(a + 1, n):
                system.add((0, a, b))
        
        system_list = [list(t) for t in system]
        
        # Now try to add triples not containing 0
        candidates = [t for t in all_triples if 0 not in t]
        random.shuffle(candidates)
        
        for t in candidates:
            if time.time() > deadline - 10:
                break
            if t in system:
                continue
            if not check_violation(system, t):
                system.add(t)
                system_list.append(list(t))
        
        return system_list
    
    structured = try_structured_build()
    if len(structured) > best_count:
        best_count = len(structured)
        best_system = structured
    
    # Try local search: start from best found, try to swap triples to improve
    def local_search(start_system, max_iters=5000):
        system = set(tuple(sorted(t)) for t in start_system)
        system_list = [list(t) for t in system]
        current_count = len(system)
        
        # Build a set of all triples not in system
        remaining = [t for t in all_triples if t not in system]
        random.shuffle(remaining)
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 15:
            improved = False
            iters += 1
            
            # Try adding a random remaining triple
            for t in remaining[:100]:  # Try up to 100 candidates
                if time.time() > deadline - 15:
                    break
                if t in system:
                    continue
                if not check_violation(system, t):
                    system.add(t)
                    system_list.append(list(t))
                    current_count += 1
                    improved = True
                    # Remove from remaining
                    if t in remaining:
                        remaining.remove(t)
            
            # Try removing a triple and adding two others
            if not improved and len(system) > 10:
                # Pick a random triple to potentially remove
                remove_candidates = list(system)
                random.shuffle(remove_candidates)
                for rt in remove_candidates[:20]:
                    if time.time() > deadline - 15:
                        break
                    system.discard(rt)
                    if rt in system_list:
                        system_list.remove(list(rt))
                    
                    # Try to add two triples that were blocked
                    added = 0
                    for t in remaining[:50]:
                        if time.time() > deadline - 15:
                            break
                        if t in system:
                            continue
                        if not check_violation(system, t):
                            system.add(t)
                            system_list.append(list(t))
                            added += 1
                            if t in remaining:
                                remaining.remove(t)
                            if added >= 2:
                                break
                    
                    if added >= 2:
                        improved = True
                    else:
                        # Revert
                        system.add(rt)
                        system_list.append(list(rt))
        
        return system_list
    
    if best_system and time.time() < deadline - 20:
        improved = local_search(best_system)
        if len(improved) > best_count:
            best_count = len(improved)
            best_system = improved
    
    # Fallback
    if best_system is None:
        best_system = [[0, a, b] for a in range(1, n) for b in range(a + 1, n)]
    
    return best_system
