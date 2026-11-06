import time
import random
from itertools import combinations

def _precompute_k222s(n):
    """
    Precompute all K(2,2,2) configurations and triple-to-K mapping.
    Returns:
    - all_triples: list of tuples
    - triple_to_k222s: list of lists (for each triple index, list of K indices it belongs to)
    - k222_list: list of frozensets of triple indices
    """
    all_triples = list(combinations(range(n), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    triple_to_k222s = [[] for _ in range(len(all_triples))]
    k222_list = []
    
    # Generate all sets of 6 points
    for points in combinations(range(n), 6):
        # Generate all perfect matchings of these 6 points
        # There are 15 perfect matchings for 6 points
        def gen_matchings(pts):
            if not pts:
                yield []
                return
            first = pts[0]
            rest = pts[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_rest = rest[:i] + rest[i+1:]
                for m in gen_matchings(new_rest):
                    yield [(first, second)] + m
        
        for matching in gen_matchings(points):
            # matching is list of 3 tuples: (a,a'), (b,b'), (c,c')
            k_triples = []
            for x in [0, 1]:
                for y in [0, 1]:
                    for z in [0, 1]:
                        t = tuple(sorted((matching[0][x], matching[1][y], matching[2][z])))
                        k_triples.append(t)
            
            k_indices = set()
            for t in k_triples:
                k_indices.add(triple_to_idx[t])
            
            k222_idx = len(k222_list)
            k222_list.append(frozenset(k_indices))
            
            for idx in k_indices:
                triple_to_k222s[idx].append(k222_idx)
    
    return all_triples, triple_to_k222s, k222_list

def _get_conflicts(triple_idx, present_set, triple_to_k222s, k222_list):
    """
    Returns a list of K(2,2,2) indices that would be completed if triple_idx were added.
    """
    conflicts = []
    for k_idx in triple_to_k222s[triple_idx]:
        k_set = k222_list[k_idx]
        # Check if all other members are in present_set
        if all(other in present_set for other in k_set if other != triple_idx):
            conflicts.append(k_idx)
    return conflicts

def _greedy_init(n_triples, triple_to_k222s, k222_list):
    """
    Fast greedy initialization using a static random order.
    """
    order = list(range(n_triples))
    random.shuffle(order)
    present = set()
    for idx in order:
        if not _get_conflicts(idx, present, triple_to_k222s, k222_list):
            present.add(idx)
    return present

def _local_search_add_remove(current_set, all_indices, triple_to_k222s, k222_list, deadline, max_iters=5000):
    """
    Local search: try to add a random triple, if conflict, remove a conflicting triple.
    """
    best_set = set(current_set)
    best_size = len(current_set)
    iters = 0
    
    while time.time() < deadline and iters < max_iters:
        iters += 1
        # Try to add a random triple
        u = random.choice(all_indices)
        if u in current_set:
            continue
        
        conflicts = _get_conflicts(u, current_set, triple_to_k222s, k222_list)
        
        if not conflicts:
            # No conflicts, add it
            current_set.add(u)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = set(current_set)
        else:
            # Has conflicts. Try to resolve by removing one of the conflicting triples.
            # Pick a random conflict
            k_idx = random.choice(conflicts)
            k_set = k222_list[k_idx]
            
            # Candidates to remove: any triple in k_set except u
            candidates = [t for t in k_set if t != u and t in current_set]
            
            if candidates:
                rem_candidate = random.choice(candidates)
                # Try removing rem_candidate and adding u
                current_set.remove(rem_candidate)
                current_set.add(u)
                # Check if u is now valid
                if not _get_conflicts(u, current_set, triple_to_k222s, k222_list):
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = set(current_set)
                else:
                    # Still conflicts, revert
                    current_set.remove(u)
                    current_set.add(rem_candidate)
    
    return best_set

def _hill_climbing_multi(current_set, all_indices, triple_to_k222s, k222_list, deadline, max_rounds=100):
    """
    Multi-round hill climbing with random restarts.
    """
    best_set = set(current_set)
    best_size = len(current_set)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Start from best_set
        current = set(best_set)
        
        # Do local search
        local_best = _local_search_add_remove(current, all_indices, triple_to_k222s, k222_list, deadline, max_iters=1000)
        
        if len(local_best) > best_size:
            best_size = len(local_best)
            best_set = set(local_best)
        
        # Occasionally perturb the best solution to escape local optima
        if round_num % 5 == 4 and len(best_set) > 10:
            perturbed = set(best_set)
            # Remove 2-3 random elements
            for _ in range(random.randint(2, 3)):
                if perturbed:
                    perturbed.remove(random.choice(list(perturbed)))
            # Add 2-3 random valid elements
            for _ in range(random.randint(2, 3)):
                u = random.choice(all_indices)
                if u not in perturbed and not _get_conflicts(u, perturbed, triple_to_k222s, k222_list):
                    perturbed.add(u)
            # Run local search from perturbed state
            perturbed_best = _local_search_add_remove(perturbed, all_indices, triple_to_k222s, k222_list, deadline, max_iters=500)
            if len(perturbed_best) > best_size:
                best_size = len(perturbed_best)
                best_set = set(perturbed_best)
    
    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    n = 16
    
    all_triples, triple_to_k222s, k222_list = _precompute_k222s(n)
    n_triples = len(all_triples)
    
    # Initial solution via greedy
    current_set = _greedy_init(n_triples, triple_to_k222s, k222_list)
    best_set = set(current_set)
    best_size = len(current_set)
    
    # Multi-round hill climbing with perturbations
    best_set = _hill_climbing_multi(best_set, list(range(n_triples)), triple_to_k222s, k222_list, deadline)
    best_size = len(best_set)
    
    # Final local search to polish
    final_set = _local_search_add_remove(best_set, list(range(n_triples)), triple_to_k222s, k222_list, deadline, max_iters=20000)
    if len(final_set) > best_size:
        best_set = set(final_set)
        best_size = len(final_set)
    
    return [list(all_triples[i]) for i in sorted(best_set)]

if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    t1 = time.time()
    print(f"Found {len(res)} triples in {t1-t0:.2f}s")
