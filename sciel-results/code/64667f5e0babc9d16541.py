"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

N = 128

def _construct_baseline():
    """
    Construct the known strong baseline: {65..128} U {1,3,...,31, 64}.
    """
    mask = 0
    for v in range(65, 129):
        mask |= (1 << v)
    for v in [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 64]:
        mask |= (1 << v)
    return mask

def _precompute_constraints():
    """
    Precompute constraint masks for fast validity checking.
    For each pair (a,b) with a<b and a+b<=N, compute the mask of c values
    (c>b) such that if a,b,a+b,c are all in the set, we have a forbidden triple.
    """
    # forbidden_c[(a,b)] = bitmask of c values that would create a forbidden triple
    # if a, b, a+b, and c are all selected
    forbidden_c = {}
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            s_ab = a + b
            if s_ab > N:
                break
            # c must satisfy: c > b, a+c <= N, b+c <= N
            max_c = min(N - a, N - b)
            if max_c <= b:
                continue
            c_mask = 0
            for c in range(b + 1, max_c + 1):
                c_mask |= (1 << c)
            if c_mask:
                forbidden_c[(a, b)] = c_mask
    return forbidden_c

def _is_valid_fast(mask, forbidden_c):
    """
    Fast validity check using precomputed constraint masks.
    """
    for (a, b), c_mask in forbidden_c.items():
        if ((mask >> a) & 1) and ((mask >> b) & 1) and ((mask >> (a + b)) & 1):
            if (mask & c_mask) != 0:
                return False
    return True

def _greedy_fill(mask, order, forbidden_c):
    """Try to add elements from 'order' to mask if valid."""
    for v in order:
        bit = 1 << v
        if mask & bit:
            continue
        trial = mask | bit
        if _is_valid_fast(trial, forbidden_c):
            mask = trial
    return mask

def _local_search(mask, deadline, forbidden_c):
    """Local search to improve mask size."""
    current_mask = mask
    current_size = current_mask.bit_count()
    
    while time.time() < deadline:
        improved = False
        
        # Phase 1: Try adding single elements
        for v in range(1, N + 1):
            bit = 1 << v
            if current_mask & bit:
                continue
            trial = current_mask | bit
            if _is_valid_fast(trial, forbidden_c):
                current_mask = trial
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try 'remove one, add many'
        found_better = False
        best_candidate = current_mask
        best_candidate_size = current_size
        
        for v in range(1, N + 1):
            bit = 1 << v
            if not (current_mask & bit):
                continue
            
            reduced_mask = current_mask & ~bit
            candidate = reduced_mask
            for add_v in range(1, N + 1):
                add_bit = 1 << add_v
                if candidate & add_bit:
                    continue
                trial = candidate | add_bit
                if _is_valid_fast(trial, forbidden_c):
                    candidate = trial
            
            if candidate.bit_count() > best_candidate_size:
                best_candidate = candidate
                best_candidate_size = candidate.bit_count()
                found_better = True
        
        if found_better:
            current_mask = best_candidate
            current_size = best_candidate_size
        else:
            break
    
    return current_mask

def _bron_kerbosch_max_clique(start_mask, deadline, forbidden_c, current_best_size):
    """
    Bron-Kerbosch style search to find larger valid sets.
    We search for cliques in the "extendable" space.
    """
    best_mask = start_mask
    best_size = start_mask.bit_count()
    
    # Get the set of elements in start_mask
    in_set = set()
    for i in range(1, N + 1):
        if (start_mask >> i) & 1:
            in_set.add(i)
    
    # Elements not in set
    not_in_set = [i for i in range(1, N + 1) if i not in in_set]
    
    # Try to build a larger set by exploring which elements to include
    # This is a simplified Bron-Kerbosch: we branch on elements to add/remove
    
    def _explore(mask, remaining, depth, deadline):
        nonlocal best_mask, best_size
        
        if time.time() > deadline:
            return
        
        if not remaining:
            if mask.bit_count() > best_size:
                best_mask = mask
                best_size = mask.bit_count()
            return
        
        # Pivot selection: pick element that maximizes pruning
        # For simplicity, just pick the first element
        pivot = remaining[0]
        
        # Branch 1: Include pivot
        bit_pivot = 1 << pivot
        trial = mask | bit_pivot
        if _is_valid_fast(trial, forbidden_c):
            # Filter remaining: only elements that can coexist with pivot in current context
            new_remaining = []
            for r in remaining[1:]:
                if r == pivot:
                    continue
                # Check if adding r to trial is valid
                if not (trial & (1 << r)):
                    new_remaining.append(r)
            _explore(trial, new_remaining, depth + 1, deadline)
        
        # Branch 2: Exclude pivot
        _explore(mask, remaining[1:], depth + 1, deadline)
    
    # This naive approach is too slow. Instead, use a guided search.
    # We'll do a series of local improvements with random perturbations.
    
    current_mask = start_mask
    current_size = start_mask.bit_count()
    
    while time.time() < deadline and current_size <= best_size:
        # Try to improve via local search
        improved_mask = _local_search(current_mask, deadline, forbidden_c)
        if improved_mask.bit_count() > best_size:
            best_mask = improved_mask
            best_size = improved_mask.bit_count()
            current_mask = improved_mask
            current_size = best_size
        else:
            # Perturb: remove 3-5 random elements, then try to fill
            if current_size < 3:
                break
            elems = [i for i in range(1, N + 1) if (current_mask >> i) & 1]
            k = min(3, len(elems))
            to_remove = random.sample(elems, k)
            perturbed = current_mask
            for r in to_remove:
                perturbed &= ~(1 << r)
            
            # Greedy fill on perturbed
            random_order = list(range(1, N + 1))
            random.shuffle(random_order)
            filled = _greedy_fill(perturbed, random_order, forbidden_c)
            
            # Local search on filled
            searched = _local_search(filled, deadline, forbidden_c)
            
            if searched.bit_count() > best_size:
                best_mask = searched
                best_size = searched.bit_count()
                current_mask = searched
                current_size = best_size
            else:
                current_mask = searched
                current_size = searched.bit_count()
    
    return best_mask

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute constraints
    forbidden_c = _precompute_constraints()
    
    # 1. Start with strong baseline
    best_mask = _construct_baseline()
    best_size = best_mask.bit_count()
    
    # 2. Greedy fill
    if time.time() < deadline - 15:
        filled_mask = _greedy_fill(best_mask, list(range(1, N + 1)), forbidden_c)
        if filled_mask.bit_count() > best_size:
            best_mask = filled_mask
            best_size = filled_mask.bit_count()
    
    # 3. Local search
    if time.time() < deadline - 10:
        searched_mask = _local_search(best_mask, deadline - 10, forbidden_c)
        if searched_mask.bit_count() > best_size:
            best_mask = searched_mask
            best_size = searched_mask.bit_count()
    
    # 4. Bron-Kerbosch guided search with random restarts
    random.seed(42)
    if time.time() < deadline - 5:
        for restart in range(100):
            if time.time() > deadline - 5:
                break
            
            # Perturb the best mask
            elems = [i for i in range(1, N + 1) if (best_mask >> i) & 1]
            if len(elems) >= 3:
                k = random.randint(2, 5)
                to_remove = random.sample(elems, min(k, len(elems)))
                perturbed = best_mask
                for r in to_remove:
                    perturbed &= ~(1 << r)
            else:
                perturbed = best_mask
            
            # Try different fill orders
            perm = list(range(1, N + 1))
            random.shuffle(perm)
            filled_p = _greedy_fill(perturbed, perm, forbidden_c)
            
            # Local search
            searched_p = _local_search(filled_p, deadline - 5, forbidden_c)
            
            if searched_p.bit_count() > best_size:
                best_mask = searched_p
                best_size = searched_p.bit_count()
    
    # Final validity check
    if not _is_valid_fast(best_mask, forbidden_c):
        best_mask = _construct_baseline()
    
    # Convert mask to list of ints
    result = []
    for i in range(1, N + 1):
        if (best_mask >> i) & 1:
            result.append(i)
    
    return result
# EVOLVE-BLOCK-END
