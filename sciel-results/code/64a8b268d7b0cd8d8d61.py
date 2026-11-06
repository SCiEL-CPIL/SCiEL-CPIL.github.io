"""Improved program: MODEL-AND-OPTIMIZE for small maximal Sidon sets in [1,256].

Strategy:
1. Start from strong baselines (greedy ascending, descending, random).
2. MIS-based vertex elimination: build conflict graph, eliminate high-degree vertices.
3. Use combined local search:
   a. Remove elements one by one if the set remains maximal.
   b. Try swaps: remove one element, add a missing element, check Sidon + maximality.
4. Multiple randomized restarts to escape local optima.
5. Deadline guard with safety margin.
"""

import time
import random

N = 256

# EVOLVE-BLOCK-START

def _is_sidon(set_list):
    """Check if set_list is a Sidon set. Returns True if all pair sums a+b (a<=b) are distinct."""
    s = set(set_list)
    if len(s) != len(set_list):
        return False
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sm = lst[i] + lst[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _compute_sums(set_list):
    """Compute the set of all pair sums."""
    sums = set()
    lst = sorted(set_list)
    for i in range(len(lst)):
        for j in range(i, len(lst)):
            sums.add(lst[i] + lst[j])
    return sums

def _is_maximal_fast(set_list, sums, N_val=N):
    """Faster maximality check using precomputed sums."""
    s = set(set_list)
    for x in range(1, N_val + 1):
        if x in s:
            continue
        if 2 * x in sums:
            continue
        conflict = False
        for elem in set_list:
            if (x + elem) in sums:
                conflict = True
                break
        if not conflict:
            return False
    return True

def _greedy_ascending():
    """Original greedy: pick smallest possible element."""
    selected = []
    sums = set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_descending():
    """Greedy descending: pick largest possible element."""
    selected = []
    sums = set()
    for x in range(N, 0, -1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _greedy_random(seed):
    """Greedy with random ordering."""
    rng = random.Random(seed)
    order = list(range(1, N + 1))
    rng.shuffle(order)
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _build_conflict_graph():
    """
    Build a conflict graph where edge (u,v) exists if u and v cannot coexist in a Sidon set.
    Two elements u,v conflict if 2*u == 2*v (impossible for distinct) or u+v equals some other pair sum.
    Actually, u and v conflict if adding both would create duplicate sums.
    More precisely: u,v conflict if there exist a,b in {u,v} such that a+b = c+d for distinct pairs.
    For just two elements u,v: they conflict if 2*u == u+v (i.e., u==v, impossible) or 2*v == u+v (impossible) 
    or u+v == 2*u or u+v == 2*v. So u and v never directly conflict as a pair!
    
    Wait - the conflict is more subtle. In a Sidon set, ALL pair sums must be distinct.
    So u and v conflict if there's some third element w such that adding u,v,w creates duplicate sums.
    
    For MIS formulation: we want a maximal independent set where independent = Sidon.
    The "conflict" between u and v in the context of the full problem is that they can't both be in any Sidon set.
    But actually, any two distinct elements CAN be in a Sidon set together (just {u,v} is always Sidon).
    
    So the conflict graph approach needs rethinking. Let me use a different approach:
    Greedy elimination based on "how many other elements would be blocked" if we include this element.
    """
    # For each element x, compute how many other elements y it "conflicts with" in the sense
    # that including x makes it harder to include y later.
    # Actually, let's just use the degree in a different sense:
    # For element x, its "degree" = number of elements y such that {x,y} is NOT extendable to a large Sidon set.
    # This is hard to compute. Let's use a simpler heuristic.
    
    # Alternative: For each x, compute the number of sums 2*x and x+y for y in [1,N] that would block other elements.
    # Higher "blocking power" = higher degree = eliminate first (don't include).
    
    blocking = [0] * (N + 1)
    for x in range(1, N + 1):
        # 2*x blocks element x itself (diagonal)
        # x + y for y in [1,N], y != x: these sums block other elements
        # Count how many distinct sums are created
        sums_created = set()
        sums_created.add(2 * x)
        for y in range(1, N + 1):
            if y != x:
                sums_created.add(x + y)
        blocking[x] = len(sums_created)
    
    return blocking

def _mis_greedy_elimination(deadline, rng):
    """
    MIS-based greedy: iteratively decide to include or exclude vertices.
    Start with all vertices available.
    At each step, pick the vertex with highest "degree" (blocking power) among remaining.
    If we exclude it, mark it as removed.
    If we include it, add to solution and remove all its neighbors (conflicting elements).
    
    But since any two elements can coexist, we need a different conflict definition.
    
    Revised approach: Use greedy construction but with a twist.
    Instead of picking the smallest/largest/random, pick based on a scoring function.
    Score(x) = number of elements that would still be addable after adding x.
    Pick x that maximizes this (keeps options open) or minimizes it (forces commitment).
    """
    # Let's try: greedy that picks elements to MINIMIZE the final set size.
    # Strategy: at each step, for each candidate x, compute how many remaining elements
    # would still be addable if we pick x. Pick x that leaves the FEWEST options (most restrictive).
    
    available = set(range(1, N + 1))
    selected = []
    sums = set()
    
    while available and time.time() < deadline:
        best_x = None
        best_options = float('inf')
        
        for x in list(available):
            # Try adding x
            new_sums = [x + a for a in selected] + [2 * x]
            if len(set(new_sums)) != len(new_sums):
                continue
            if sums.intersection(set(new_sums)):
                continue
            
            # Count how many remaining elements could still be added after x
            temp_sums = sums | set(new_sums)
            options = 0
            for y in available:
                if y == x:
                    continue
                added_y = [y + a for a in selected] + [y + x] + [2 * y]
                # Check if y can be added given selected + x
                if len(set(added_y)) != len(added_y):
                    continue
                if temp_sums.intersection(set(added_y)):
                    continue
                options += 1
            
            if options < best_options:
                best_options = options
                best_x = x
        
        if best_x is None:
            break
        
        selected.append(best_x)
        available.remove(best_x)
        new_sums = [best_x + a for a in selected[:-1]] + [2 * best_x]
        sums.update(new_sums)
    
    return selected

def _make_maximal(current, deadline):
    """
    Ensure the set is maximal by adding any missing element that doesn't break Sidon.
    Returns the maximal set.
    """
    current = list(current)
    current_set = set(current)
    current_sums = _compute_sums(current)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, N + 1):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            
            # Check if adding x keeps it Sidon
            new_sums = [x + a for a in current] + [2 * x]
            # Check for internal conflicts in new_sums
            if len(set(new_sums)) != len(new_sums):
                continue
            # Check conflict with existing sums
            if current_sums.intersection(set(new_sums)):
                continue
            
            # It is a valid addition. Add it.
            current.append(x)
            current_set.add(x)
            current_sums.update(new_sums)
            changed = True
            break
            
    return current

def _local_search_reduce(start_set, deadline, rng):
    """
    Local search: iteratively removes elements that do not break maximality.
    This is the primary optimization step.
    """
    current = list(start_set)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                continue
            
            cand_sums = _compute_sums(candidate)
            if _is_maximal_fast(candidate, cand_sums):
                current = candidate
                improved = True
                break

    return current

def _local_search_swap(start_set, deadline, rng, best_size):
    """
    Try to improve by swapping: remove one, add another, check Sidon + maximal.
    Accept only if size decreases or if it enables further reduction.
    """
    current = list(start_set)
    current_size = len(current)
    
    if current_size < 2:
        return current
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline and iterations < 20:
        improved = False
        iterations += 1
        current_set = set(current)
        # Try all pairs (remove r, add a)
        remove_candidates = list(current)
        rng.shuffle(remove_candidates)
        add_candidates = [x for x in range(1, N+1) if x not in current_set]
        rng.shuffle(add_candidates)
        
        for r in remove_candidates:
            if time.time() > deadline:
                break
            for a in add_candidates:
                if time.time() > deadline:
                    break
                candidate = [x for x in current if x != r] + [a]
                candidate = sorted(set(candidate))
                if len(candidate) != current_size:
                    continue
                # Check Sidon
                if not _is_sidon(candidate):
                    continue
                # Check maximal
                cand_sums = _compute_sums(candidate)
                if _is_maximal_fast(candidate, cand_sums):
                    # Accept swap (same size but maybe enables further reductions)
                    current = candidate
                    improved = True
                    break
            if improved:
                break
    return current

def _greedy_most_restrictive(deadline, rng=None):
    """
    Greedy that picks the most restrictive element at each step.
    For each candidate x, compute how many other elements would still be addable.
    Pick x that leaves the fewest options.
    """
    available = list(range(1, N + 1))
    if rng:
        rng.shuffle(available)
    selected = []
    sums = set()
    
    while available and time.time() < deadline:
        best_x = None
        best_options = float('inf')
        best_idx = -1
        
        for i, x in enumerate(available):
            # Try adding x
            new_sums_list = [x + a for a in selected] + [2 * x]
            if len(set(new_sums_list)) != len(new_sums_list):
                continue
            if sums.intersection(set(new_sums_list)):
                continue
            
            # Count how many remaining elements could still be added after x
            temp_sums = sums | set(new_sums_list)
            options = 0
            for y in available:
                if y == x:
                    continue
                # Check if y can be added given selected + x
                added_y = [y + a for a in selected] + [y + x] + [2 * y]
                if len(set(added_y)) != len(added_y):
                    continue
                if temp_sums.intersection(set(added_y)):
                    continue
                options += 1
            
            if options < best_options:
                best_options = options
                best_x = x
                best_idx = i
        
        if best_x is None:
            break
        
        selected.append(best_x)
        available.pop(best_idx)
        new_sums_list = [best_x + a for a in selected[:-1]] + [2 * best_x]
        sums.update(new_sums_list)
    
    return selected

def construct():
    start_time = time.time()
    # Use 28s deadline to be safe against 30s hard limit
    deadline = start_time + 28.0
    
    # Generate diverse starting sets
    candidates = []
    
    # 1. Greedy ascending
    candidates.append(_greedy_ascending())
    
    # 2. Greedy descending
    candidates.append(_greedy_descending())
    
    # 3. Most restrictive greedy (MIS-inspired)
    candidates.append(_greedy_most_restrictive(deadline))
    
    # 4. Multiple random greedy
    for seed in range(50):
        if time.time() > deadline:
            break
        candidates.append(_greedy_random(seed))
    
    # 5. MIS greedy with different random seeds
    for seed in range(20):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        candidates.append(_greedy_most_restrictive(deadline, rng))
    
    # Filter valid Sidon sets
    valid_candidates = []
    for c in candidates:
        if _is_sidon(c):
            valid_candidates.append(c)
            
    if not valid_candidates:
        return _greedy_ascending()
        
    # Sort by size to prioritize smaller sets
    valid_candidates.sort(key=len)
    
    best_set = valid_candidates[0]
    best_size = len(best_set)
    
    # Apply local search to top candidates
    rng = random.Random(42)
    
    # Process top 40 candidates
    for c in valid_candidates[:40]:
        if time.time() > deadline:
            break
        
        # First, try reduction
        reduced = _local_search_reduce(c, deadline, rng)
        
        # Verify the result is still a valid maximal Sidon set
        if _is_sidon(reduced) and _is_maximal_fast(reduced, _compute_sums(reduced)):
            if len(reduced) < best_size:
                best_size = len(reduced)
                best_set = reduced
        
        # If we still have time, try swaps on the best found so far
        if time.time() < deadline - 5.0 and len(best_set) > 10:
            swapped = _local_search_swap(best_set, deadline, rng, best_size)
            if _is_sidon(swapped) and _is_maximal_fast(swapped, _compute_sums(swapped)):
                if len(swapped) < best_size:
                    best_size = len(swapped)
                    best_set = swapped
    
    # Final verification
    if not _is_sidon(best_set):
        return _greedy_ascending()
        
    # Ensure maximality (safety check)
    sums = _compute_sums(best_set)
    if not _is_maximal_fast(best_set, sums):
        best_set = _make_maximal(best_set, deadline)
        
    return best_set

# EVOLVE-BLOCK-END
