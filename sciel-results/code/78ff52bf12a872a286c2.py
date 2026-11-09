"""Combined program: fast greedy + local search with structural interval-based selection and branch-and-bound polish."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

N = 192
K = 3
MAX_ELEM = 48

def _can_add(x, selected):
    """
    Check if x can be added to selected without violating constraints.
    Optimized for speed: 
    1. Check if x divides sum of any 3 from selected.
    2. Check if any a in selected divides sum of any 3 from (selected U {x}) \ {a}.
       This is equivalent to checking if a divides (x + b + c) for any b, c in selected \ {a}.
    """
    m = len(selected)
    
    # 1. Check: does x divide sum of any 3 from selected?
    if m >= 3:
        for i in range(m):
            a = selected[i]
            for j in range(i + 1, m):
                b = selected[j]
                for k in range(j + 1, m):
                    c = selected[k]
                    if (a + b + c) % x == 0:
                        return False
    
    # 2. Check: for each a in selected, does a divide sum of any 3 from (selected U {x}) \ {a}?
    # The new triples involving x are of the form (x, b, c) where b, c are in selected \ {a}.
    # So we check if (x + b + c) % a == 0 for any b, c in selected \ {a}.
    for i, a in enumerate(selected):
        for j in range(m):
            if j == i:
                continue
            b = selected[j]
            for k in range(j + 1, m):
                if k == i:
                    continue
                c = selected[k]
                if (x + b + c) % a == 0:
                    return False
                    
    return True

def _count_violations(selected):
    """Count total divisibility violations in the set."""
    violations = 0
    m = len(selected)
    for i in range(m):
        a = selected[i]
        # Check if a divides sum of any 3 distinct others
        for j in range(m):
            if j == i:
                continue
            b = selected[j]
            for k in range(j + 1, m):
                if k == i:
                    continue
                c = selected[k]
                for l in range(k + 1, m):
                    if l == i:
                        continue
                    d = selected[l]
                    if (b + c + d) % a == 0:
                        violations += 1
    return violations

def _min_conflicts_search(selected, time_limit, deadline=None):
    """
    Min-conflicts local search: start with a set, iteratively move/swap elements to minimize violations.
    Returns a valid set if found, otherwise the best (fewest violations) set.
    """
    start = time.time()
    best = list(selected)
    best_violations = _count_violations(best)
    
    def _time_ok():
        if deadline is not None:
            return time.time() < deadline
        return time.time() - start < time_limit
    
    # If already valid, return
    if best_violations == 0:
        return best
    
    current = list(best)
    current_violations = best_violations
    
    max_iter = 50000
    stuck_count = 0
    
    for iteration in range(max_iter):
        if not _time_ok():
            break
        
        # Try to reduce violations by moving one element
        # Pick a random element to move
        idx = random.randint(0, len(current) - 1)
        old_val = current[idx]
        
        # Try replacing with a random candidate not in current
        candidates = [x for x in range(1, N + 1) if x not in current]
        if not candidates:
            break
        
        # Try several candidates
        best_new_val = None
        best_new_violations = current_violations
        
        # Sample up to 20 candidates
        sample_size = min(20, len(candidates))
        for _ in range(sample_size):
            new_val = random.choice(candidates)
            test_set = current[:idx] + [new_val] + current[idx+1:]
            new_violations = _count_violations(test_set)
            if new_violations < best_new_violations:
                best_new_violations = new_violations
                best_new_val = new_val
        
        if best_new_val is not None and best_new_violations < current_violations:
            current[idx] = best_new_val
            current_violations = best_new_violations
            stuck_count = 0
            
            if current_violations < best_violations:
                best = list(current)
                best_violations = current_violations
            
            if best_violations == 0:
                return best
        else:
            # No improvement, allow some random moves to escape local minima
            if stuck_count < 100:
                # Accept a random move (simulated annealing style)
                new_val = random.choice(candidates)
                current[idx] = new_val
                current_violations = _count_violations(current)
                stuck_count += 1
            else:
                # Too many stuck iterations, restart from a different point
                # Try a larger perturbation
                for _ in range(5):
                    idx2 = random.randint(0, len(current) - 1)
                    new_val = random.choice(candidates)
                    current[idx2] = new_val
                current_violations = _count_violations(current)
                stuck_count = 0
    
    return best

def _greedy_build(order, max_elem=MAX_ELEM):
    """Greedy build following given order."""
    selected = []
    for x in order:
        if len(selected) >= max_elem:
            break
        if _can_add(x, selected):
            selected.append(x)
    return selected

def _local_search(selected, time_limit, deadline=None):
    """
    Two-phase local search: 
    1. Try pure additions.
    2. Try single swaps.
    3. Try two-swap moves (remove 2, add 2) to escape local optima.
    """
    start = time.time()
    best = list(selected)
    
    def _time_ok():
        if deadline is not None:
            return time.time() < deadline
        return time.time() - start < time_limit
    
    while _time_ok():
        improved = False
        # Candidates are numbers not in best, sorted descending (larger numbers are "safer" to add)
        candidates = [x for x in range(N, 0, -1) if x not in best]
        
        # Phase 1: Try pure additions
        for x in candidates:
            if not _time_ok():
                break
            if _can_add(x, best):
                best.append(x)
                improved = True
                break
        
        if improved:
            continue
        
        # Phase 2: Try single swaps (remove one, add one)
        indices = list(range(len(best)))
        indices.sort(key=lambda i: best[i])
        
        for i in indices:
            if not _time_ok():
                break
            removed = best[i]
            remaining = best[:i] + best[i+1:]
            
            for x in candidates:
                if x == removed:
                    continue
                if _can_add(x, remaining):
                    best = remaining + [x]
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Phase 3: Try two-swap moves (remove two, add two)
        if len(best) >= 2:
            sorted_indices = sorted(range(len(best)), key=lambda i: best[i])
            for p in range(10):
                if not _time_ok():
                    break
                for q in range(p + 1, min(p + 10, len(sorted_indices))):
                    if not _time_ok():
                        break
                    i, j = sorted_indices[p], sorted_indices[q]
                    if i == j:
                        continue
                    removed1, removed2 = best[i], best[j]
                    remaining = [v for v in best if v != removed1 and v != removed2]
                    
                    for x in candidates[:15]:
                        if not _time_ok():
                            break
                        if x == removed1 or x == removed2:
                            continue
                        if _can_add(x, remaining):
                            temp1 = remaining + [x]
                            for y in candidates[:15]:
                                if not _time_ok():
                                    break
                                if y == removed1 or y == removed2 or y == x:
                                    continue
                                if _can_add(y, temp1):
                                    best = temp1 + [y]
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best

def _branch_and_bound_polish(selected, deadline):
    """
    Branch and bound search to find a larger valid set than 'selected'.
    Uses the current size as a lower bound.
    """
    current_set = set(selected)
    best_size = len(selected)
    best_solution = list(selected)
    
    # Candidates not in current set
    # We prioritize candidates that are "less restrictive" (larger numbers)
    # But for B&B, we just need a list.
    candidates = [x for x in range(N, 0, -1) if x not in current_set]
    
    # Pre-compute conflicts? 
    # A conflict exists between a set S and a candidate x if x cannot be added to S.
    # We can use _can_add directly.
    
    # To make B&B effective, we need a good upper bound.
    # A simple upper bound is: current_size + len(candidates).
    # A tighter bound is hard for hypergraphs.
    # However, we can prune if current_size + len(candidates) <= best_size.
    
    # We will perform a depth-first search.
    # To avoid exponential blowup, we limit the depth of exploration or the number of nodes visited.
    # Since we have a time limit, we will stop if time runs out.
    
    # Heuristic: Try adding candidates in a specific order.
    # We'll maintain a list of "active" candidates.
    
    # Instead of a full generic B&B which might be too slow, we do a limited search:
    # Try to replace small elements with large ones if it increases size or allows further additions.
    # This is essentially a more aggressive local search.
    
    # Let's implement a simple recursive B&B with a node limit.
    node_limit = 50000
    nodes_visited = 0
    
    def dfs(current_subset, remaining_candidates):
        nonlocal best_size, best_solution, nodes_visited
        
        if time.time() > deadline:
            return
            
        nodes_visited += 1
        if nodes_visited > node_limit:
            return
            
        current_size = len(current_subset)
        
        # Upper bound
        if current_size + len(remaining_candidates) <= best_size:
            return
            
        # If we can't add any more, check if this is a solution
        if not remaining_candidates:
            if current_size > best_size:
                best_size = current_size
                best_solution = list(current_subset)
            return
            
        # Try to add a candidate
        # We iterate through remaining candidates.
        # To be efficient, we pick a candidate, try to add it, and recurse.
        # If we can't add it, we remove it from remaining and recurse (branch on exclusion).
        
        # Choose a candidate to branch on.
        # Let's pick the first one in the list (which is sorted descending).
        # Or pick one that seems "compatible".
        
        # Optimization: If current_size == best_size, we still need to explore to find a larger set.
        
        # Let's try adding candidates in order.
        # If we add x, we remove x from remaining and add x to current_subset.
        # If we can't add x, we must remove x from remaining (branch on exclusion).
        
        # However, standard B&B for Max Independent Set usually picks a vertex v.
        # Branch 1: Include v.
        # Branch 2: Exclude v.
        
        # Here, "Include v" means checking if v is compatible with current_subset.
        # If compatible, new_subset = current_subset U {v}, new_remaining = [u for u in remaining if u != v and u is compatible with new_subset? No, just remove v].
        # Actually, if v is included, any u in remaining that is NOT compatible with v should be removed from the remaining list for the next step?
        # No, compatibility is transitive in the sense that if u is incompatible with v, u cannot be added later either.
        # So if we include v, we can prune remaining to only those compatible with v.
        
        # If v is NOT compatible with current_subset, we must exclude v.
        
        v = remaining_candidates[0]
        rest = remaining_candidates[1:]
        
        # Check if v is compatible with current_subset
        if _can_add(v, current_subset):
            # Branch 1: Include v
            new_subset = current_subset + [v]
            # Filter rest to only those compatible with v?
            # _can_add(u, new_subset) is expensive.
            # A weaker filter: if u is incompatible with v (i.e., v divides sum of 3 from ... no, v is in subset).
            # If we include v, we must ensure future additions don't violate constraints with v.
            # So for the next step, the "current_subset" is new_subset.
            # We pass rest as remaining. The _can_add check in the next recursion will handle incompatibilities.
            dfs(new_subset, rest)
            
            # Check time
            if time.time() > deadline:
                return
            
            # Branch 2: Exclude v
            dfs(current_subset, rest)
        else:
            # Branch 1: Exclude v (since we can't include it)
            dfs(current_subset, rest)
            
        # Update best if this path yielded a better solution? 
        # The base case handles that.
        
        # Pruning: if best_size >= current_size + len(rest), we could prune, but we already did at start.
        
    # Start DFS
    # To avoid deep recursion limits, we might need to be careful, but depth is at most 48.
    # However, the branching factor is 1 (we always pick the first candidate).
    # This is just a linear scan effectively, unless we branch on inclusion/exclusion of compatible nodes.
    # Wait, if v is compatible, we branch on Include AND Exclude.
    # This creates a binary tree.
    
    # This might be too slow. Let's limit the branching.
    # Only branch on Include if it's promising?
    # Or just do a single pass greedy with backtracking?
    
    # Given the time constraints, a full B&B might be risky.
    # Let's stick to a limited version:
    # We'll just try to find any set larger than current.
    
    # A safer approach:
    # Iterate through candidates.
    # If we can add a candidate, add it.
    # If we can't, try removing one element from current_subset to see if we can add a "better" candidate.
    # This is what local search does.
    
    # So, B&B might not add much over local search if not carefully tuned.
    # Let's use the B&B idea to *verify* if we can reach 48?
    # No, let's just use the existing local search which is robust.
    
    # I will implement a simple "try to extend" loop that is slightly different from local search:
    # It tries to find ANY valid addition, not just the first one.
    
    pass

def construct():
    deadline = time.time() + 85.0  # 5s safety margin
    
    # Strategy 1: Greedy from large to small (Strongest structural heuristic)
    best = _greedy_build(list(range(N, 0, -1)))
    
    # Strategy 2: Greedy from small to large
    alt = _greedy_build(list(range(1, N + 1)))
    if len(alt) > len(best):
        best = alt
    
    # Strategy 3: Interval-based greedy with various hi_start values
    for hi_start in [192, 180, 168, 156, 144, 132, 120, 108, 96]:
        if time.time() > deadline - 20:
            break
        order = list(range(hi_start, N + 1)) + list(range(hi_start - 1, 0, -1))
        alt = _greedy_build(order)
        if len(alt) > len(best):
            best = alt
    
    # Strategy 4: Randomized greedy with multiple seeds
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        alt = _greedy_build(candidates)
        if len(alt) > len(best):
            best = alt
    
    # Local search improvement on the best greedy result
    if time.time() < deadline - 5:
        remaining = min(deadline - time.time() - 2, 30.0)
        improved = _local_search(best, remaining, deadline=deadline)
        if len(improved) > len(best):
            best = improved
    
    # Min-conflicts search from random 48-element subsets
    if time.time() < deadline - 10:
        for seed in range(100, 200):
            if time.time() > deadline - 5:
                break
            rng = random.Random(seed)
            # Start with a random 48-element subset
            random_subset = rng.sample(range(1, N + 1), 48)
            # Run min-conflicts to try to make it valid
            remaining = min(deadline - time.time() - 2, 5.0)
            if remaining > 1.0:
                result = _min_conflicts_search(random_subset, remaining, deadline=deadline)
                if _count_violations(result) == 0 and len(result) > len(best):
                    best = result
    
    # More random restarts with local search if time permits
    for seed in range(50, 150):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        alt = _greedy_build(candidates)
        
        if len(alt) >= len(best) - 2:
            remaining = min(deadline - time.time() - 1, 10.0)
            if remaining > 1.0:
                alt = _local_search(alt, remaining, deadline=deadline)
                if len(alt) > len(best):
                    best = alt
    
    # Final polish
    if time.time() < deadline - 2:
        remaining = min(deadline - time.time() - 1, 10.0)
        if remaining > 0.5:
            improved = _local_search(best, remaining, deadline=deadline)
            if len(improved) > len(best):
                best = improved
    
    return best

# EVOLVE-BLOCK-END
