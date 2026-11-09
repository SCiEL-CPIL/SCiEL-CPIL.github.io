"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _find_violations(selected):
    """
    Find all violations in the selected set.
    Returns a list of tuples (a, b, c, d) where a divides b+c+d.
    Also returns a count of how many violations each element is involved in.
    """
    n = len(selected)
    if n < 4:
        return [], {}
    
    violations = []
    counts = {x: 0 for x in selected}
    
    for i, a in enumerate(selected):
        others = [v for j, v in enumerate(selected) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            s = t[0] + t[1] + t[2]
            if s % a == 0:
                violations.append((a, t[0], t[1], t[2]))
                counts[a] += 1
                counts[t[0]] += 1
                counts[t[1]] += 1
                counts[t[2]] += 1
                
    return violations, counts

def _is_valid_set(selected):
    """Full validation of a set. Used only for final check."""
    for a in selected:
        others = [v for v in selected if v != a]
        if len(others) >= 3:
            for t in combinations(others, 3):
                if sum(t) % a == 0:
                    return False
    return True

def _valid_addition(x, selected):
    """Check if adding x to selected keeps the set valid.
    Optimized: Check x against existing triples, and existing elements against new triples involving x.
    """
    sel = selected
    # Condition 1: x must not divide sum of any 3 distinct others in sel
    if len(sel) >= 3:
        for t in combinations(sel, 3):
            if sum(t) % x == 0:
                return False
    
    # Condition 2: For each a in sel, x + (two others) must not be divisible by a
    # This is equivalent to: for all pairs (u, v) in sel, u != v, (x + u + v) % a != 0 for all a in sel?
    # No, the condition is: a does not divide (x + u + v) where u, v are distinct from a and x.
    # So for each a in sel, we check pairs (u, v) from sel \ {a}.
    for a in sel:
        others = [v for v in sel if v != a]
        if len(others) >= 2:
            for i in range(len(others)):
                for j in range(i + 1, len(others)):
                    if (x + others[i] + others[j]) % a == 0:
                        return False
    return True

def _greedy_removal(initial_set, time_limit=10.0):
    """
    Start with initial_set and greedily remove elements to satisfy constraints.
    """
    selected = list(initial_set)
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        violations, counts = _find_violations(selected)
        
        if not violations:
            return selected
            
        # Find element with max count
        max_count = max(counts.values())
        candidates = [x for x in selected if counts[x] == max_count]
        to_remove = min(candidates)
        selected.remove(to_remove)
        
    return selected

def _simple_local_search(selected, time_limit, candidates):
    """Fast local search: try adding candidates, then remove-one-add-two if stuck."""
    selected = list(selected)
    deadline = time.time() + time_limit
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Step 1: Try to add any candidate
        for x in candidates:
            if time.time() >= deadline:
                break
            if x not in selected:
                if _valid_addition(x, selected):
                    selected.append(x)
                    improved = True
                    break
        
        # Step 2: If no addition possible, try remove-one-add-two
        if not improved:
            for i in range(len(selected)):
                if time.time() >= deadline:
                    break
                a = selected[i]
                sel_without_a = selected[:i] + selected[i+1:]
                added = []
                # Try to add 2 candidates
                for x in candidates:
                    if time.time() >= deadline:
                        break
                    if x == a or x in sel_without_a or x in added:
                        continue
                    if _valid_addition(x, sel_without_a + added):
                        added.append(x)
                        if len(added) == 2:
                            break
                if len(added) == 2:
                    selected = sel_without_a + added
                    improved = True
                    break
    return selected

def _min_conflicts_search(initial_set, time_limit, candidates):
    """
    Min-conflicts local search: start with initial_set, try to fix violations by swapping elements.
    """
    selected = list(initial_set)
    deadline = time.time() + time_limit
    best_set = list(selected)
    best_size = len(selected)
    
    # First, ensure initial set is valid or close to valid
    # If not valid, we might need to remove elements first, but min-conflicts usually starts from a solution.
    # Let's start with a valid set from greedy removal.
    
    while time.time() < deadline:
        violations, counts = _find_violations(selected)
        if not violations:
            if len(selected) > best_size:
                best_set = list(selected)
                best_size = len(selected)
            # Try to add
            added = False
            for x in candidates:
                if x in selected:
                    continue
                if _valid_addition(x, selected):
                    selected.append(x)
                    added = True
                    break
            if not added:
                # No more additions, try swaps?
                # For simplicity, break if we can't add and no violations
                # But we can try swapping a low-degree element with a candidate?
                # Actually, if valid and can't add, we are at a local max for this set.
                # Let's try a random swap to escape local max.
                if len(selected) > 0:
                    i = random.randint(0, len(selected) - 1)
                    # Remove selected[i]
                    removed = selected.pop(i)
                    # Try to add a random candidate
                    candidates_not_in = [x for x in candidates if x not in selected]
                    if candidates_not_in:
                        new_x = random.choice(candidates_not_in)
                        # Check if valid with new_x
                        if _valid_addition(new_x, selected):
                            selected.append(new_x)
                        else:
                            # Revert
                            selected.append(removed)
                            # Try another candidate?
                            # For now, just revert and continue loop
                    else:
                        selected.append(removed)
        else:
            # Has violations. Pick a random violating element and try to swap it.
            # Pick element with max count
            max_count = max(counts.values())
            candidates_to_swap = [x for x in selected if counts[x] == max_count]
            to_swap = random.choice(candidates_to_swap)
            
            # Try to replace to_swap with a candidate
            success = False
            random.shuffle(candidates)
            for x in candidates:
                if x in selected:
                    continue
                # Check if replacing to_swap with x reduces violations or keeps valid
                new_selected = [v for v in selected if v != to_swap]
                if _valid_addition(x, new_selected):
                    new_selected.append(x)
                    # Check if new_selected has fewer violations or is valid
                    new_violations, _ = _find_violations(new_selected)
                    if len(new_violations) < len(violations) or (not new_violations):
                        selected = new_selected
                        success = True
                        break
            if not success:
                # Try just removing the element?
                # Or try a different element to swap
                pass

    return best_set

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 60s limit, keep 5s margin
    
    # Strategy: Greedy removal from [144, 192]
    # This range has 49 elements.
    initial_pool = list(range(144, 193))
    
    best = _greedy_removal(initial_pool, time_limit=5.0)
    
    # If we didn't get 48, try other pools or local search
    if len(best) < 48 and time.time() < deadline:
        # Try slightly wider pool
        for start_val in [143, 142, 141, 140]:
            if time.time() >= deadline:
                break
            pool = list(range(start_val, 193))
            # We can limit the removal process to keep it fast
            candidate = _greedy_removal(pool, time_limit=5.0)
            if len(candidate) > len(best):
                best = candidate
                
    # Local search to add more elements
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 2
        # Candidates are all numbers not in best
        candidates = [x for x in range(1, 193) if x not in best]
        # Sort candidates descending to prioritize larger numbers (heuristic)
        candidates.sort(reverse=True)
        best = _simple_local_search(best, remaining, candidates)
        
    # Final check: is best valid?
    # The greedy removal ensures validity at the end.
    # But let's double check.
    violations, _ = _find_violations(best)
    if violations:
        # Should not happen if _greedy_removal worked correctly
        # Fallback to a simple valid set
        # Just return a small valid set to avoid 0 score
        best = [192, 191, 190, 189] # Check if valid?
        # 192 | 191+190+189 = 570? 570/192 no.
        # 191 | 570? 570/191 no.
        # 190 | 570? 570/190 = 2.94 no.
        # 189 | 570? 570/189 no.
        # So [189,190,191,192] is valid.
        
    return best

# EVOLVE-BLOCK-END
