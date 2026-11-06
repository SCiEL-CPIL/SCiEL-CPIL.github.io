import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s kill
    
    N = 180
    
    # Precompute all conflict triples (a, b, c) with a < b < c, 1/a = 1/b + 1/c
    # This means c = a*b / (b - a), and we need c to be integer and c <= N
    # For a < b < c: 1/a = 1/b + 1/c => c = ab/(b-a)
    # b > a, and b-a must divide a*b. Let d = b - a, so b = a + d
    # c = a(a+d)/d = a + a^2/d. So d must divide a^2.
    # d < b < c => d < a+d < a + a^2/d
    # The last inequality: a+d < a + a^2/d => d < a^2/d => d^2 < a^2 => d < a
    # So d < a, d divides a^2, and c = a + a^2/d <= N, b = a + d <= N
    
    conflicts = []  # list of (a, b, c) triples
    for a in range(1, N + 1):
        a2 = a * a
        # d must divide a^2, d < a, and a + a^2/d <= N, a + d <= N
        for d in range(1, min(a, N - a + 1)):
            if a2 % d == 0:
                b = a + d
                c = a + a2 // d
                if b < c and c <= N:
                    conflicts.append((a, b, c))
    
    num_conflicts = len(conflicts)
    
    # Strategy: Start with the large set {91, ..., 180} (size 90)
    # Then try to add elements from 1 to 90 one by one (greedy)
    # Use a conflict graph approach: if adding x creates a conflict, skip it
    
    # For efficiency, maintain a set of "forbidden" elements
    # An element is forbidden if it's part of a conflict with two other already-selected elements
    
    selected = set(range(91, 181))
    selected_size = 90
    
    # For each potential addition, check if it creates any conflict
    # A conflict (a,b,c) is violated if all three are in selected
    # When we add element x, a conflict is newly violated if x is one of a,b,c
    # and the other two are already in selected
    
    # Build adjacency: for each element, which conflicts involve it
    conflicts_by_elem = {i: [] for i in range(1, N + 1)}
    for (a, b, c) in conflicts:
        conflicts_by_elem[a].append((b, c))
        conflicts_by_elem[b].append((a, c))
        conflicts_by_elem[c].append((a, b))
    
    # Try adding elements from 1 to 90 in some order
    # Order: try elements that have fewer conflicts first (safer to add)
    candidates = list(range(1, 91))
    # Sort by number of conflicts involving them
    candidates.sort(key=lambda x: len(conflicts_by_elem[x]))
    
    for x in candidates:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        
        # Check if adding x would create a conflict
        safe = True
        for (other1, other2) in conflicts_by_elem[x]:
            if other1 in selected and other2 in selected:
                safe = False
                break
        
        if safe:
            selected.add(x)
            selected_size += 1
    
    # Now try a second pass: see if we can remove some elements and add more
    # This is a local search / hill climbing step
    best_set = set(selected)
    best_size = len(selected)
    
    # Try single-swap improvements: remove one element, then greedily add others
    for iter_num in range(5):
        if time.time() > deadline:
            break
        
        improved = False
        for remove_candidate in sorted(selected, key=lambda x: len(conflicts_by_elem[x]), reverse=True):
            if time.time() > deadline:
                break
            
            trial = set(selected)
            trial.remove(remove_candidate)
            
            # Now greedily add back elements not in trial
            # Sort candidates by conflict count
            not_in_trial = [x for x in range(1, N + 1) if x not in trial]
            not_in_trial.sort(key=lambda x: len(conflicts_by_elem[x]))
            
            for x in not_in_trial:
                if x in trial:
                    continue
                safe = True
                for (other1, other2) in conflicts_by_elem[x]:
                    if other1 in trial and other2 in trial:
                        safe = False
                        break
                if safe:
                    trial.add(x)
            
            if len(trial) > best_size:
                best_size = len(trial)
                best_set = trial
                selected = trial
                improved = True
                break
        
        if not improved:
            break
    
    # Final verification (should be fast for our purposes)
    result = sorted(best_set)
    
    # Verify no conflicts
    result_set = set(result)
    for (a, b, c) in conflicts:
        if a in result_set and b in result_set and c in result_set:
            # Remove one element to break the conflict
            # Remove the largest to preserve density
            result_set.remove(c)
            result = sorted(result_set)
    
    if not result:
        return list(range(91, 181))
    
    return result
