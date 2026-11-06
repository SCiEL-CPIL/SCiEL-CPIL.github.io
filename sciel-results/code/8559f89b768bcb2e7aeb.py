"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from itertools import combinations

def _is_valid(selected):
    """Check if selected set satisfies the constraint."""
    n = len(selected)
    if n < 4:
        return True
    sel_set = set(selected)
    for i in range(n):
        a = selected[i]
        others = [selected[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _can_add_fast(selected, x):
    """Fast check if x can be added to selected without violating constraints."""
    n = len(selected)
    
    # Check x against existing triples: no triple from S sums to multiple of x
    if n >= 3:
        for triple in combinations(selected, 3):
            if sum(triple) % x == 0:
                return False
    
    # Check each existing element a against pairs + x
    # For each a in S, for each pair (s1, s2) from S\{a}, check (s1+s2+x) % a != 0
    if n >= 2:
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i:
                    continue
                s1 = selected[j]
                for k in range(j+1, n):
                    if k == i:
                        continue
                    s2 = selected[k]
                    if (s1 + s2 + x) % a == 0:
                        return False
    
    return True

def _conflict_count(x, selected):
    """Count how many constraints x would violate if added to selected."""
    n = len(selected)
    conflicts = 0
    
    # Check x against existing triples
    if n >= 3:
        for triple in combinations(selected, 3):
            if sum(triple) % x == 0:
                conflicts += 1
    
    # Check each existing element a against pairs + x
    if n >= 2:
        for i in range(n):
            a = selected[i]
            for j in range(n):
                if j == i:
                    continue
                s1 = selected[j]
                for k in range(j+1, n):
                    if k == i:
                        continue
                    s2 = selected[k]
                    if (s1 + s2 + x) % a == 0:
                        conflicts += 1
    
    return conflicts

def _greedy_heuristic(deadline):
    """Greedy that picks the element with fewest conflicts at each step."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining and time.time() < deadline:
        best_x = None
        best_conflicts = float('inf')
        
        for x in remaining:
            c = _conflict_count(x, selected)
            if c < best_conflicts:
                best_conflicts = c
                best_x = x
            elif c == best_conflicts and best_x is not None and x > best_x:
                # Tie-break: prefer larger numbers
                best_x = x
        
        if best_x is not None and best_conflicts == 0:
            selected.append(best_x)
            remaining.discard(best_x)
        else:
            break
    
    return selected

def _greedy_order(order, deadline):
    """Greedy construction with given order."""
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if _can_add_fast(selected, x):
            selected.append(x)
    return selected

def _local_improve(selected, deadline):
    """Local search: try to add more elements or swap."""
    selected = list(selected)
    sel_set = set(selected)
    
    # Try to add elements - try all candidates, prefer those with fewer conflicts
    candidates = [x for x in range(1, 193) if x not in sel_set]
    
    # Sort candidates by conflict count (fewer conflicts first)
    def sort_key(x):
        return _conflict_count(x, selected)
    
    candidates.sort(key=sort_key)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sel_set = set(selected)
        for x in candidates[:]:
            if time.time() > deadline:
                break
            if x not in sel_set:
                if _can_add_fast(selected, x):
                    selected.append(x)
                    sel_set.add(x)
                    candidates.remove(x)
                    improved = True
    
    # Try 1-for-2 swaps: remove one element, try to add two new ones
    for i in range(len(selected)):
        if time.time() > deadline:
            break
        removed = selected[i]
        remaining = selected[:i] + selected[i+1:]
        rem_set = set(remaining)
        
        candidates_for_swap = [x for x in range(1, 193) if x not in rem_set and x != removed]
        # Sort by conflict count
        candidates_for_swap.sort(key=lambda x: _conflict_count(x, remaining))
        
        added = []
        for x in candidates_for_swap:
            if time.time() > deadline:
                break
            if len(added) == 2:
                break
            if _can_add_fast(remaining + added, x):
                added.append(x)
        
        if len(added) >= 2:
            new_selected = remaining + added
            if len(new_selected) > len(selected):
                selected = new_selected
    
    # Try 2-for-3 swaps: remove two elements, try to add three new ones
    for i in range(len(selected)):
        if time.time() > deadline:
            break
        for j in range(i+1, len(selected)):
            if time.time() > deadline:
                break
            removed = [selected[i], selected[j]]
            remaining = [x for x in selected if x not in removed]
            rem_set = set(remaining)
            
            candidates_for_swap = [x for x in range(1, 193) if x not in rem_set and x not in removed]
            candidates_for_swap.sort(key=lambda x: _conflict_count(x, remaining))
            
            added = []
            for x in candidates_for_swap:
                if time.time() > deadline:
                    break
                if len(added) == 3:
                    break
                if _can_add_fast(remaining + added, x):
                    added.append(x)
            
            if len(added) >= 3:
                new_selected = remaining + added
                if len(new_selected) > len(selected):
                    selected = new_selected
                    break
    
    return selected

def _simulated_annealing(selected, deadline):
    """Simulated annealing style local search."""
    selected = list(selected)
    current_size = len(selected)
    best = list(selected)
    best_size = current_size
    
    temp = 30.0
    cooling = 0.997
    
    while time.time() < deadline and temp > 0.5:
        # Try a random move
        r = random.random()
        if r < 0.35:
            move_type = 'add'
        elif r < 0.55:
            move_type = 'remove'
        elif r < 0.85:
            move_type = 'swap'
        else:
            move_type = 'double_swap'
        
        if move_type == 'add':
            candidates = [x for x in range(1, 193) if x not in set(selected)]
            if not candidates:
                temp *= cooling
                continue
            x = random.choice(candidates)
            new_selected = selected + [x]
        elif move_type == 'remove':
            if len(selected) <= 1:
                temp *= cooling
                continue
            i = random.randint(0, len(selected) - 1)
            new_selected = selected[:i] + selected[i+1:]
        elif move_type == 'swap':
            if len(selected) < 1:
                temp *= cooling
                continue
            i = random.randint(0, len(selected) - 1)
            removed = selected[i]
            remaining = selected[:i] + selected[i+1:]
            candidates = [x for x in range(1, 193) if x not in set(remaining) and x != removed]
            if not candidates:
                temp *= cooling
                continue
            x = random.choice(candidates)
            new_selected = remaining + [x]
        else:  # double_swap: remove one, add two
            if len(selected) < 1:
                temp *= cooling
                continue
            i = random.randint(0, len(selected) - 1)
            removed = selected[i]
            remaining = selected[:i] + selected[i+1:]
            candidates = [x for x in range(1, 193) if x not in set(remaining) and x != removed]
            if len(candidates) < 2:
                temp *= cooling
                continue
            x1, x2 = random.sample(candidates, 2)
            new_selected = remaining + [x1, x2]
        
        # Check validity
        if _is_valid(new_selected):
            new_size = len(new_selected)
            delta = new_size - current_size
            if delta >= 0 or random.random() < math.exp(delta / temp):
                selected = new_selected
                current_size = new_size
                if current_size > best_size:
                    best = list(selected)
                    best_size = current_size
        
        temp *= cooling
    
    return best

def _random_greedy(deadline):
    """Randomized greedy with multiple restarts."""
    best = []
    
    # Try several random orderings
    for attempt in range(40):
        if time.time() > deadline:
            break
        rand_order = list(range(1, 193))
        random.shuffle(rand_order)
        result = _greedy_order(rand_order, deadline)
        if len(result) > len(best):
            best = result
    
    # Structured orderings
    structured_orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
        list(range(96, 0, -1)) + list(range(97, 193)),
        [x for x in range(192, 0, -1) if x > 64],
        [x for x in range(2, 193) if all(x % i != 0 for i in range(2, int(x**0.5)+1))],
        [x for x in range(192, 0, -1) if x % 2 != 0 and x % 3 != 0],
        # Numbers in upper half
        list(range(192, 96, -1)),
        # Numbers not divisible by 2, 3, or 5
        [x for x in range(192, 0, -1) if x % 2 != 0 and x % 3 != 0 and x % 5 != 0],
    ]
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        result = _greedy_order(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Heuristic greedy
    if time.time() < deadline:
        result = _greedy_heuristic(deadline)
        if len(result) > len(best):
            best = result
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 15 second safety margin
    
    # Phase 1: Randomized greedy to find good initial solution
    best = _random_greedy(deadline)
    
    # Phase 2: Local improvement
    if time.time() < deadline:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 3: Simulated annealing
    if time.time() < deadline - 8:
        improved = _simulated_annealing(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 4: More local improvement rounds
    if time.time() < deadline - 4:
        improved = _local_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Phase 5: Another SA round if time permits
    if time.time() < deadline - 2:
        improved = _simulated_annealing(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if not _is_valid(best):
        # Fallback to simple greedy
        best = _greedy_order(list(range(192, 0, -1)), deadline)
    
    return best
# EVOLVE-BLOCK-END
