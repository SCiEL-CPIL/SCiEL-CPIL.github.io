import time
import random
import sys
from collections import defaultdict

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """
    Precompute all conflict triples (a, b, c) where a, b, c are distinct,
    1 <= a,b,c <= 180, and 1/a = 1/b + 1/c.
    This means b*c = a*(b+c).
    """
    conflicts = []
    for a in range(1, 181):
        for b in range(a + 1, 181):
            # 1/a = 1/b + 1/c => c = a*b/(b-a)
            # b-a must divide a*b
            d = b - a
            if (a * b) % d != 0:
                continue
            c = (a * b) // d
            if c <= 0 or c > 180:
                continue
            if c == a or c == b:
                continue
            conflicts.append((a, b, c))
    return conflicts

def _build_conflict_data(conflicts):
    """
    Build data structures for efficient conflict checking.
    For each element x, store the set of pairs (y, z) such that {x, y, z} is a conflict.
    """
    # For each x, store a set of frozensets of the other two elements
    elem_conflict_pairs = {i: set() for i in range(1, 181)}
    conflict_triples = []
    
    for (a, b, c) in conflicts:
        triple = frozenset((a, b, c))
        conflict_triples.append(triple)
        elem_conflict_pairs[a].add(frozenset((b, c)))
        elem_conflict_pairs[b].add(frozenset((a, c)))
        elem_conflict_pairs[c].add(frozenset((a, b)))
    
    return conflict_triples, elem_conflict_pairs

def _can_add(current, x, elem_conflict_pairs):
    """Check if x can be added to current without creating a conflict."""
    for pair in elem_conflict_pairs[x]:
        if pair.issubset(current):
            return False
    return True

def _greedy_fixed_order(elements, elem_conflict_pairs, deadline):
    """Greedy: add elements in given order if possible."""
    current = set()
    for x in elements:
        if time.time() > deadline:
            break
        if x in current:
            continue
        if _can_add(current, x, elem_conflict_pairs):
            current.add(x)
    return current

def _greedy_min_degree(elem_conflict_pairs, deadline):
    """
    Minimum-degree greedy: repeatedly pick the element with fewest conflicts
    among remaining elements that can be added.
    """
    current = set()
    remaining = set(range(1, 181))
    
    while remaining and time.time() < deadline:
        best_x = None
        best_degree = float('inf')
        
        for x in remaining:
            if not _can_add(current, x, elem_conflict_pairs):
                continue
            # Count how many potential conflicts x has with remaining elements
            degree = 0
            for pair in elem_conflict_pairs[x]:
                # Check if either element of the pair is in remaining
                for y in pair:
                    if y in remaining:
                        degree += 1
                        break
            if degree < best_degree:
                best_degree = degree
                best_x = x
        
        if best_x is None:
            break
        
        current.add(best_x)
        remaining.discard(best_x)
    
    return current

def _local_search_1swap(current, elem_conflict_pairs, all_elements, deadline):
    """
    Local search: try removing one element and adding another (or more).
    """
    current = set(current)
    best = set(current)
    all_set = set(all_elements)
    
    for _ in range(200):
        if time.time() > deadline:
            break
        
        improved = False
        in_list = list(current)
        random.shuffle(in_list)
        
        for x in in_list:
            if time.time() > deadline:
                break
            temp = current - {x}
            missing = [y for y in all_elements if y not in temp]
            random.shuffle(missing)
            
            # Try to add as many as possible
            added = 0
            for y in missing:
                if _can_add(temp, y, elem_conflict_pairs):
                    temp.add(y)
                    added += 1
                    if added >= 2:
                        break
            
            if len(temp) > len(current):
                current = temp
                improved = True
                if len(current) > len(best):
                    best = set(current)
                break
        
        if not improved:
            break
    
    return best

def _local_search_2swap(current, elem_conflict_pairs, all_elements, deadline):
    """
    Local search: try removing two elements and adding up to three.
    """
    current = set(current)
    best = set(current)
    all_set = set(all_elements)
    
    for _ in range(100):
        if time.time() > deadline:
            break
        
        improved = False
        in_list = list(current)
        random.shuffle(in_list)
        
        for i in range(len(in_list)):
            if time.time() > deadline:
                break
            x1 = in_list[i]
            for j in range(i + 1, len(in_list)):
                x2 = in_list[j]
                temp = current - {x1, x2}
                missing = [y for y in all_elements if y not in temp]
                random.shuffle(missing)
                
                # Try to add up to 3
                added = 0
                for y in missing:
                    if added >= 3:
                        break
                    if _can_add(temp, y, elem_conflict_pairs):
                        temp.add(y)
                        added += 1
                
                if len(temp) > len(current):
                    current = temp
                    improved = True
                    if len(current) > len(best):
                        best = set(current)
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best

def _simulated_annealing(current, elem_conflict_pairs, all_elements, deadline):
    """
    Simulated annealing: accept worse solutions with decreasing probability.
    """
    current = set(current)
    best = set(current)
    all_set = set(all_elements)
    
    T = 1.0
    cooling = 0.995
    
    while time.time() < deadline and T > 0.01:
        # Pick a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try to add a random missing element
            missing = [x for x in all_elements if x not in current]
            if not missing:
                T *= cooling
                continue
            x = random.choice(missing)
            if _can_add(current, x, elem_conflict_pairs):
                new_set = current | {x}
                if len(new_set) > len(best):
                    best = set(new_set)
                current = new_set
            else:
                # Reject or accept based on temperature (here we just reject)
                pass
        elif move_type < 0.6:
            # Try to remove a random element
            in_list = list(current)
            if not in_list:
                T *= cooling
                continue
            x = random.choice(in_list)
            new_set = current - {x}
            # Always accept removal if it allows future additions
            current = new_set
        else:
            # Try a swap: remove one, add one
            in_list = list(current)
            missing = [x for x in all_elements if x not in current]
            if not in_list or not missing:
                T *= cooling
                continue
            x_remove = random.choice(in_list)
            x_add = random.choice(missing)
            temp = current - {x_remove}
            if _can_add(temp, x_add, elem_conflict_pairs):
                new_set = temp | {x_add}
                if len(new_set) > len(best):
                    best = set(new_set)
                current = new_set
        
        T *= cooling
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    conflict_triples, elem_conflict_pairs = _build_conflict_data(conflicts)
    
    all_elements = list(range(1, 181))
    
    best_set = set()
    best_size = 0
    
    # Fixed-order greedy: ascending
    current = _greedy_fixed_order(all_elements, elem_conflict_pairs, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
    
    # Fixed-order greedy: descending
    current = _greedy_fixed_order(all_elements[::-1], elem_conflict_pairs, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
    
    # Minimum-degree greedy
    current = _greedy_min_degree(elem_conflict_pairs, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
    
    # Random restarts with fixed-order greedy
    for restart in range(30):
        if time.time() > deadline:
            break
        random.seed(1000 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        current = _greedy_fixed_order(shuffled, elem_conflict_pairs, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Local search from best_set
    improved = _local_search_1swap(best_set, elem_conflict_pairs, all_elements, deadline)
    if len(improved) > best_size:
        best_size = len(improved)
        best_set = improved
    
    improved = _local_search_2swap(best_set, elem_conflict_pairs, all_elements, deadline)
    if len(improved) > best_size:
        best_size = len(improved)
        best_set = improved
    
    # More restarts with local search
    for restart in range(20):
        if time.time() > deadline:
            break
        random.seed(2000 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        current = _greedy_fixed_order(shuffled, elem_conflict_pairs, deadline)
        improved = _local_search_1swap(current, elem_conflict_pairs, all_elements, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
        improved = _local_search_2swap(improved, elem_conflict_pairs, all_elements, deadline)
        if len(improved) > best_size:
            best_size = len(improved)
            best_set = improved
    
    # Simulated annealing from best
    improved = _simulated_annealing(best_set, elem_conflict_pairs, all_elements, deadline)
    if len(improved) > best_size:
        best_size = len(improved)
        best_set = improved
    
    return list(best_set)

# EVOLVE-BLOCK-END
