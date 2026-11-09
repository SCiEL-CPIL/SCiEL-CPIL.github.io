import time
import random
import sys

# EVOLVE-BLOCK-START

def _precompute_conflicts():
    """
    Precompute all conflict triples (a, b, c) where a, b, c are distinct,
    1 <= a,b,c <= 180, and 1/a = 1/b + 1/c.
    Returns a list of (a, b, c) tuples.
    """
    conflicts = []
    for a in range(1, 181):
        a2 = a * a
        # d = b - a, so b = a + d, c = a + a2/d
        # d must divide a2, d != 0, d != a (to ensure b != c)
        # b = a + d >= 1, c = a + a2//d >= 1
        # b, c <= 180
        # d > 0 (since b > a for the equation to hold with positive numbers)
        for d in range(1, 181):  # d = b - a, b up to 180, a >= 1, so d up to 179
            if a2 % d != 0:
                continue
            c_val = a + a2 // d
            b_val = a + d
            if b_val > 180 or c_val > 180:
                continue
            if b_val <= 0 or c_val <= 0:
                continue
            if b_val == a or c_val == a:
                continue
            if b_val == c_val:
                continue
            conflicts.append((a, b_val, c_val))
    return conflicts

def _build_conflict_graph(conflicts):
    """
    Build a graph where edge (x, y) means x and y cannot both be in the set.
    Actually, the constraint is: for each conflict (a, b, c), we cannot have all three.
    But it's a hypergraph. For efficiency in checking, we can precompute for each pair
    which conflicts involve them, or just store the list of conflicts.
    
    A better approach: for each element x, precompute the set of (y, z) such that
    (x, y, z) is a conflict (in any order). Then when adding x, check if any conflict
    is completed.
    
    Let's build: for each x, a list of pairs (y, z) where (x, y, z) or (y, x, z) or (y, z, x)
    is a conflict triple. Actually, the equation 1/a = 1/b + 1/c is not symmetric in all three,
    but the constraint is: no three distinct a, b, c in the set satisfy the equation.
    
    Let's just store all conflict triples as frozensets for quick lookup, and also
    for each element, the list of conflict triples it participates in.
    """
    # conflict_triples: list of frozenset({a,b,c})
    # elem_to_conflicts: dict x -> list of indices into conflict_triples
    conflict_triples = []
    elem_to_conflicts = {i: [] for i in range(1, 181)}
    
    for (a, b, c) in conflicts:
        triple = frozenset((a, b, c))
        idx = len(conflict_triples)
        conflict_triples.append(triple)
        elem_to_conflicts[a].append(idx)
        elem_to_conflicts[b].append(idx)
        elem_to_conflicts[c].append(idx)
    
    return conflict_triples, elem_to_conflicts

def _check_valid_with_conflicts(s_set, conflict_triples, elem_to_conflicts):
    """
    Check if s_set (a set of ints) has no conflict triple fully contained.
    """
    for idx, triple in enumerate(conflict_triples):
        if triple.issubset(s_set):
            return False
    return True

def _incremental_check_valid(s_set, new_elem, conflict_triples, elem_to_conflicts):
    """
    Check if adding new_elem to s_set creates any conflict.
    Only need to check conflicts that involve new_elem.
    """
    for idx in elem_to_conflicts[new_elem]:
        triple = conflict_triples[idx]
        # Check if the other two elements of the triple are in s_set
        others = triple - {new_elem}
        if others.issubset(s_set):
            return False
    return True

def _greedy_multi_order(elements, conflict_triples, elem_to_conflicts, order_func, deadline):
    """
    Greedily build a valid set by adding elements in the order specified by order_func.
    """
    current = set()
    for x in elements:
        if time.time() > deadline:
            break
        if x in current:
            continue
        if _incremental_check_valid(current, x, conflict_triples, elem_to_conflicts):
            current.add(x)
    return current

def _local_search(current_set, conflict_triples, elem_to_conflicts, all_elements, deadline, max_rounds=50):
    """
    Local search: try to add missing elements, and if stuck, try swaps.
    Returns the best set found.
    """
    current = set(current_set)
    best = set(current)
    all_elems_set = set(all_elements)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Try to add any missing element
        missing = [x for x in all_elements if x not in current]
        random.shuffle(missing)
        added_any = False
        for m in missing:
            if time.time() > deadline:
                break
            if _incremental_check_valid(current, m, conflict_triples, elem_to_conflicts):
                current.add(m)
                added_any = True
                if len(current) > len(best):
                    best = set(current)
        
        if added_any:
            continue
        
        # Stuck: try a swap or a remove+add-2
        # Option 1: Remove one, add one (net 0) - might unblock future adds
        # Option 2: Remove one, add two (net +1)
        
        # Try Option 2 first: remove x from current, try to add two from missing
        in_set = list(current)
        random.shuffle(in_set)
        found_improvement = False
        
        for x in in_set:
            if time.time() > deadline:
                break
            temp = current - {x}
            missing_after = [y for y in all_elements if y not in temp]
            random.shuffle(missing_after)
            
            # Try to add two
            added = 0
            temp_set = set(temp)
            for y in missing_after:
                if added >= 2:
                    break
                if _incremental_check_valid(temp_set, y, conflict_triples, elem_to_conflicts):
                    temp_set.add(y)
                    added += 1
            
            if len(temp_set) > len(current):
                current = temp_set
                found_improvement = True
                if len(current) > len(best):
                    best = set(current)
                break
        
        if found_improvement:
            continue
        
        # Try Option 1: Remove one, add one (net 0), then try to add more
        # Actually, let's try: remove x, then greedily add as many as possible
        for x in in_set:
            if time.time() > deadline:
                break
            temp_set = current - {x}
            # Greedy add
            missing_after = [y for y in all_elements if y not in temp_set]
            random.shuffle(missing_after)
            for y in missing_after:
                if time.time() > deadline:
                    break
                if _incremental_check_valid(temp_set, y, conflict_triples, elem_to_conflicts):
                    temp_set.add(y)
            
            if len(temp_set) > len(current):
                current = temp_set
                found_improvement = True
                if len(current) > len(best):
                    best = set(current)
                break
        
        if not found_improvement:
            # No improvement in this round, break
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s margin before 90s kill
    
    # Precompute conflicts
    conflicts = _precompute_conflicts()
    conflict_triples, elem_to_conflicts = _build_conflict_graph(conflicts)
    
    all_elements = list(range(1, 181))
    
    best_set = set()
    best_size = 0
    
    # Multiple greedy orders
    # Order 1: descending (180 to 1)
    current = _greedy_multi_order(all_elements[::-1], conflict_triples, elem_to_conflicts, None, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
    
    # Order 2: ascending (1 to 180)
    current = _greedy_multi_order(all_elements, conflict_triples, elem_to_conflicts, None, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = set(current)
    
    # Order 3: random restarts
    for restart in range(20):
        if time.time() > deadline:
            break
        random.seed(200 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        current = _greedy_multi_order(shuffled, conflict_triples, elem_to_conflicts, None, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
    
    # Local search from best_set
    improved_set = _local_search(best_set, conflict_triples, elem_to_conflicts, all_elements, deadline)
    if len(improved_set) > best_size:
        best_size = len(improved_set)
        best_set = improved_set
    
    # More local search restarts from different greedy solutions
    for restart in range(10):
        if time.time() > deadline:
            break
        random.seed(300 + restart)
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        current = _greedy_multi_order(shuffled, conflict_triples, elem_to_conflicts, None, deadline)
        improved_set = _local_search(current, conflict_triples, elem_to_conflicts, all_elements, deadline)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return list(best_set)

# EVOLVE-BLOCK-END
