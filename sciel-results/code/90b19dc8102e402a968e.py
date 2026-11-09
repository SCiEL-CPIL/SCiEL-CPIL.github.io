"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _precompute_bad_triples(max_n):
    """Precompute all bad triples (a,b,c) with 1<=a<b<c<=max_n."""
    n = max_n
    # Precompute lcm table
    lcm_table = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[i][j] = _lcm(i, j)
            if i != j:
                lcm_table[j][i] = lcm_table[i][j]
    
    bad_triples = []
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            ab = lcm_table[a][b]
            for c in range(b + 1, n + 1):
                if lcm_table[a][c] == ab and lcm_table[b][c] == ab:
                    bad_triples.append((a, b, c))
    return bad_triples

def _build_conflict_map(bad_triples, max_n):
    """Build conflict structures for fast validity checking."""
    bad_set = set()
    for (a, b, c) in bad_triples:
        bad_set.add(frozenset((a, b, c)))
    
    # For each element, the set of pairs that conflict with it
    elem_conflicts = {i: [] for i in range(1, max_n + 1)}
    for (a, b, c) in bad_triples:
        elem_conflicts[a].append((b, c))
        elem_conflicts[b].append((a, c))
        elem_conflicts[c].append((a, b))
    
    return bad_set, elem_conflicts

def _greedy_with_order(order, elem_conflicts):
    """Greedy: process elements in given order, add if no bad triple completed."""
    selected = set()
    for x in order:
        can_add = True
        for (a, b) in elem_conflicts[x]:
            if a in selected and b in selected:
                can_add = False
                break
        if can_add:
            selected.add(x)
    return selected

def _can_add(x, selected, elem_conflicts):
    """Check if x can be added to selected without creating a bad triple."""
    for (a, b) in elem_conflicts[x]:
        if a in selected and b in selected:
            return False
    return True

def _local_search(selected, elem_conflicts, max_n, deadline):
    """Try to improve by adding elements or swapping."""
    selected = set(selected)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each unselected element
        for x in range(1, max_n + 1):
            if x in selected:
                continue
            if _can_add(x, selected, elem_conflicts):
                selected.add(x)
                improved = True
        # Try swaps: remove one, add another
        if not improved:
            for x in range(1, max_n + 1):
                if x in selected:
                    continue
                # Find which pairs block x
                blocking_pairs = []
                for (a, b) in elem_conflicts[x]:
                    if a in selected and b in selected:
                        blocking_pairs.append((a, b))
                if not blocking_pairs:
                    continue
                # Try removing one element from a blocking pair
                for (a, b) in blocking_pairs:
                    for y in (a, b):
                        if y in selected:
                            new_selected = selected - {y}
                            if _can_add(x, new_selected, elem_conflicts):
                                new_selected.add(x)
                                selected = new_selected
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
    return selected

def _simulated_annealing(initial_set, elem_conflicts, max_n, deadline, rng):
    """Simulated annealing to escape local optima."""
    current = set(initial_set)
    best = set(current)
    best_size = len(current)
    
    # Temperature schedule
    t_start = 1.0
    t_end = 0.01
    num_steps = 2000
    cooling_rate = (t_end / t_start) ** (1.0 / num_steps)
    
    current_temp = t_start
    step = 0
    
    while time.time() < deadline and step < num_steps:
        step += 1
        current_temp *= cooling_rate
        
        # Choose a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random unselected element
            unselected = [x for x in range(1, max_n + 1) if x not in current]
            if not unselected:
                continue
            x = rng.choice(unselected)
            if _can_add(x, current, elem_conflicts):
                new_set = current | {x}
                delta = 1
            else:
                # Can't add directly; try removing one blocker then adding
                blockers = []
                for (a, b) in elem_conflicts[x]:
                    if a in current and b in current:
                        blockers.extend([a, b])
                if not blockers:
                    continue
                y = rng.choice(set(blockers))
                candidate = current - {y} | {x}
                # Verify validity
                valid = True
                for z in candidate:
                    for (a, b) in elem_conflicts[z]:
                        if a in candidate and b in candidate:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    new_set = candidate
                    delta = 0  # swap: same size
                else:
                    continue
        elif move_type < 0.7:
            # Try to remove a random element
            if not current:
                continue
            y = rng.choice(list(current))
            new_set = current - {y}
            delta = -1
        else:
            # Try a swap: remove one, add another
            if not current:
                continue
            y = rng.choice(list(current))
            unselected = [x for x in range(1, max_n + 1) if x not in current]
            if not unselected:
                continue
            x = rng.choice(unselected)
            candidate = current - {y} | {x}
            # Verify validity
            valid = True
            for z in candidate:
                for (a, b) in elem_conflicts[z]:
                    if a in candidate and b in candidate:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            new_set = candidate
            delta = 0
        
        # Acceptance criterion
        if delta >= 0:
            current = new_set
            if len(current) > best_size:
                best_size = len(current)
                best = set(current)
        else:
            # Accept with probability exp(delta / temp)
            prob = 2.718281828459045 ** (delta / current_temp) if current_temp > 0 else 0
            if rng.random() < prob:
                current = new_set
                if len(current) > best_size:
                    best_size = len(current)
                    best = set(current)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for safety (90s limit)
    
    max_n = 128
    bad_triples = _precompute_bad_triples(max_n)
    bad_set, elem_conflicts = _build_conflict_map(bad_triples, max_n)
    
    best = set()
    best_size = 0
    
    # First, do a deterministic ascending greedy as baseline
    order = list(range(1, max_n + 1))
    candidate = _greedy_with_order(order, elem_conflicts)
    candidate = _local_search(candidate, elem_conflicts, max_n, deadline)
    if len(candidate) > best_size:
        best = candidate
        best_size = len(candidate)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 10:
        return sorted(best)
    
    # Randomized restarts with greedy + local search
    rng = random.Random(42)
    restart_count = 0
    while time.time() < deadline - 20:
        restart_count += 1
        # Random order
        order = list(range(1, max_n + 1))
        rng.shuffle(order)
        
        candidate = _greedy_with_order(order, elem_conflicts)
        candidate = _local_search(candidate, elem_conflicts, max_n, deadline - 15)
        
        if len(candidate) > best_size:
            best = candidate
            best_size = len(candidate)
    
    # Simulated annealing on the best set found
    if time.time() < deadline - 5:
        sa_result = _simulated_annealing(best, elem_conflicts, max_n, deadline - 2, rng)
        if len(sa_result) > best_size:
            best = sa_result
            best_size = len(best)
    
    # Final local search on best
    if time.time() < deadline - 1:
        best = _local_search(best, elem_conflicts, max_n, deadline - 0.5)
    
    # Final validation
    S_list = sorted(best)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _lcm(a, b) == _lcm(a, c) == _lcm(b, c):
            valid = False
            break
    
    if not valid:
        # Fallback: use the simplest valid set
        best = {1}
    
    if not best:
        best = {1}
    
    return sorted(best)
# EVOLVE-BLOCK-END
