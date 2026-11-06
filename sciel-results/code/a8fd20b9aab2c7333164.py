"""Improved: faster conflict checking, better local search with repair moves, more diversification."""
import time
import random
from math import gcd
from collections import defaultdict

def _lcm(a, b):
    return a * b // gcd(a, b)

def construct():
    start_time = time.time()
    DEADLINE = 88.0

    n = 128
    universe = list(range(1, n + 1))

    # Precompute LCM table
    lcm_table = {}
    for i in range(1, n + 1):
        for j in range(i, n + 1):
            lcm_table[(i, j)] = _lcm(i, j)

    # Find all bad triples: (a, b, c) with a < b < c where lcm(a,b)==lcm(a,c)==lcm(b,c)
    lcm_to_pairs = defaultdict(list)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            l = _lcm(i, j)
            lcm_to_pairs[l].append((i, j))

    bad_triples = []
    for l, pairs in lcm_to_pairs.items():
        candidates = [x for x in range(1, n + 1) if l % x == 0]
        if len(candidates) >= 3:
            for idx_a in range(len(candidates)):
                a = candidates[idx_a]
                for idx_b in range(idx_a + 1, len(candidates)):
                    b = candidates[idx_b]
                    if _lcm(a, b) != l:
                        continue
                    for idx_c in range(idx_b + 1, len(candidates)):
                        c = candidates[idx_c]
                        if _lcm(a, c) == l and _lcm(b, c) == l:
                            bad_triples.append((a, b, c))

    # For each x, forbidden_pairs[x] = list of (a,b) pairs such that {a,b,x} is a bad triple
    forbidden_pairs = [set() for _ in range(n + 1)]
    for (a, b, c) in bad_triples:
        forbidden_pairs[c].add((a, b))
        forbidden_pairs[a].add((b, c))
        forbidden_pairs[b].add((a, c))

    # Convert to sorted lists for faster iteration
    forbidden_pairs_list = [sorted(forbidden_pairs[x]) for x in range(n + 1)]
    
    # Precompute for each x: the set of elements that appear in any forbidden pair with x
    # This helps identify which elements to consider for repair
    conflict_elements = [set() for _ in range(n + 1)]
    for x in range(1, n + 1):
        for (a, b) in forbidden_pairs[x]:
            conflict_elements[x].add(a)
            conflict_elements[x].add(b)

    # Constraint degree
    constraint_degree = [0] * (n + 1)
    for x in range(1, n + 1):
        constraint_degree[x] = len(forbidden_pairs[x])

    # Fast can_add using list iteration
    def can_add(x, selected_set):
        for (a, b) in forbidden_pairs_list[x]:
            if a in selected_set and b in selected_set:
                return False
        return True

    # Greedy baseline
    def greedy(order):
        selected = []
        selected_set = set()
        for x in order:
            if can_add(x, selected_set):
                selected.append(x)
                selected_set.add(x)
        return selected

    # Get initial solution from greedy
    order_greedy = sorted(universe, key=lambda x: (constraint_degree[x], x))
    init_sol = greedy(order_greedy)
    best_size = len(init_sol)
    best_solution = list(init_sol)

    # Improved local search with repair moves
    def local_search_v3(initial_set, max_iters=80000):
        """Local search with 1-add, 2-swap, and repair moves."""
        current = set(initial_set)
        best = set(initial_set)
        best_size_local = len(current)
        T = 1.5  # temperature
        cooling = 0.99995
        stuck_count = 0
        
        # Precompute list of universe elements not in current for quick access
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            # Determine move type with adaptive probabilities
            r = random.random()
            if r < 0.4:
                move_type = 0  # 1-add
            elif r < 0.7:
                move_type = 1  # 2-swap
            else:
                move_type = 2  # repair: remove 1-2 elements then add 1-2
            
            new_set = set(current)
            
            if move_type == 0:  # 1-add
                candidates = [x for x in universe if x not in current]
                if not candidates:
                    continue
                # Try a few random candidates, pick the first that works
                random.shuffle(candidates)
                added = False
                for x in candidates[:5]:
                    if can_add(x, current):
                        new_set.add(x)
                        added = True
                        break
                if not added:
                    stuck_count += 1
                    continue
            elif move_type == 1:  # 2-swap: remove one, add one
                if not current:
                    continue
                to_remove = random.choice(list(current))
                candidates = [x for x in universe if x not in current]
                if not candidates:
                    continue
                # Try several candidates for to_add
                random.shuffle(candidates)
                added = False
                for to_add in candidates[:8]:
                    new_set.discard(to_remove)
                    if can_add(to_add, new_set):
                        new_set.add(to_add)
                        added = True
                        break
                if not added:
                    continue
            else:  # repair: remove 1-2 conflicting elements, then add 1-2 new ones
                if len(current) < 2:
                    continue
                # Identify elements to potentially remove: those involved in conflicts
                # with elements we might want to add
                conflict_set = set()
                for x in current:
                    conflict_set.update(conflict_elements[x])
                
                # Pick 1-2 elements to remove from current that are in conflict_set
                removable = [x for x in current if x in conflict_set]
                if not removable:
                    removable = list(current)
                
                num_remove = random.randint(1, min(2, len(removable)))
                to_remove = random.sample(removable, num_remove)
                
                for x in to_remove:
                    new_set.discard(x)
                
                # Now try to add 1-2 new elements
                candidates = [x for x in universe if x not in new_set]
                if not candidates:
                    continue
                random.shuffle(candidates)
                num_add = min(num_remove + random.randint(0, 1), len(candidates))
                added_count = 0
                for x in candidates[:15]:
                    if added_count >= num_add:
                        break
                    if can_add(x, new_set):
                        new_set.add(x)
                        added_count += 1
                
                if added_count == 0:
                    continue
            
            delta = len(new_set) - len(current)
            
            # Accept if improvement, or with probability based on temperature
            if delta >= 0:
                current = new_set
                stuck_count = 0
                if len(current) > best_size_local:
                    best_size_local = len(current)
                    best = set(current)
            else:
                # Simulated annealing acceptance
                if random.random() < pow(2.718281828, delta / max(T, 0.01)):
                    current = new_set
                else:
                    stuck_count += 1
            
            T *= cooling
            
            # If stuck too long, restart with perturbation
            if stuck_count > 200:
                # Perturb: remove 3-5 random elements and try to refill
                for _ in range(random.randint(3, 5)):
                    if current:
                        current.discard(random.choice(list(current)))
                # Greedily refill
                for x in sorted(universe, key=lambda y: (constraint_degree[y], y)):
                    if x not in current and can_add(x, current):
                        current.add(x)
                stuck_count = 0
                T = 1.5  # reset temperature
        
        return list(best)

    # Run multiple greedy + local search combinations
    all_solutions = []
    
    # Greedy with different orderings
    strategies = [
        lambda: sorted(universe, key=lambda x: (constraint_degree[x], x)),
        lambda: sorted(universe, key=lambda x: (-constraint_degree[x], x)),
        lambda: list(reversed(universe)),
        lambda: sorted(universe, key=lambda x: (x % 2, x)),
        lambda: sorted(universe, key=lambda x: (-x.bit_count(), x)),
        lambda: sorted(universe, key=lambda x: (x.bit_count(), x)),
        lambda: sorted(universe, key=lambda x: (constraint_degree[x] % 7, x)),
    ]
    
    for i, strat in enumerate(strategies):
        if time.time() - start_time > DEADLINE:
            break
        order = strat()
        sol = greedy(order)
        all_solutions.append(sol)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)
    
    # Randomized greedy with more iterations
    for i in range(50):
        if time.time() - start_time > DEADLINE - 5:
            break
        random.seed(i * 31337 + 7)
        order = list(universe)
        random.shuffle(order)
        sol = greedy(order)
        all_solutions.append(sol)
        if len(sol) > best_size:
            best_size = len(sol)
            best_solution = list(sol)
    
    # Local search on top solutions
    all_solutions.sort(key=lambda s: -len(s))
    
    # Run local search on top 10 solutions with more iterations
    for sol in all_solutions[:10]:
        if time.time() - start_time > DEADLINE - 3:
            break
        improved = local_search_v3(sol, max_iters=50000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = list(improved)
    
    # Final polish on best solution with longer run
    if time.time() - start_time < DEADLINE - 1:
        improved = local_search_v3(best_solution, max_iters=40000)
        if len(improved) > best_size:
            best_size = len(improved)
            best_solution = list(improved)
    
    if not best_solution:
        best_solution = [1]
    
    return best_solution
