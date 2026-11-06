"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
from math import gcd

def _lcm(a, b):
    return a * b // gcd(a, b)

def _build_conflicts():
    """
    Precompute conflicts for each vertex.
    conflicts[v] is a set of frozensets {u, w} such that {v, u, w} is a bad triple.
    """
    conflicts = {v: set() for v in range(1, 129)}
    nums = list(range(1, 129))
    
    # We store bad triples as sorted tuples to avoid duplicates
    bad_triples = set()
    
    for a, b, c in combinations(nums, 3):
        lab = _lcm(a, b)
        lac = _lcm(a, c)
        lbc = _lcm(b, c)
        if lab == lac == lbc:
            bad_triples.add((a, b, c))
            
    # Populate conflicts map
    for a, b, c in bad_triples:
        # For each vertex in the triple, add the pair of the other two to its conflict list
        conflicts[a].add(frozenset((b, c)))
        conflicts[b].add(frozenset((a, c)))
        conflicts[c].add(frozenset((a, b)))
        
    return conflicts

def _can_add(selected_set, v, conflicts):
    """
    Check if v can be added to selected_set without forming a bad triple.
    """
    # If any conflict pair of v is fully contained in selected_set, we cannot add v
    for pair in conflicts[v]:
        if pair <= selected_set:
            return False
    return True

def _greedy_construct(conflicts, order):
    """
    Greedy construction using a specific order of vertices.
    """
    selected = set()
    for v in order:
        if _can_add(selected, v, conflicts):
            selected.add(v)
    return selected

def _local_improve(selected, conflicts, time_limit=10):
    """
    Try to improve the selected set by adding vertices.
    If a vertex cannot be added, try to remove one of its conflicting neighbors to make space.
    """
    selected = set(selected)
    best_size = len(selected)
    start = time.time()
    
    # List of vertices not in selected
    candidates = [v for v in range(1, 129) if v not in selected]
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Shuffle candidates to break symmetry
        random.shuffle(candidates)
        
        for v in candidates:
            if v in selected:
                continue
            
            if _can_add(selected, v, conflicts):
                selected.add(v)
                best_size = len(selected)
                improved = True
                # Remove v from candidates if it's there, or just let the loop handle it
                # We break to restart the loop with the new state
                break
        
        if not improved:
            # Try swap moves: remove u, add v
            # This is more expensive. We only try if we have time.
            if time.time() - start < time_limit - 1:
                for v in candidates:
                    if v in selected:
                        continue
                    # Check if v is blocked
                    blocking_pairs = [p for p in conflicts[v] if p <= selected]
                    if not blocking_pairs:
                        # Should have been added above
                        selected.add(v)
                        improved = True
                        break
                    
                    # Try to remove one vertex from a blocking pair
                    # We pick the first blocking pair and try removing one of its elements
                    # To be more effective, we might want to try removing a vertex that is in many blocking pairs
                    # For simplicity, we try removing 'u' from the first blocking pair
                    for pair in blocking_pairs:
                        for u in pair:
                            if u in selected:
                                # Tentatively remove u
                                selected.remove(u)
                                if _can_add(selected, v, conflicts):
                                    selected.add(v)
                                    # Size is same, but structure changed.
                                    # We accept this swap to potentially enable future additions.
                                    # However, to prevent cycling, we should only accept if it leads to improvement soon?
                                    # For now, let's just do the swap and mark improved to retry additions.
                                    # This might cycle. A simple heuristic: only swap if it reduces the "potential" of the set?
                                    # Or just do it and rely on the outer loop to find additions.
                                    # To avoid infinite loops of swaps, we can limit the number of swaps or ensure we don't swap back immediately.
                                    # Given the time limit, simple swaps are risky. 
                                    # Let's only accept swaps if we can immediately add ANOTHER vertex? No, that's 2-opt.
                                    # Let's stick to pure additions for stability, and rely on restarts for global search.
                                    selected.remove(v)
                                    selected.add(u)
                                else:
                                    selected.add(u) # Put u back
                                
                                # If we did a swap, we break to restart the addition phase
                                if selected != {x for x in selected}: # This check is trivial
                                     pass
                                
                                # Actually, let's disable swaps for now to ensure robustness.
                                # The greedy restarts are usually sufficient for N=128.
                                break

    return list(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # 1. Precompute conflicts
    conflicts = _build_conflicts()
    
    # Calculate degrees for ordering
    degrees = {v: len(conflicts[v]) for v in range(1, 129)}
    
    best_solution = []
    best_size = 0
    
    # 2. Initial Greedy with sorted order (lowest degree first)
    order_sorted = sorted(range(1, 129), key=lambda x: degrees[x])
    sol = _greedy_construct(conflicts, order_sorted)
    best_solution = sol
    best_size = len(sol)
    
    # 3. Multiple Random Restarts
    random.seed(42)
    restart_count = 0
    while time.time() < deadline - 5:
        restart_count += 1
        order = list(range(1, 129))
        random.shuffle(order)
        
        # Mix sorted and random?
        # Sometimes a hybrid works well.
        # Let's just use random for diversity.
        
        sol = _greedy_construct(conflicts, order)
        
        if len(sol) > best_size:
            best_solution = sol
            best_size = len(sol)
            
        # Local improvement on the new solution
        if len(sol) >= best_size - 1:
            improved_sol = _local_improve(sol, conflicts, time_limit=2)
            if len(improved_sol) > best_size:
                best_solution = improved_sol
                best_size = len(best_solution)
                
    # 4. Final local improvement on the best solution found
    if time.time() < deadline - 2:
        improved_best = _local_improve(best_solution, conflicts, time_limit=10)
        if len(improved_best) > best_size:
            best_solution = improved_best
            best_size = len(best_solution)

    if not best_solution:
        return [1]
        
    return best_solution
# EVOLVE-BLOCK-END
