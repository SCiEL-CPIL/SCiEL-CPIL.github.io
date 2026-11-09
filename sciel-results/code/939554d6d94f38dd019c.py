"""Rewritten program: branch-and-bound on minimum maximal no-isosceles set."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np
from collections import defaultdict

DEADLINE = 325  # seconds, safety margin before 330s kill

def _d2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy

def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, chosen):
    """Check if p can be added to chosen without creating an isosceles triangle."""
    if not chosen:
        return True
    n = len(chosen)
    for i in range(n):
        for j in range(i + 1, n):
            d1 = _d2(p, chosen[i])
            d2 = _d2(p, chosen[j])
            d3 = _d2(chosen[i], chosen[j])
            if d1 == d2 or d1 == d3 or d2 == d3:
                return False
    return True

def _is_maximal(chosen, all_points):
    """Check if no point outside chosen can be added."""
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True

def _greedy_random(all_points, rng):
    """Randomized greedy: pick points in random order, add if possible."""
    order = list(all_points)
    rng.shuffle(order)
    chosen = []
    for p in order:
        if _can_add(p, chosen):
            chosen.append(p)
    return chosen

def _try_remove(chosen, all_points, rng, max_iters=200):
    """Try to remove points while maintaining maximality."""
    chosen = list(chosen)
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        indices = list(range(len(chosen)))
        rng.shuffle(indices)
        for idx in indices:
            candidate = chosen[:idx] + chosen[idx+1:]
            if _is_maximal(candidate, all_points):
                chosen = candidate
                improved = True
                break
    return chosen

def _swap_search(chosen, all_points, rng, max_iters=50):
    """Try to swap points to reduce size while maintaining maximality."""
    chosen_set = set(chosen)
    chosen = list(chosen)
    remaining = [p for p in all_points if p not in chosen_set]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if not remaining:
            break
        
        chosen_indices = list(range(len(chosen)))
        rng.shuffle(chosen_indices)
        
        for ci in chosen_indices:
            if time.time() - _swap_search.start_time > DEADLINE * 0.8:
                return chosen
            
            without_ci = chosen[:ci] + chosen[ci+1:]
            without_ci_set = set(without_ci)
            
            addable = []
            for p in remaining:
                if p in without_ci_set:
                    continue
                if _can_add(p, without_ci):
                    addable.append(p)
            
            if not addable:
                continue
            
            rng.shuffle(addable)
            for p in addable[:20]:
                candidate = without_ci + [p]
                if _is_maximal(candidate, all_points):
                    chosen = candidate
                    chosen_set = set(chosen)
                    remaining = [p for p in all_points if p not in chosen_set]
                    improved = True
                    break
            
            if improved:
                break
    
    return chosen

def _precompute_conflicts(all_points):
    """
    Precompute for each point p, the set of pairs (a_idx, b_idx) such that {p, a, b} is isosceles.
    This allows fast checking of whether p can be added to a chosen set.
    """
    n_pts = len(all_points)
    iso_pairs = [set() for _ in range(n_pts)]
    
    for p_idx in range(n_pts):
        p = all_points[p_idx]
        for a_idx in range(n_pts):
            if a_idx == p_idx:
                continue
            a = all_points[a_idx]
            for b_idx in range(a_idx + 1, n_pts):
                if b_idx == p_idx:
                    continue
                b = all_points[b_idx]
                if _is_iso(p, a, b):
                    iso_pairs[p_idx].add((a_idx, b_idx))
    
    return iso_pairs

def _is_maximal_with_iso_pairs(chosen_indices, iso_pairs, all_points):
    """Check maximality using precomputed iso_pairs."""
    chosen_set = set(chosen_indices)
    n_pts = len(all_points)
    for p_idx in range(n_pts):
        if p_idx in chosen_set:
            continue
        # p is blocked iff there exists (a,b) in chosen×chosen such that (a,b) in iso_pairs[p_idx]
        blocked = False
        for (a_idx, b_idx) in iso_pairs[p_idx]:
            if a_idx in chosen_set and b_idx in chosen_set:
                blocked = True
                break
        if not blocked:
            return False
    return True

def _can_add_with_iso_pairs(p_idx, chosen_indices, iso_pairs):
    """Check if p_idx can be added to chosen_indices."""
    chosen_set = set(chosen_indices)
    for (a_idx, b_idx) in iso_pairs[p_idx]:
        if a_idx in chosen_set and b_idx in chosen_set:
            return False
    return True

def _greedy_random_iso(all_points, iso_pairs, rng):
    """Randomized greedy using iso_pairs for speed."""
    n_pts = len(all_points)
    order = list(range(n_pts))
    rng.shuffle(order)
    chosen = []
    chosen_set = set()
    for p_idx in order:
        if _can_add_with_iso_pairs(p_idx, chosen_set, iso_pairs):
            chosen.append(p_idx)
            chosen_set.add(p_idx)
    return [all_points[i] for i in chosen]

def _try_remove_iso(chosen, all_points, iso_pairs, rng, max_iters=200):
    """Try to remove points while maintaining maximality, using iso_pairs."""
    n_pts = len(all_points)
    chosen_indices = [all_points.index(p) for p in chosen]
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        indices = list(range(len(chosen_indices)))
        rng.shuffle(indices)
        for idx in indices:
            candidate = chosen_indices[:idx] + chosen_indices[idx+1:]
            candidate_set = set(candidate)
            if _is_maximal_with_iso_pairs(candidate_set, iso_pairs, all_points):
                chosen_indices = candidate
                improved = True
                break
    return [all_points[i] for i in chosen_indices]

def _swap_search_iso(chosen, all_points, iso_pairs, rng, max_iters=50):
    """Try to swap points to reduce size while maintaining maximality, using iso_pairs."""
    n_pts = len(all_points)
    chosen_indices = [all_points.index(p) for p in chosen]
    chosen_set = set(chosen_indices)
    remaining = [i for i in range(n_pts) if i not in chosen_set]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        if not remaining:
            break
        
        chosen_indices_list = list(chosen_indices)
        rng.shuffle(chosen_indices_list)
        
        for ci in chosen_indices_list:
            if time.time() - _swap_search.start_time > DEADLINE * 0.8:
                return [all_points[i] for i in chosen_indices]
            
            without_ci = [i for i in chosen_indices if i != ci]
            without_ci_set = set(without_ci)
            
            addable = []
            for p in remaining:
                if p in without_ci_set:
                    continue
                if _can_add_with_iso_pairs(p, without_ci_set, iso_pairs):
                    addable.append(p)
            
            if not addable:
                continue
            
            rng.shuffle(addable)
            for p in addable[:20]:
                candidate = without_ci + [p]
                candidate_set = set(candidate)
                if _is_maximal_with_iso_pairs(candidate_set, iso_pairs, all_points):
                    chosen_indices = candidate
                    chosen_set = candidate_set
                    remaining = [i for i in range(n_pts) if i not in chosen_set]
                    improved = True
                    break
            
            if improved:
                break
    
    return [all_points[i] for i in chosen_indices]

def _simulated_annealing(chosen, all_points, iso_pairs, rng, start_time, deadline, max_iters=5000):
    """Simulated annealing to minimize set size while maintaining maximality."""
    n_pts = len(all_points)
    current_indices = [all_points.index(p) for p in chosen]
    current_set = set(current_indices)
    current_size = len(current_indices)
    
    best_indices = list(current_indices)
    best_size = current_size
    best_set = set(current_indices)
    
    T = 1.0
    cooling = 0.999
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - 10:
            break
        
        # Generate a neighbor: remove a random point and possibly add another
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to remove a point
            if not current_indices:
                continue
            idx = rng.randint(0, len(current_indices) - 1)
            candidate = current_indices[:idx] + current_indices[idx+1:]
            candidate_set = set(candidate)
            if not _is_maximal_with_iso_pairs(candidate_set, iso_pairs, all_points):
                continue
            new_size = len(candidate)
            delta = new_size - current_size
        elif move_type < 0.7:
            # Try to swap: remove one, add another
            if not current_indices:
                continue
            remaining = [i for i in range(n_pts) if i not in current_set]
            if not remaining:
                continue
            idx = rng.randint(0, len(current_indices) - 1)
            remove_idx = current_indices[idx]
            without_ci = current_indices[:idx] + current_indices[idx+1:]
            without_ci_set = set(without_ci)
            
            addable = [p for p in remaining if _can_add_with_iso_pairs(p, without_ci_set, iso_pairs)]
            if not addable:
                continue
            add_idx = rng.choice(addable)
            candidate = without_ci + [add_idx]
            candidate_set = set(candidate)
            if not _is_maximal_with_iso_pairs(candidate_set, iso_pairs, all_points):
                continue
            new_size = len(candidate)
            delta = new_size - current_size
        else:
            # Try to add a point (may increase size, helps escape local minima)
            remaining = [i for i in range(n_pts) if i not in current_set]
            if not remaining:
                continue
            addable = [p for p in remaining if _can_add_with_iso_pairs(p, current_set, iso_pairs)]
            if not addable:
                continue
            add_idx = rng.choice(addable)
            candidate = current_indices + [add_idx]
            candidate_set = set(candidate)
            # Maximality is automatically maintained when adding a point
            new_size = len(candidate)
            delta = new_size - current_size
        
        # Accept or reject
        if delta <= 0:
            # Better or equal, always accept
            current_indices = candidate
            current_set = candidate_set
            current_size = new_size
            if new_size < best_size:
                best_indices = list(current_indices)
                best_size = new_size
                best_set = set(current_indices)
        else:
            # Worse, accept with probability exp(-delta/T)
            if rng.random() < np.exp(-delta / T):
                current_indices = candidate
                current_set = candidate_set
                current_size = new_size
        
        T *= cooling
    
    return [all_points[i] for i in best_indices]

def _tabu_search(chosen, all_points, iso_pairs, rng, start_time, deadline, max_iters=3000, tabu_len=50):
    """Tabu search to minimize set size while maintaining maximality."""
    n_pts = len(all_points)
    current_indices = [all_points.index(p) for p in chosen]
    current_set = set(current_indices)
    current_size = len(current_indices)
    
    best_indices = list(current_indices)
    best_size = current_size
    best_set = set(current_indices)
    
    tabu_list = []  # List of (point, iteration) that are tabu
    iteration = 0
    
    for it in range(max_iters):
        if time.time() - start_time > deadline - 10:
            break
        iteration += 1
        
        # Clean up old tabu entries
        tabu_list = [(p, i) for p, i in tabu_list if iteration - i < tabu_len]
        tabu_set = {p for p, i in tabu_list}
        
        # Generate neighbors
        best_neighbor = None
        best_neighbor_size = current_size + 1
        
        # Try removals
        for idx in range(len(current_indices)):
            if current_indices[idx] in tabu_set:
                continue
            candidate = current_indices[:idx] + current_indices[idx+1:]
            candidate_set = set(candidate)
            if not _is_maximal_with_iso_pairs(candidate_set, iso_pairs, all_points):
                continue
            new_size = len(candidate)
            if new_size < best_neighbor_size:
                best_neighbor_size = new_size
                best_neighbor = (candidate, candidate_set, 'remove', current_indices[idx])
        
        # Try swaps
        remaining = [i for i in range(n_pts) if i not in current_set]
        for idx in range(len(current_indices)):
            remove_idx = current_indices[idx]
            if remove_idx in tabu_set:
                continue
            without_ci = current_indices[:idx] + current_indices[idx+1:]
            without_ci_set = set(without_ci)
            
            for p in remaining:
                if p in tabu_set:
                    continue
                if not _can_add_with_iso_pairs(p, without_ci_set, iso_pairs):
                    continue
                candidate = without_ci + [p]
                candidate_set = set(candidate)
                if not _is_maximal_with_iso_pairs(candidate_set, iso_pairs, all_points):
                    continue
                new_size = len(candidate)
                if new_size < best_neighbor_size:
                    best_neighbor_size = new_size
                    best_neighbor = (candidate, candidate_set, 'swap', remove_idx)
        
        if best_neighbor is None:
            break
        
        candidate, candidate_set, move_type, tabu_point = best_neighbor
        
        # Accept the best neighbor (even if it's worse, to escape local minima)
        current_indices = candidate
        current_set = candidate_set
        current_size = len(candidate)
        
        # Add to tabu list
        if move_type == 'remove':
            tabu_list.append((tabu_point, iteration))
        elif move_type == 'swap':
            tabu_list.append((tabu_point, iteration))
        
        if current_size < best_size:
            best_indices = list(current_indices)
            best_size = current_size
            best_set = set(current_indices)
    
    return [all_points[i] for i in best_indices]

def construct():
    start_time = time.time()
    _swap_search.start_time = start_time
    all_points = list(itertools.product(range(N), repeat=2))
    n_pts = len(all_points)
    rng = random.Random(42)
    
    # Precompute iso_pairs
    iso_pairs = _precompute_conflicts(all_points)
    
    best = None
    best_size = n_pts + 1
    
    # Phase 1: Multiple randomized greedy starts
    while time.time() - start_time < DEADLINE * 0.25:
        candidate = _greedy_random_iso(all_points, iso_pairs, rng)
        if len(candidate) < best_size:
            best = candidate
            best_size = len(candidate)
    
    # Phase 2: Local search with removal and swaps
    if best is not None:
        while time.time() - start_time < DEADLINE * 0.6:
            improved = _try_remove_iso(best, all_points, iso_pairs, rng, max_iters=100)
            if len(improved) < best_size:
                best = improved
                best_size = len(best)
            
            swapped = _swap_search_iso(best, all_points, iso_pairs, rng, max_iters=30)
            if len(swapped) < best_size:
                best = swapped
                best_size = len(best)
            
            # Occasionally try a new greedy start
            if time.time() - start_time < DEADLINE * 0.4:
                candidate = _greedy_random_iso(all_points, iso_pairs, rng)
                candidate = _try_remove_iso(candidate, all_points, iso_pairs, rng, max_iters=50)
                if len(candidate) < best_size:
                    best = candidate
                    best_size = len(candidate)
    
    # Phase 3: Simulated annealing
    if best is not None and time.time() - start_time < DEADLINE * 0.8:
        sa_result = _simulated_annealing(best, all_points, iso_pairs, rng, start_time, DEADLINE * 0.8, max_iters=3000)
        if len(sa_result) < best_size:
            best = sa_result
            best_size = len(best)
    
    # Phase 4: Tabu search
    if best is not None and time.time() - start_time < DEADLINE * 0.9:
        tabu_result = _tabu_search(best, all_points, iso_pairs, rng, start_time, DEADLINE * 0.9, max_iters=2000)
        if len(tabu_result) < best_size:
            best = tabu_result
            best_size = len(best)
    
    # Final: ensure maximality
    if best is not None and time.time() - start_time < DEADLINE - 5:
        if not _is_maximal(best, all_points):
            remaining = [p for p in all_points if p not in set(best)]
            rng.shuffle(remaining)
            for p in remaining:
                if _can_add(p, best):
                    best.append(p)
                if _is_maximal(best, all_points):
                    break
    
    return best if best is not None else []
# EVOLVE-BLOCK-END
