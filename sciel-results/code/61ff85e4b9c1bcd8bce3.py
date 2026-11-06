"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import random
import time
import math

def _dist2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

def _is_isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(point, chosen_set):
    """
    Check if 'point' can be added to 'chosen_set' without forming an isosceles triangle.
    chosen_set should be a set of points for O(1) membership, but we need pairs, so we iterate.
    For efficiency, we might pass a list if we know it's small, but set is safer for generic use.
    Here we assume chosen_set is a set.
    """
    chosen_list = list(chosen_set)
    n = len(chosen_list)
    for i in range(n):
        for j in range(i + 1, n):
            if _is_isosceles(point, chosen_list[i], chosen_list[j]):
                return False
    return True

def _is_maximal(chosen_set, all_points):
    """
    Check if chosen_set is maximal.
    A set is maximal if no point outside the set can be added.
    """
    for p in all_points:
        if p not in chosen_set:
            if _can_add(p, chosen_set):
                return False
    return True

def _energy(chosen_set, all_points):
    """
    Energy function for Simulated Annealing.
    We want to minimize size, but the set MUST be maximal.
    We use a penalty for non-maximality.
    E = size + penalty * (1 if not maximal else 0)
    Actually, using count of addable points is smoother.
    """
    size = len(chosen_set)
    # Count addable points
    addable = 0
    for p in all_points:
        if p not in chosen_set:
            if _can_add(p, chosen_set):
                addable += 1
    # If addable > 0, it's not maximal.
    # We want to drive addable to 0.
    # Large penalty to ensure maximality is prioritized over size reduction initially.
    penalty = 100
    return size + penalty * addable

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    all_points = list(itertools.product(range(N), repeat=2))
    all_points_set = set(all_points)
    
    best_set = None
    best_energy = float('inf')
    best_size = float('inf')
    
    # 1. Initial Greedy Solution to get a valid starting point
    # Greedy: add points in random order if valid
    random.shuffle(all_points)
    current_set = set()
    for p in all_points:
        if _can_add(p, current_set):
            current_set.add(p)
    
    # Make it maximal? Greedy usually results in a maximal set if we iterate all points?
    # Not necessarily. If we skip a point because it conflicted, later points might not block it?
    # No, if p conflicted with existing points, it will still conflict.
    # So greedy construction yields a maximal set.
    # Let's verify.
    # If p cannot be added, it forms isosceles with some pair in S.
    # That pair remains in S. So p can never be added.
    # Thus, the set of all points that *can* be added is exactly the set we built.
    # So current_set is maximal.
    
    current_energy = _energy(current_set, all_points)
    if current_energy < best_energy:
        best_energy = current_energy
        best_set = set(current_set)
        best_size = len(current_set)
    
    # 2. Simulated Annealing
    # State: current_set (a set of points)
    # We will try moves: Add, Remove, Swap
    
    T = 100.0
    cooling_rate = 0.995
    min_T = 1.0
    
    while time.time() < deadline and T > min_T:
        T *= cooling_rate
        
        # Generate neighbor
        move_type = random.random()
        new_set = set(current_set)
        
        if move_type < 0.3:
            # Remove a point
            if current_set:
                p = random.choice(list(current_set))
                new_set.remove(p)
            else:
                continue
        elif move_type < 0.6:
            # Add a point (if valid)
            candidates = [p for p in all_points if p not in current_set]
            if candidates:
                p = random.choice(candidates)
                if _can_add(p, current_set):
                    new_set.add(p)
                else:
                    continue # Skip invalid move
            else:
                continue
        else:
            # Swap a point
            if current_set and len(current_set) < len(all_points):
                p_in = random.choice(list(current_set))
                candidates = [p for p in all_points if p not in current_set]
                if candidates:
                    p_out = random.choice(candidates)
                    # Check if swapping is valid
                    # Remove p_in, add p_out
                    temp_set = set(current_set)
                    temp_set.remove(p_in)
                    if _can_add(p_out, temp_set):
                        temp_set.add(p_out)
                        new_set = temp_set
                    else:
                        continue
                else:
                    continue
            else:
                continue
        
        # Calculate energies
        # Optimization: If new_set is same size and we just swapped, energy might be similar.
        # But we need to check maximality penalty.
        new_energy = _energy(new_set, all_points)
        old_energy = current_energy
        
        delta_E = new_energy - old_energy
        
        if delta_E < 0:
            # Accept
            current_set = new_set
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_set = set(current_set)
                if len(current_set) < best_size:
                    best_size = len(current_set)
        else:
            # Accept with probability exp(-delta_E / T)
            if random.random() < math.exp(-delta_E / T):
                current_set = new_set
                current_energy = new_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_set = set(current_set)
                    if len(current_set) < best_size:
                        best_size = len(current_set)
    
    # 3. Local Search Refinement on Best Set Found
    # Try to remove points from best_set if it remains maximal
    if best_set:
        refined_set = set(best_set)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle points to avoid bias
            points_list = list(refined_set)
            random.shuffle(points_list)
            for p in points_list:
                trial_set = set(refined_set)
                trial_set.remove(p)
                if _is_maximal(trial_set, all_points):
                    refined_set = trial_set
                    improved = True
                    break
        
        if len(refined_set) < best_size:
            best_size = len(refined_set)
            best_set = refined_set
    
    # Ensure we return a valid maximal set
    if best_set and _is_maximal(best_set, all_points):
        return list(best_set)
    
    # Fallback: Return the initial greedy set (which is maximal)
    return list(current_set)
# EVOLVE-BLOCK-END
