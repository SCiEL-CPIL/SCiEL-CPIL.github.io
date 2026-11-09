import time
import math
import random
from math import gcd

MAXIMUM = 128
DEADLINE = 85.0
SAFETY_MARGIN = 5.0

# Precompute gcd table for moduli 2..128
_gcd_table = {}
for m1 in range(2, MAXIMUM + 1):
    for m2 in range(m1, MAXIMUM + 1):
        g = gcd(m1, m2)
        _gcd_table[(m1, m2)] = g
        if m1 != m2:
            _gcd_table[(m2, m1)] = g

def _check_pair(r1, m1, r2, m2):
    """Check if two residue classes are disjoint."""
    g = _gcd_table.get((m1, m2), gcd(m1, m2))
    return r1 % g != r2 % g

def _is_valid_solution(solution):
    """Check if a solution is valid."""
    if not solution:
        return False
    seen_moduli = set()
    for r, m in solution:
        if m < 2 or m > MAXIMUM:
            return False
        if m in seen_moduli:
            return False
        seen_moduli.add(m)
        if r < 0 or r >= m:
            return False
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            r1, m1 = solution[i]
            r2, m2 = solution[j]
            if not _check_pair(r1, m1, r2, m2):
                return False
    return True

def _build_conflict_matrix():
    """
    Build a fast conflict lookup. For each pair of moduli (m1, m2),
    precompute which residues conflict.
    Returns a dict: (m1, m2) -> set of r1 values that conflict with SOME r2 in [0, m2)
    Actually, we need: for given (r1, m1, m2), which r2 values in [0,m2) are compatible?
    Let's precompute: compat[m1][m2][r1] = bitmask of compatible r2 values
    """
    # For each pair (m1, m2), for each r1 in [0, m1), compute which r2 in [0, m2) are compatible
    # Two classes (r1, m1) and (r2, m2) are disjoint iff r1 % gcd(m1,m2) != r2 % gcd(m1,m2)
    # So for fixed (m1, m2, r1), the forbidden r2 values are those with r2 % g == r1 % g where g = gcd(m1,m2)
    
    compat = {}
    for m1 in range(2, MAXIMUM + 1):
        for m2 in range(2, MAXIMUM + 1):
            if m1 == m2:
                continue
            g = _gcd_table.get((m1, m2), gcd(m1, m2))
            # For each r1, compute bitmask of compatible r2
            mask_list = []
            for r1 in range(m1):
                target = r1 % g
                mask = 0
                for r2 in range(m2):
                    if r2 % g != target:
                        mask |= (1 << r2)
                mask_list.append(mask)
            compat[(m1, m2)] = mask_list
    return compat

_COMPAT = _build_conflict_matrix()

def _fast_compatible(r_new, m_new, solution):
    """Check if (r_new, m_new) is compatible with all in solution using precomputed masks."""
    for (r_prev, m_prev) in solution:
        if m_new == m_prev:
            return False
        # Get mask of compatible r_prev values for given (m_new, r_new)
        # We need: is r_prev compatible with r_new?
        # From _COMPAT[(m_new, m_prev)][r_new], check if bit r_prev is set
        mask = _COMPAT.get((m_new, m_prev))
        if mask is None:
            if not _check_pair(r_new, m_new, r_prev, m_prev):
                return False
        else:
            if not (mask[r_new] >> r_prev) & 1:
                return False
    return True

def _greedy_construct(mod_order, deadline):
    """Greedy construction with fast compatibility check."""
    solution = []
    used_moduli = set()
    
    for m in mod_order:
        if time.time() > deadline:
            break
        if m in used_moduli:
            continue
        
        for r in range(m):
            if _fast_compatible(r, m, solution):
                solution.append([r, m])
                used_moduli.add(m)
                break
    
    return solution

def _simulated_annealing(initial_solution, deadline):
    """
    Simulated annealing: state is a set of (r, m) pairs.
    Moves: 
      1. Add a new (r, m) pair that's compatible
      2. Remove a random (r, m) pair
      3. Change the residue of an existing pair
    Objective: maximize |solution|
    """
    if not initial_solution:
        return initial_solution
    
    # Start with a greedy solution
    solution = [list(x) for x in initial_solution]
    used_moduli = set(m for _, m in solution)
    
    best_solution = [list(x) for x in solution]
    best_len = len(solution)
    
    start_temp = 1.0
    end_temp = 0.01
    total_steps = 500000
    
    # Estimate time per step to set number of iterations
    t_start = time.time()
    # Do a few steps to estimate
    for _ in range(100):
        pass
    elapsed = time.time() - t_start
    if elapsed > 0:
        time_per_step = elapsed / 100
    else:
        time_per_step = 1e-6
    
    max_steps = min(total_steps, int((deadline - time.time()) / time_per_step * 0.8))
    max_steps = max(max_steps, 1000)
    
    all_moduli = list(range(2, MAXIMUM + 1))
    
    for step in range(max_steps):
        if time.time() > deadline:
            break
        
        temp = start_temp * (end_temp / start_temp) ** (step / max_steps)
        
        # Choose a move
        move_type = random.random()
        
        if move_type < 0.3 and len(solution) < MAXIMUM - 1:
            # Add a new pair
            available_moduli = [m for m in all_moduli if m not in used_moduli]
            if not available_moduli:
                continue
            m_new = random.choice(available_moduli)
            r_new = random.randint(0, m_new - 1)
            
            if _fast_compatible(r_new, m_new, solution):
                new_solution = solution + [[r_new, m_new]]
                new_used = used_moduli | {m_new}
                delta = 1  # improved
                solution = new_solution
                used_moduli = new_used
                if len(solution) > best_len:
                    best_len = len(solution)
                    best_solution = [list(x) for x in solution]
        
        elif move_type < 0.5 and len(solution) > 1:
            # Remove a random pair
            idx = random.randint(0, len(solution) - 1)
            r_rem, m_rem = solution[idx]
            new_solution = solution[:idx] + solution[idx+1:]
            new_used = used_moduli - {m_rem}
            delta = -1
            # Accept with probability exp(delta/temp)
            if random.random() < math.exp(delta / temp):
                solution = new_solution
                used_moduli = new_used
        
        elif move_type < 0.8 and len(solution) > 0:
            # Change residue of a random pair
            idx = random.randint(0, len(solution) - 1)
            r_old, m_old = solution[idx]
            r_new = random.randint(0, m_old - 1)
            if r_new == r_old:
                continue
            
            # Check if new residue is compatible with all other pairs
            others = solution[:idx] + solution[idx+1:]
            if _fast_compatible(r_new, m_old, others):
                new_solution = solution[:idx] + [[r_new, m_old]] + solution[idx+1:]
                delta = 0  # same length, but might enable future additions
                # Always accept neutral moves with some probability
                if random.random() < 0.5:
                    solution = new_solution
        
        else:
            # Swap: remove one, add another
            if len(solution) == 0:
                continue
            idx = random.randint(0, len(solution) - 1)
            r_rem, m_rem = solution[idx]
            others = solution[:idx] + solution[idx+1:]
            
            available_moduli = [m for m in all_moduli if m not in used_moduli or m == m_rem]
            if not available_moduli:
                continue
            m_new = random.choice(available_moduli)
            r_new = random.randint(0, m_new - 1)
            
            if _fast_compatible(r_new, m_new, others):
                new_solution = others + [[r_new, m_new]]
                new_used = (used_moduli - {m_rem}) | {m_new}
                delta = 0  # same length
                if random.random() < 0.5:
                    solution = new_solution
                    used_moduli = new_used
        
        if len(solution) > best_len:
            best_len = len(solution)
            best_solution = [list(x) for x in solution]
    
    return best_solution

def _local_search_improve(solution, deadline):
    """Try to improve the solution by adding new moduli."""
    if not solution:
        return solution
    
    best = [list(x) for x in solution]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        used_moduli = set(m for _, m in best)
        
        for m_new in range(2, MAXIMUM + 1):
            if time.time() > deadline:
                break
            if m_new in used_moduli:
                continue
            
            for r_new in range(m_new):
                if _fast_compatible(r_new, m_new, best):
                    new_solution = [list(x) for x in best]
                    new_solution.append([r_new, m_new])
                    best = new_solution
                    used_moduli.add(m_new)
                    improved = True
                    break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_solution = []
    all_moduli = list(range(2, MAXIMUM + 1))
    
    # Phase 1: Multiple greedy strategies with different orderings
    orderings = [
        all_moduli,
        list(reversed(all_moduli)),
        sorted(all_moduli, key=lambda m: m.bit_count()),
        sorted(all_moduli, key=lambda m: -m.bit_count()),
        sorted(all_moduli, key=lambda m: min(p for p in range(2, m+1) if m%p==0)),
    ]
    
    for mod_order in orderings:
        if time.time() > deadline:
            break
        solution = _greedy_construct(mod_order, deadline)
        if len(solution) > len(best_solution):
            best_solution = [list(x) for x in solution]
    
    # Phase 2: Simulated annealing from best greedy solution
    if best_solution and time.time() < deadline:
        sa_solution = _simulated_annealing(best_solution, deadline)
        if len(sa_solution) > len(best_solution):
            best_solution = sa_solution
    
    # Phase 3: Randomized multi-start with SA
    if time.time() < deadline:
        random.seed(42)
        num_attempts = 200
        for attempt in range(num_attempts):
            if time.time() > deadline:
                break
            
            mod_order = all_moduli[:]
            random.shuffle(mod_order)
            
            solution = _greedy_construct(mod_order, deadline)
            if len(solution) > len(best_solution):
                best_solution = [list(x) for x in solution]
            
            if time.time() < deadline and len(solution) > 0:
                sa_solution = _simulated_annealing(solution, deadline)
                if len(sa_solution) > len(best_solution):
                    best_solution = sa_solution
    
    # Phase 4: Final local search
    if best_solution and time.time() < deadline:
        final = _local_search_improve(best_solution, deadline)
        if len(final) > len(best_solution):
            best_solution = final
    
    if not best_solution:
        return [[0, 2]]
    
    return best_solution
