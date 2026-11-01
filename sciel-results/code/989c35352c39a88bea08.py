<<<<<<< SEARCH
import time
import itertools
import random
import numpy as np

_DEADLINE = 140.0
=======
import time
import itertools
import random
import numpy as np
import heapq

_DEADLINE = 140.0
>>>>>>> REPLACE

<<<<<<< SEARCH
def _build_spheres_bitmask(q, n, radius):
    """Build bitmask spheres for all words. Returns list of integers (bitmasks)."""
    total = q ** n
    powers = [q ** i for i in range(n)]
    spheres = [0] * total
    for x in range(total):
        mask = 1 << x
        if radius >= 1:
            for i in range(n):
                digit = (x // powers[i]) % q
                for val in range(q):
                    if val == digit:
                        continue
                    nb = x + (val - digit) * powers[i]
                    mask |= (1 << nb)
        spheres[x] = mask
    return spheres
=======
def _build_spheres_bitmask(q, n, radius):
    """Build bitmask spheres for all words. Returns list of integers (bitmasks)."""
    total = q ** n
    powers = [q ** i for i in range(n)]
    spheres = [0] * total
    for x in range(total):
        mask = 1 << x
        if radius >= 1:
            for i in range(n):
                digit = (x // powers[i]) % q
                for val in range(q):
                    if val == digit:
                        continue
                    nb = x + (val - digit) * powers[i]
                    mask |= (1 << nb)
        spheres[x] = mask
    return spheres


def _compute_gains(spheres, covered, total):
    """Compute gain for each candidate given current coverage."""
    gains = [0] * total
    for i in range(total):
        gains[i] = (spheres[i] & ~covered).bit_count()
    return gains
>>>>>>> REPLACE

<<<<<<< SEARCH
def _greedy_bitmask(spheres, full_mask, time_budget, start_time, random_order=None):
    """Greedy set-cover using bitmask operations."""
    if random_order is not None:
        candidates = list(range(len(spheres)))
        random.shuffle(candidates)
    else:
        candidates = list(range(len(spheres)))
    
    covered = 0
    chosen = []
    while covered != full_mask:
        if time.time() - start_time > time_budget:
            break
        best_gain = -1
        best_idx = -1
        for idx in candidates:
            new_covered = spheres[idx] & ~covered
            gain = new_covered.bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        covered |= spheres[best_idx]
    return chosen
=======
def _greedy_bitmask(spheres, full_mask, time_budget, start_time, random_order=None):
    """Greedy set-cover using bitmask operations with tie-breaking via random order."""
    total = len(spheres)
    if random_order is not None:
        candidates = list(range(total))
        random.shuffle(candidates)
    else:
        candidates = list(range(total))
    
    covered = 0
    chosen = []
    while covered != full_mask:
        if time.time() - start_time > time_budget:
            break
        best_gain = -1
        best_idx = -1
        for idx in candidates:
            gain = (spheres[idx] & ~covered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_gain <= 0:
            break
        chosen.append(best_idx)
        covered |= spheres[best_idx]
    return chosen


def _local_search_prune(code_indices, spheres, full_mask, time_budget, start_time):
    """Local search: try removing codewords and swapping to reduce size while keeping coverage."""
    current = list(code_indices)
    if not current:
        return current
    
    # Compute coverage
    covered = 0
    for idx in current:
        covered |= spheres[idx]
    
    best = list(current)
    best_len = len(current)
    
    # Phase 1: Try removing each codeword if coverage is maintained
    changed = True
    while changed:
        changed = False
        for i in range(len(current)):
            if time.time() - start_time > time_budget:
                return best
            # Temporarily remove
            new_covered = 0
            for j, idx in enumerate(current):
                if j != i:
                    new_covered |= spheres[idx]
            if new_covered == full_mask:
                current.pop(i)
                covered = new_covered
                changed = True
                if len(current) < best_len:
                    best_len = len(current)
                    best = list(current)
                break
    
    # Phase 2: Try swaps - remove one, add another that covers more
    improved = True
    while improved:
        improved = False
        for i in range(len(current)):
            if time.time() - start_time > time_budget:
                return best
            # Compute coverage without current[i]
            covered_without_i = 0
            for j, idx in enumerate(current):
                if j != i:
                    covered_without_i |= spheres[idx]
            
            # Find best replacement
            best_gain = 0
            best_new = -1
            for new_idx in range(len(spheres)):
                if new_idx in current:
                    continue
                gain = (spheres[new_idx] & ~covered_without_i).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_new = new_idx
            
            if best_new >= 0 and best_gain > 0:
                # Check if this swap actually helps (we need same or fewer words)
                # Since we're replacing one with one, size stays same
                # But it might enable further removals
                new_current = [c for j, c in enumerate(current) if j != i] + [best_new]
                new_covered = covered_without_i | spheres[best_new]
                if new_covered == full_mask:
                    current = new_current
                    covered = new_covered
                    improved = True
                    break
    
    if len(current) < best_len:
        best = list(current)
    
    return best
>>>>>>> REPLACE

<<<<<<< SEARCH
def _simulated_annealing(code_indices, spheres, full_mask, time_budget, start_time):
    """Simulated annealing to minimize |code| + |uncovered| using bitmask operations."""
    current_code = list(code_indices)
    current_covered = 0
    for idx in current_code:
        current_covered |= spheres[idx]
    
    current_uncovered_count = full_mask.bit_count() - current_covered.bit_count()
    current_objective = len(current_code) + current_uncovered_count
    
    best_code = list(current_code)
    best_objective = current_objective
    
    # Temperature schedule
    t0 = 2.0
    t_end = 0.01
    n_steps = 50000
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Cooling
        progress = step / n_steps
        temperature = t0 * (t_end / t0) ** progress
        
        # Choose a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try removing a codeword
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            new_code = [c for c in current_code if c != idx_to_remove]
            new_covered = 0
            for idx in new_code:
                new_covered |= spheres[idx]
            new_uncovered = full_mask.bit_count() - new_covered.bit_count()
            new_objective = len(new_code) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = new_code
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
                    
        elif move_type < 0.6:
            # Try adding a random codeword
            new_idx = random.randint(0, len(spheres) - 1)
            if new_idx in current_code:
                continue
            new_code = current_code + [new_idx]
            new_covered = current_covered | spheres[new_idx]
            new_uncovered = full_mask.bit_count() - new_covered.bit_count()
            new_objective = len(new_code) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = new_code
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
                    
        else:
            # Try swapping: remove one, add another
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            new_idx = random.randint(0, len(spheres) - 1)
            if new_idx == idx_to_remove or new_idx in current_code:
                continue
            new_code = [c for c in current_code if c != idx_to_remove] + [new_idx]
            new_covered = 0
            for idx in new_code:
                new_covered |= spheres[idx]
            new_uncovered = full_mask.bit_count() - new_covered.bit_count()
            new_objective = len(new_code) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = new_code
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
    
    return best_code
=======
def _simulated_annealing(code_indices, spheres, full_mask, time_budget, start_time):
    """Simulated annealing to minimize |code| + |uncovered| using bitmask operations."""
    current_code = list(code_indices)
    current_covered = 0
    for idx in current_code:
        current_covered |= spheres[idx]
    
    total_bits = full_mask.bit_count()
    current_uncovered_count = total_bits - current_covered.bit_count()
    current_objective = len(current_code) + current_uncovered_count
    
    best_code = list(current_code)
    best_objective = current_objective
    
    # Temperature schedule
    t0 = 2.0
    t_end = 0.01
    n_steps = 50000
    total = len(spheres)
    
    for step in range(n_steps):
        if time.time() - start_time > time_budget:
            break
        
        # Cooling
        progress = step / n_steps
        temperature = t0 * (t_end / t0) ** progress
        
        # Choose a random move: add, remove, or swap
        move_type = random.random()
        
        if move_type < 0.3:
            # Try removing a codeword
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            # Compute new coverage
            new_covered = 0
            for idx in current_code:
                if idx != idx_to_remove:
                    new_covered |= spheres[idx]
            new_uncovered = total_bits - new_covered.bit_count()
            new_objective = (len(current_code) - 1) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = [c for c in current_code if c != idx_to_remove]
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
                    
        elif move_type < 0.6:
            # Try adding a random codeword
            new_idx = random.randint(0, total - 1)
            if new_idx in current_code:
                continue
            new_covered = current_covered | spheres[new_idx]
            new_uncovered = total_bits - new_covered.bit_count()
            new_objective = (len(current_code) + 1) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = current_code + [new_idx]
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
                    
        else:
            # Try swapping: remove one, add another
            if len(current_code) == 0:
                continue
            idx_to_remove = random.choice(current_code)
            new_idx = random.randint(0, total - 1)
            if new_idx == idx_to_remove or new_idx in current_code:
                continue
            # Compute coverage after swap
            new_covered = 0
            for idx in current_code:
                if idx != idx_to_remove:
                    new_covered |= spheres[idx]
            new_covered |= spheres[new_idx]
            new_uncovered = total_bits - new_covered.bit_count()
            new_objective = len(current_code) + new_uncovered
            delta = new_objective - current_objective
            if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                current_code = [c for c in current_code if c != idx_to_remove] + [new_idx]
                current_covered = new_covered
                current_uncovered_count = new_uncovered
                current_objective = new_objective
                if current_objective < best_objective:
                    best_objective = current_objective
                    best_code = list(current_code)
    
    return best_code
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Multiple greedy restarts
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() - start > _DEADLINE - 20:
            break
        
        random.seed(r)
        candidates = list(range(total))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        while covered != full_mask:
            if time.time() - start > _DEADLINE - 20:
                break
            best_gain = -1
            best_idx = -1
            for idx in candidates:
                new_covered = spheres[idx] & ~covered
                gain = new_covered.bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain <= 0:
                break
            chosen.append(best_idx)
            covered |= spheres[best_idx]
        
        # Prune the greedy result
        pruned_greedy = _prune_bitmask(chosen, spheres, full_mask)
        
        # Try simulated annealing on the pruned result
        if time.time() - start < _DEADLINE - 30:
            sa_result = _simulated_annealing(pruned_greedy, spheres, full_mask, 
                                            _DEADLINE - 30 - (time.time() - start), start)
            if len(sa_result) < len(pruned_greedy):
                pruned_greedy = sa_result
        
        if len(pruned_greedy) < best_len:
            best_len = len(pruned_greedy)
            best = [all_words[i] for i in pruned_greedy]
=======
    # Multiple greedy restarts with local search polish
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() - start > _DEADLINE - 20:
            break
        
        random.seed(r)
        candidates = list(range(total))
        random.shuffle(candidates)
        
        covered = 0
        chosen = []
        while covered != full_mask:
            if time.time() - start > _DEADLINE - 20:
                break
            best_gain = -1
            best_idx = -1
            for idx in candidates:
                gain = (spheres[idx] & ~covered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            if best_gain <= 0:
                break
            chosen.append(best_idx)
            covered |= spheres[best_idx]
        
        # Prune the greedy result
        pruned_greedy = _prune_bitmask(chosen, spheres, full_mask)
        
        # Local search polish
        if time.time() - start < _DEADLINE - 15:
            ls_result = _local_search_prune(pruned_greedy, spheres, full_mask,
                                            _DEADLINE - 15 - (time.time() - start), start)
            if len(ls_result) < len(pruned_greedy):
                pruned_greedy = ls_result
        
        # Try simulated annealing on the polished result
        if time.time() - start < _DEADLINE - 5:
            sa_result = _simulated_annealing(pruned_greedy, spheres, full_mask, 
                                            _DEADLINE - 5 - (time.time() - start), start)
            if len(sa_result) < len(pruned_greedy):
                pruned_greedy = sa_result
        
        if len(pruned_greedy) < best_len:
            best_len = len(pruned_greedy)
            best = [all_words[i] for i in pruned_greedy]
>>>>>>> REPLACE

