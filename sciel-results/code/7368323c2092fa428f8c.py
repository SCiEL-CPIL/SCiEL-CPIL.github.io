"""Odd covering system for moduli dividing 51975.
Strategy: randomized greedy set-packing with adaptive probabilistic look-ahead.
"""
import time
import random
import numpy as np

BASE = 51975  # 3^3 * 5^2 * 7 * 11

# Precompute all divisors > 1
def _get_divisors():
    divs = []
    for a in range(4):      # 3^a
        for b in range(3):  # 5^b
            for c in range(2):  # 7^c
                for d in range(2):  # 11^d
                    m = (3**a) * (5**b) * (7**c) * (11**d)
                    if m > 1:
                        divs.append(m)
    return sorted(divs)

ALL_MODULI = _get_divisors()
NUM_MODULI = len(ALL_MODULI)

# Precompute residue lists for each modulus: residues[m] = list of x in [0, BASE) with x % m == r
# We'll use a mapping from modulus to a numpy array of its residue positions
# Actually, for each modulus m, the class (m, r) covers positions r, r+m, r+2m, ...
# We precompute for each modulus m the full set of positions as a 2D structure.

# Precompute: for each modulus m, a list of arrays where arr[r] = positions covered by (m, r)
# To save memory, store as list of numpy arrays per modulus.
_mod_positions = {}
for m in ALL_MODULI:
    arrs = []
    for r in range(m):
        arrs.append(np.arange(r, BASE, m, dtype=np.int32))
    _mod_positions[m] = arrs

# Index map: modulus -> index in ALL_MODULI
_mod_index = {m: i for i, m in enumerate(ALL_MODULI)}


def _greedy_with_lookahead(rng, deadline, initial_order=None):
    """
    Randomized greedy with adaptive probabilistic look-ahead.
    At each step, for each remaining modulus, compute marginal gain.
    Select next modulus with probability proportional to gain^alpha (soft-max),
    with alpha adapting based on remaining budget.
    """
    covered = np.zeros(BASE, dtype=bool)
    used_mods = set()
    classes = []
    remaining = list(ALL_MODULI)
    
    if initial_order:
        # Use a shuffled version of the given order as tie-breaking seed
        remaining = initial_order[:]
        rng.shuffle(remaining)
    
    # Precompute gain for each (m, r) pair lazily; we'll compute on the fly
    # For efficiency, precompute the positions arrays (done above)
    
    # Adaptive alpha: start high (greedy), decrease over time (more random)
    total_steps = NUM_MODULI
    step = 0
    
    while remaining and time.time() < deadline:
        step += 1
        # Adaptive temperature: alpha decreases as we go (more exploration later)
        # alpha = alpha_max * (1 - step/total_steps) + alpha_min
        alpha_max = 5.0
        alpha_min = 0.3
        alpha = alpha_max * (1.0 - step / total_steps) + alpha_min
        
        # Compute marginal gain for each remaining modulus
        # For each modulus m, find best r and its gain
        gains = {}
        best_r_for_m = {}
        
        for m in remaining:
            positions_list = _mod_positions[m]
            best_gain = -1
            best_r = 0
            for r in range(m):
                pos = positions_list[r]
                if len(pos) == 0:
                    continue
                # Count how many are not yet covered
                gain = np.count_nonzero(~covered[pos])
                if gain > best_gain:
                    best_gain = gain
                    best_r = r
            gains[m] = best_gain
            best_r_for_m[m] = best_r
        
        # Remove moduli with zero gain (they can't help)
        active = [m for m in remaining if gains[m] > 0]
        if not active:
            break
        
        # Probabilistic selection: P(m) proportional to gains[m]^alpha
        # Use soft-max (log-sum-exp) for numerical stability
        gain_arr = np.array([gains[m] for m in active], dtype=np.float64)
        scaled = gain_arr * alpha
        # log-sum-exp for stability
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        total_exp = np.sum(exp_scaled)
        probs = exp_scaled / total_exp
        
        # Sample
        idx = rng.choice(len(active), p=probs)
        chosen_m = active[idx]
        chosen_r = best_r_for_m[chosen_m]
        
        # Apply
        pos = _mod_positions[chosen_m][chosen_r]
        covered[pos] = True
        classes.append([chosen_m, chosen_r])
        used_mods.add(chosen_m)
        
        # Remove from remaining
        remaining = [m for m in remaining if m != chosen_m]
    
    # Add any remaining moduli with gain 0 (they don't affect score but must be distinct)
    # Actually, the task says we can choose which moduli to use. Let's only keep useful ones.
    # But wait - the task says "with distinct moduli m > 1 dividing 51975". It doesn't say we MUST use all.
    # Let me re-read: "construct() -> classes [[m, r], ...] with distinct moduli m > 1 dividing 51975"
    # It seems we can choose any subset. Let's only keep the ones we added.
    
    return classes


def _local_search(classes, rng, deadline, max_rounds=5):
    """Local search: try to reassign residues of existing classes to reduce uncovered count."""
    if not classes:
        return classes, BASE
    
    covered = np.zeros(BASE, dtype=bool)
    for m, r in classes:
        pos = _mod_positions[m][r]
        covered[pos] = True
    
    current_uncovered = np.count_nonzero(~covered)
    
    for round_num in range(max_rounds):
        if time.time() > deadline:
            break
        improved = False
        # Try reassigning each class's residue
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            m, old_r = classes[i]
            positions_list = _mod_positions[m]
            
            # Try a few random residues
            for trial in range(3):
                new_r = rng.randrange(m)
                if new_r == old_r:
                    continue
                
                # Compute effect: remove old class contribution, add new
                # Temporarily unmark old, mark new, check if it helps
                # But we need to be careful: other classes may cover those positions too.
                # Simple approach: just compute the delta in uncovered count.
                # Positions that were ONLY covered by this class (m, old_r) will become uncovered.
                # Positions that are newly covered by (m, new_r) will become covered.
                
                # For simplicity, just try it and measure
                old_pos = positions_list[old_r]
                new_pos = positions_list[new_r]
                
                # Count: how many positions in old_pos are not covered by any other class?
                # This is expensive. Let's use a simpler heuristic:
                # Just check if new_r covers more currently-uncovered positions than old_r does.
                old_gain = np.count_nonzero(~covered[old_pos])
                new_gain = np.count_nonzero(~covered[new_pos])
                
                if new_gain > old_gain:
                    # Apply the change
                    covered[old_pos] = False
                    # Re-apply other classes that cover old_pos positions
                    for j in range(len(classes)):
                        if j == i:
                            continue
                        mj, rj = classes[j]
                        covered[_mod_positions[mj][rj]] = True
                    covered[new_pos] = True
                    classes[i][1] = new_r
                    current_uncovered = np.count_nonzero(~covered)
                    improved = True
                    break
        
        if not improved:
            break
    
    return classes, current_uncovered


def _try_remove_redundant(classes, deadline):
    """Try to remove classes that don't contribute to coverage."""
    if not classes:
        return classes, BASE
    
    covered = np.zeros(BASE, dtype=bool)
    for m, r in classes:
        covered[_mod_positions[m][r]] = True
    
    current_uncovered = np.count_nonzero(~covered)
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(classes) - 1, -1, -1):
            if time.time() > deadline:
                break
            m, r = classes[i]
            # Remove this class and see if uncovered count changes
            temp_covered = np.zeros(BASE, dtype=bool)
            for j in range(len(classes)):
                if j == i:
                    continue
                mj, rj = classes[j]
                temp_covered[_mod_positions[mj][rj]] = True
            
            new_uncovered = np.count_nonzero(~temp_covered)
            if new_uncovered <= current_uncovered:
                classes.pop(i)
                if new_uncovered < current_uncovered:
                    current_uncovered = new_uncovered
                covered = temp_covered
                improved = True
                break
    
    return classes, current_uncovered


def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_classes = None
    best_score = BASE
    
    rng = random.Random(42)
    
    # Multiple randomized greedy runs with different seeds
    num_restarts = 30
    for restart in range(num_restarts):
        if time.time() > deadline - 10:  # Leave time for local search
            break
        
        seed = 42 + restart * 7919
        local_rng = random.Random(seed)
        
        classes = _greedy_with_lookahead(local_rng, deadline - 10)
        
        if not classes:
            continue
        
        # Compute score
        covered = np.zeros(BASE, dtype=bool)
        for m, r in classes:
            covered[_mod_positions[m][r]] = True
        score = np.count_nonzero(~covered)
        
        if score < best_score:
            best_score = score
            best_classes = [c[:] for c in classes]
    
    # Local search on best solution
    if best_classes and time.time() < deadline - 5:
        local_rng = random.Random(999)
        best_classes, best_score = _local_search(best_classes, local_rng, deadline - 5, max_rounds=8)
    
    # Try to remove redundant classes
    if best_classes and time.time() < deadline - 2:
        best_classes, best_score = _try_remove_redundant(best_classes, deadline - 2)
    
    # Final fallback
    if best_classes is None:
        best_classes = [[m, 0] for m in ALL_MODULI]
    
    return best_classes
