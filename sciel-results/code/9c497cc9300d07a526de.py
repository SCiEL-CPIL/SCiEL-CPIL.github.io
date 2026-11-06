import time
import random
from itertools import combinations

# Precompute all 4-subsets of [1,16] as bitmasks (0-indexed bits for 0..15)
_ALL_MASKS = []
_MASK_TO_IDX = {}
for c in combinations(range(16), 4):
    mask = 0
    for i in c:
        mask |= (1 << i)
    _ALL_MASKS.append(mask)
    _MASK_TO_IDX[mask] = len(_ALL_MASKS) - 1

_N = len(_ALL_MASKS)  # 1820
_ALL_MASK = (1 << 16) - 1

# Cache for conflict sets
_conflict_cache = {}

def _get_conflicts(i, j):
    """
    Returns list of indices k such that {i, j, k} is a sunflower.
    """
    key = (i, j) if i < j else (j, i)
    if key in _conflict_cache:
        return _conflict_cache[key]
    
    mi = _ALL_MASKS[i]
    mj = _ALL_MASKS[j]
    K = mi & mj
    k_size = bin(K).count('1')
    
    if k_size > 4:
        _conflict_cache[key] = []
        return []
        
    # Free elements are those not in mi or mj
    # Mk must contain K and be disjoint from (mi \ K) and (mj \ K)
    # So Mk \ K must be a subset of (All \ (mi | mj))
    free_mask = _ALL_MASK & ~(mi | mj)
    num_to_choose = 4 - k_size
    
    if num_to_choose < 0:
        _conflict_cache[key] = []
        return []
        
    free_indices = [idx for idx in range(16) if (free_mask >> idx) & 1]
    
    conflicts = []
    if num_to_choose == 0:
        # Mk must be exactly K
        mk = K
        idx_k = _MASK_TO_IDX.get(mk, -1)
        if idx_k != -1 and idx_k != i and idx_k != j:
            conflicts.append(idx_k)
    else:
        for combo in combinations(free_indices, num_to_choose):
            y_mask = 0
            for idx in combo:
                y_mask |= (1 << idx)
            mk = K | y_mask
            idx_k = _MASK_TO_IDX.get(mk, -1)
            if idx_k != -1 and idx_k != i and idx_k != j:
                conflicts.append(idx_k)
                
    _conflict_cache[key] = conflicts
    return conflicts

def _greedy_build(order, deadline):
    """
    Build a family using greedy addition.
    Uses forbidden_count for O(1) checks.
    """
    forbidden_count = [0] * _N
    selected = set()
    family = []
    
    def can_add(c):
        return c not in selected and forbidden_count[c] == 0
        
    def add(c):
        selected.add(c)
        family.append(c)
        # Update forbidden counts
        for y in family:
            if y == c:
                continue
            conflicts = _get_conflicts(c, y)
            for k in conflicts:
                forbidden_count[k] += 1
                
    def remove(c):
        selected.discard(c)
        # Remove from family list
        if c in family:
            family.remove(c)
        # Update forbidden counts
        for y in family:
            if y == c:
                continue
            conflicts = _get_conflicts(c, y)
            for k in conflicts:
                forbidden_count[k] -= 1
                
    for idx in order:
        if time.time() > deadline:
            break
        if can_add(idx):
            add(idx)
            
    return family

def _local_improve(family, deadline):
    """
    Improve family by trying to add more elements, and if stuck, swapping.
    """
    forbidden_count = [0] * _N
    selected = set(family)
    # Initialize forbidden counts
    for i in range(len(family)):
        for j in range(i + 1, len(family)):
            conflicts = _get_conflicts(family[i], family[j])
            for k in conflicts:
                forbidden_count[k] += 1
                
    def can_add(c):
        return c not in selected and forbidden_count[c] == 0
        
    def add(c):
        selected.add(c)
        family.append(c)
        for y in family:
            if y == c:
                continue
            conflicts = _get_conflicts(c, y)
            for k in conflicts:
                forbidden_count[k] += 1
                
    def remove(c):
        selected.discard(c)
        family.remove(c)
        for y in family:
            if y == c:
                continue
            conflicts = _get_conflicts(c, y)
            for k in conflicts:
                forbidden_count[k] -= 1
                
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any available
        for c in range(_N):
            if time.time() > deadline:
                break
            if can_add(c):
                add(c)
                improved = True
                # Reset loop to try more adds
                break
                
        if not improved:
            # Try swaps
            # For each x in family, try to swap with some y
            # We want to find a swap that allows further additions
            # Heuristic: try random swaps
            rng = random.Random()
            for _ in range(100): # Increased swap attempts
                if time.time() > deadline:
                    break
                if not family:
                    break
                x = rng.choice(family)
                y = rng.randint(0, _N - 1)
                if y in selected:
                    continue
                    
                # Try swap
                remove(x)
                if can_add(y):
                    add(y)
                    improved = True
                    # After swap, try to add more in next loop iteration
                    break
                else:
                    # Revert
                    add(x)
                    
    return family

def _simulated_annealing(family, deadline):
    """
    Structural change: Use Simulated Annealing to escape local optima.
    Moves: Add, Remove, Swap.
    Energy: -size (we want to maximize size).
    """
    current_family = list(family)
    current_size = len(current_family)
    best_family = list(current_family)
    best_size = current_size
    
    # Build initial forbidden counts
    forbidden_count = [0] * _N
    selected = set(current_family)
    for i in range(len(current_family)):
        for j in range(i + 1, len(current_family)):
            conflicts = _get_conflicts(current_family[i], current_family[j])
            for k in conflicts:
                forbidden_count[k] += 1
                
    def can_add(c):
        return c not in selected and forbidden_count[c] == 0
        
    def add(c):
        selected.add(c)
        current_family.append(c)
        for y in current_family:
            if y == c:
                continue
            conflicts = _get_conflicts(c, y)
            for k in conflicts:
                forbidden_count[k] += 1
                
    def remove(c):
        selected.discard(c)
        current_family.remove(c)
        for y in current_family:
            if y == c:
                continue
            conflicts = _get_conflicts(c, y)
            for k in conflicts:
                forbidden_count[k] -= 1
                
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    rng = random.Random(123)
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.3:
            # Try to add a random element
            if not current_family:
                c = rng.randint(0, _N - 1)
            else:
                c = rng.randint(0, _N - 1)
                
            if c in selected:
                continue
                
            # Check if valid
            if can_add(c):
                add(c)
                delta_energy = -1.0 # Improvement
                # Accept
                if len(current_family) > best_size:
                    best_size = len(current_family)
                    best_family = list(current_family)
            else:
                # Reject (energy increases)
                # But in SA, we might accept if T is high enough?
                # Since we can't add invalid moves, we just skip.
                # Or we could try a swap instead.
                pass
                
        elif move_type < 0.6:
            # Try to remove a random element
            if not current_family:
                continue
            c = rng.choice(current_family)
            remove(c)
            delta_energy = 1.0 # Worse
            # Accept with probability exp(-delta/T)
            if rng.random() < 2.71828 ** (-delta_energy / T):
                pass # Accepted
            else:
                add(c) # Rejected, revert
                
        else:
            # Try a swap: remove one, add one
            if not current_family:
                continue
            c1 = rng.choice(current_family)
            c2 = rng.randint(0, _N - 1)
            if c2 in selected:
                continue
                
            remove(c1)
            if can_add(c2):
                add(c2)
                # Size unchanged, but structure changed. 
                # This allows escaping local minima in the "add" space.
                # We accept this move always (or with prob 1) to explore.
                # Actually, let's treat it as neutral.
            else:
                # Revert
                add(c1)
                
        T *= cooling
        
    return best_family

def construct():
    start_time = time.time()
    deadline = start_time + 85 # Safety margin
    
    best_family = []
    best_size = 0
    
    # Multiple random restarts
    rng = random.Random(42)
    
    # 1. Deterministic Greedy
    fam = _greedy_build(list(range(_N)), deadline - 60)
    if len(fam) > best_size:
        best_size = len(fam)
        best_family = fam
    
    # 2. Random Greedy Restarts
    for _ in range(50):
        if time.time() > deadline - 30:
            break
        order = list(range(_N))
        rng.shuffle(order)
        fam = _greedy_build(order, deadline - 30)
        if len(fam) > best_size:
            best_size = len(fam)
            best_family = fam
            
    # 3. Local Improvement on Best
    if time.time() < deadline - 10:
        improved_fam = _local_improve(list(best_family), deadline - 10)
        if len(improved_fam) > best_size:
            best_size = len(improved_fam)
            best_family = improved_fam
            
    # 4. Simulated Annealing (Structural Change)
    if time.time() < deadline - 5:
        sa_fam = _simulated_annealing(list(best_family), deadline - 2)
        if len(sa_fam) > best_size:
            best_size = len(sa_fam)
            best_family = sa_fam
            
    # 5. Final Polish: Try to add any remaining
    if time.time() < deadline - 1:
        # Re-initialize counters for final add pass
        forbidden_count = [0] * _N
        selected = set(best_family)
        for i in range(len(best_family)):
            for j in range(i + 1, len(best_family)):
                conflicts = _get_conflicts(best_family[i], best_family[j])
                for k in conflicts:
                    forbidden_count[k] += 1
        
        for c in range(_N):
            if time.time() > deadline - 0.5:
                break
            if c not in selected and forbidden_count[c] == 0:
                # Add it
                best_family.append(c)
                selected.add(c)
                for y in best_family:
                    if y == c:
                        continue
                    conflicts = _get_conflicts(c, y)
                    for k in conflicts:
                        forbidden_count[k] += 1
                        
    # Convert back to sorted lists of 1-based integers
    result = []
    for idx in best_family:
        mask = _ALL_MASKS[idx]
        elems = []
        for i in range(16):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
