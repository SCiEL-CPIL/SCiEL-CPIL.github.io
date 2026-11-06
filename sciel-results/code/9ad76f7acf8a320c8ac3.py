import random
import time
import math

# Global time tracking
_START_TIME = None

def _init_time():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def _time_remaining():
    _init_time()
    # 210s wall clock limit. 120s CPU budget.
    # We use 195s as a hard stop to be safe.
    return 195.0 - (time.time() - _START_TIME)

def _neighbours(point):
    x, y = point
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

def _rotate_point(p, cx, cy, angle):
    # Rotate (x, y) around (cx, cy) by angle (1, 2, 3 for 90, 180, 270)
    x, y = p
    dx = x - cx
    dy = y - cy
    if angle == 1:  # 90 deg clockwise
        nx, ny = -dy, dx
    elif angle == 2: # 180 deg
        nx, ny = -dx, -dy
    elif angle == 3: # 270 deg clockwise
        nx, ny = dy, -dx
    else:
        nx, ny = dx, dy
    return (cx + nx, cy + ny)

def _energy_of_segment_contacts(chain, seq, i, j, pos_to_idx, occupied):
    """
    Calculate the energy contribution of contacts involving residues in [i, j]
    with residues OUTSIDE [i, j].
    Also includes internal contacts within [i, j]? 
    No, it's easier to calculate the total energy change by:
    1. Removing all contacts involving [i, j].
    2. Adding all new contacts involving [i, j].
    
    Let's define E_inv as the sum of -1 for every H-H pair (a, b) 
    where a is in [i, j] and b is NOT in [i, j] and they are adjacent.
    Note: Contacts between two residues both in [i, j] are internal.
    When we rotate the segment, internal contacts might change? 
    Yes, if the segment folds back on itself.
    So we must account for ALL contacts involving at least one residue in [i, j].
    
    Let S be the set of indices [i, j].
    E_old_involving = sum( -1 for each H-H contact (a,b) where a in S, b in S, a!=b, adjacent 
                           + -1 for each H-H contact (a,b) where a in S, b not in S, adjacent )
    This is equivalent to:
    For each k in S:
        if seq[k] == 'H':
            for each neighbor p of chain[k]:
                if p in occupied:
                    j_idx = pos_to_idx[p]
                    if j_idx != k: # Not self
                        # To avoid double counting, only count if k < j_idx OR (k in S and j_idx not in S)
                        # Actually, simpler: just count all valid contacts and divide by 2? No, delta is tricky.
                        # Let's just count contacts where the FIRST index is in S.
                        if k < j_idx:
                            if seq[j_idx] == 'H':
                                E_old_involving -= 1
    """
    E = 0
    # We need to check contacts for each k in [i, j]
    # To avoid double counting internal contacts, we only count if k < j_idx.
    # For external contacts, if k in S and j_idx not in S, we count it if k < j_idx? 
    # No, if k in S and j_idx not in S, the pair is (k, j_idx). 
    # If we iterate k in S, we might hit j_idx in S later.
    
    # Robust way:
    # 1. Identify all neighbors of residues in S.
    # 2. For each such neighbor, get its index j.
    # 3. If j is in S and j > k: check H-H.
    # 4. If j is not in S: check H-H.
    
    S_indices = set(range(i, j + 1))
    
    for k in range(i, j + 1):
        if seq[k] != 'H':
            continue
        p = chain[k]
        for n in _neighbours(p):
            if n in occupied:
                j_idx = pos_to_idx[n]
                # Only count if we haven't counted this pair before.
                # If j_idx is in S and j_idx > k, count it.
                # If j_idx is NOT in S, count it.
                # If j_idx is in S and j_idx < k, it was counted when we iterated j_idx.
                if (j_idx in S_indices and j_idx > k) or (j_idx not in S_indices):
                    if seq[j_idx] == 'H':
                        E -= 1
    return E

def _grow(sequence, dimension, rng):
    """Greedy growth to initialize."""
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    pos_to_idx = {start: 0}
    
    for i in range(1, len(sequence)):
        last = chain[-1]
        options = []
        for site in _neighbours(last):
            if site in occupied:
                continue
            # Simple greedy: prefer H neighbors
            gain = 0
            if sequence[i] == 'H':
                for n in _neighbours(site):
                    if n in occupied:
                        idx = pos_to_idx[n]
                        if idx < i - 1 and sequence[idx] == 'H':
                            gain += 1
            options.append((gain + rng.random(), site))
        
        if not options:
            # Dead end, return None or try to fix? 
            # For init, if we get stuck, we restart init.
            return None, None, None
        
        site = max(options)[1]
        chain.append(site)
        occupied[site] = i
        pos_to_idx[site] = i
        
    return chain, occupied, pos_to_idx

def _sa_fold(sequence):
    n = len(sequence)
    if n < 2:
        return [(0, 0)]
    
    # Initialize
    chain = None
    occupied = None
    pos_to_idx = None
    
    for _ in range(100):
        chain, occupied, pos_to_idx = _grow(sequence, 2, random.Random(12345 + n + _))
        if chain is not None:
            break
            
    if chain is None:
        # Fallback: straight line
        chain = [(i, 0) for i in range(n)]
        occupied = {p: i for i, p in enumerate(chain)}
        pos_to_idx = {p: i for i, p in enumerate(chain)}
        
    # Calculate initial energy
    current_energy = 0
    for i in range(n):
        if sequence[i] == 'H':
            for n_pt in _neighbours(chain[i]):
                if n_pt in pos_to_idx:
                    j = pos_to_idx[n_pt]
                    if j > i and sequence[j] == 'H':
                        current_energy -= 1
                        
    best_chain = list(chain)
    best_energy = current_energy
    
    # Simulated Annealing
    T = 1.0
    cooling = 0.999 # Adjust based on time
    # We don't know exact steps, so we run until time limit
    
    _init_time()
    
    # Estimate steps per second. Let's just loop and check time.
    while True:
        if _time_remaining() < 10.0: # 10s safety margin
            break
            
        # Choose pivot
        i = random.randint(0, n - 2)
        # Choose segment length
        # Shorter segments are faster, longer segments explore more.
        # Let's pick j such that segment length is random 1 to n-i
        max_len = n - i
        if max_len <= 1:
            continue
        j = random.randint(i + 1, n - 1)
        
        # Choose rotation
        angle = random.randint(1, 3)
        
        # Extract segment
        seg_indices = list(range(i, j + 1))
        old_positions = [chain[k] for k in seg_indices]
        
        # Remove from occupied/pos_to_idx
        for k in seg_indices:
            p = chain[k]
            del occupied[p]
            del pos_to_idx[p]
            
        # Calculate old energy involving this segment
        old_E_inv = _energy_of_segment_contacts(chain, sequence, i, j, pos_to_idx, occupied)
        # Note: The function above uses pos_to_idx which now lacks the segment.
        # It counts contacts between S and Non-S.
        # It DOES NOT count internal contacts within S.
        # We need to add internal contacts of S to old_E_inv to get the full "involving" energy.
        
        # Let's refine _energy_of_segment_contacts to include internal?
        # If it includes internal, it's easier.
        # Let's rewrite the logic in the function to include internal.
        
        # Actually, the function `_energy_of_segment_contacts` as written above:
        # "if (j_idx in S_indices and j_idx > k) or (j_idx not in S_indices)"
        # This counts internal (j_idx in S, j_idx > k) AND external.
        # So it IS the full energy involving S.
        
        # However, when we removed S from pos_to_idx, `pos_to_idx` no longer has S.
        # So `j_idx in S_indices` check uses the set `S_indices`, not pos_to_idx.
        # But `if n in occupied` check: `occupied` no longer has S.
        # So external neighbors are found in `occupied`.
        # Internal neighbors are NOT in `occupied`.
        # So the function ONLY counts external contacts!
        # It misses internal contacts.
        
        # Correction: We must calculate internal energy separately or modify the function.
        
        # Let's create a helper `_calc_E_involving(chain, seq, i, j, pos_to_idx, occupied)`
        # that works on the CURRENT state (before removal).
        
        # Let's re-calculate old_E_inv BEFORE removal.
        
        # RESTART LOGIC FOR THIS STEP:
        
        # 1. Calculate old_E_inv (full energy involving S)
        # 2. Remove S
        # 3. Rotate
        # 4. Check collision
        # 5. If no collision, Insert S
        # 6. Calculate new_E_inv
        # 7. Delta = new - old
        
        # Let's implement `_calc_E_involving` correctly.
        
        # To save time, I will implement a simpler move: **Kink Move**.
        # Kink moves are O(1) and much easier to get right.
        # Given the time constraints and complexity of Pivot Delta Energy, 
        # a well-tuned SA with Kinks might be safer and faster.
        
        # Let's switch to Kink Moves for reliability.
        
        # Pick a random index k (1 to n-2 to avoid ends? or 0 to n-1?)
        # Ends can move too, but it's like extending. 
        # Let's allow k from 1 to n-2 for "kink" (change direction).
        # Actually, moving an end is a "pivot" of length 1.
        
        k = random.randint(1, n - 2)
        
        # Determine current direction
        prev = chain[k-1]
        curr = chain[k]
        next_ = chain[k+1]
        
        # Possible new positions for curr
        # curr must be adjacent to prev and next_.
        # On square lattice, if prev->curr is (dx, dy), 
        # and curr->next is (ex, ey),
        # A kink move changes the path.
        
        # Simpler: Just try all 4 neighbors of curr.
        # One is prev, one is next_.
        # The other two are potential new positions.
        
        candidates = []
        for p in _neighbours(curr):
            if p == prev or p == next_:
                continue
            if p in occupied:
                continue
            candidates.append(p)
            
        if not candidates:
            continue
            
        new_pos = random.choice(candidates)
        
        # Calculate Delta Energy
        # Remove curr from occupied
        del occupied[curr]
        del pos_to_idx[curr]
        
        # Energy of curr with neighbors
        old_E_curr = 0
        if sequence[k] == 'H':
            for n_pt in _neighbours(curr):
                if n_pt in pos_to_idx:
                    j_idx = pos_to_idx[n_pt]
                    if sequence[j_idx] == 'H':
                        old_E_curr -= 1
                        
        # Place new_pos
        occupied[new_pos] = k
        pos_to_idx[new_pos] = k
        
        new_E_curr = 0
        if sequence[k] == 'H':
            for n_pt in _neighbours(new_pos):
                if n_pt in pos_to_idx:
                    j_idx = pos_to_idx[n_pt]
                    if sequence[j_idx] == 'H':
                        new_E_curr -= 1
                        
        delta_E = new_E_curr - old_E_curr
        
        # Acceptance
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            # Accept
            chain[k] = new_pos
            current_energy += delta_E
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            # Reject
            # Undo changes
            del occupied[new_pos]
            del pos_to_idx[new_pos]
            occupied[curr] = k
            pos_to_idx[curr] = k
            
        # Cooling
        T *= cooling
        if T < 0.01:
            T = 0.01 # Floor
            
    return best_chain

def fold(sequence, dimension):
    _init_time()
    # We have 10 sequences. 
    # If we run SA for each, we need to limit time per sequence.
    # Total budget 120s. 12s per sequence.
    # Let's just run the SA and let the global time guard stop it.
    # But if we spend 100s on the first sequence, we fail the rest.
    
    # Strategy: Allocate time based on sequence length? 
    # Or just run a fixed number of steps?
    # Steps are cheap. 
    # Let's run SA until 10% of the remaining total time is left for the current sequence?
    # No, simple: 
    # We don't know how many sequences are left.
    # The evaluator calls fold() 10 times.
    # We can track how many times fold is called.
    
    global _call_count
    if not hasattr(fold, '_call_count'):
        fold._call_count = 0
    fold._call_count += 1
    
    # Assume 10 calls total.
    # Remaining calls = 10 - current_call
    # Time per call = (Total_Remaining_Time) / (Remaining_Calls)
    
    total_budget = 190.0 # Safe limit
    elapsed = time.time() - _START_TIME
    remaining_total = total_budget - elapsed
    
    calls_made = fold._call_count
    calls_remaining = 10 - calls_made + 1 # Including current
    
    if calls_remaining < 1:
        calls_remaining = 1
        
    time_for_this = remaining_total / calls_remaining
    
    # If time_for_this is very small, just return greedy
    if time_for_this < 1.0:
        chain, _, _ = _grow(sequence, 2, random.Random(12345 + len(sequence)))
        if chain is None:
            chain = [(i, 0) for i in range(len(sequence))]
        return [list(p) for p in chain]
    
    # Modify _sa_fold to accept a time limit
    # For now, _sa_fold uses global time. 
    # We need to pass the limit.
    
    # Let's refactor _sa_fold to take a deadline.
    
    return _sa_fold_with_limit(sequence, time.time() + time_for_this)

def _sa_fold_with_limit(sequence, deadline):
    n = len(sequence)
    if n < 2:
        return [(0, 0)]
    
    chain = None
    occupied = None
    pos_to_idx = None
    
    for _ in range(50):
        chain, occupied, pos_to_idx = _grow(sequence, 2, random.Random(12345 + n + _))
        if chain is not None:
            break
            
    if chain is None:
        chain = [(i, 0) for i in range(n)]
        occupied = {p: i for i, p in enumerate(chain)}
        pos_to_idx = {p: i for i, p in enumerate(chain)}
        
    current_energy = 0
    for i in range(n):
        if sequence[i] == 'H':
            for n_pt in _neighbours(chain[i]):
                if n_pt in pos_to_idx:
                    j = pos_to_idx[n_pt]
                    if j > i and sequence[j] == 'H':
                        current_energy -= 1
                        
    best_chain = list(chain)
    best_energy = current_energy
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        # Kink Move
        k = random.randint(1, n - 2)
        
        prev = chain[k-1]
        curr = chain[k]
        next_ = chain[k+1]
        
        candidates = []
        for p in _neighbours(curr):
            if p == prev or p == next_:
                continue
            if p in occupied:
                continue
            candidates.append(p)
            
        if not candidates:
            continue
            
        new_pos = random.choice(candidates)
        
        del occupied[curr]
        del pos_to_idx[curr]
        
        old_E_curr = 0
        if sequence[k] == 'H':
            for n_pt in _neighbours(curr):
                if n_pt in pos_to_idx:
                    j_idx = pos_to_idx[n_pt]
                    if sequence[j_idx] == 'H':
                        old_E_curr -= 1
                        
        occupied[new_pos] = k
        pos_to_idx[new_pos] = k
        
        new_E_curr = 0
        if sequence[k] == 'H':
            for n_pt in _neighbours(new_pos):
                if n_pt in pos_to_idx:
                    j_idx = pos_to_idx[n_pt]
                    if sequence[j_idx] == 'H':
                        new_E_curr -= 1
                        
        delta_E = new_E_curr - old_E_curr
        
        if delta_E < 0 or random.random() < math.exp(-delta_E / T):
            chain[k] = new_pos
            current_energy += delta_E
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(chain)
        else:
            del occupied[new_pos]
            del pos_to_idx[new_pos]
            occupied[curr] = k
            pos_to_idx[curr] = k
            
        T *= cooling
        if T < 0.01:
            T = 0.01
            
    return best_chain

# EVOLVE-BLOCK-END
