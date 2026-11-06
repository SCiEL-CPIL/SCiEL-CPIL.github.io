"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math
import sys

_GLOBAL_START = time.time()
_GLOBAL_LIMIT = 195.0  # Safety margin for 210s kill


def _time_left():
    return _GLOBAL_LIMIT - (time.time() - _GLOBAL_START)


def _energy(sequence, chain):
    """Compute the HP contact energy of a conformation.
    Energy = - (number of H-H contacts)
    """
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    n = len(chain)
    for i, p in enumerate(chain):
        if sequence[i] == "H":
            x, y = p
            # Check 4 neighbors
            for q in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                j = where.get(q)
                if j is not None and j > i + 1 and sequence[j] == "H":
                    contacts += 1
    return -contacts


def _apply_pivot(chain, pivot_idx, angle):
    """
    Apply a pivot move at pivot_idx with given angle (90, -90, 180).
    Returns None if invalid (collision), else new chain list.
    """
    n = len(chain)
    if pivot_idx < 1 or pivot_idx >= n - 1:
        return None
    
    pivot = chain[pivot_idx]
    fixed = chain[:pivot_idx + 1]
    moving_start = pivot_idx + 1
    segment = chain[moving_start:]
    
    if not segment:
        return chain
    
    fixed_set = set(fixed)
    
    rotated_segment = []
    px, py = pivot
    
    valid = True
    for p in segment:
        dx, dy = p[0] - px, p[1] - py
        
        if angle == 90:
            ndx, ndy = -dy, dx
        elif angle == -90:
            ndx, ndy = dy, -dx
        else: # 180
            ndx, ndy = -dx, -dy
            
        new_p = (px + ndx, py + ndy)
        
        if new_p in fixed_set:
            valid = False
            break
            
        rotated_segment.append(new_p)
        
    if not valid:
        return None
        
    return fixed + rotated_segment


def _apply_slither(chain, rng):
    """
    Slither move: Move the last residue to a different valid neighbor of the second-to-last.
    """
    n = len(chain)
    if n < 2:
        return None
    
    prev = chain[-2]
    last = chain[-1]
    occupied = set(chain[:-1])
    
    candidates = []
    for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
        site = (prev[0]+dx, prev[1]+dy)
        if site != last and site not in occupied:
            candidates.append(site)
            
    if not candidates:
        return None
        
    new_last = rng.choice(candidates)
    return chain[:-1] + [new_last]


def _select_core_contacts(sequence, max_contacts=10):
    """
    Select a set of non-conflicting H-H contacts to target.
    Conflict: Two contacts (i,j) and (k,l) conflict if they share an index or 
    if the geometric path between them is likely to be blocked (simplified: just index disjointness 
    and not adjacent in sequence).
    For simplicity, we pick pairs of H residues that are far apart in sequence, 
    ensuring they don't share indices.
    """
    h_indices = [i for i, c in enumerate(sequence) if c == 'H']
    if len(h_indices) < 2:
        return []
    
    potential_contacts = []
    # Generate pairs of H indices that are not adjacent in sequence
    for i in range(len(h_indices)):
        for j in range(i+1, len(h_indices)):
            idx1 = h_indices[i]
            idx2 = h_indices[j]
            # Must be non-adjacent in chain to be a contact
            if abs(idx1 - idx2) > 1:
                # Weight by distance? Longer range contacts are harder but more valuable if satisfied
                potential_contacts.append((idx1, idx2))
                
    if not potential_contacts:
        return []
        
    # Greedy selection of non-overlapping contacts
    # Sort by "difficulty" or just random shuffle to add variety
    random.shuffle(potential_contacts)
    
    selected = []
    used_indices = set()
    
    for i, j in potential_contacts:
        if i in used_indices or j in used_indices:
            continue
        # Check if this contact is "compatible" with already selected ones
        # Simple compatibility: indices are distinct (already checked)
        # A stricter check would be geometric, but we don't have geometry yet.
        # We just want a set of disjoint pairs.
        selected.append((i, j))
        used_indices.add(i)
        used_indices.add(j)
        
        if len(selected) >= max_contacts:
            break
            
    return selected


def _backtrack_fold_guided(sequence, core_contacts, rng, time_limit):
    """
    Backtracking search that tries to satisfy core_contacts.
    """
    n = len(sequence)
    if n == 0:
        return []
    
    start = (0, 0)
    chain = [start]
    occupied = {start: 0}
    
    # Pre-compute target positions for core contacts?
    # No, we don't know where they will be.
    # Instead, during growth, if we place a residue i, and i is part of a core contact (i, j),
    # we might bias the direction of j later.
    # For now, we just use standard backtracking with a heuristic that prefers
    # moves that keep the chain compact and close to other H residues.
    
    best_chain = None
    best_score = -1000
    
    # Recursive backtracking is risky for N=100. 
    # We use a depth-limited backtracking or iterative deepening.
    # Given time constraints, we'll do a guided greedy growth with random restarts 
    # but biased towards the core contacts.
    
    # Actually, a true backtracking is better for "satisfying" constraints.
    # Let's try a limited backtracking.
    
    def backtrack(idx, current_chain, current_occupied):
        nonlocal best_chain, best_score
        
        if time.time() > time_limit:
            return False # Stop
        
        if idx == n:
            # Calculate a "score" based on how many core contacts are satisfied
            # and general energy
            e = _energy(sequence, current_chain)
            
            # Count satisfied core contacts
            sat = 0
            where = {p: i for i, p in enumerate(current_chain)}
            for i, j in core_contacts:
                if i in where and j in where:
                    p1 = current_chain[i]
                    p2 = current_chain[j]
                    # Check if adjacent
                    if abs(p1[0]-p2[0]) + abs(p1[1]-p2[1]) == 1:
                        sat += 1
                        
            # Score: prioritize satisfied core contacts, then energy
            score = sat * 100 + e
            
            if score > best_score:
                best_score = score
                best_chain = list(current_chain)
            return True # Found a valid full chain
            
        curr = current_chain[-1]
        cx, cy = curr
        
        # Neighbors
        neighbors = [(cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)]
        random.shuffle(neighbors)
        
        # Heuristic: If current residue is H and part of a core contact with a future residue,
        # we might want to move towards where that future residue *might* be.
        # This is hard to predict. Instead, we just prefer moves that keep the chain tight.
        
        # Calculate centroid
        if len(current_chain) > 1:
            sum_x = sum(p[0] for p in current_chain)
            sum_y = sum(p[1] for p in current_chain)
            cx_avg = sum_x / len(current_chain)
            cy_avg = sum_y / len(current_chain)
        else:
            cx_avg, cy_avg = 0, 0
            
        scored_neighbors = []
        for nx, ny in neighbors:
            if (nx, ny) in current_occupied:
                continue
                
            # Distance to centroid
            dist = math.sqrt((nx - cx_avg)**2 + (ny - cy_avg)**2)
            
            # Bonus if this move creates a contact with an existing H
            bonus = 0
            if sequence[idx] == 'H':
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    q = (nx+dx, ny+dy)
                    if q in current_occupied:
                        j = current_occupied[q]
                        if j < idx - 1 and sequence[j] == 'H':
                            bonus += 10 # Strong bonus for immediate contact
                
            # Penalty for moving away from centroid
            score = -dist + bonus + rng.random() * 2.0
            scored_neighbors.append((score, (nx, ny)))
            
        if not scored_neighbors:
            return False
            
        # Sort by score, try best first
        scored_neighbors.sort(key=lambda x: x[0], reverse=True)
        
        for score, (nx, ny) in scored_neighbors:
            current_chain.append((nx, ny))
            current_occupied[(nx, ny)] = idx
            
            # Recurse
            # To prevent deep recursion stack overflow or excessive time,
            # we could limit depth, but N=100 is okay for Python if careful.
            # However, backtracking can explode. 
            # We add a check: if we are deep and time is running out, stop.
            if not backtrack(idx + 1, current_chain, current_occupied):
                return False # Time limit reached
            
            # Backtrack
            current_chain.pop()
            del current_occupied[(nx, ny)]
            
            # If we found a solution, we might want to keep searching for a better one?
            # For now, we return the first "good" one found by the heuristic ordering.
            # To allow multiple solutions, we could continue, but time is tight.
            # Let's just return the best one found so far in this branch.
            if best_chain is not None:
                # We found a valid chain. Let's see if we should keep searching.
                # If we have time, we can try other branches.
                if time.time() < time_limit - 0.5:
                    continue
                else:
                    return True # Time up, return best found
    
    # Run backtracking
    time_limit = time.time() + time_limit
    backtrack(1, chain, occupied)
    
    if best_chain is not None:
        return best_chain
    return None


def _local_search(sequence, chain, time_limit, rng):
    """
    Simulated Annealing local search.
    """
    current_chain = list(chain)
    current_energy = _energy(sequence, current_chain)
    best_chain = list(current_chain)
    best_energy = current_energy
    
    n = len(current_chain)
    
    T = 5.0
    T_min = 0.01
    
    if n > 80:
        cooling = 0.9999
    elif n > 50:
        cooling = 0.9997
    else:
        cooling = 0.9995
        
    max_iters = 10000
    
    start_time = time.time()
    
    for iter in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        if _time_left() < 15:
            break
            
        r = rng.random()
        new_chain = None
        
        if r < 0.8:
            pivot_idx = rng.randint(1, n - 2)
            angle = rng.choice([90, -90, 180])
            new_chain = _apply_pivot(current_chain, pivot_idx, angle)
        else:
            new_chain = _apply_slither(current_chain, rng)
            
        if new_chain is None:
            continue
            
        new_energy = _energy(sequence, new_chain)
        delta = new_energy - current_energy
        
        if delta < 0:
            current_chain = new_chain
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_chain = list(current_chain)
        else:
            if T > T_min:
                exp_val = math.exp(-delta / T)
                if rng.random() < exp_val:
                    current_chain = new_chain
                    current_energy = new_energy
        
        T *= cooling
        
    return best_chain


def fold(sequence, dimension):
    """
    Main entry point.
    """
    seq_len = len(sequence)
    remaining = _time_left()
    
    if remaining < 5:
        return [[i, 0] for i in range(seq_len)]
        
    # Time allocation
    # Backtracking is expensive. Limit it.
    # Local search is fast.
    
    if remaining < 20:
        time_for_bt = 2.0
        time_for_ls = 5.0
    else:
        time_for_bt = 5.0
        time_for_ls = 10.0
        
    rng = random.Random(12345 + seq_len)
    
    best_chain = None
    best_energy = 0
    
    # Phase 1: Guided Backtracking
    # Select core contacts
    core_contacts = _select_core_contacts(sequence, max_contacts=8)
    
    bt_start = time.time()
    bt_chain = _backtrack_fold_guided(sequence, core_contacts, rng, time_for_bt)
    
    if bt_chain:
        e = _energy(sequence, bt_chain)
        best_chain = bt_chain
        best_energy = e
        
    # If backtracking failed or didn't find a good chain, fallback to greedy
    if best_chain is None:
        # Simple greedy growth
        chain = [[0,0]]
        occupied = {(0,0): 0}
        for i in range(1, seq_len):
            curr = chain[-1]
            options = []
            for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                site = (curr[0]+dx, curr[1]+dy)
                if site not in occupied:
                    options.append(site)
            if not options:
                break
            site = rng.choice(options)
            chain.append(site)
            occupied[site] = i
        best_chain = chain
        best_energy = _energy(sequence, best_chain)

    # Phase 2: Local Search
    if _time_left() > 15:
        ls_time = min(time_for_ls, _time_left() - 15)
        improved = _local_search(sequence, best_chain, ls_time, rng)
        if improved:
            e_imp = _energy(sequence, improved)
            if e_imp < best_energy:
                best_energy = e_imp
                best_chain = improved

    if best_chain is None:
        best_chain = [[i, 0] for i in range(seq_len)]
        
    return best_chain
# EVOLVE-BLOCK-END
