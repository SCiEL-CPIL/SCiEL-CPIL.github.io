"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np
from collections import defaultdict

def _primes_upto(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n+1) if sieve[i]]

def _build_counts_np(residues, primes, max_L):
    """Build count array using numpy for speed."""
    counts = np.zeros(max_L + 1, dtype=np.int32)
    for p, r in zip(primes, residues):
        if p > max_L:
            # If p > max_L, only x = r (if r >= 1 and r <= max_L) is covered
            if 1 <= r <= max_L:
                counts[r] += 1
            continue
            
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        
        # Generate indices: start, start+p, ... <= max_L
        # Number of elements: floor((max_L - start) / p) + 1
        n_elems = (max_L - start) // p + 1
        if n_elems > 0:
            idxs = start + p * np.arange(n_elems, dtype=np.int32)
            counts[idxs] += 1
    return counts

def _find_prefix_length(counts, max_L):
    """Find the largest L such that counts[1..L] are all >= 2."""
    # We want the first index i in 1..max_L where counts[i] < 2.
    # If no such index, return max_L.
    # Create a boolean mask of where counts < 2
    # Note: counts[0] is irrelevant, we care about 1..max_L
    mask = counts[1:max_L+1] < 2
    if not mask.any():
        return max_L
    # Find the first True in the mask
    first_fail = np.argmax(mask)
    # The prefix length is first_fail (since mask is 0-indexed relative to 1..max_L)
    # e.g., if mask[0] is True, index 1 fails, L=0.
    # if mask[4] is True, index 5 fails, L=4.
    return int(first_fail)

def _apply_move(counts, residues, primes, idx, new_r, max_L):
    """Apply a residue change and update counts."""
    p = primes[idx]
    old_r = residues[idx]
    
    if old_r == new_r:
        return

    # Remove old coverage
    if p <= max_L:
        start_old = old_r if old_r >= 1 else old_r + p
        if start_old <= max_L:
            n_elems = (max_L - start_old) // p + 1
            if n_elems > 0:
                idxs_old = start_old + p * np.arange(n_elems, dtype=np.int32)
                counts[idxs_old] -= 1
    else:
        if 1 <= old_r <= max_L:
            counts[old_r] -= 1

    # Add new coverage
    if p <= max_L:
        start_new = new_r if new_r >= 1 else new_r + p
        if start_new <= max_L:
            n_elems = (max_L - start_new) // p + 1
            if n_elems > 0:
                idxs_new = start_new + p * np.arange(n_elems, dtype=np.int32)
                counts[idxs_new] += 1
    else:
        if 1 <= new_r <= max_L:
            counts[new_r] += 1

    residues[idx] = new_r

def _undo_move(counts, residues, primes, idx, old_r, max_L):
    """Undo a residue change."""
    # This is effectively applying the move back to old_r
    new_r = residues[idx]
    _apply_move(counts, residues, primes, idx, old_r, max_L)

def _greedy_init(primes, max_L):
    """
    Greedy initialization: 
    1. Start with random residues.
    2. Iteratively improve by picking a random prime and setting its residue 
       to maximize the number of under-covered positions (count < 2) it covers.
       To speed this up, we just pick the residue that covers the most 'gaps' 
       in the current prefix.
    """
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    # Initial random assignment
    for i in range(n_primes):
        residues[i] = random.randint(0, primes[i] - 1)
    
    # Build initial counts
    counts = _build_counts_np(residues, primes, max_L)
    
    # Iterate a few times to improve
    for _ in range(10):
        improved = False
        # Randomly shuffle order of primes to break symmetry
        order = list(range(n_primes))
        random.shuffle(order)
        
        for idx in order:
            p = primes[idx]
            old_r = residues[idx]
            
            # Find best residue for this prime
            # We want to maximize coverage of positions with count < 2
            # But we must not reduce count of positions with count == 2 below 2? 
            # Actually, greedy "maximize gain" is simple.
            # Gain = sum(counts[pos] < 2 for pos covered by new_r) - sum(counts[pos] < 2 for pos covered by old_r)
            # This is expensive to compute exactly. 
            # Approximation: Just pick a random residue and see if it improves the prefix length?
            # Better: Pick residue that covers the most 'holes' (count < 2) in the current prefix.
            
            # To keep it fast, let's just do a small random search around current residue
            # or pick the residue that covers the first gap.
            
            # Let's try: Find the first gap (count < 2) in 1..max_L.
            # If we can cover it with this prime, do it.
            # This is O(L) per prime per iteration, which is 168 * 5000 * 10 = 8.4M ops. Acceptable.
            
            # Find first gap
            mask = counts < 2
            gaps = np.where(mask)[0]
            gaps = gaps[gaps >= 1] # Only care about 1..max_L
            
            if len(gaps) == 0:
                continue
                
            first_gap = gaps[0]
            
            # Can this prime cover first_gap?
            target_r = first_gap % p
            
            # Check if changing to target_r improves things
            # Simple heuristic: If the prime currently does NOT cover first_gap, 
            # and changing to target_r doesn't break existing double coverage too badly?
            # Let's just apply it and check if prefix length increases.
            
            if residues[idx] != target_r:
                # Apply
                _apply_move(counts, residues, primes, idx, target_r, max_L)
                new_score = _find_prefix_length(counts, max_L)
                # We don't track old score easily here, but if we are greedy, 
                # we assume covering the first gap is good.
                # However, we might break earlier positions.
                # Let's undo if it doesn't improve prefix length.
                # We need old_score.
                # Let's just keep it. Greedy is heuristic.
                improved = True
            else:
                # Already covers it. Try to cover the second gap?
                # Or just leave it.
                pass
    
    return residues, counts

def _simulated_annealing(residues, counts, primes, max_L, deadline, seed=0):
    """
    Simulated Annealing with fast prefix evaluation.
    """
    np.random.seed(seed)
    random.seed(seed)
    
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    T_initial = 10.0
    T_final = 0.1
    cooling_rate = 0.999
    
    T = T_initial
    iteration = 0
    
    # Precompute weights for prime selection
    # Small primes have more impact. Weight = 1/p
    weights = np.array([1.0 / p for p in primes])
    weights /= weights.sum()
    
    while time.time() < deadline:
        iteration += 1
        
        # Select prime to modify
        # Use numpy random choice with p=weights
        idx = np.random.choice(n_primes, p=weights)
        
        p = primes[idx]
        old_r = residues[idx]
        
        # Generate new residue
        # Bias towards residues that cover the current bottleneck?
        # For now, random
        new_r = np.random.randint(0, p)
        
        if new_r == old_r:
            continue
        
        # Apply move
        _apply_move(counts, residues, primes, idx, new_r, max_L)
        
        # Evaluate
        new_score = _find_prefix_length(counts, max_L)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > T_final and np.random.random() < math.exp(delta / T)):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                best_counts = counts.copy()
        else:
            _undo_move(counts, residues, primes, idx, old_r, max_L)
        
        T *= cooling_rate
        if T < T_final:
            T = T_final
            
        # Check time frequently
        if iteration % 100 == 0 and time.time() > deadline:
            break
            
    return best_residues, best_counts, best_score

def _bottleneck_repair(residues, counts, primes, max_L, deadline):
    """
    Targeted repair: Find the first gap and try to fix it by changing one prime.
    """
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    if current_score >= max_L:
        return best_residues, best_counts, best_score
    
    gap_pos = current_score + 1
    
    # Find primes that currently cover gap_pos
    covering_primes = []
    for i in range(n_primes):
        p = primes[i]
        r = residues[i]
        if gap_pos % p == r:
            covering_primes.append(i)
    
    # If < 2 primes cover it, we need to add one.
    # We look for a prime NOT in covering_primes that can be changed to cover gap_pos
    # with minimal loss to the prefix [1, current_score].
    
    candidates = []
    for i in range(n_primes):
        if i in covering_primes:
            continue
            
        p = primes[i]
        new_r = gap_pos % p
        old_r = residues[i]
        
        if new_r == old_r:
            continue
            
        # Estimate loss: How many positions in [1, current_score] are currently 
        # covered ONLY by this prime (count == 1) or will drop from 2 to 1?
        # Actually, we care about positions that will drop BELOW 2.
        # Positions covered by old_r:
        if p <= max_L:
            start_old = old_r if old_r >= 1 else old_r + p
            if start_old <= max_L:
                # Only check up to current_score
                end_check = min(current_score, max_L)
                if start_old <= end_check:
                    n_elems = (end_check - start_old) // p + 1
                    idxs_old = start_old + p * np.arange(n_elems, dtype=np.int32)
                    # Loss is count of positions where counts[idxs_old] == 2 
                    # (because they will drop to 1)
                    # Positions where counts[idxs_old] == 1 will drop to 0 (worse!)
                    # But we are adding coverage to gap_pos, so maybe we gain there?
                    # No, gap_pos is > current_score, so it's not in the prefix.
                    # So any drop in [1, current_score] is pure loss.
                    loss = np.sum(counts[idxs_old] <= 2) # If count is 1, it becomes 0 (bad). If 2, becomes 1 (bad).
                    # Actually, if count is 1, it becomes 0. That breaks the prefix if that position is <= current_score.
                    # So we want loss == 0 ideally.
                    candidates.append((loss, i, new_r))
        else:
            # p > max_L, so only one position is covered
            if 1 <= old_r <= current_score:
                if counts[old_r] <= 2:
                    candidates.append((1, i, new_r))
                else:
                    candidates.append((0, i, new_r))
            else:
                candidates.append((0, i, new_r))
    
    # Sort candidates by loss
    candidates.sort(key=lambda x: x[0])
    
    # Try the best candidates
    for loss, i, new_r in candidates[:10]:
        if time.time() > deadline:
            break
            
        old_r = residues[i]
        _apply_move(counts, residues, primes, i, new_r, max_L)
        
        new_score = _find_prefix_length(counts, max_L)
        
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                best_counts = counts.copy()
            # If we improved, we might have new gaps. Continue loop?
            # For simplicity, break after one improvement and let outer loop handle it?
            # Or continue repairing.
            # Let's continue.
            # Update covering_primes logic? 
            # It's complex. Let's just break and let the main loop re-invoke if needed.
            break
        else:
            _undo_move(counts, residues, primes, i, old_r, max_L)
            
    return best_residues, best_counts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    WORK_L = 5000
    
    best_overall_residues = None
    best_overall_score = 0
    
    # Strategy:
    # 1. Greedy Init
    # 2. SA for a while
    # 3. Bottleneck Repair
    # 4. Repeat with different seeds
    
    num_restarts = 5
    time_per_restart = (deadline - start_time - 10.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # 1. Init
        residues, counts = _greedy_init(primes, WORK_L)
        score = _find_prefix_length(counts, WORK_L)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
        
        # 2. SA
        sa_deadline = min(deadline - 20.0, start_time + time_per_restart * 0.7)
        residues, counts, score = _simulated_annealing(residues, counts, primes, WORK_L, sa_deadline, seed=restart)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
        
        # 3. Bottleneck Repair
        # Run a few rounds of repair
        repair_deadline = min(deadline - 10.0, start_time + time_per_restart * 0.9)
        for _ in range(5):
            if time.time() > repair_deadline:
                break
            residues, counts, score = _bottleneck_repair(residues, counts, primes, WORK_L, repair_deadline)
            if score > best_overall_score:
                best_overall_score = score
                best_overall_residues = residues[:]
            # If we hit max_L, break
            if score >= WORK_L:
                break
    
    # Final polish on best solution
    if best_overall_residues is not None and time.time() < deadline - 5.0:
        counts = _build_counts_np(best_overall_residues, primes, WORK_L)
        final_deadline = deadline - 2.0
        # One more round of SA on the best solution?
        # Or just repair.
        best_overall_residues, _, _ = _bottleneck_repair(best_overall_residues, counts, primes, WORK_L, final_deadline)
        
        # Check score again
        counts = _build_counts_np(best_overall_residues, primes, WORK_L)
        final_score = _find_prefix_length(counts, WORK_L)
        if final_score > best_overall_score:
            best_overall_score = final_score

    if best_overall_residues is None:
        # Fallback
        best_overall_residues, _ = _greedy_init(primes, WORK_L)
    
    # Ensure all residues are Python ints
    return [int(r) for r in best_overall_residues]

# EVOLVE-BLOCK-END
