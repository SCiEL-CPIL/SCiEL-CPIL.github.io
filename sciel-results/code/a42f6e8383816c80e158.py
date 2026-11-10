"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

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
        start = r if r >= 1 else r + p
        if start > max_L:
            continue
        idxs = np.arange(start, max_L + 1, p)
        counts[idxs] += 1
    return counts

def _find_prefix_length(counts, max_L):
    """Find prefix length from counts array."""
    L = 0
    for x in range(1, max_L + 1):
        if counts[x] >= 2:
            L = x
        else:
            break
    return L

def _apply_move(counts, residues, primes, idx, new_r, max_L):
    """Apply a residue change and update counts. Returns new_score delta info."""
    p = primes[idx]
    old_r = residues[idx]
    
    # Remove old coverage
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] -= 1
    
    # Add new coverage
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] += 1
    
    residues[idx] = new_r

def _undo_move(counts, residues, primes, idx, old_r, max_L):
    """Undo a residue change."""
    p = primes[idx]
    new_r = residues[idx]
    
    # Remove new coverage
    start_new = new_r if new_r >= 1 else new_r + p
    if start_new <= max_L:
        idxs_new = np.arange(start_new, max_L + 1, p)
        counts[idxs_new] -= 1
    
    # Add back old coverage
    start_old = old_r if old_r >= 1 else old_r + p
    if start_old <= max_L:
        idxs_old = np.arange(start_old, max_L + 1, p)
        counts[idxs_old] += 1
    
    residues[idx] = old_r

def _greedy_init(primes, max_L):
    """Greedy initialization: choose residue for each prime to maximize coverage of under-covered positions."""
    n_primes = len(primes)
    residues = [0] * n_primes
    counts = np.zeros(max_L + 1, dtype=np.int32)
    
    for idx in range(n_primes):
        p = primes[idx]
        best_r = 0
        best_metric = -1
        
        if p <= max_L:
            for r in range(p):
                start_x = r if r >= 1 else r + p
                if start_x > max_L:
                    gain = 0
                else:
                    idxs = np.arange(start_x, max_L + 1, p)
                    gain = np.sum(counts[idxs] < 2)
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
        else:
            best_r = 0
            best_metric = -1
            for r in range(1, min(p, max_L + 1)):
                gain = 1 if counts[r] < 2 else 0
                if gain > best_metric:
                    best_metric = gain
                    best_r = r
            if best_metric <= 0:
                best_r = 0
        
        residues[idx] = best_r
        start_x = best_r if best_r >= 1 else best_r + p
        if start_x <= max_L:
            idxs = np.arange(start_x, max_L + 1, p)
            counts[idxs] += 1
    
    return residues, counts

def _simulated_annealing(residues, counts, primes, max_L, deadline, iterations_budget):
    """Run simulated annealing to improve the solution."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    T_initial = 5.0
    T_final = 0.05
    cooling_rate = 0.9995
    T = T_initial
    
    iteration = 0
    while time.time() < deadline - 15.0 and iteration < iterations_budget:
        iteration += 1
        
        # Choose a prime to modify - bias toward small primes (more impact)
        if random.random() < 0.7:
            # Weighted toward small primes
            weights = [1.0 / primes[i] for i in range(n_primes)]
            total_w = sum(weights)
            r = random.random() * total_w
            cum = 0
            idx = 0
            for i in range(n_primes):
                cum += weights[i]
                if r <= cum:
                    idx = i
                    break
        else:
            idx = random.randint(0, n_primes - 1)
        
        p = primes[idx]
        old_r = residues[idx]
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
        
        # Apply move
        _apply_move(counts, residues, primes, idx, new_r, max_L)
        new_score = _find_prefix_length(counts, max_L)
        
        # Acceptance
        if new_score > current_score:
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_residues = residues[:]
                best_counts = counts.copy()
        else:
            delta = new_score - current_score
            if delta >= 0 or (T > T_final and random.random() < math.exp(delta / T)):
                current_score = new_score
            else:
                _undo_move(counts, residues, primes, idx, old_r, max_L)
        
        T *= cooling_rate
        if T < T_final:
            T = T_final
    
    return best_residues, best_counts, best_score

def _local_search(residues, counts, primes, max_L, deadline):
    """Targeted local search to fix gaps."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        current_score = _find_prefix_length(counts, max_L)
        
        if current_score >= max_L:
            break
        
        gap_pos = current_score + 1
        
        # Find primes covering gap_pos
        covering = []
        for i in range(n_primes):
            p = primes[i]
            r = residues[i]
            if gap_pos % p == r % p:
                covering.append(i)
        
        if len(covering) < 2:
            # Try to change a non-covering prime to cover gap_pos
            candidates = []
            for i in range(n_primes):
                if i in covering:
                    continue
                p = primes[i]
                old_r = residues[i]
                new_r = gap_pos % p
                if new_r == old_r:
                    continue
                # Check loss
                start_old = old_r if old_r >= 1 else old_r + p
                loss = 0
                if start_old <= max_L:
                    idxs = np.arange(start_old, min(current_score, max_L) + 1, p)
                    loss = np.sum(counts[idxs] == 2)
                if loss == 0:
                    candidates.append((i, new_r))
            
            if candidates:
                # Pick the first candidate
                i, new_r = candidates[0]
                _apply_move(counts, residues, primes, i, new_r, max_L)
                improved = True
        
        if improved:
            new_score = _find_prefix_length(counts, max_L)
            if new_score > best_score:
                best_score = new_score
                best_residues = residues[:]
                best_counts = counts.copy()
    
    return best_residues, best_counts, best_score

def _pairwise_local_search(residues, counts, primes, max_L, deadline):
    """Try pairwise swaps to improve coverage."""
    n_primes = len(primes)
    current_score = _find_prefix_length(counts, max_L)
    best_score = current_score
    best_residues = residues[:]
    best_counts = counts.copy()
    
    while time.time() < deadline - 10.0:
        current_score = _find_prefix_length(counts, max_L)
        if current_score >= max_L:
            break
        
        # Try a few random pairwise changes
        found_improvement = False
        for _ in range(50):
            if time.time() > deadline - 10.0:
                break
            i = random.randint(0, n_primes - 1)
            j = random.randint(0, n_primes - 1)
            if i == j:
                continue
            
            p_i = primes[i]
            p_j = primes[j]
            old_ri = residues[i]
            old_rj = residues[j]
            
            # Try swapping residues
            new_ri = old_rj % p_i
            new_rj = old_ri % p_j
            
            if new_ri == old_ri and new_rj == old_rj:
                continue
            
            # Apply both moves
            _apply_move(counts, residues, primes, i, new_ri, max_L)
            _apply_move(counts, residues, primes, j, new_rj, max_L)
            
            new_score = _find_prefix_length(counts, max_L)
            
            if new_score > current_score:
                current_score = new_score
                found_improvement = True
                if current_score > best_score:
                    best_score = current_score
                    best_residues = residues[:]
                    best_counts = counts.copy()
            else:
                # Undo
                _undo_move(counts, residues, primes, j, new_rj, max_L)
                _undo_move(counts, residues, primes, i, new_ri, max_L)
        
        if not found_improvement:
            break
    
    return best_residues, best_counts, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s margin
    
    primes = _primes_upto(1000)
    n_primes = len(primes)
    
    WORK_L = 2000
    
    best_overall_residues = None
    best_overall_score = 0
    
    # Multiple restarts
    num_restarts = 5
    time_per_restart = (deadline - start_time - 20.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Greedy initialization
        residues, counts = _greedy_init(primes, WORK_L)
        score = _find_prefix_length(counts, WORK_L)
        
        # Simulated annealing
        sa_deadline = min(deadline - 25.0, start_time + time_per_restart * 0.6)
        iterations = int(200000 * (time_per_restart * 0.6) / 10.0)
        residues, counts, score = _simulated_annealing(residues, counts, primes, WORK_L, sa_deadline, iterations)
        
        # Local search
        ls_deadline = min(deadline - 20.0, start_time + time_per_restart * 0.8)
        residues, counts, score = _local_search(residues, counts, primes, WORK_L, ls_deadline)
        
        # Pairwise local search
        pl_deadline = min(deadline - 15.0, start_time + time_per_restart * 0.95)
        residues, counts, score = _pairwise_local_search(residues, counts, primes, WORK_L, pl_deadline)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_residues = residues[:]
    
    # Final local search on best solution
    if best_overall_residues is not None and time.time() < deadline - 10.0:
        counts = _build_counts_np(best_overall_residues, primes, WORK_L)
        final_deadline = deadline - 5.0
        best_overall_residues, _, _ = _local_search(best_overall_residues, counts, primes, WORK_L, final_deadline)
    
    if best_overall_residues is None:
        # Fallback
        best_overall_residues, _ = _greedy_init(primes, WORK_L)
    
    # Ensure all residues are Python ints
    return [int(r) for r in best_overall_residues]
# EVOLVE-BLOCK-END
