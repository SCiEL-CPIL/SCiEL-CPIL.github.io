"""HP lattice protein folding on the square lattice: multi-strategy search with segment moves."""
# EVOLVE-BLOCK-START
import time
import random
import heapq
from collections import deque

_T0 = time.time()
_DEADLINE = 195.0  # 210s wall-clock limit, 15s safety margin


def _remaining():
    return _DEADLINE - (time.time() - _T0)


def _energy(seq, chain):
    """Compute energy: -number of H-H non-bonded contacts."""
    n = len(chain)
    if n < 3:
        return 0
    pos_to_idx = {p: i for i, p in enumerate(chain)}
    h_positions = [i for i, c in enumerate(seq) if c == 'H']
    contacts = 0
    for i in h_positions:
        x, y = chain[i]
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((x + dx, y + dy))
            if j is not None and j > i + 1 and seq[j] == 'H':
                contacts += 1
    return -contacts


def _greedy_grow(seq, dim, rng, n_iter=50):
    """Randomized greedy chain growth."""
    best_chain, best_energy = None, 0
    for _ in range(n_iter):
        start = (0, 0)
        chain = [start]
        occupied = {start: 0}
        success = True
        for i in range(1, len(seq)):
            neighbors = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                site = (chain[-1][0] + dx, chain[-1][1] + dy)
                if site not in occupied:
                    gain = 0
                    if seq[i] == 'H':
                        for qdx, qdy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            q = (site[0] + qdx, site[1] + qdy)
                            if q in occupied and occupied[q] < i - 1 and seq[occupied[q]] == 'H':
                                gain += 1
                    free = sum(1 for qdx, qdy in ((1, 0), (-1, 0), (0, 1), (0, -1))
                               if (site[0] + qdx, site[1] + qdy) not in occupied)
                    score = gain + 0.1 * free + rng.random() * 0.5
                    neighbors.append((score, site))
            if not neighbors:
                success = False
                break
            neighbors.sort(key=lambda x: -x[0])
            # Pick from top candidates with some randomness
            k = min(3, len(neighbors))
            chosen = rng.choice(neighbors[:k])[1]
            occupied[chosen] = i
            chain.append(chosen)
        if success and len(chain) == len(seq):
            e = _energy(seq, chain)
            if best_chain is None or e < best_energy:
                best_chain, best_energy = chain, e
    return best_chain


def _segment_move(seq, chain, rng, n_trials=20, max_seg_len=15):
    """Try to relocate a segment of the chain to a new location."""
    n = len(chain)
    if n < 4:
        return None
    
    best_chain, best_energy = None, _energy(seq, chain)
    
    for _ in range(n_trials):
        # Pick a random segment
        seg_len = rng.randint(2, min(max_seg_len, n - 2))
        start_idx = rng.randint(1, n - seg_len - 1)
        
        # Extract segment
        segment = chain[start_idx:start_idx + seg_len + 1]
        
        # Remove segment from chain
        new_chain = chain[:start_idx] + chain[start_idx + seg_len + 1:]
        occupied = {p: i for i, p in enumerate(new_chain)}
        
        # Try to reinsert segment at various positions
        for _ in range(5):
            # Try different starting positions for the segment
            trial_chain = list(new_chain[:start_idx])
            trial_occupied = dict(occupied)
            trial_occupied.update({p: i for i, p in enumerate(new_chain[start_idx:])})
            
            # Random walk to place the segment
            success = True
            current_pos = None
            
            # Find a valid starting position adjacent to the predecessor
            if start_idx > 0:
                pred_pos = new_chain[start_idx - 1]
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    pos = (pred_pos[0] + dx, pred_pos[1] + dy)
                    if pos not in trial_occupied:
                        current_pos = pos
                        break
                if current_pos is None:
                    success = False
            
            if not success:
                continue
            
            trial_chain.append(current_pos)
            trial_occupied[current_pos] = start_idx
            
            # Continue the segment
            for j in range(1, seg_len + 1):
                placed = False
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -4)):
                    pos = (current_pos[0] + dx, current_pos[1] + dy)
                    if pos not in trial_occupied:
                        # Check if this position is valid (not colliding)
                        trial_chain.append(pos)
                        trial_occupied[pos] = start_idx + j
                        current_pos = pos
                        placed = True
                        break
                if not placed:
                    success = False
                    break
            
            if not success:
                continue
            
            # Check if the segment end connects to the successor
            if start_idx + seg_len + 1 < n:
                succ_pos = new_chain[start_idx + seg_len + 1]
                last_pos = trial_chain[-1]
                if abs(last_pos[0] - succ_pos[0]) + abs(last_pos[1] - succ_pos[1]) != 1:
                    continue
            
            # Verify self-avoidance
            if len(set(trial_chain)) != len(trial_chain):
                continue
            
            # Compute energy
            e = _energy(seq, trial_chain)
            if best_chain is None or e < best_energy:
                best_chain, best_energy = trial_chain, e
    
    return best_chain


def _local_optimize(seq, chain, max_iters=100):
    """Local optimization via single-residue moves."""
    n = len(chain)
    current_energy = _energy(seq, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    for _ in range(max_iters):
        if _remaining() < 5:
            break
        
        improved = False
        for i in range(1, n - 1):
            # Try moving residue i to a neighboring position
            x, y = chain[i]
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                new_pos = (x + dx, y + dy)
                
                # Check if new position is valid
                if new_pos in chain and chain.index(new_pos) != i:
                    continue
                if new_pos == chain[i - 1] or new_pos == chain[i + 1]:
                    continue
                
                # Create trial chain
                trial_chain = list(chain)
                trial_chain[i] = new_pos
                
                # Check self-avoidance
                if len(set(trial_chain)) != len(trial_chain):
                    continue
                
                # Compute energy
                e = _energy(seq, trial_chain)
                if e < best_energy:
                    best_chain = trial_chain
                    best_energy = e
                    chain = trial_chain
                    improved = True
        
        if not improved:
            break
    
    return best_chain


def fold(sequence, dimension):
    """Best conformation using multiple strategies."""
    if dimension != 2:
        # Fallback for non-2D (shouldn't happen)
        return [[i] + [0] * (dimension - 1) for i in range(len(sequence))]
    
    n = len(sequence)
    if n <= 2:
        return [[0, 0], [1, 0]][:n]
    
    rng = random.Random(42 + len(sequence))
    
    best_chain = None
    best_energy = 0
    
    # Strategy 1: Greedy growth
    chain = _greedy_grow(sequence, 2, rng, n_iter=30)
    if chain is not None:
        e = _energy(sequence, chain)
        best_chain, best_energy = chain, e
    
    # Strategy 2: Local optimization
    if best_chain is not None and _remaining() > 10:
        optimized = _local_optimize(sequence, best_chain, max_iters=50)
        e = _energy(sequence, optimized)
        if e < best_energy:
            best_chain, best_energy = optimized, e
    
    # Strategy 3: Segment moves
    if best_chain is not None and _remaining() > 5:
        moved = _segment_move(sequence, best_chain, rng, n_trials=15, max_seg_len=12)
        if moved is not None:
            e = _energy(sequence, moved)
            if e < best_energy:
                best_chain, best_energy = moved, e
    
    # Strategy 4: More greedy growth with different seeds
    if _remaining() > 10:
        for seed in [100, 200, 300]:
            if _remaining() < 5:
                break
            rng2 = random.Random(seed + len(sequence))
            chain2 = _greedy_grow(sequence, 2, rng2, n_iter=10)
            if chain2 is not None:
                e = _energy(sequence, chain2)
                if e < best_energy:
                    best_chain, best_energy = chain2, e
                    # Quick local optimization
                    if _remaining() > 5:
                        opt = _local_optimize(sequence, best_chain, max_iters=20)
                        e_opt = _energy(sequence, opt)
                        if e_opt < best_energy:
                            best_chain, best_energy = opt, e_opt
    
    # Final local optimization
    if best_chain is not None and _remaining() > 3:
        final = _local_optimize(sequence, best_chain, max_iters=30)
        e = _energy(sequence, final)
        if e < best_energy:
            best_chain, best_energy = final, e
    
    if best_chain is None:
        # Fallback: straight line
        best_chain = [[i, 0] for i in range(n)]
    
    return best_chain
# EVOLVE-BLOCK-END
