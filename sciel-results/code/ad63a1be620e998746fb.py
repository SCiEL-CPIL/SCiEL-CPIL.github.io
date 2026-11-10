"""HP lattice protein folding on the square lattice."""
# EVOLVE-BLOCK-START
import random
import time
import math

# Explicit deadline: evaluator kills after 210s wall-clock; keep 30s safety margin.
_DEADLINE = time.time() + 175.0

def _neighbours(point):
    """Yield the 4 unit-step neighbours of a 2D point (tuple)."""
    x, y = point
    yield (x + 1, y)
    yield (x - 1, y)
    yield (x, y + 1)
    yield (x, y - 1)

def _compute_energy(sequence, chain):
    """Compute the negative number of non-adjacent H-H contacts."""
    if not chain:
        return 0
    where = {p: i for i, p in enumerate(chain)}
    contacts = 0
    for i, p in enumerate(chain):
        if sequence[i] == 'H':
            for q in _neighbours(p):
                j = where.get(q)
                if j is not None and abs(j - i) > 1 and sequence[j] == 'H':
                    contacts += 1
    return -contacts // 2

def _greedy_grow(sequence, dimension, rng, temperature=0.0):
    """
    Randomized greedy chain growth.
    Prefer sites that create new H-H contacts, then sites with more free space.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)]
    
    chain = [(0, 0)]
    occupied = {(0, 0): 0}
    current_dir = 0 # 0: Up, 1: Right, 2: Down, 3: Left
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    # Pre-compute H positions for quick lookup? No, just check sequence[i]
    
    for i in range(1, n):
        last = chain[-1]
        options = []
        
        for d in range(4):
            # Avoid going back (180 degree turn) to maintain some structure? 
            # Actually, for HP, 180 turns are allowed but usually suboptimal for packing.
            # However, standard SAWs allow all 4. Let's allow all 4 to be safe, 
            # but bias against 180 if possible? 
            # Actually, the prompt says "unit steps". 180 is a unit step back.
            # But self-avoidance prevents going to the previous position.
            
            nx = last[0] + dx[d]
            ny = last[1] + dy[d]
            pos = (nx, ny)
            
            if pos in occupied:
                continue
            
            # Score this move
            score = 0
            
            # 1. Contact score: Does this H residue touch existing H residues?
            if sequence[i] == 'H':
                for q in _neighbours(pos):
                    if q in occupied:
                        j = occupied[q]
                        if abs(j - i) > 1 and sequence[j] == 'H':
                            score += 10 # Strong preference for contacts
            
            # 2. Space score: How many free neighbors does this new position have?
            # More free neighbors = less likely to trap the chain
            free_count = 0
            for q in _neighbours(pos):
                if q not in occupied:
                    free_count += 1
            score += free_count * 0.1
            
            # 3. Direction bias: Prefer straight or slight turns over sharp U-turns?
            # If d is opposite to current_dir, it's a U-turn (backwards)
            if (d - current_dir) % 4 == 2:
                score -= 1 # Small penalty for U-turns
            
            options.append((score, d))
        
        if not options:
            # Dead end, restart or fail
            return None
        
        # Select move based on temperature
        if temperature > 0:
            # Boltzmann distribution
            max_score = max(s for s, _ in options)
            weights = [math.exp((s - max_score) / temperature) for s, _ in options]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_d = options[-1][1]
            for (s, d), w in zip(options, weights):
                cum += w
                if r <= cum:
                    chosen_d = d
                    break
        else:
            # Greedy: pick best score
            best_score, chosen_d = max(options, key=lambda x: x[0])
        
        nx = last[0] + dx[chosen_d]
        ny = last[1] + dy[chosen_d]
        pos = (nx, ny)
        
        chain.append(pos)
        occupied[pos] = i
        current_dir = chosen_d
        
    return chain

def _regrow_suffix(sequence, chain, cut_index, rng, temperature=0.0, max_attempts=10):
    """
    Remove the suffix of the chain starting from cut_index and regrow it.
    Returns new chain if successful, else None.
    """
    n = len(sequence)
    if cut_index >= n:
        return chain
    
    # Keep prefix up to cut_index
    prefix = chain[:cut_index]
    if not prefix:
        return None
        
    last = prefix[-1]
    
    # Rebuild occupied set from prefix
    occupied = {p: i for i, p in enumerate(prefix)}
    
    # Determine direction of last step in prefix
    if len(prefix) > 1:
        prev = prefix[-2]
        dx_val = last[0] - prev[0]
        dy_val = last[1] - prev[1]
        if dx_val == 0 and dy_val == 1:
            current_dir = 0
        elif dx_val == 1 and dy_val == 0:
            current_dir = 1
        elif dx_val == 0 and dy_val == -1:
            current_dir = 2
        elif dx_val == -1 and dy_val == 0:
            current_dir = 3
        else:
            current_dir = 0 # Should not happen
    else:
        current_dir = rng.randint(0, 3)
        
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    for i in range(cut_index, n):
        last = prefix[-1]
        options = []
        
        for d in range(4):
            nx = last[0] + dx[d]
            ny = last[1] + dy[d]
            pos = (nx, ny)
            
            if pos in occupied:
                continue
            
            score = 0
            if sequence[i] == 'H':
                for q in _neighbours(pos):
                    if q in occupied:
                        j = occupied[q]
                        if abs(j - i) > 1 and sequence[j] == 'H':
                            score += 10
            
            free_count = 0
            for q in _neighbours(pos):
                if q not in occupied:
                    free_count += 1
            score += free_count * 0.1
            
            if (d - current_dir) % 4 == 2:
                score -= 1
            
            options.append((score, d))
        
        if not options:
            return None
        
        if temperature > 0:
            max_score = max(s for s, _ in options)
            weights = [math.exp((s - max_score) / temperature) for s, _ in options]
            total_w = sum(weights)
            r = rng.random() * total_w
            cum = 0
            chosen_d = options[-1][1]
            for (s, d), w in zip(options, weights):
                cum += w
                if r <= cum:
                    chosen_d = d
                    break
        else:
            best_score, chosen_d = max(options, key=lambda x: x[0])
        
        nx = last[0] + dx[chosen_d]
        ny = last[1] + dy[chosen_d]
        pos = (nx, ny)
        
        prefix.append(pos)
        occupied[pos] = i
        current_dir = chosen_d
        
    return prefix

def _sa_loop(sequence, rng):
    """
    Simulated Annealing loop with regrowth moves.
    """
    n = len(sequence)
    if n < 2:
        return [(0, 0)]
    
    # Initialize with greedy growth
    chain = _greedy_grow(sequence, 2, rng, temperature=0.0)
    if chain is None:
        # Fallback to straight line if greedy fails (unlikely)
        chain = [(i, 0) for i in range(n)]
        
    current_energy = _compute_energy(sequence, chain)
    best_chain = list(chain)
    best_energy = current_energy
    
    # SA Parameters
    start_temp = 5.0
    end_temp = 0.01
    cooling_rate = 0.995
    
    temp = start_temp
    
    while temp > end_temp:
        if time.time() > _DEADLINE - 10:
            break
            
        # Choose move type
        # 1. Pivot (small local change)
        # 2. Regrowth (large local change)
        
        move_type = rng.random()
        
        if move_type < 0.5:
            # Pivot move: pick a random internal index, rotate the suffix
            # This is complex to implement correctly for SAWs. 
            # Instead, let's stick to Regrowth which is robust.
            # Or simple "kick": move one residue if possible.
            pass # We will just use regrowth for simplicity and robustness
            
        # Regrowth move
        # Choose cut index. Bias towards later parts? Or random?
        # Random cut index from 2 to n-2 (leave room for regrowth)
        if n > 4:
            cut_index = rng.randint(2, n - 2)
        else:
            cut_index = 1
            
        new_chain = _regrow_suffix(sequence, chain, cut_index, rng, temperature=temp)
        
        if new_chain is not None:
            new_energy = _compute_energy(sequence, new_chain)
            delta_e = new_energy - current_energy
            
            # Acceptance criterion
            if delta_e < 0 or rng.random() < math.exp(-delta_e / temp):
                chain = new_chain
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_chain = list(chain)
        
        temp *= cooling_rate
        
    return best_chain

def fold(sequence, dimension):
    """
    Fold the sequence using Simulated Annealing with regrowth moves.
    """
    rng = random.Random(42)
    
    n = len(sequence)
    if n < 2:
        return [[0, 0]]
    
    # Run SA
    chain = _sa_loop(sequence, rng)
    
    # Fallback
    if chain is None:
        chain = [(i, 0) for i in range(n)]
        
    return [list(p) for p in chain]
# EVOLVE-BLOCK-END
