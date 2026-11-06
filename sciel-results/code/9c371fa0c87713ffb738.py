"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _score_set(s):
    """Count distinct d>0 such that a, a+d, a+2d all in s."""
    s_set = set(s)
    diffs = set()
    for a in s:
        for d in range(1, 256):
            if (a + d) in s_set and (a + 2*d) in s_set:
                diffs.add(d)
    return len(diffs)

def _score_bitmask(mask):
    """Count distinct d>0 using bitmask representation. mask has bits for positions 0..511."""
    diffs = 0
    # For each d, check if there exists a such that bits a, a+d, a+2d are set
    # a ranges from 0 to 511-2d
    for d in range(1, 256):
        # We need mask & (mask >> d) & (mask >> 2*d) to be non-zero
        # But we need to be careful about bit positions
        # Let's use a different approach: for each d, check all a
        # Actually, the bitmask approach: (mask & (mask >> d) & (mask >> 2*d)) != 0
        if mask & (mask >> d) & (mask >> 2*d):
            diffs += 1
    return diffs

def _random_neighbor(mask, rng):
    """Generate a random neighbor by swapping one bit in with one bit out."""
    # Find set bits
    set_bits = []
    unset_bits = []
    for i in range(512):
        if mask & (1 << i):
            set_bits.append(i)
        else:
            unset_bits.append(i)
    
    if not set_bits or not unset_bits:
        return mask
    
    i = rng.choice(set_bits)
    j = rng.choice(unset_bits)
    new_mask = mask ^ (1 << i) ^ (1 << j)
    return new_mask

def _simulated_annealing(mask, rng, time_limit):
    """Run simulated annealing to improve the set."""
    start_time = time.time()
    current_score = _score_bitmask(mask)
    best_mask = mask
    best_score = current_score
    
    T = 50.0
    T_min = 0.1
    alpha = 0.999
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        new_mask = _random_neighbor(mask, rng)
        new_score = _score_bitmask(new_mask)
        
        delta = new_score - current_score
        if delta > 0 or rng.random() < (2.718281828 ** (delta / T)):
            mask = new_mask
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
        
        T *= alpha
    
    return best_mask, best_score

def _hill_climbing(mask, rng, time_limit):
    """Run hill climbing with restarts."""
    start_time = time.time()
    current_score = _score_bitmask(mask)
    best_mask = mask
    best_score = current_score
    no_improve = 0
    
    while time.time() - start_time < time_limit:
        new_mask = _random_neighbor(mask, rng)
        new_score = _score_bitmask(new_mask)
        
        if new_score > current_score:
            mask = new_mask
            current_score = new_score
            no_improve = 0
            if current_score > best_score:
                best_score = current_score
                best_mask = mask
        else:
            no_improve += 1
            if no_improve > 100:
                # Restart from a random point
                mask = 0
                positions = rng.sample(range(512), 32)
                for p in positions:
                    mask |= (1 << p)
                current_score = _score_bitmask(mask)
                no_improve = 0
    
    return best_mask, best_score

def _greedy_build(rng):
    """Build a set greedily, prioritizing positions that create many progressions."""
    mask = 0
    positions = list(range(512))
    rng.shuffle(positions)
    
    # Start with a random seed
    for _ in range(5):
        p = positions.pop()
        mask |= (1 << p)
    
    # Greedily add positions
    for p in positions:
        if time.time() > 145:  # Safety margin
            break
        candidate = mask | (1 << p)
        # Only add if it improves score
        if _score_bitmask(candidate) > _score_bitmask(mask):
            mask = candidate
            if _score_bitmask(mask) >= 20:
                break
    
    return mask

def _complete_progression_moves(mask, rng, time_limit):
    """Use heuristic moves that complete 3-term progressions."""
    start_time = time.time()
    current_score = _score_bitmask(mask)
    best_mask = mask
    best_score = current_score
    
    # Find all pairs (a, b) in the set, and for each pair, the third element c = 2b - a
    # or c = 2a - b that would complete a progression
    set_bits = [i for i in range(512) if mask & (1 << i)]
    
    no_improve = 0
    while time.time() - start_time < time_limit and no_improve < 200:
        # Generate candidate moves based on completing progressions
        candidates = set()
        
        # For each pair in the set, try to add the third element
        for a, b in combinations(set_bits, 2):
            c1 = 2 * b - a
            c2 = 2 * a - b
            if 0 <= c1 <= 511 and not (mask & (1 << c1)):
                candidates.add(c1)
            if 0 <= c2 <= 511 and not (mask & (1 << c2)):
                candidates.add(c2)
        
        if not candidates:
            # Fall back to random move
            new_mask = _random_neighbor(mask, rng)
            new_score = _score_bitmask(new_mask)
            if new_score > current_score:
                mask = new_mask
                current_score = new_score
                set_bits = [i for i in range(512) if mask & (1 << i)]
                no_improve = 0
                if current_score > best_score:
                    best_score = current_score
                    best_mask = mask
            else:
                no_improve += 1
            continue
        
        # Try each candidate
        improved = False
        for c in candidates:
            if time.time() - start_time > time_limit:
                break
            # We need to remove one existing element and add c
            # Try removing each element and see if score improves
            for a in set_bits:
                new_mask = mask ^ (1 << a) | (1 << c)
                new_score = _score_bitmask(new_mask)
                if new_score > current_score:
                    mask = new_mask
                    current_score = new_score
                    set_bits = [i for i in range(512) if mask & (1 << i)]
                    no_improve = 0
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_mask = mask
                    break
            if improved:
                break
        
        if not improved:
            no_improve += 1
    
    return best_mask, best_score

def construct():
    rng = random.Random(42)
    start_time = time.time()
    
    # Initial construction: try several strategies
    best_mask = 0
    best_score = 0
    
    # Strategy 1: Greedy build
    mask = _greedy_build(rng)
    score = _score_bitmask(mask)
    if score > best_score:
        best_score = score
        best_mask = mask
    
    # Strategy 2: Random + hill climbing
    for _ in range(3):
        if time.time() - start_time > 140:
            break
        mask = 0
        positions = rng.sample(range(512), 32)
        for p in positions:
            mask |= (1 << p)
        score = _score_bitmask(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
        
        # Improve with progression-based moves
        mask, score = _complete_progression_moves(mask, rng, 30)
        if score > best_score:
            best_score = score
            best_mask = mask
    
    # Final polish with simulated annealing
    if time.time() - start_time < 145:
        mask, score = _simulated_annealing(best_mask, rng, 40)
        if score > best_score:
            best_score = score
            best_mask = mask
    
    # Convert bitmask to list
    result = [i + 1 for i in range(512) if best_mask & (1 << i)]
    
    # Ensure exactly 32 elements
    if len(result) != 32:
        # Fallback: use the first 32
        result = list(range(1, 33))
    
    return result
# EVOLVE-BLOCK-END
