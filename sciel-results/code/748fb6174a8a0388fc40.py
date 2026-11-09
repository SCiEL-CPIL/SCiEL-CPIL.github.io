import time
import random
from collections import deque

N = 12

def compute_threshold_fast(trans):
    """
    Fast BFS on the subset automaton using bitmask representation.
    trans: [list0, list1]
    Returns shortest reset word length or inf if not synchronizing.
    """
    full_mask = (1 << N) - 1
    singleton_masks = set(1 << i for i in range(N))
    
    # Precompute next state for each mask for both letters
    # This is O(2^N * N) which is ~50k ops, negligible.
    next_a = [0] * (1 << N)
    next_b = [0] * (1 << N)
    
    # Optimization: Compute transitions for all masks at once
    # For each mask, we need to apply trans[0] and trans[1] to each bit
    # We can do this more efficiently by iterating bits.
    
    for mask in range(1 << N):
        m_a = 0
        m_b = 0
        # Iterate through set bits
        # A simple loop over N is fine for N=12
        for i in range(N):
            if mask & (1 << i):
                m_a |= (1 << trans[0][i])
                m_b |= (1 << trans[1][i])
        next_a[mask] = m_a
        next_b[mask] = m_b
        
    visited = {full_mask: 0}
    queue = deque([full_mask])
    
    while queue:
        current_mask = queue.popleft()
        dist = visited[current_mask]
        
        if current_mask in singleton_masks:
            return dist
            
        # Try 'a'
        next_mask_a = next_a[current_mask]
        if next_mask_a not in visited:
            visited[next_mask_a] = dist + 1
            queue.append(next_mask_a)
            
        # Try 'b'
        next_mask_b = next_b[current_mask]
        if next_mask_b not in visited:
            visited[next_mask_b] = dist + 1
            queue.append(next_mask_b)
            
    return float('inf')

def generate_pawlik_base():
    """
    Standard Černý (Pawlik) automaton.
    a: i -> (i + 1) % N
    b: 0 -> 0, i -> i - 1 for i > 0
    """
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] * N
    for i in range(1, N):
        b_map[i] = i - 1
    return [a_map, b_map]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # 1. Start with the known optimal Pawlik automaton
    base = generate_pawlik_base()
    best_trans = base
    best_score = compute_threshold_fast(base)
    
    # 2. Local Search / Hill Climbing
    # Strategy: Randomly perturb the transition maps, check if it still synchronizes,
    # and if the threshold is improved.
    # We focus on perturbing the 'b' map (the reset-like map) or the 'a' map (the permutation).
    # Since 'a' is a permutation in Pawlik, we can try other permutations or non-permutations.
    # However, keeping 'a' as a permutation is a strong structural constraint that often yields high thresholds.
    # Let's try random mutations on both, but with bias towards keeping structure.
    
    # To speed up evaluation, we rely on compute_threshold_fast.
    # For N=12, one evaluation takes ~1-2ms.
    # In 80 seconds, we can potentially run ~40,000 evaluations.
    
    # We will use a simple hill climbing with restarts or just continuous search.
    # Given the "stagnation" warning, we should try a more aggressive search or different neighborhood.
    
    # Neighborhood: Change one transition in one of the two maps.
    # If it improves, accept. If it stagnates, maybe try a larger jump?
    # For now, simple hill climbing with random restarts if stuck.
    
    current_trans = [list(base[0]), list(base[1])]
    current_score = best_score
    no_improve_count = 0
    max_no_improve = 2000  # Restart if stuck for this many steps
    
    while time.time() < deadline:
        # Pick a random map (0 or 1) and a random state
        map_idx = random.randint(0, 1)
        state_idx = random.randint(0, N - 1)
        
        # Generate a new destination
        new_dest = random.randint(0, N - 1)
        
        # Save old value
        old_dest = current_trans[map_idx][state_idx]
        
        # Apply mutation
        current_trans[map_idx][state_idx] = new_dest
        
        # Evaluate
        new_score = compute_threshold_fast(current_trans)
        
        if new_score < float('inf'):  # Must synchronize
            if new_score > current_score:
                current_score = new_score
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_trans = [list(current_trans[0]), list(current_trans[1])]
            else:
                no_improve_count += 1
                # If no improvement, revert the change (strict hill climbing)
                # Or, we could accept equal scores to explore plateaus?
                # Strict is safer to avoid drifting away from good structures.
                current_trans[map_idx][state_idx] = old_dest
        else:
            # Not synchronizing, revert
            current_trans[map_idx][state_idx] = old_dest
            no_improve_count += 1
            
        # If stuck, try a more aggressive restart?
        # For now, just keep going. The random nature might escape local minima eventually.
        # If we want to be smarter, we could restart from best_trans if stuck too long.
        if no_improve_count > max_no_improve:
            # Restart from best found so far
            current_trans = [list(best_trans[0]), list(best_trans[1])]
            current_score = best_score
            no_improve_count = 0
            
    return best_trans

if __name__ == "__main__":
    # Simple self-test
    maps = construct()
    print("Constructed automaton.")
    print("Threshold:", compute_threshold_fast(maps))
