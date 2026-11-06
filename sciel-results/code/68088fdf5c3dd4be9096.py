import time
import random
import numpy as np
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

def _compute_threshold(trans0, trans1):
    """
    Computes the exact shortest reset word length using BFS on the subset automaton.
    Returns -1 if not synchronizing.
    """
    # Precompute transitions for all subsets
    # next_subsets[s][0] is the subset reached from s by '0'
    # next_subsets[s][1] is the subset reached from s by '1'
    
    # Optimization: Use bit operations
    # For a subset s, the transition is the union of trans[bit] for all bits in s.
    
    # We can precompute this for all 2^N subsets.
    # 2^12 = 4096 subsets.
    # For each subset, we need to compute 2 transitions.
    # Each transition is a union of up to N bits.
    
    next_subsets = np.zeros((1 << N, 2), dtype=np.int32)
    
    # Precompute single-bit transitions
    t0_bits = [1 << trans0[i] for i in range(N)]
    t1_bits = [1 << trans1[i] for i in range(N)]
    
    for s in range(1 << N):
        # Compute transition for 0
        res0 = 0
        temp = s
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            res0 |= t0_bits[idx]
            temp ^= lsb
        next_subsets[s, 0] = res0
        
        # Compute transition for 1
        res1 = 0
        temp = s
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            res1 |= t1_bits[idx]
            temp ^= lsb
        next_subsets[s, 1] = res1

    # BFS from FULL_MASK
    visited = np.full(1 << N, -1, dtype=np.int32)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        dist = visited[current]
        
        # Check if singleton
        if current != 0 and (current & (current - 1)) == 0:
            return dist
            
        for letter in range(2):
            nxt = next_subsets[current, letter]
            if visited[nxt] == -1:
                visited[nxt] = dist + 1
                queue.append(nxt)
                
    return -1

def _is_synchronizing(trans0, trans1):
    """
    Quick check: does the automaton synchronize?
    Returns True if it synchronizes, False otherwise.
    Uses BFS but stops early if a singleton is found.
    """
    # Similar to _compute_threshold but returns bool
    t0_bits = [1 << trans0[i] for i in range(N)]
    t1_bits = [1 << trans1[i] for i in range(N)]
    
    next_subsets = np.zeros((1 << N, 2), dtype=np.int32)
    
    for s in range(1 << N):
        res0 = 0
        temp = s
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            res0 |= t0_bits[idx]
            temp ^= lsb
        next_subsets[s, 0] = res0
        
        res1 = 0
        temp = s
        while temp:
            lsb = temp & -temp
            idx = lsb.bit_length() - 1
            res1 |= t1_bits[idx]
            temp ^= lsb
        next_subsets[s, 1] = res1

    visited = np.full(1 << N, -1, dtype=np.int32)
    queue = deque()
    
    start = FULL_MASK
    visited[start] = 0
    queue.append(start)
    
    while queue:
        current = queue.popleft()
        
        if current != 0 and (current & (current - 1)) == 0:
            return True
            
        for letter in range(2):
            nxt = next_subsets[current, letter]
            if visited[nxt] == -1:
                visited[nxt] = 0
                queue.append(nxt)
                
    return False

def _construct_ladder(rng, target_length):
    """
    Constructs an automaton that follows a specific 'ladder' of subsets.
    We want a path: S_0 (size N) -> S_1 (size N-1) -> ... -> S_k (size 1).
    This guarantees a reset word of length at least k.
    
    Strategy:
    1. Define a sequence of subsets S_0, S_1, ..., S_k.
    2. For each step i, choose a letter l_i such that delta(S_i, l_i) = S_{i+1}.
    3. This imposes constraints on trans0 and trans1.
    4. Solve for trans0 and trans1 satisfying these constraints.
    """
    # Generate a random chain of subsets
    # S_0 = {0, ..., N-1}
    # S_{i+1} is a random subset of S_i of size |S_i| - 1
    
    subsets = []
    current_set = list(range(N))
    subsets.append(current_set[:])
    
    for _ in range(target_length - 1):
        # Remove a random element
        if not current_set:
            break
        idx_to_remove = rng.randint(0, len(current_set) - 1)
        current_set.pop(idx_to_remove)
        subsets.append(current_set[:])
    
    k = len(subsets) - 1
    
    # Now we have a chain of k+1 subsets.
    # We need to assign letters to the transitions S_i -> S_{i+1}.
    # Let letter_seq[i] be the letter used to go from S_i to S_{i+1}.
    
    # Constraints:
    # For each i, delta(S_i, letter_seq[i]) = S_{i+1}.
    # delta(S, l) = { trans_l(x) for x in S }
    
    # We have 2 letters.
    # Let's try to assign letters such that we can satisfy the constraints.
    # This is a constraint satisfaction problem.
    
    # Simplification:
    # Assume the letter sequence is fixed (e.g., alternating 0, 1, 0, 1...).
    # Or random.
    # Then we solve for trans0 and trans1.
    
    # Let's try a few random letter sequences.
    
    best_trans0 = None
    best_trans1 = None
    best_score = -1
    
    for attempt in range(20):
        # Random letter sequence
        letters = [rng.randint(0, 1) for _ in range(k)]
        
        # Initialize trans0, trans1 with None
        trans0 = [None] * N
        trans1 = [None] * N
        
        # Constraints:
        # For each step i:
        # Let S_i be the set of states.
        # Let l = letters[i].
        # Let T_l be the transition function for l.
        # We need { T_l(x) for x in S_i } = S_{i+1}.
        
        # This implies:
        # 1. For all x in S_i, T_l(x) must be in S_{i+1}.
        # 2. For all y in S_{i+1}, there exists x in S_i such that T_l(x) = y.
        
        # This is a bipartite matching problem for each step?
        # Actually, since it's a function, each x maps to exactly one y.
        # So we need a surjection from S_i to S_{i+1} using T_l.
        # Since |S_i| = |S_{i+1}| + 1, one element in S_i maps to the same target as another, or some element maps outside? No, must map INSIDE S_{i+1}.
        # So two elements in S_i must map to the same element in S_{i+1}.
        
        # Let's collect constraints for each letter.
        # For each letter l, we have a set of constraints on T_l.
        
        # This is getting complex. Let's try a simpler heuristic.
        
        # Heuristic:
        # Pick T_0 to be a "shift" on the states, and T_1 to be a "reset" or "collapse".
        # Or, construct T_0 and T_1 randomly, then check if they support the ladder.
        
        # Alternative Strategy: "Direct Assignment"
        # For each state x, and each letter l, T_l(x) is a variable.
        # Constraints are linear-ish (set equality).
        
        # Let's try a specific construction known to work:
        # "Pinwheel" or "Cerny-like" but tailored.
        
        # Actually, let's just use the GA approach but with a smarter seed.
        # The "Ladder" idea is good for *verifying* a candidate, but constructing one from scratch is hard.
        
        # Let's switch to a "Local Search" strategy starting from a good seed.
        
        pass # Skip this complex construction for now, use hill climbing.

    return None, None, -1

def _hill_climb(trans0, trans1, rng, time_limit):
    """
    Improves the automaton using hill climbing.
    Tries single-step mutations and keeps them if they increase the threshold.
    """
    current0 = trans0[:]
    current1 = trans1[:]
    current_score = _compute_threshold(current0, current1)
    
    if current_score < 0:
        return current0, current1, -1
        
    best0 = current0[:]
    best1 = current1[:]
    best_score = current_score
    
    start_time = time.time()
    no_improve_count = 0
    
    while time.time() - start_time < time_limit:
        # Generate a neighbor
        mut0 = current0[:]
        mut1 = current1[:]
        
        # Pick a random state and letter, change its destination
        state = rng.randint(0, N - 1)
        letter = rng.randint(0, 1)
        
        if letter == 0:
            mut0[state] = rng.randint(0, N - 1)
        else:
            mut1[state] = rng.randint(0, N - 1)
            
        # Quick check: is it synchronizing?
        # If not, skip (score -1)
        # To save time, we can do a fast BFS check that returns early if not synchronizing?
        # No, BFS to check synchronizability is similar cost to computing threshold.
        # But if it's not synchronizing, we don't need the exact threshold.
        # Let's just compute the threshold.
        
        new_score = _compute_threshold(mut0, mut1)
        
        if new_score > current_score:
            current0 = mut0
            current1 = mut1
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best0 = current0[:]
                best1 = current1[:]
                
            no_improve_count = 0
        else:
            # No improvement, maybe revert or just continue
            # If we accept worse moves (simulated annealing), we might escape local optima.
            # But hill climbing is simpler. Let's stick to strict improvement for now.
            # If stuck, we might need to restart.
            no_improve_count += 1
            
            # If stuck for too long, try a larger jump
            if no_improve_count > 50:
                # Restart from best with a random perturbation
                current0 = best0[:]
                current1 = best1[:]
                current_score = best_score
                
                # Perturb
                for _ in range(rng.randint(2, 5)):
                    state = rng.randint(0, N - 1)
                    letter = rng.randint(0, 1)
                    if letter == 0:
                        current0[state] = rng.randint(0, N - 1)
                    else:
                        current1[state] = rng.randint(0, N - 1)
                        
                new_score = _compute_threshold(current0, current1)
                if new_score > 0:
                    current_score = new_score
                else:
                    # If perturbation broke it, revert to best
                    current0 = best0[:]
                    current1 = best1[:]
                    current_score = best_score
                    
                no_improve_count = 0

    return best0, best1, best_score

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin
    
    rng = random.Random(12345)
    
    best_overall_score = -1
    best_overall_0 = []
    best_overall_1 = []
    
    # Initialize with a few diverse seeds
    seeds = []
    
    # Seed 1: Cerny Automaton
    c0 = [(i + 1) % N for i in range(N)]
    c1 = [0] * N
    for i in range(1, N):
        c1[i] = i - 1
    seeds.append((c0, c1))
    
    # Seed 2: Random
    for _ in range(10):
        r0 = [rng.randint(0, N - 1) for _ in range(N)]
        r1 = [rng.randint(0, N - 1) for _ in range(N)]
        # Check if synchronizing
        if _is_synchronizing(r0, r1):
            seeds.append((r0, r1))
            
    # If no random seeds are synchronizing, add more Cerny variants
    if len(seeds) < 5:
        for _ in range(5):
            # Perturb Cerny
            p0 = c0[:]
            p1 = c1[:]
            for _ in range(rng.randint(1, 3)):
                state = rng.randint(0, N - 1)
                letter = rng.randint(0, 1)
                if letter == 0:
                    p0[state] = rng.randint(0, N - 1)
                else:
                    p1[state] = rng.randint(0, N - 1)
            if _is_synchronizing(p0, p1):
                seeds.append((p0, p1))
                
    # Run hill climbing on each seed
    # Allocate time based on remaining budget
    time_per_seed = (deadline - (time.time() - start_time)) / max(1, len(seeds))
    
    for seed0, seed1 in seeds:
        if time.time() > deadline:
            break
            
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
            
        # Allocate at least 2s, at most 20s per seed
        alloc = min(20.0, max(2.0, remaining_time / 2))
        
        res0, res1, score = _hill_climb(seed0, seed1, rng, alloc)
        
        if score > best_overall_score:
            best_overall_score = score
            best_overall_0 = res0
            best_overall_1 = res1
            
    return [best_overall_0, best_overall_1]

# EVOLVE-BLOCK-END
