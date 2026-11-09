"""Simulated annealing search for divisibility-incomparable subset sums."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _compute_subset_sums(nums):
    """Compute all nonempty subset sums as a sorted list."""
    sums = set()
    for x in nums:
        new_sums = set()
        for s in sums:
            new_sums.add(s + x)
        new_sums.add(x)
        sums |= new_sums
    return sorted(sums)

def _is_valid_fast(nums):
    """Fast validity check using bitset for subset sums."""
    n = len(nums)
    if n == 0:
        return True
    if n > 16:
        return False
    
    # Use bitset: bit i is set if sum i is achievable
    # Max possible sum is 16 * 256 = 4096
    max_sum = sum(nums)
    bitset = 1  # bit 0 set (empty sum)
    for x in nums:
        bitset |= (bitset << x)
    
    # Extract all achievable sums
    sums = []
    for i in range(1, max_sum + 1):
        if bitset & (1 << i):
            sums.append(i)
    
    # Check distinctness: should have exactly 2^n - 1 sums
    if len(sums) != (1 << n) - 1:
        return False
    
    # Check divisibility: no two sums divide each other
    sums_set = set(sums)
    max_s = sums[-1] if sums else 0
    for s in sums:
        for k in range(2, max_s // s + 1):
            if k * s in sums_set:
                return False
    return True

def _score(nums):
    """Score function: higher is better. Returns (validity, size) for comparison."""
    if not _is_valid_fast(nums):
        return (0, len(nums))
    return (1, len(nums))

def _try_add(nums, candidate):
    """Try adding candidate to nums. Returns new list if valid, else None."""
    if candidate in nums:
        return None
    trial = nums + [candidate]
    if _is_valid_fast(trial):
        return trial
    return None

def _try_swap(nums, idx, replacement):
    """Try replacing nums[idx] with replacement. Returns new list if valid, else None."""
    if replacement in nums:
        return None
    trial = nums[:]
    trial[idx] = replacement
    if _is_valid_fast(trial):
        return trial
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    rng = random.Random(42)
    all_candidates = list(range(1, 257))
    
    # Initial solution: greedy construction
    best = []
    best_sums = set()
    for num in sorted(all_candidates, reverse=True):
        trial = best + [num]
        if _is_valid_fast(trial):
            best = trial
    
    best_score = _score(best)
    best_valid = best_score[0]
    best_size = best_score[1]
    
    # Simulated annealing
    current = best[:]
    current_score = best_score
    T = 10.0
    T_min = 0.1
    cooling = 0.995
    
    while time.time() < deadline - 5 and T > T_min:
        # Generate a random move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Try to add a random element
            candidate = rng.choice(all_candidates)
            new_state = _try_add(current, candidate)
            if new_state is None:
                T *= cooling
                continue
            new_score = _score(new_state)
        elif move_type < 0.8:
            # Try to swap a random element
            idx = rng.randint(0, len(current) - 1)
            replacement = rng.choice(all_candidates)
            new_state = _try_swap(current, idx, replacement)
            if new_state is None:
                T *= cooling
                continue
            new_score = _score(new_state)
        else:
            # Try to remove a random element (to allow re-adding different ones)
            if len(current) <= 3:
                T *= cooling
                continue
            idx = rng.randint(0, len(current) - 1)
            new_state = current[:idx] + current[idx+1:]
            new_score = _score(new_state)
        
        # Acceptance criterion
        delta = (new_score[0] - current_score[0], new_score[1] - current_score[1])
        
        if new_score >= current_score:
            # Accept if better or equal
            current = new_state
            current_score = new_score
            if current_score > best_score:
                best = current[:]
                best_score = current_score
                best_valid = current_score[0]
                best_size = current_score[1]
        else:
            # Accept worse state with probability exp(delta / T)
            # For our score, delta is negative in the relevant dimension
            if delta[0] == -1:
                # Lost validity - very unlikely to accept
                prob = math.exp(-5.0 / T)
            elif delta[1] < 0:
                # Lost size - accept with some probability
                prob = math.exp(delta[1] / T)
            else:
                prob = 0.0
            
            if rng.random() < prob:
                current = new_state
                current_score = new_score
        
        T *= cooling
    
    # Final greedy extension from best
    if time.time() < deadline - 2:
        extended = best[:]
        for num in sorted(all_candidates, reverse=True):
            if num in extended:
                continue
            trial = extended + [num]
            if _is_valid_fast(trial):
                extended = trial
        if len(extended) > best_size:
            best = extended
    
    # Final validation
    if not _is_valid_fast(best):
        best = [31, 32, 34]
    
    return list(best)
# EVOLVE-BLOCK-END
