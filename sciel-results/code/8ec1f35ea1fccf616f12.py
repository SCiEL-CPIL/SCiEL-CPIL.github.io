"""
Strategy: Fast randomized multi-restart greedy search with O(1) incremental collision checking.
Instead of slow backtracking over all 256 values, we run many fast greedy passes
with random tie-breaking and random value skipping. Each pass is O(n^2) worst-case
but with small constant factors, allowing thousands of attempts in 90s.
"""

import time
import random

_DEADLINE = 85.0
_START_TIME = time.time()


def _remaining_budget():
    return _DEADLINE - (time.time() - _START_TIME)


def _greedy_pass(seq_start, sums_set, suffix_sums, min_val, max_val, 
                 randomize=False, skip_prob=0.0, rng=None):
    """
    One greedy pass: try to extend the sequence by picking valid values.
    
    seq_start: current sequence (list)
    sums_set: set of all current block sums
    suffix_sums: list where suffix_sums[i] = sum of seq[i:] (for current seq)
    min_val: minimum value to try
    max_val: maximum value to try (inclusive)
    randomize: if True, shuffle candidate order
    skip_prob: probability of skipping a valid candidate (for diversity)
    rng: random.Random instance
    
    Returns: (extended_seq, extended_sums_set, extended_suffix_sums)
    """
    seq = list(seq_start)
    sums = set(sums_set)
    suffix = list(suffix_sums)
    
    if rng is None:
        rng = random.Random()
    
    # Build list of candidate values
    candidates = list(range(min_val, max_val + 1))
    if randomize:
        rng.shuffle(candidates)
    
    for v in candidates:
        if _remaining_budget() < 0.5:
            break
        
        # Skip with some probability for diversity
        if skip_prob > 0 and rng.random() < skip_prob:
            continue
        
        # Check if v can be appended
        # New blocks: for each i in 0..len(seq), block (i, len(seq)+1) has sum = suffix[i] + v
        # We need all these sums to be distinct from each other and from existing sums
        
        n = len(seq)
        # suffix currently has n+1 entries: suffix[0]..suffix[n], where suffix[n]=0
        # New block sums: suffix[i] + v for i in 0..n
        
        new_block_sums = set()
        valid = True
        for i in range(n + 1):
            s = suffix[i] + v
            if s in sums or s in new_block_sums:
                valid = False
                break
            new_block_sums.add(s)
        
        if not valid:
            continue
        
        # Accept v
        seq.append(v)
        sums |= new_block_sums
        # Update suffix sums: new suffix has n+2 entries
        # new_suffix[i] = suffix[i] + v for i in 0..n, new_suffix[n+1] = 0
        new_suffix = [0] * (n + 2)
        for i in range(n + 1):
            new_suffix[i] = suffix[i] + v
        new_suffix[n + 1] = 0
        suffix = new_suffix
    
    return seq, sums, suffix


def _greedy_pass_simple(min_val, max_val, randomize=False, skip_prob=0.0, rng=None):
    """
    Simplified greedy pass starting from empty sequence.
    Returns the constructed sequence.
    """
    seq = []
    sums = set()
    suffix = [0]  # suffix[0] = 0 for empty sequence
    
    if rng is None:
        rng = random.Random()
    
    candidates = list(range(min_val, max_val + 1))
    if randomize:
        rng.shuffle(candidates)
    
    for v in candidates:
        if _remaining_budget() < 0.5:
            break
        
        if skip_prob > 0 and rng.random() < skip_prob:
            continue
        
        n = len(seq)
        new_block_sums = set()
        valid = True
        for i in range(n + 1):
            s = suffix[i] + v
            if s in sums or s in new_block_sums:
                valid = False
                break
            new_block_sums.add(s)
        
        if not valid:
            continue
        
        seq.append(v)
        sums |= new_block_sums
        new_suffix = [0] * (n + 2)
        for i in range(n + 1):
            new_suffix[i] = suffix[i] + v
        new_suffix[n + 1] = 0
        suffix = new_suffix
    
    return seq


def _local_improve(seq):
    """
    Try to improve a sequence by removing one element and re-adding a different one.
    This is a simple local search step.
    Returns improved sequence or original if no improvement.
    """
    if _remaining_budget() < 1.0:
        return seq
    
    best = list(seq)
    best_len = len(seq)
    
    for remove_idx in range(len(seq)):
        if _remaining_budget() < 1.0:
            break
        
        # Remove element at remove_idx
        candidate_seq = seq[:remove_idx] + seq[remove_idx+1:]
        
        # Now try to extend from the end
        # Rebuild sums and suffix for candidate_seq
        n = len(candidate_seq)
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += candidate_seq[j]
                if s in sums:
                    # Collision in base sequence, skip
                    break
                sums.add(s)
        
        # Check if base sequence is valid
        valid_base = len(sums) == n * (n + 1) // 2
        if not valid_base:
            continue
        
        # Build suffix sums
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + candidate_seq[i]
        
        # Try to extend from position n
        # The next value must be > candidate_seq[-1] if n > 0, else >= 1
        min_val = candidate_seq[-1] + 1 if n > 0 else 1
        
        # Try to add one or more elements
        extended = list(candidate_seq)
        ext_sums = set(sums)
        ext_suffix = list(suffix)
        
        for v in range(min_val, 257):
            if _remaining_budget() < 1.0:
                break
            
            m = len(extended)
            new_block_sums = set()
            valid = True
            for i in range(m + 1):
                s = ext_suffix[i] + v
                if s in ext_sums or s in new_block_sums:
                    valid = False
                    break
                new_block_sums.add(s)
            
            if not valid:
                continue
            
            extended.append(v)
            ext_sums |= new_block_sums
            new_ext_suffix = [0] * (m + 2)
            for i in range(m + 1):
                new_ext_suffix[i] = ext_suffix[i] + v
            new_ext_suffix[m + 1] = 0
            ext_suffix = new_ext_suffix
        
        if len(extended) > best_len:
            best = extended
            best_len = len(extended)
    
    return best


def _validate(seq):
    """
    Validate that seq is strictly increasing in [1,256] with distinct consecutive block sums.
    """
    n = len(seq)
    if n == 0:
        return True
    
    # Check strictly increasing
    for i in range(1, n):
        if seq[i] <= seq[i-1]:
            return False
    
    # Check range
    for v in seq:
        if v < 1 or v > 256:
            return False
    
    # Check distinct block sums
    seen = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in seen:
                return False
            seen.add(s)
    
    return True


def construct():
    """
    Return a strictly increasing sequence of integers in [1,256] with distinct
    consecutive block sums, maximizing sequence length.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    best_seq = []
    best_len = 0
    
    # Phase 1: Run many fast randomized greedy passes
    # Each pass is fast, so we can do thousands
    
    num_passes = 0
    while _remaining_budget() > 2.0:
        num_passes += 1
        
        # Vary parameters for diversity
        rng = random.Random(num_passes * 7919)
        
        # Different strategies:
        # 1. Simple greedy with random shuffle
        # 2. Greedy with skip probability
        # 3. Start from different min values
        
        strategy = num_passes % 5
        
        if strategy == 0:
            # Pure random shuffle
            seq = _greedy_pass_simple(1, 256, randomize=True, skip_prob=0.0, rng=rng)
        elif strategy == 1:
            # Random shuffle with skip
            seq = _greedy_pass_simple(1, 256, randomize=True, skip_prob=0.1, rng=rng)
        elif strategy == 2:
            # Higher skip probability
            seq = _greedy_pass_simple(1, 256, randomize=True, skip_prob=0.3, rng=rng)
        elif strategy == 3:
            # Start from a higher min value
            min_v = rng.randint(1, 20)
            seq = _greedy_pass_simple(min_v, 256, randomize=True, skip_prob=0.0, rng=rng)
        else:
            # No randomization, standard greedy (deterministic baseline)
            seq = _greedy_pass_simple(1, 256, randomize=False, skip_prob=0.0, rng=rng)
        
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq[:]
            
            # Try local improvement on the best found
            if _remaining_budget() > 5.0:
                improved = _local_improve(best_seq)
                if len(improved) > best_len:
                    best_len = len(improved)
                    best_seq = improved[:]
        
        # Check deadline
        if _remaining_budget() < 2.0:
            break
    
    # Phase 2: Try to extend the best sequence further with targeted search
    if _remaining_budget() > 3.0 and best_len > 0:
        # Try to add more elements to the end
        n = len(best_seq)
        min_val = best_seq[-1] + 1
        
        # Rebuild sums and suffix
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += best_seq[j]
                sums.add(s)
        
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + best_seq[i]
        
        # Try to extend
        extended = list(best_seq)
        ext_sums = set(sums)
        ext_suffix = list(suffix)
        
        for v in range(min_val, 257):
            if _remaining_budget() < 2.0:
                break
            
            m = len(extended)
            new_block_sums = set()
            valid = True
            for i in range(m + 1):
                s = ext_suffix[i] + v
                if s in ext_sums or s in new_block_sums:
                    valid = False
                    break
                new_block_sums.add(s)
            
            if not valid:
                continue
            
            extended.append(v)
            ext_sums |= new_block_sums
            new_ext_suffix = [0] * (m + 2)
            for i in range(m + 1):
                new_ext_suffix[i] = ext_suffix[i] + v
            new_ext_suffix[m + 1] = 0
            ext_suffix = new_ext_suffix
        
        if len(extended) > best_len:
            best_len = len(extended)
            best_seq = extended
    
    # Final validation
    if not _validate(best_seq):
        # Fallback: simple deterministic greedy
        best_seq = _greedy_pass_simple(1, 256, randomize=False, skip_prob=0.0)
    
    if not _validate(best_seq):
        # Ultimate fallback: return a known valid short sequence
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
    
    return best_seq
