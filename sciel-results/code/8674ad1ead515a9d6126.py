"""Program for increasing sequences with unique consecutive-block sums."""
import random
import math

# EVOLVE-BLOCK-START

def _compute_block_sums(seq):
    """Compute all nonempty consecutive block sums for a sequence."""
    n = len(seq)
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return sums

def _count_duplicate_sums(seq):
    """Count the number of duplicate block sums (excess over distinct count)."""
    n = len(seq)
    total_blocks = n * (n + 1) // 2
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            sums.add(s)
    return total_blocks - len(sums)

def _is_valid(seq):
    """Check if sequence has all distinct block sums and is strictly increasing in [1,256]."""
    if not seq:
        return True
    if seq[0] < 1 or seq[-1] > 256:
        return False
    for i in range(1, len(seq)):
        if seq[i] <= seq[i-1]:
            return False
    return _count_duplicate_sums(seq) == 0

def _greedy_construct():
    """Greedy construction to get a good initial sequence."""
    values = []
    sums = set()
    for x in range(1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    return values

def _simulated_annealing(initial_seq, max_iters=50000, start_temp=10.0, end_temp=0.01):
    """Simulated annealing on difference vectors to maximize sequence length."""
    seq = list(initial_seq)
    n = len(seq)
    
    # Represent as difference vector: d[i] = seq[i+1] - seq[i] >= 1
    # seq[0] is the base value
    base = seq[0]
    diffs = [seq[i+1] - seq[i] for i in range(n-1)]
    
    def rebuild(base, diffs):
        s = [base]
        for d in diffs:
            s.append(s[-1] + d)
        return s
    
    def get_seq(base, diffs):
        return rebuild(base, diffs)
    
    # Initial cost
    current_seq = get_seq(base, diffs)
    current_cost = _count_duplicate_sums(current_seq)
    best_seq = list(current_seq)
    best_cost = current_cost
    
    temp = start_temp
    cooling = (end_temp / start_temp) ** (1.0 / max_iters)
    
    for it in range(max_iters):
        temp *= cooling
        
        # Perturbation: modify one or two differences
        move_type = random.random()
        
        new_base = base
        new_diffs = list(diffs)
        
        if move_type < 0.3:
            # Change one difference
            idx = random.randrange(len(new_diffs))
            delta = random.randint(-3, 3)
            if delta != 0:
                new_diffs[idx] = max(1, new_diffs[idx] + delta)
        elif move_type < 0.5:
            # Change base value
            delta = random.randint(-2, 2)
            if delta != 0:
                new_base = max(1, base + delta)
        elif move_type < 0.7:
            # Swap two differences
            if len(new_diffs) >= 2:
                i, j = random.sample(range(len(new_diffs)), 2)
                new_diffs[i], new_diffs[j] = new_diffs[j], new_diffs[i]
        elif move_type < 0.85:
            # Try to extend the sequence
            # Add a new element at the end
            if new_base + sum(new_diffs) < 256:
                last_val = new_base + sum(new_diffs)
                # Try adding a new difference
                new_d = random.randint(1, 256 - last_val)
                new_diffs.append(new_d)
        else:
            # Try to remove last element (shrink)
            if len(new_diffs) > 1:
                new_diffs.pop()
        
        # Validate bounds
        new_seq = get_seq(new_base, new_diffs)
        if not new_seq or new_seq[0] < 1 or new_seq[-1] > 256:
            continue
        # Check strictly increasing
        valid_inc = all(new_seq[i] < new_seq[i+1] for i in range(len(new_seq)-1))
        if not valid_inc:
            continue
        
        new_cost = _count_duplicate_sums(new_seq)
        
        # Acceptance criterion
        delta_cost = new_cost - current_cost
        if delta_cost <= 0:
            # Accept
            base = new_base
            diffs = new_diffs
            current_cost = new_cost
            current_seq = new_seq
            if current_cost < best_cost:
                best_cost = current_cost
                best_seq = list(current_seq)
        else:
            # Probabilistic acceptance
            if temp > 0:
                prob = math.exp(-delta_cost / temp)
                if random.random() < prob:
                    base = new_base
                    diffs = new_diffs
                    current_cost = new_cost
                    current_seq = new_seq
    
    return best_seq

def _extend_sequence(seq):
    """Try to extend a valid sequence by trying all possible next values."""
    if not seq:
        return [1]
    last = seq[-1]
    if last >= 256:
        return seq
    
    # Get existing block sums
    sums = _compute_block_sums(seq)
    
    for x in range(last + 1, 257):
        tails = []
        total = x
        tails.append(total)
        for a in reversed(seq):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            seq.append(x)
            sums.update(tails)
            # Recursively try to extend further
            seq = _extend_sequence(seq)
            return seq
    
    return seq

def construct():
    """Construct a strictly increasing sequence in [1,256] with distinct consecutive block sums."""
    # Start with greedy
    greedy_seq = _greedy_construct()
    
    # Try to extend greedy sequence
    extended = _extend_sequence(list(greedy_seq))
    best_seq = extended
    
    # Run simulated annealing to try to improve
    # Use multiple restarts
    best_len = len(best_seq)
    
    random.seed(42)
    
    for restart in range(5):
        # Start from current best or a variant
        if restart == 0:
            initial = list(greedy_seq)
        else:
            # Perturb the greedy sequence
            initial = list(greedy_seq)
            # Randomly modify some elements
            for _ in range(random.randint(1, 3)):
                if len(initial) > 1:
                    idx = random.randrange(len(initial) - 1)
                    if random.random() < 0.5:
                        # Increase
                        if initial[idx] < 255:
                            initial[idx] += random.randint(1, 5)
                    else:
                        # Decrease
                        if initial[idx] > 2:
                            initial[idx] -= random.randint(1, 5)
            # Ensure strictly increasing
            for i in range(1, len(initial)):
                if initial[i] <= initial[i-1]:
                    initial[i] = initial[i-1] + 1
            # Ensure within bounds
            if initial[-1] > 256:
                initial = list(greedy_seq)
        
        sa_result = _simulated_annealing(initial, max_iters=30000, start_temp=8.0, end_temp=0.05)
        
        # Validate and extend
        if len(sa_result) > len(best_seq):
            # Check if valid
            if _is_valid(sa_result):
                extended_sa = _extend_sequence(list(sa_result))
                if len(extended_sa) > len(best_seq):
                    best_seq = extended_sa
                    best_len = len(best_seq)
    
    # Also try a different approach: start from smaller sequences and grow
    for start_val in [1, 2, 3, 5, 8]:
        trial = [start_val]
        trial = _extend_sequence(trial)
        if len(trial) > len(best_seq):
            best_seq = trial
    
    # Final extension
    best_seq = _extend_sequence(list(best_seq))
    
    return best_seq

# EVOLVE-BLOCK-END
