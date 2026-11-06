"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    def compute_all_block_sums(seq):
        """Compute all nonempty consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        """Check if sequence is strictly increasing in [1,256] with distinct block sums."""
        if not seq:
            return True
        if any(x < 1 or x > 256 for x in seq):
            return False
        for i in range(1, len(seq)):
            if seq[i] <= seq[i-1]:
                return False
        sums = compute_all_block_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    # Greedy baseline
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    best_seq = values[:]
    best_len = len(best_seq)

    # Simulated Annealing on the sequence
    # Representation: list of integers, strictly increasing, each in [1,256]
    # Objective: maximize length (primary), then minimize number of duplicate block sums (secondary)
    
    def get_score(seq):
        """Score: primary is length, secondary is -number_of_duplicate_sums."""
        if not is_valid(seq):
            # Count how many distinct sums we have vs expected
            n = len(seq)
            expected = n * (n + 1) // 2
            actual = len(compute_all_block_sums(seq))
            # Also check strictly increasing
            inc = all(seq[i] > seq[i-1] for i in range(1, len(seq)))
            in_range = all(1 <= x <= 256 for x in seq)
            if not inc or not in_range:
                return -1000000 - len(seq)
            return -1000000 - (expected - actual) * 100 - len(seq)
        return len(seq)

    def get_duplicate_count(seq):
        """Number of duplicate block sums (0 if all distinct)."""
        n = len(seq)
        expected = n * (n + 1) // 2
        actual = len(compute_all_block_sums(seq))
        return expected - actual

    # Start from greedy solution
    current = best_seq[:]
    current_score = get_score(current)
    
    # Temperature schedule
    T = 50.0
    cooling = 0.999
    min_T = 0.1
    
    # Move operators
    def move_insert(seq):
        """Insert a random element at a random position."""
        if len(seq) >= 256:
            return None
        # Choose a value not in seq
        existing = set(seq)
        candidates = [x for x in range(1, 257) if x not in existing]
        if not candidates:
            return None
        val = random.choice(candidates)
        pos = random.randint(0, len(seq))
        new_seq = seq[:pos] + [val] + seq[pos:]
        # Check strictly increasing
        for i in range(1, len(new_seq)):
            if new_seq[i] <= new_seq[i-1]:
                return None
        return new_seq

    def move_remove(seq):
        """Remove a random element."""
        if len(seq) <= 1:
            return None
        pos = random.randint(0, len(seq) - 1)
        return seq[:pos] + seq[pos+1:]

    def move_swap(seq):
        """Swap two elements (maintaining increasing order by re-sorting)."""
        if len(seq) < 2:
            return None
        i, j = random.sample(range(len(seq)), 2)
        new_seq = seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        new_seq.sort()
        # Check strictly increasing (should be after sort, but check for duplicates)
        for k in range(1, len(new_seq)):
            if new_seq[k] <= new_seq[k-1]:
                return None
        return new_seq

    def move_adjust(seq):
        """Adjust a random element by a small amount."""
        if not seq:
            return None
        pos = random.randint(0, len(seq) - 1)
        delta = random.choice([-3, -2, -1, 1, 2, 3])
        new_val = seq[pos] + delta
        if new_val < 1 or new_val > 256:
            return None
        new_seq = seq[:]
        new_seq[pos] = new_val
        new_seq.sort()
        # Check strictly increasing
        for k in range(1, len(new_seq)):
            if new_seq[k] <= new_seq[k-1]:
                return None
        return new_seq

    # Run simulated annealing
    iteration = 0
    max_iter = 50000
    
    while time.time() < deadline - 2.0 and iteration < max_iter:
        iteration += 1
        
        # Choose a move
        move_type = random.random()
        if move_type < 0.3:
            new_seq = move_insert(current)
        elif move_type < 0.5:
            new_seq = move_remove(current)
        elif move_type < 0.7:
            new_seq = move_swap(current)
        else:
            new_seq = move_adjust(current)
        
        if new_seq is None:
            continue
        
        new_score = get_score(new_seq)
        delta = new_score - current_score
        
        if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 0.01)):
            current = new_seq
            current_score = new_score
            if current_score > best_len:
                best_seq = current[:]
                best_len = current_score
        
        T *= cooling
        if T < min_T:
            T = min_T
    
    # Local search refinement
    def local_search(seq, max_iters=10000):
        """Try to improve sequence via local moves."""
        best = seq[:]
        best_sc = get_score(best)
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for move_fn in [move_insert, move_remove, move_swap, move_adjust]:
                new_seq = move_fn(best)
                if new_seq is None:
                    continue
                sc = get_score(new_seq)
                if sc > best_sc:
                    best = new_seq
                    best_sc = sc
                    improved = True
                    break
            if not improved:
                break
        return best, best_sc
    
    refined, refined_score = local_search(current)
    if refined_score > best_len:
        best_seq = refined[:]
        best_len = refined_score

    # Try multiple SA restarts with different initial sequences
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
        
        # Create a perturbed starting point
        perturbed = best_seq[:]
        for _ in range(random.randint(3, 8)):
            move_type = random.random()
            if move_type < 0.4:
                new = move_insert(perturbed)
            elif move_type < 0.7:
                new = move_remove(perturbed)
            else:
                new = move_adjust(perturbed)
            if new is not None:
                perturbed = new
        
        # Run short SA from this point
        T = 20.0
        for _ in range(5000):
            if time.time() > deadline - 3.0:
                break
            move_type = random.random()
            if move_type < 0.3:
                new_seq = move_insert(perturbed)
            elif move_type < 0.5:
                new_seq = move_remove(perturbed)
            elif move_type < 0.7:
                new_seq = move_swap(perturbed)
            else:
                new_seq = move_adjust(perturbed)
            
            if new_seq is None:
                continue
            
            new_score = get_score(new_seq)
            delta = new_score - get_score(perturbed)
            
            if delta > 0 or random.random() < pow(2.718281828, delta / max(T, 0.01)):
                perturbed = new_seq
            
            T *= 0.999
            if T < 0.1:
                T = 0.1
        
        if get_score(perturbed) > best_len:
            best_seq = perturbed[:]
            best_len = get_score(perturbed)

    return best_seq
# EVOLVE-BLOCK-END
