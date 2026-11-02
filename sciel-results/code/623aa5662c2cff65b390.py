import itertools
import random
import time
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    total = q ** n
    words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(words)}
    
    # Precompute spheres (balls) for radius 1
    spheres = []
    for i, w in enumerate(words):
        ball = [i]
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    new_w_list = list(w)
                    new_w_list[pos] = new_val
                    new_w_tuple = tuple(new_w_list)
                    ball.append(word_to_idx[new_w_tuple])
        spheres.append(ball)
    
    # Precompute for each word, the list of codeword indices that cover it
    # Actually, we'll work with coverage counts directly.
    
    def coverage_counts(code_set):
        """Return an array of coverage counts for each word in the space."""
        counts = [0] * total
        for ci in code_set:
            for w in spheres[ci]:
                counts[w] += 1
        return counts
    
    def is_covered(counts):
        """Check if all words are covered (count > 0)."""
        for c in counts:
            if c == 0:
                return False
        return True
    
    def uncovered_count(counts):
        """Return the number of uncovered words."""
        return sum(1 for c in counts if c == 0)
    
    # Build initial code: tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = set()
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            w = tuple(h + list(tail))
            initial_code.add(word_to_idx[w])
    
    # Remove redundant codewords from initial code
    def remove_redundant(code_set):
        code_list = sorted(code_set)
        if not code_list:
            return code_set
        
        counts = coverage_counts(code_set)
        changed = True
        while changed:
            changed = False
            new_code = set()
            for ci in code_list:
                # Check if ci is redundant: all words in its sphere have count > 1
                redundant = True
                for w in spheres[ci]:
                    if counts[w] <= 1:
                        redundant = False
                        break
                if redundant:
                    for w in spheres[ci]:
                        counts[w] -= 1
                    changed = True
                else:
                    new_code.add(ci)
            code_list = sorted(new_code)
        return new_code
    
    best_code = remove_redundant(initial_code)
    best_counts = coverage_counts(best_code)
    best_size = len(best_code)
    
    # 1-swap local search: try swapping each codeword with every non-codeword
    # A swap is valid if the new code still covers the space.
    # We prefer swaps that reduce the size (impossible with 1-swap, size stays same)
    # or that increase the minimum coverage (more robust).
    # Actually, 1-swap keeps size the same. So we need a different objective.
    
    # Better: use a "quality" metric = sum of coverage counts (total coverage mass).
    # Higher quality means more redundancy, which might allow further reduction.
    # But we want to minimize size. So let's try: after swap, try to remove redundant.
    
    # Alternative: simulated annealing on the set of codewords.
    # State: set of codewords.
    # Move: add a random codeword, or remove a random codeword, or swap.
    # Objective: minimize size, subject to coverage.
    # Penalty: if not covered, add large penalty.
    
    def objective(code_set):
        counts = coverage_counts(code_set)
        uncov = uncovered_count(counts)
        # Penalty for uncovered words
        return len(code_set) + 1000 * uncov
    
    # Simulated annealing
    # Start from best_code
    current_code = set(best_code)
    current_obj = objective(current_code)
    
    # Temperature schedule
    T_start = 50.0
    T_end = 0.5
    cooling_rate = 0.995
    
    T = T_start
    step = 0
    max_steps = 50000
    
    # Precompute list of all indices for random selection
    all_indices = list(range(total))
    
    while step < max_steps and time.time() < deadline - 10:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.33:
            # Add a random codeword
            new_code = set(current_code)
            new_code.add(random.choice(all_indices))
        elif move_type < 0.66:
            # Remove a random codeword
            if not current_code:
                continue
            new_code = set(current_code)
            to_remove = random.choice(list(current_code))
            new_code.discard(to_remove)
        else:
            # Swap: remove one, add one
            if not current_code:
                continue
            new_code = set(current_code)
            to_remove = random.choice(list(current_code))
            new_code.discard(to_remove)
            new_code.add(random.choice(all_indices))
        
        # Evaluate new state
        new_obj = objective(new_code)
        
        # Acceptance criterion
        delta_obj = new_obj - current_obj
        if delta_obj < 0 or (T > 0 and random.random() < math.exp(-delta_obj / T)):
            current_code = new_code
            current_obj = new_obj
            
            # Try to remove redundant codewords
            current_code = remove_redundant(current_code)
            current_obj = objective(current_code)
            
            # Update best if valid and smaller
            counts_check = coverage_counts(current_code)
            if is_covered(counts_check) and len(current_code) < best_size:
                best_code = set(current_code)
                best_size = len(current_code)
        
        step += 1
        T *= cooling_rate
        if T < T_end:
            T = T_end
    
    # Final local search: 1-swap to improve quality (increase min coverage)
    # Then remove redundant again.
    # Do a few rounds of swap + reduce.
    for round_num in range(20):
        if time.time() > deadline - 5:
            break
        
        improved = False
        code_list = sorted(best_code)
        counts = coverage_counts(best_code)
        
        for i, ci in enumerate(code_list):
            # Try swapping ci with a random non-codeword
            for _ in range(5):  # Try 5 random swaps per codeword
                if time.time() > deadline - 5:
                    break
                candidate = random.choice(all_indices)
                if candidate in best_code:
                    continue
                
                new_code = set(best_code)
                new_code.discard(ci)
                new_code.add(candidate)
                
                new_counts = coverage_counts(new_code)
                if not is_covered(new_counts):
                    continue
                
                # Check if this swap allows further reduction
                new_code_reduced = remove_redundant(new_code)
                if len(new_code_reduced) < len(best_code):
                    best_code = new_code_reduced
                    best_size = len(best_code)
                    improved = True
                    break
                elif len(new_code_reduced) == len(best_code):
                    # Check if min coverage increased
                    old_min = min(counts[w] for w in range(total) if counts[w] > 0) if any(counts[w] > 0 for w in range(total)) else 0
                    new_min = min(new_counts[w] for w in range(total) if new_counts[w] > 0) if any(new_counts[w] > 0 for w in range(total)) else 0
                    if new_min > old_min:
                        best_code = new_code_reduced
                        improved = True
                        break
        
        if not improved:
            break
    
    # Final verification
    final_counts = coverage_counts(best_code)
    if not is_covered(final_counts):
        # Fallback: greedy repair
        uncov = [i for i, c in enumerate(final_counts) if c == 0]
        uncov_set = set(uncov)
        while uncov_set:
            best_c = -1
            best_cnt = -1
            for c in all_indices:
                cnt = sum(1 for w in spheres[c] if w in uncov_set)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_c = c
            if best_c == -1 or best_cnt == 0:
                break
            best_code.add(best_c)
            for w in spheres[best_c]:
                uncov_set.discard(w)
        best_code = remove_redundant(best_code)
    
    return [list(words[i]) for i in sorted(best_code)]
