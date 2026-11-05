import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    if q != 3 or n != 7 or radius != 1:
        # Fallback for other parameters
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]
    
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s safety margin
    
    total_words = 3**n
    
    # Precompute base-3 digits for all words
    all_digits = np.zeros((total_words, n), dtype=np.int32)
    for i in range(total_words):
        temp = i
        for j in range(n-1, -1, -1):
            all_digits[i, j] = temp % 3
            temp //= 3
    
    # Precompute neighbor indices for each word
    # Each word has 1 + n*(q-1) = 1 + 7*2 = 15 neighbors
    neighbors = np.zeros((total_words, 15), dtype=np.int32)
    for i in range(total_words):
        neighbors[i, 0] = i
        idx = 1
        for pos in range(n):
            orig_val = all_digits[i, pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                # Compute the difference in integer representation
                # Since we're using base-3, changing digit at position pos by delta
                # means changing the integer by delta * 3^(n-1-pos)
                power = 3 ** (n - 1 - pos)
                delta = new_val - orig_val
                neighbors[i, idx] = i + delta * power
                idx += 1
    
    # Initial solution: tetracode product (243 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_codewords = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            word = h + list(tail)
            val = 0
            for d in word:
                val = val * 3 + d
            initial_codewords.append(val)
    
    # Coverage tracking
    coverage = np.zeros(total_words, dtype=np.int32)
    for c in initial_codewords:
        coverage[neighbors[c]] += 1
    
    current_codewords = set(initial_codewords)
    best_size = len(current_codewords)
    best_codewords = list(current_codewords)
    
    # Helper functions
    def is_valid():
        return np.all(coverage >= 1)
    
    def try_remove(c):
        """Try to remove codeword c. Returns True if successful."""
        coverage[neighbors[c]] -= 1
        if is_valid():
            current_codewords.remove(c)
            return True
        else:
            coverage[neighbors[c]] += 1
            return False
    
    def try_swap(c1, c2):
        """Try to swap c1 for c2. Returns True if successful."""
        if c1 == c2:
            return False
        coverage[neighbors[c1]] -= 1
        coverage[neighbors[c2]] += 1
        if is_valid():
            current_codewords.remove(c1)
            current_codewords.add(c2)
            return True
        else:
            coverage[neighbors[c1]] += 1
            coverage[neighbors[c2]] -= 1
            return False
    
    # Phase 1: Greedy deletion - try to remove words in random order
    random.seed(42)
    no_improve = 0
    max_no_improve = 200
    
    while time.time() < deadline - 30:  # Reserve 30s for ILS phase
        improved = False
        current_list = list(current_codewords)
        random.shuffle(current_list)
        
        for c in current_list:
            if time.time() >= deadline - 30:
                break
            if c not in current_codewords:
                continue
            if try_remove(c):
                improved = True
                if len(current_codewords) < best_size:
                    best_size = len(current_codewords)
                    best_codewords = list(current_codewords)
        
        if not improved:
            no_improve += 1
            if no_improve > max_no_improve:
                break
        else:
            no_improve = 0
    
    # Phase 2: ILS (Iterated Local Search) - perturb and repair
    while time.time() < deadline:
        current_size = len(current_codewords)
        
        # Perturbation: remove k random codewords and add k random new codewords
        k = random.randint(1, 3)
        perturbed = set(current_codewords)
        
        # Remove k random codewords
        to_remove = random.sample(list(perturbed), min(k, len(perturbed)))
        for c in to_remove:
            perturbed.remove(c)
        
        # Add k random codewords not in perturbed set
        to_add = []
        while len(to_add) < k:
            c = random.randint(0, total_words - 1)
            if c not in perturbed:
                to_add.append(c)
                perturbed.add(c)
        
        # Check if perturbed solution is valid
        coverage_pert = np.zeros(total_words, dtype=np.int32)
        for c in perturbed:
            coverage_pert[neighbors[c]] += 1
        
        if not is_valid_perturbed(coverage_pert):
            # Repair: add codewords until valid
            uncovered = np.where(coverage_pert == 0)[0]
            while len(uncovered) > 0:
                if time.time() >= deadline:
                    break
                # Pick a random uncovered word
                target = random.choice(uncovered)
                # Find a codeword that covers it
                # Try all words that are neighbors of target
                candidates = neighbors[target]
                random.shuffle(candidates)
                for c in candidates:
                    if c not in perturbed:
                        coverage_pert[neighbors[c]] += 1
                        perturbed.add(c)
                        uncovered = np.where(coverage_pert == 0)[0]
                        break
                else:
                    # No single addition fixes it, pick random
                    c = random.randint(0, total_words - 1)
                    if c not in perturbed:
                        coverage_pert[neighbors[c]] += 1
                        perturbed.add(c)
                        uncovered = np.where(coverage_pert == 0)[0]
        
        # If perturbed solution is valid, try to improve it via deletion
        if len(perturbed) < best_size or random.random() < 0.3:
            # Update current solution
            for c in current_codewords - perturbed:
                coverage[neighbors[c]] -= 1
            for c in perturbed - current_codewords:
                coverage[neighbors[c]] += 1
            current_codewords = set(perturbed)
            
            # Try to improve via deletion
            improved = True
            while improved and time.time() < deadline:
                improved = False
                current_list = list(current_codewords)
                random.shuffle(current_list)
                for c in current_list:
                    if time.time() >= deadline:
                        break
                    if c not in current_codewords:
                        continue
                    if try_remove(c):
                        improved = True
                        if len(current_codewords) < best_size:
                            best_size = len(current_codewords)
                            best_codewords = list(current_codewords)
    
    # Convert best_codewords back to list of lists
    result = []
    for c in best_codewords:
        digits = all_digits[c].tolist()
        result.append(digits)
    
    return result

def is_valid_perturbed(coverage_pert):
    return np.all(coverage_pert >= 1)
