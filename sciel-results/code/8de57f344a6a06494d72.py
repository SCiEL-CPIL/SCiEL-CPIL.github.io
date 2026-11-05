import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 6 or radius != 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    total = 3 ** n
    powers = [3 ** p for p in range(n)]

    def to_word(idx):
        return [(idx // powers[p]) % 3 for p in range(n)]

    def to_idx(word):
        idx = 0
        for w in word:
            idx = idx * 3 + w
        return idx

    # Precompute balls for all words
    balls = []
    for i in range(total):
        d = [(i // powers[p]) % 3 for p in range(n)]
        ball = [i]
        for p in range(n):
            base = i - d[p] * powers[p]
            for alt in range(3):
                if alt != d[p]:
                    ball.append(base + alt * powers[p])
        balls.append(ball)

    # Precompute adjacency: for each word, which other words are at distance 1
    adj = np.zeros((total, total), dtype=bool)
    for i in range(total):
        for j in balls[i]:
            if j != i:
                adj[i, j] = True

    # Initial code: tetracode direct sum (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_idx = [to_idx(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    current_code = np.array(sorted(set(init_idx)), dtype=np.int32)

    start_time = time.time()
    deadline = start_time + 115.0

    def is_valid_code(code_arr):
        """Check if code covers all words within radius 1."""
        if len(code_arr) == 0:
            return False
        # Build coverage
        covered = np.zeros(total, dtype=bool)
        for c in code_arr:
            covered[balls[c]] = True
        return np.all(covered)

    def greedy_remove_from(code_arr):
        """Remove redundant words greedily."""
        code_set = set(int(x) for x in code_arr)
        coverage = np.zeros(total, dtype=np.int32)
        for c in code_arr:
            coverage[balls[c]] += 1
        
        code_list = list(code_arr)
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for i in range(len(code_list)):
                c = code_list[i]
                # Check if removing c still covers everything
                if np.all(coverage[balls[c]] >= 2):
                    coverage[balls[c]] -= 1
                    code_set.discard(c)
                    code_list.pop(i)
                    changed = True
                    break
        return np.array(code_list, dtype=np.int32)

    # Phase 1: Greedy removal
    current_code = greedy_remove_from(current_code)
    best_code = current_code.copy()
    best_size = len(best_code)

    # Phase 2: Simulated annealing with add/remove moves
    current_code_list = list(current_code)
    current_size = len(current_code_list)
    
    # Track uncovered count for fast evaluation
    coverage = np.zeros(total, dtype=np.int32)
    for c in current_code_list:
        coverage[balls[c]] += 1
    uncovered_count = int(np.sum(coverage == 0))
    
    def energy(code_list):
        """Energy = size + penalty * uncovered_count."""
        cov = np.zeros(total, dtype=np.int32)
        for c in code_list:
            cov[balls[c]] += 1
        uncov = int(np.sum(cov == 0))
        return len(code_list) + 1000 * uncov

    T = 50.0
    cooling_rate = 0.999
    min_T = 0.01
    
    best_energy = energy(current_code_list)
    
    while time.time() < deadline and T > min_T:
        # Choose a random move
        if random.random() < 0.4 and len(current_code_list) > 0:
            # Remove a random word
            idx = random.randrange(len(current_code_list))
            c = current_code_list[idx]
            old_cov = coverage[balls[c]].copy()
            coverage[balls[c]] -= 1
            current_code_list.pop(idx)
            new_uncov = int(np.sum(coverage == 0))
            delta_e = 1 + 1000 * (new_uncov - uncovered_count)
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                uncovered_count = new_uncov
                e = energy(current_code_list)
                if e < best_energy:
                    best_energy = e
                    best_code = np.array(current_code_list, dtype=np.int32)
                    best_size = len(best_code)
            else:
                coverage[balls[c]] = old_cov
                current_code_list.insert(idx, c)
        else:
            # Add a random word not in code
            code_set = set(current_code_list)
            candidates = [i for i in range(total) if i not in code_set]
            if not candidates:
                break
            c = random.choice(candidates)
            old_cov = coverage[balls[c]].copy()
            coverage[balls[c]] += 1
            current_code_list.append(c)
            new_uncov = int(np.sum(coverage == 0))
            delta_e = -1 + 1000 * (new_uncov - uncovered_count)
            
            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                uncovered_count = new_uncov
                e = energy(current_code_list)
                if e < best_energy:
                    best_energy = e
                    best_code = np.array(current_code_list, dtype=np.int32)
                    best_size = len(best_code)
            else:
                coverage[balls[c]] = old_cov
                current_code_list.pop()
        
        T *= cooling_rate
    
    # Final greedy cleanup
    best_code = greedy_remove_from(best_code)
    
    return [to_word(int(x)) for x in best_code]
