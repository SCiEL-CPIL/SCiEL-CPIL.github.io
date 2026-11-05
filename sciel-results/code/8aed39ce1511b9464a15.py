import time
import itertools
import random
import numpy as np
from scipy.optimize import linprog

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    total_words = q ** n
    
    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)
        
    # Precompute neighbors (balls)
    neighbors = [set() for _ in range(total_words)]
    for i in range(total_words):
        d = digits_of[i]
        neigh = {i}
        
        for r in range(1, radius + 1):
            for pos_comb in itertools.combinations(range(n), r):
                for variation in itertools.product(range(q), repeat=r):
                    new_d = list(d)
                    is_diff = False
                    for k, pos in enumerate(pos_comb):
                        new_val = variation[k]
                        if new_val != d[pos]:
                            is_diff = True
                        new_d[pos] = new_val
                    if not is_diff:
                        continue
                    idx = 0
                    for j in range(n):
                        idx = idx * q + new_d[j]
                    neigh.add(idx)
        neighbors[i] = neigh

    # Convert neighbors to numpy arrays for faster operations
    neighbor_arrays = [np.array(list(neigh)) for neigh in neighbors]
    
    # Greedy initialization to get a good starting point
    uncovered = set(range(total_words))
    initial_code = []
    while uncovered and time.time() < deadline - 60.0:
        best_idx = -1
        max_cover = 0
        for i in range(total_words):
            if i in initial_code:
                continue
            cnt = len(neighbors[i] & uncovered)
            if cnt > max_cover:
                max_cover = cnt
                best_idx = i
        if max_cover == 0:
            break
        initial_code.append(best_idx)
        uncovered.difference_update(neighbors[best_idx])
    
    if not initial_code:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    best_code = initial_code[:]
    best_size = len(best_code)
    
    # Branch and Bound with symmetry breaking
    # We use a simplified B&B that tries to remove codewords
    # Lower bound: greedy lower bound based on maximum coverage
    
    def greedy_lower_bound(uncovered_set, available_codewords):
        if not uncovered_set:
            return 0
        # Simple greedy lower bound
        lb = 0
        remaining = set(uncovered_set)
        while remaining:
            best_cover = 0
            for cw in available_codewords:
                if cw in remaining:
                    cover = len(neighbors[cw] & remaining)
                    if cover > best_cover:
                        best_cover = cover
            if best_cover == 0:
                return lb + len(remaining)
            lb += 1
            # Remove the best covered set (approximation)
            # For simplicity, just reduce remaining
            remaining = set(random.sample(list(remaining), max(1, len(remaining) - best_cover)))
        return lb
    
    # Local search to improve initial code
    current_code = initial_code[:]
    current_size = len(current_code)
    
    # Try to remove codewords
    for _ in range(100):
        if time.time() > deadline - 10.0:
            break
        removed_any = False
        for i in range(len(current_code)):
            candidate = current_code[i]
            # Check if removing candidate still covers all
            temp_code = current_code[:i] + current_code[i+1:]
            covered = set()
            for cw in temp_code:
                covered.update(neighbors[cw])
            if len(covered) == total_words:
                current_code = temp_code
                current_size -= 1
                removed_any = True
                break
        if not removed_any:
            break
    
    if current_size < best_size:
        best_size = current_size
        best_code = current_code[:]
    
    # Try to swap codewords for better coverage
    for _ in range(50):
        if time.time() > deadline - 5.0:
            break
        improved = False
        for i in range(len(best_code)):
            if time.time() > deadline - 5.0:
                break
            # Try to replace best_code[i] with a random other codeword
            for _ in range(10):
                new_cw = random.randint(0, total_words - 1)
                if new_cw in best_code:
                    continue
                new_code = best_code[:i] + [new_cw] + best_code[i+1:]
                covered = set()
                for cw in new_code:
                    covered.update(neighbors[cw])
                if len(covered) == total_words:
                    # Check if this allows removal of another codeword
                    for j in range(len(new_code)):
                        if j == i:
                            continue
                        temp = new_code[:j] + new_code[j+1:]
                        temp_covered = set()
                        for cw in temp:
                            temp_covered.update(neighbors[cw])
                        if len(temp_covered) == total_words:
                            best_code = temp
                            best_size -= 1
                            improved = True
                            break
                    if improved:
                        break
        if not improved:
            break
    
    return [list(digits_of[i]) for i in best_code]
