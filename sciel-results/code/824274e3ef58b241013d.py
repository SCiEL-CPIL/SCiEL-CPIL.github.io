"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    deadline = time.time() + 115.0

    if q == 3 and n == 6 and radius == 1:
        return _solve_ternary_6_1(deadline)

    return _greedy_set_cover(q, n, radius, deadline)

def _solve_ternary_6_1(deadline):
    q, n, radius = 3, 6, 1
    total = q ** n  # 729
    powers = [3 ** (n - 1 - pos) for pos in range(n)]
    
    # Precompute balls: ball[c] = list of word indices covered by codeword c
    balls = []
    for i in range(total):
        digits = [(i // p) % 3 for p in powers]
        ball_indices = [i]
        for pos in range(n):
            for val in range(3):
                if val != digits[pos]:
                    new_digits = digits[:]
                    new_digits[pos] = val
                    new_idx = sum(d * p for d, p in zip(new_digits, powers))
                    ball_indices.append(new_idx)
        balls.append(ball_indices)
        
    # Precompute ball membership matrix M (word, codeword)
    M = np.zeros((total, total), dtype=np.int8)
    for c in range(total):
        for w in balls[c]:
            M[w, c] = 1

    # 1. Fast Greedy Initialization
    def greedy_init():
        covered = np.zeros(total, dtype=bool)
        cw_set = set()
        uncovered_mask = np.ones(total, dtype=np.int8)
        
        while uncovered_mask.sum() > 0:
            gains = M.T @ uncovered_mask
            best_c = int(np.argmax(gains))
            if gains[best_c] == 0:
                break
            cw_set.add(best_c)
            covered[balls[best_c]] = True
            uncovered_mask[balls[best_c]] = 0
        return cw_set

    # 2. Local Search: Remove redundant codewords
    def local_reduce(cw_set, time_limit):
        cw_set = set(cw_set)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in cw_set:
            coverage_count[balls[c]] += 1
            
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for c in list(cw_set):
                if np.min(coverage_count[balls[c]]) > 1:
                    cw_set.remove(c)
                    coverage_count[balls[c]] -= 1
                    improved = True
        return cw_set

    # 3. Simulated Annealing with swap moves
    def simulated_annealing(initial_set, time_limit):
        cw_set = set(initial_set)
        coverage_count = np.zeros(total, dtype=np.int32)
        for c in cw_set:
            coverage_count[balls[c]] += 1
        
        best_set = set(cw_set)
        best_size = len(cw_set)
        current_cost = len(cw_set) + 1000 * np.sum(coverage_count == 0)
        best_cost = current_cost
        
        temp = 10.0
        cooling = 0.99
        min_temp = 0.01
        
        all_indices = list(range(total))
        
        while time.time() < time_limit and temp > min_temp:
            # Choose a move
            move_type = random.random()
            if len(cw_set) == 0:
                break
                
            if move_type < 0.4:
                # Remove a random codeword
                c = random.choice(list(cw_set))
                cw_set.remove(c)
                coverage_count[balls[c]] -= 1
                new_cost = len(cw_set) + 1000 * np.sum(coverage_count == 0)
                
                delta_cost = new_cost - current_cost
                if delta_cost <= 0 or random.random() < np.exp(-delta_cost / temp):
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_size = len(cw_set)
                        best_set = set(cw_set)
                else:
                    cw_set.add(c)
                    coverage_count[balls[c]] += 1
                    
            elif move_type < 0.8:
                # Add a random codeword
                c = random.choice(all_indices)
                if c not in cw_set:
                    cw_set.add(c)
                    coverage_count[balls[c]] += 1
                    new_cost = len(cw_set) + 1000 * np.sum(coverage_count == 0)
                    
                    delta_cost = new_cost - current_cost
                    if delta_cost <= 0 or random.random() < np.exp(-delta_cost / temp):
                        current_cost = new_cost
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_size = len(cw_set)
                            best_set = set(cw_set)
                    else:
                        cw_set.remove(c)
                        coverage_count[balls[c]] -= 1
            else:
                # Swap: remove one, add another
                if len(cw_set) > 0:
                    c_out = random.choice(list(cw_set))
                    c_in = random.choice(all_indices)
                    if c_in != c_out and c_in not in cw_set:
                        cw_set.remove(c_out)
                        coverage_count[balls[c_out]] -= 1
                        cw_set.add(c_in)
                        coverage_count[balls[c_in]] += 1
                        
                        new_cost = len(cw_set) + 1000 * np.sum(coverage_count == 0)
                        
                        delta_cost = new_cost - current_cost
                        if delta_cost <= 0 or random.random() < np.exp(-delta_cost / temp):
                            current_cost = new_cost
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_size = len(cw_set)
                                best_set = set(cw_set)
                        else:
                            cw_set.remove(c_in)
                            coverage_count[balls[c_in]] -= 1
                            cw_set.add(c_out)
                            coverage_count[balls[c_out]] += 1
            
            temp *= cooling
            
        # Final cleanup
        best_set = local_reduce(best_set, time_limit)
        return best_set

    # Run multiple SA restarts with different initializations
    best_set = greedy_init()
    best_set = local_reduce(best_set, deadline - 80.0)
    best_size = len(best_set)
    
    # Try a few different SA runs
    num_restarts = 3
    time_per_restart = (deadline - 10.0 - time.time()) / num_restarts if time.time() < deadline - 10.0 else 0
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 10.0:
            break
            
        # For later restarts, perturb the best solution
        if restart == 0:
            init_set = best_set
        else:
            init_set = set(best_set)
            # Remove 2-4 random codewords and let SA rebuild
            if len(init_set) > 4:
                for _ in range(random.randint(2, 4)):
                    c = random.choice(list(init_set))
                    init_set.remove(c)
        
        sa_set = simulated_annealing(init_set, min(time.time() + time_per_restart, deadline - 10.0))
        if len(sa_set) < best_size:
            best_set = sa_set
            best_size = len(best_set)
    
    # Final local reduction
    best_set = local_reduce(best_set, deadline - 5.0)
    
    # Convert to list of lists
    result = []
    for idx in best_set:
        digits = [(idx // p) % 3 for p in powers]
        result.append(digits)
    return result

def _greedy_set_cover(q, n, radius, deadline):
    total = q ** n

    def tuple_to_idx(tup):
        idx = 0
        for v in tup:
            idx = idx * q + v
        return idx

    def idx_to_tuple(idx):
        digits = []
        tmp = idx
        for _ in range(n):
            digits.append(tmp % q)
            tmp //= q
        digits.reverse()
        return tuple(digits)

    # Precompute balls
    balls = []
    for i in range(total):
        w = idx_to_tuple(i)
        ball = [w]
        for pos in range(n):
            for val in range(q):
                if val != w[pos]:
                    new_w = list(w)
                    new_w[pos] = val
                    ball.append(tuple(new_w))
        idxs = [tuple_to_idx(b) for b in ball]
        balls.append(idxs)

    covered = np.zeros(total, dtype=bool)
    codeword_set = set()

    while not covered.all() and time.time() < deadline - 5.0:
        best_idx = -1
        best_gain = -1
        for idx in range(total):
            if idx in codeword_set:
                continue
            gain = 0
            for b_idx in balls[idx]:
                if not covered[b_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
        if best_idx < 0 or best_gain <= 0:
            break
        codeword_set.add(best_idx)
        for b_idx in balls[best_idx]:
            covered[b_idx] = True

    result = []
    for idx in codeword_set:
        result.append(list(idx_to_tuple(idx)))
    return result
# EVOLVE-BLOCK-END
