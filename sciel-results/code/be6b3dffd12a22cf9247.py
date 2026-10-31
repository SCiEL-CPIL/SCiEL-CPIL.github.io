"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START

def _compute_score_fast(U):
    """Compute the score for set U using FFT for efficiency."""
    U = sorted(set(U))
    if not U or 0 not in U:
        return -1
    max_u = max(U)
    if max_u == 0:
        return 0
    n = len(U)
    
    if n < 12:
        # Direct computation for small sets
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        n_sum = len(sumset)
        n_diff = len(diffset)
    else:
        # FFT-based
        # Sumset: |U+U|
        len_sum = 2 * max_u + 1
        fft_size = 1
        while fft_size < 2 * len_sum:
            fft_size *= 2
        f = np.zeros(fft_size, dtype=np.float64)
        for u in U:
            f[u] = 1.0
        F = np.fft.fft(f)
        conv = np.fft.ifft(F * F).real
        n_sum = int(np.sum(conv > 0.5))
        
        # Difference set: |U-U|
        # U-U ranges from -max_u to max_u
        # Use convolution with reversed set
        f_rev = np.zeros(fft_size, dtype=np.float64)
        for u in U:
            f_rev[max_u - u] = 1.0
        F_rev = np.fft.fft(f_rev)
        conv_diff = np.fft.ifft(F * F_rev).real
        n_diff = int(np.sum(conv_diff > 0.5))
    
    if n_sum == 0:
        return -1
    base = 2 * max_u + 1
    if base <= 1:
        return 0
    ratio = n_diff / n_sum
    if ratio <= 1:
        return 1.0
    return 1 + math.log(ratio) / math.log(base)


def _local_search(U, time_limit, seed=42):
    """Local search with add/remove/swap moves to maximize score."""
    rng = random.Random(seed)
    U = sorted(set(U))
    if 0 not in U:
        U.append(0)
        U = sorted(U)
    current_score = _compute_score_fast(U)
    best_score = current_score
    best_set = list(U)
    
    deadline = time.time() + time_limit
    no_improve = 0
    max_no_improve = 50
    
    while no_improve < max_no_improve and time.time() < deadline:
        improved = False
        max_cur = max(U) if U else 0
        n = len(U)
        
        # Generate candidate moves
        moves = []
        
        # Add moves
        if n < 30:
            add_candidates = set()
            for el in U:
                for d in range(-4, 5):
                    new_el = el + d
                    if new_el > 0 and new_el not in U:
                        add_candidates.add(new_el)
            for new_el in range(max_cur + 1, max_cur + 15):
                add_candidates.add(new_el)
            if max_cur > 0:
                for mult in [2, 3]:
                    add_candidates.add(max_cur * mult)
            # Random additions
            for _ in range(10):
                add_candidates.add(rng.randint(1, max_cur + 30))
            
            for new_el in sorted(add_candidates):
                moves.append(('add', new_el, None))
        
        # Remove moves
        if n > 4:
            for i in range(1, n):  # don't remove 0
                moves.append(('remove', None, U[i]))
        
        # Swap moves
        if n >= 3:
            for i in range(1, n):
                old_el = U[i]
                for new_el in range(max(1, old_el - 8), min(max_cur + 12, old_el + 9)):
                    if new_el != old_el and new_el not in U:
                        moves.append(('swap', new_el, old_el))
        
        # Shuffle moves for diversity
        rng.shuffle(moves)
        
        for move in moves:
            if time.time() > deadline:
                break
            mtype, new_el, old_el = move
            if mtype == 'add':
                cand = sorted(U + [new_el])
            elif mtype == 'remove':
                cand = [x for x in U if x != old_el]
            else:  # swap
                cand = sorted(set([x if x != old_el else new_el for x in U]))
            
            if not cand or 0 not in cand:
                continue
            
            s = _compute_score_fast(cand)
            if s > current_score + 1e-12:
                U = cand
                current_score = s
                if s > best_score:
                    best_score = s
                    best_set = list(U)
                improved = True
                break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
    
    return best_set, best_score


def _golomb_ruler(size, max_val=None):
    """Construct a Golomb ruler of given size."""
    if max_val is None:
        max_val = size * size
    gr = [0, 1]
    used_diffs = {1}
    for _ in range(size - 2):
        best_next = -1
        for candidate in range(gr[-1] + 1, max_val + 1):
            valid = True
            for existing in gr:
                d = candidate - existing
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                best_next = candidate
                break
        if best_next == -1:
            break
        gr.append(best_next)
        for existing in gr[:-1]:
            used_diffs.add(gr[-1] - existing)
    return gr


def _construct_seed_sets():
    """Generate diverse seed sets for local search."""
    seeds = []
    
    # Known Golomb rulers
    known_gr = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 10, 12],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 25, 34],
        [0, 1, 4, 10, 18, 23, 25, 34, 42],
        [0, 1, 4, 10, 18, 23, 25, 34, 42, 52],
    ]
    for gr in known_gr:
        seeds.append(list(gr))
    
    # Powers of 2
    for base in [2, 3]:
        for size in range(4, 12):
            seeds.append([base ** i for i in range(size)])
    
    # Geometric sequences
    for ratio in [1.5, 2.0, 2.5, 3.0]:
        for size in range(4, 15):
            seq = [0]
            val = 1
            for _ in range(size - 1):
                seq.append(int(val))
                val *= ratio
            seeds.append(seq)
    
    # Sparse random sets
    rng = random.Random(99)
    for _ in range(20):
        size = rng.randint(5, 20)
        max_u = rng.randint(20, 200)
        seeds.append([0] + sorted(rng.sample(range(1, max_u + 1), min(size - 1, max_u))))
    
    # Golomb rulers of various sizes
    for size in range(4, 14):
        gr = _golomb_ruler(size, max_val=size * size)
        if len(gr) == size:
            seeds.append(gr)
    
    return seeds


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_score = 0
    best_set = [0, 1, 3]
    
    # Phase 1: Evaluate all seed sets
    seeds = _construct_seed_sets()
    
    for seed in seeds:
        if time.time() > deadline - 30:
            break
        seed = sorted(set(seed))
        if not seed or 0 not in seed:
            continue
        s = _compute_score_fast(seed)
        if s > best_score:
            best_score = s
            best_set = list(seed)
    
    # Phase 2: Local search from top seeds
    # Sort seeds by score and search from the best ones
    seed_scores = []
    for seed in seeds:
        seed = sorted(set(seed))
        if not seed or 0 not in seed:
            continue
        s = _compute_score_fast(seed)
        seed_scores.append((s, seed))
    
    seed_scores.sort(reverse=True)
    
    # Search from top 10 seeds
    search_budget = deadline - time.time() - 10
    if search_budget > 5:
        per_seed_time = max(1.0, search_budget / 10)
        
        for i, (s, seed) in enumerate(seed_scores[:10]):
            if time.time() > deadline - 10:
                break
            remaining = min(per_seed_time, deadline - time.time() - 5)
            if remaining < 0.5:
                break
            
            result_set, result_score = _local_search(seed, remaining, seed=100 + i)
            if result_score > best_score:
                best_score = result_score
                best_set = result_set
    
    # Phase 3: Additional random restarts with local search
    if time.time() < deadline - 15:
        rng = random.Random(777)
        remaining_time = deadline - time.time() - 10
        if remaining_time > 5:
            num_restarts = min(30, int(remaining_time / 3))
            per_restart = remaining_time / num_restarts
            
            for r in range(num_restarts):
                if time.time() > deadline - 10:
                    break
                
                # Random structured initialization
                init_type = rng.randint(0, 4)
                
                if init_type == 0:
                    size = rng.randint(5, 15)
                    gr = _golomb_ruler(size, max_val=size * size + 20)
                    if len(gr) >= 3:
                        cand = gr
                    else:
                        cand = [0, 1, 3]
                elif init_type == 1:
                    size = rng.randint(5, 15)
                    ratio = rng.uniform(1.3, 3.0)
                    cand = [0]
                    val = rng.uniform(0.5, 2.0)
                    for _ in range(size - 1):
                        cand.append(int(val))
                        val *= ratio
                elif init_type == 2:
                    size = rng.randint(5, 25)
                    max_u = rng.randint(20, 300)
                    cand = [0] + sorted(rng.sample(range(1, max_u + 1), min(size - 1, max_u)))
                elif init_type == 3:
                    size = rng.randint(4, 12)
                    base = rng.choice([2, 3, 4, 5])
                    cand = [base ** i for i in range(size)]
                else:
                    # Perturbed Golomb ruler
                    size = rng.randint(4, 12)
                    gr = _golomb_ruler(size, max_val=size * size + 10)
                    if len(gr) >= 3:
                        cand = list(gr)
                        # Perturb some elements
                        for _ in range(rng.randint(1, 3)):
                            idx = rng.randint(1, len(cand) - 1)
                            cand[idx] += rng.randint(-3, 3)
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            cand.append(0)
                        cand = sorted(cand)
                    else:
                        cand = [0, 1, 3]
                
                cand = sorted(set(cand))
                if len(cand) < 3 or 0 not in cand:
                    continue
                
                s = _compute_score_fast(cand)
                if s > best_score:
                    best_score = s
                    best_set = list(cand)
                
                # Quick local search
                if s > best_score - 0.02 and time.time() < deadline - 10:
                    t_budget = min(per_restart, deadline - time.time() - 5)
                    if t_budget > 0.5:
                        result_set, result_score = _local_search(cand, t_budget, seed=200 + r)
                        if result_score > best_score:
                            best_score = result_score
                            best_set = result_set
    
    # Final: ensure best_set is valid
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set.append(0)
        best_set = sorted(best_set)
    
    return best_set

# EVOLVE-BLOCK-END
