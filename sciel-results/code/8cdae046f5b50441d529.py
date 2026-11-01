"""
Combined program: FFT-based scoring with heuristic-guided local search.
Uses the efficiency of FFT from the current program and the search guidance
from the inspiration to maximize the sumset/difference-set exponent.
"""
import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 145s.
    deadline = start_time + 145.0
    
    best_score = 0.0
    best_set = [0, 1, 3]
    
    def compute_score(U):
        """
        Computes the score using FFT for sumset and difference set sizes.
        U: list of non-negative integers containing 0.
        """
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1.0
        max_u = max(U)
        if max_u == 0:
            return 0.0
        
        # Create indicator vector
        # Size needs to be at least 2*max_u + 1 for sums (0..2*max_u)
        size = 1
        while size < 2 * max_u + 1:
            size *= 2
            
        # Indicator vector
        vec = np.zeros(size)
        for x in U:
            vec[x] = 1.0
            
        # Sumset: Convolution of vec with vec
        fft_vec = np.fft.rfft(vec)
        conv = np.fft.irfft(fft_vec * fft_vec, n=size)
        
        # The sumset is supported on [0, 2*max_u].
        # Due to floating point errors, we threshold.
        sum_counts = conv[:2*max_u + 1]
        n_sum = np.sum(sum_counts > 0.5)
        
        # Difference set: Autocorrelation
        ac = np.fft.irfft(fft_vec * np.conj(fft_vec), n=size)
        
        # Count non-zero entries in ac[1..max_u] for positive differences
        pos_diffs = ac[1:max_u+1]
        n_pos_diffs = np.sum(pos_diffs > 0.5)
        n_diff = 1 + 2 * n_pos_diffs
        
        if n_sum == 0:
            return -1.0
            
        base = 2 * max_u + 1
        if base <= 1:
            return 0.0
        
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        
        # Time check
        if time.time() > deadline - 2.0:
            return

        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Phase 1: Test known good structured sets
    # More Golomb rulers (optimal or near-optimal)
    grs = [
        [0, 1, 4, 10],
        [0, 1, 4, 10, 18],
        [0, 1, 4, 10, 18, 23],
        [0, 1, 4, 10, 18, 23, 31],
        [0, 1, 4, 10, 18, 23, 31, 37],
        [0, 1, 4, 10, 18, 23, 31, 37, 41],
        [0, 1, 4, 10, 18, 23, 31, 37, 41, 52],
        [0, 1, 4, 10, 18, 23, 31, 37, 41, 52, 58],
        [0, 1, 4, 10, 18, 23, 31, 37, 41, 52, 58, 60],
        [0, 1, 4, 10, 18, 23, 31, 37, 41, 52, 58, 60, 72],
        [0, 1, 4, 10, 18, 23, 31, 37, 41, 52, 58, 60, 72, 84],
    ]
    for gr in grs:
        try_set(gr)
        
    # Small structured sets
    for i in range(1, 8):
        for j in range(i+1, 12):
            for k in range(j+1, 16):
                if time.time() > deadline - 10.0:
                    break
                try_set([0, i, j, k])
                
    # Phase 2: Randomized Local Search
    num_starts = 25
    max_steps_per_start = 40
    
    # Generate seeds
    seeds = []
    
    # 1. Random sets of various sizes
    for size in [4, 5, 6, 8, 10, 12, 15]:
        for _ in range(3):
            max_u = random.randint(10, 150)
            elems = random.sample(range(1, max_u + 1), size - 1)
            seeds.append(sorted([0] + elems))
            
    # 2. Golomb rulers
    seeds.extend(grs)
    
    # 3. Powers of 2
    for base in [2, 3, 4]:
        p = [0, 1]
        while p[-1] * base < 150:
            p.append(p[-1] * base)
        seeds.append(p)
        
    # 4. Arithmetic progressions
    for d in range(1, 12):
        ap = list(range(0, 60, d))
        seeds.append(ap)
        
    # 5. "Sparse" random sets (larger range)
    for _ in range(5):
        size = random.randint(8, 15)
        max_u = random.randint(100, 300)
        elems = random.sample(range(1, max_u + 1), size - 1)
        seeds.append(sorted([0] + elems))

    random.shuffle(seeds)
    
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
            
        current = list(seed)
        try_set(current)
        
        # Local search
        no_improve = 0
        max_no_improve = 25
        
        while no_improve < max_no_improve and time.time() < deadline - 5.0:
            improved = False
            current_set = set(current)
            max_cur = max(current)
            
            # 1. Try replacing an element with a nearby value
            if len(current) >= 3:
                for i in range(1, len(current)): # Don't replace 0
                    orig = current[i]
                    # Try a range of new values
                    search_range = 8 + len(current)
                    for new_el in range(max(1, orig - search_range), orig + search_range + 1):
                        if new_el in current_set:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        
                        s = compute_score(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            current_set = set(current)
                            max_cur = max(current)
                            improved = True
                            break
                    if improved:
                        no_improve = 0
                        break
                        
            # 2. Try adding an element
            if len(current) < 20: # Limit size to keep FFT fast
                candidates = set()
                
                # Extend range
                for ext in range(1, 12):
                    candidates.add(max_cur + ext)
                
                # Near existing elements
                for x in current:
                    for dx in [-2, -1, 1, 2]:
                        if x + dx > 0 and x + dx not in current_set:
                            candidates.add(x + dx)
                
                # Random elements in a wider range
                for _ in range(8):
                    candidates.add(random.randint(1, max_cur + 40))
                
                for new_el in sorted(candidates):
                    if new_el in current_set:
                        continue
                    cand = sorted(current + [new_el])
                    
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        current_set = set(current)
                        max_cur = max(current)
                        improved = True
                        break
                        
                if improved:
                    no_improve = 0
                    continue

            # 3. Try removing an element
            if len(current) > 3:
                for i in range(1, len(current)):
                    cand = [x for j, x in enumerate(current) if j != i]
                    if not cand or 0 not in cand:
                        continue
                    
                    s = compute_score(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        current_set = set(current)
                        max_cur = max(current)
                        improved = True
                        break
                        
                if improved:
                    no_improve = 0
                    continue

            if not improved:
                no_improve += 1
            else:
                no_improve = 0
                
    # Phase 3: Final polish - exhaustive local search around best_set
    if time.time() < deadline - 3.0:
        current = list(best_set)
        current_set = set(current)
        max_cur = max(current)
        
        # Do several passes of local search
        for _ in range(3):
            if time.time() > deadline - 2.0:
                break
            improved = False
            no_improve = 0
            max_no_improve = 15
            
            while no_improve < max_no_improve and time.time() < deadline - 2.0:
                improved = False
                current_set = set(current)
                max_cur = max(current)
                
                # Try all single-element replacements
                if len(current) >= 3:
                    for i in range(1, len(current)):
                        orig = current[i]
                        search_range = 15
                        for new_el in range(max(1, orig - search_range), orig + search_range + 1):
                            if new_el in current_set:
                                continue
                            cand = list(current)
                            cand[i] = new_el
                            cand = sorted(set(cand))
                            if 0 not in cand:
                                continue
                            
                            s = compute_score(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)
                                current_set = set(current)
                                max_cur = max(current)
                                improved = True
                                break
                        if improved:
                            no_improve = 0
                            break
                            
                if not improved:
                    # Try additions
                    if len(current) < 20:
                        candidates = set()
                        for ext in range(1, 15):
                            candidates.add(max_cur + ext)
                        for x in current:
                            for dx in [-3, -2, -1, 1, 2, 3]:
                                if x + dx > 0 and x + dx not in current_set:
                                    candidates.add(x + dx)
                        
                        for new_el in sorted(candidates):
                            if new_el in current_set:
                                continue
                            cand = sorted(current + [new_el])
                            s = compute_score(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)
                                current_set = set(current)
                                max_cur = max(current)
                                improved = True
                                break
                                
                if not improved:
                    # Try removals
                    if len(current) > 3:
                        for i in range(1, len(current)):
                            cand = [x for j, x in enumerate(current) if j != i]
                            if not cand or 0 not in cand:
                                continue
                            s = compute_score(cand)
                            if s > best_score:
                                best_score = s
                                best_set = list(cand)
                                current = list(cand)
                                current_set = set(current)
                                max_cur = max(current)
                                improved = True
                                break
                                
                if not improved:
                    no_improve += 1
                else:
                    no_improve = 0
                    
    return list(best_set)

if __name__ == "__main__":
    # Quick self-test
    t0 = time.time()
    U = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Set: {U}")
    print(f"Score: {compute_score(U)}")
