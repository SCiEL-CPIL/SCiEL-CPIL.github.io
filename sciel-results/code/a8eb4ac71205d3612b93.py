"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
from itertools import combinations
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit
    
    best_score = 0
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score_fft(U):
        """Use FFT to compute sumset and difference set sizes in O(M log M)."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        # Sumset: convolution of indicator with itself
        # U+U has elements in [0, 2*max_u]
        # Create indicator polynomial
        size_sum = 2 * max_u + 1
        # Find next power of 2 for FFT
        fft_size = 1
        while fft_size < 2 * size_sum:
            fft_size *= 2
        
        # Indicator vector for U
        ind = np.zeros(fft_size)
        for x in U:
            ind[x] = 1.0
        
        # Convolve ind with itself
        f = np.fft.rfft(ind)
        conv = np.fft.irfft(f * f, n=fft_size)
        
        # Sumset size = number of positions where conv > 0
        n_sum = int(np.sum(conv[:size_sum] > 0.5))
        
        # Difference set: U-U has elements in [-max_u, max_u]
        # |U-U| = number of distinct differences
        # We can compute this by: for each d in [-max_u, max_u], check if exists a,b in U with a-b=d
        # Equivalently, convolve ind with reversed ind
        # rev_ind[x] = 1 if (max_u - x) in U, i.e., shift by max_u
        # Then (ind * rev_ind)[k] counts pairs (a,b) with a + (max_u - b) = k, i.e., a - b = k - max_u
        rev_ind = np.zeros(fft_size)
        for x in U:
            rev_ind[max_u - x] = 1.0
        
        f_rev = np.fft.rfft(rev_ind)
        conv_diff = np.fft.irfft(f * f_rev, n=fft_size)
        
        # Differences range from -max_u to max_u, which maps to indices 0 to 2*max_u in conv_diff
        # diff = k - max_u, so k ranges from 0 to 2*max_u
        n_diff = int(np.sum(conv_diff[:2 * max_u + 1] > 0.5))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    def compute_score(U):
        """Fallback for very small sets or as verification."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        
        sumset = set()
        diffset = set()
        for a in U:
            for b in U:
                sumset.add(a + b)
                diffset.add(a - b)
        
        n_sum = len(sumset)
        n_diff = len(diffset)
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score
    
    # Pre-compute seed
    best_score = compute_score_fft([0, 1, 3])
    
    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score_fft(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)
    
    # Phase 1: Exhaustive search for small sets (size <= 6)
    max_u_limit_small = 20
    for max_u in range(1, max_u_limit_small + 1):
        if time.time() > deadline - 5:
            break
        elements = list(range(1, max_u + 1))
        for size in range(3, 7):
            if len(elements) < size:
                continue
            for comb in combinations(elements, size):
                cand = sorted([0] + list(comb))
                try_set(cand)
    
    # Phase 2: Structured sets with larger max_u
    for max_u in range(10, 500):
        if time.time() > deadline - 5:
            break
        
        # Two blocks
        for a in range(1, 15):
            for b in range(a + 1, min(max_u - a + 1, a + 50)):
                if time.time() > deadline - 5:
                    break
                block1 = list(range(0, a + 1))
                block2 = list(range(b, b + a + 1))
                cand = sorted(set(block1 + block2))
                if max(cand) <= max_u:
                    try_set(cand)
        
        # Geometric with small ratios
        for r in [2, 3, 4, 5, 6, 7, 8]:
            geo = [0, 1]
            while geo[-1] * r <= max_u:
                geo.append(geo[-1] * r)
            try_set(geo)
            
        # Arithmetic progressions
        for d in range(1, 20):
            ap = []
            v = 0
            while v <= max_u:
                ap.append(v)
                v += d
            try_set(ap)
            
        # Mixed: {0, 1, ..., a} U {b}
        for a in range(1, 15):
            for b in range(a + 1, max_u + 1):
                cand = sorted(list(range(0, a + 1)) + [b])
                try_set(cand)
                
        # Mixed: {0, 1, ..., a} U {b, c}
        for a in range(1, 8):
            for b in range(a + 1, max_u + 1):
                for c in range(b + 1, max_u + 1):
                    cand = sorted(list(range(0, a + 1)) + [b, c])
                    try_set(cand)
                    
    # Phase 3: Local search around best set
    if time.time() < deadline - 10:
        current = list(best_set)
        no_improve = 0
        max_no_improve = 100
        
        while time.time() < deadline - 10 and no_improve < max_no_improve:
            improved = False
            
            # Try adding an element
            if len(current) < 20:
                max_cur = max(current)
                candidates = set()
                for x in current:
                    for dx in [-3, -2, -1, 1, 2, 3]:
                        candidates.add(x + dx)
                candidates.update(range(max_cur + 1, max_cur + 20))
                candidates.discard(0)
                
                for new_el in sorted(candidates):
                    if new_el in current or new_el < 0:
                        continue
                    cand = sorted(current + [new_el])
                    s = compute_score_fft(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            # Try replacing an element
            if len(current) >= 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    for new_el in range(max(1, current[i] - 10), current[i] + 11):
                        if new_el in current:
                            continue
                        cand = list(current)
                        cand[i] = new_el
                        cand = sorted(set(cand))
                        if 0 not in cand:
                            continue
                        s = compute_score_fft(cand)
                        if s > best_score:
                            best_score = s
                            best_set = list(cand)
                            current = list(cand)
                            improved = True
                            break
                    if improved:
                        break
            
            # Try removing an element
            if len(current) > 3:
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    cand = [x for j, x in enumerate(current) if j != i]
                    s = compute_score_fft(cand)
                    if s > best_score:
                        best_score = s
                        best_set = list(cand)
                        current = list(cand)
                        improved = True
                        break
            
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
    
    return list(best_set)
# EVOLVE-BLOCK-END
