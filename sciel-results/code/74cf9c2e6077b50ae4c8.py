"""Optimized construction for sumset/difference-set exponent lower bound."""
import time
import numpy as np
from math import log, log2

_START_TIME = time.time()
_DEADLINE = 150.0  # seconds, safety margin before 150s kill

def _remaining():
    return _DEADLINE - (time.time() - _START_TIME)

def _fft_set_size(conv_result):
    """Count non-zero entries in a convolution result (set cardinality)."""
    return np.count_nonzero(conv_result > 0.5)

def _compute_scores(U, max_val):
    """Compute |U-U| and |U+U| using FFT. Returns (diff_size, sum_size, score)."""
    n = max_val + 1
    # FFT size needs to be at least 2*n-1 for convolution
    fft_size = 1
    while fft_size < 2 * n - 1:
        fft_size *= 2
    
    # Create indicator vector
    vec = np.zeros(fft_size, dtype=np.float64)
    vec[U] = 1.0
    
    # FFT of vector
    F = np.fft.fft(vec)
    
    # Sumset: U+U = U * U (convolution)
    sum_fft = F * F
    sum_conv = np.fft.ifft(sum_fft).real
    sum_size = _fft_set_size(sum_conv)
    
    # Difference set: U-U = U * U_flipped
    # U-U = {a-b : a,b in U} = {a+b' : a in U, b' in -U}
    # Flip: vec_flipped[i] = vec[n-1-i] for i in 0..n-1
    vec_flipped = np.zeros(fft_size, dtype=np.float64)
    vec_flipped[:n] = vec[n-1::-1]  # reverse
    F_flip = np.fft.fft(vec_flipped)
    diff_fft = F * F_flip
    diff_conv = np.fft.ifft(diff_fft).real
    diff_size = _fft_set_size(diff_conv)
    
    if sum_size == 0:
        return 0, 0, 0.0
    
    ratio = diff_size / sum_size
    if max_val <= 0:
        return diff_size, sum_size, 1.0
    
    score = 1.0 + log(ratio) / log(2 * max_val + 1)
    return diff_size, sum_size, score

def _score_of(U):
    """Compute the score for a given set U."""
    U = sorted(set(U))
    if not U or U[0] != 0:
        U = [0] + U
    max_val = max(U)
    if max_val <= 0:
        return 1.0
    _, _, score = _compute_scores(U, max_val)
    return score

def _score_of_fast(U, max_val):
    """Fast score computation assuming U is already a sorted list with 0."""
    if max_val <= 0:
        return 1.0
    _, _, score = _compute_scores(U, max_val)
    return score

def construct():
    """Construct a set U to maximize the sumset/difference-set exponent lower bound."""
    best_set = [0, 1, 3]
    best_score = _score_of(best_set)
    
    # Start with a reasonable initial set
    # Use a geometric progression as starting point
    def make_initial_set(n_elements, max_val):
        """Create an initial set with geometric spacing."""
        if n_elements <= 1:
            return [0]
        # Geometric progression
        ratio = (max_val / (n_elements - 1)) ** (1.0 / max(1, n_elements - 2)) if n_elements > 2 else max_val
        vals = set()
        vals.add(0)
        current = 1
        for i in range(1, n_elements):
            vals.add(int(round(current)))
            current *= ratio
        vals.add(max_val)
        return sorted([v for v in vals if 0 <= v <= max_val])
    
    # Simulated annealing
    import random
    random.seed(42)
    
    # Parameters
    current_temp = 10.0
    cooling_rate = 0.995
    min_temp = 0.001
    
    # Start with a moderate-sized set
    current_set = make_initial_set(50, 2000)
    current_max = max(current_set)
    current_score = _score_of_fast(current_set, current_max)
    
    if current_score > best_score:
        best_score = current_score
        best_set = list(current_set)
    
    iteration = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while _remaining() > 5.0 and current_temp > min_temp:
        iteration += 1
        
        # Choose a random perturbation
        op = random.random()
        
        new_set = list(current_set)
        new_max = current_max
        
        if op < 0.3:
            # Add a random element
            if len(new_set) < 5000:
                new_val = random.randint(1, max(1, current_max + 100))
                if new_val not in new_set:
                    new_set.append(new_val)
                    new_set.sort()
                    new_max = max(new_max, new_val)
        elif op < 0.5:
            # Remove a random element (keep 0)
            if len(new_set) > 2:
                removable = [i for i in range(1, len(new_set))]
                if removable:
                    idx = random.choice(removable)
                    new_set.pop(idx)
                    new_max = max(new_set) if new_set else 0
        elif op < 0.7:
            # Move a random element
            if len(new_set) > 1:
                idx = random.randint(1, len(new_set) - 1)
                new_val = random.randint(1, max(1, current_max + 100))
                if new_val not in new_set:
                    new_set[idx] = new_val
                    new_set.sort()
                    new_max = max(new_max, new_val)
        elif op < 0.85:
            # Scale the set
            scale = random.uniform(0.7, 1.3)
            new_set = sorted(set([0] + [min(int(round(v * scale)), 100000) for v in new_set]))
            new_max = max(new_set) if new_set else 0
        else:
            # Add multiple elements at once
            num_add = random.randint(2, 10)
            for _ in range(num_add):
                if len(new_set) < 5000:
                    new_val = random.randint(1, max(1, current_max + 200))
                    if new_val not in new_set:
                        new_set.append(new_val)
            new_set.sort()
            new_max = max(new_set) if new_set else 0
        
        # Evaluate new set
        if not new_set or new_set[0] != 0:
            new_set = [0] + new_set
            new_set.sort()
            new_max = max(new_set)
        
        new_score = _score_of_fast(new_set, new_max)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta > 0 or random.random() < np.exp(delta / current_temp):
            current_set = new_set
            current_max = new_max
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        current_temp *= cooling_rate
        
        # Restart if stuck
        if no_improve_count > max_no_improve:
            # Try a fresh random start
            n_elems = random.randint(20, 200)
            max_val = random.randint(500, 50000)
            current_set = make_initial_set(n_elems, max_val)
            current_max = max(current_set)
            current_score = _score_of_fast(current_set, current_max)
            current_temp = 10.0
            no_improve_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_set = list(current_set)
    
    # Final polish: try local optimizations on the best set
    if _remaining() > 3.0:
        polish_set = list(best_set)
        polish_max = max(polish_set)
        polish_score = best_score
        
        # Try adding/removing elements near existing ones
        for _ in range(100):
            if _remaining() < 2.0:
                break
            improved = False
            for i in range(len(polish_set)):
                if i == 0:
                    continue
                # Try removing this element
                trial = polish_set[:i] + polish_set[i+1:]
                if len(trial) < 2:
                    continue
                trial_max = max(trial)
                trial_score = _score_of_fast(trial, trial_max)
                if trial_score > polish_score + 1e-10:
                    polish_set = trial
                    polish_max = trial_max
                    polish_score = trial_score
                    improved = True
                    break
            
            if not improved:
                # Try adding elements
                for _ in range(10):
                    if _remaining() < 2.0:
                        break
                    candidate = random.randint(1, max(1, polish_max + 50))
                    if candidate not in polish_set:
                        trial = polish_set + [candidate]
                        trial.sort()
                        trial_max = max(trial)
                        trial_score = _score_of_fast(trial, trial_max)
                        if trial_score > polish_score + 1e-10:
                            polish_set = trial
                            polish_max = trial_max
                            polish_score = trial_score
                            break
        
        if polish_score > best_score:
            best_score = polish_score
            best_set = polish_set
    
    return best_set
