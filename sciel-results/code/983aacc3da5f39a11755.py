"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

_N = 256

# Precompute all 4-term arithmetic progressions as flat index arrays
_AP_A = []
_AP_B = []
_AP_C = []
_AP_D = []
for d in range(1, _N // 3 + 1):
    for a in range(_N - 3 * d):
        _AP_A.append(a)
        _AP_B.append(a + d)
        _AP_C.append(a + 2 * d)
        _AP_D.append(a + 3 * d)

_AP_A = np.array(_AP_A, dtype=np.int32)
_AP_B = np.array(_AP_B, dtype=np.int32)
_AP_C = np.array(_AP_C, dtype=np.int32)
_AP_D = np.array(_AP_D, dtype=np.int32)

def _score(arr):
    """
    arr: numpy array of 0/1 of length 256.
    Returns the count of monochromatic 4-term APs.
    """
    ca = arr[_AP_A]
    cb = arr[_AP_B]
    cc = arr[_AP_C]
    cd = arr[_AP_D]
    mask = (ca == cb) & (cb == cc) & (cc == cd)
    return np.sum(mask)

def _evaluate(colours_list):
    """
    colours_list: list of 256 values in {0,1}
    Returns the count of monochromatic 4-term APs.
    """
    arr = np.array(colours_list, dtype=np.int8)
    return _score(arr)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    def get_time_left():
        return deadline - time.time()

    # 1. Initialization: Try various structured and random colorings
    best_colours = None
    best_score = float('inf')
    
    def gen_random(seed):
        random.seed(seed)
        return np.array([random.randint(0, 1) for _ in range(_N)], dtype=np.int8)

    def gen_block(block_size):
        return np.array([(i // block_size) % 2 for i in range(_N)], dtype=np.int8)

    def gen_thue_morse():
        return np.array([bin(i).count('1') % 2 for i in range(_N)], dtype=np.int8)

    def gen_parity():
        return np.array([i % 2 for i in range(_N)], dtype=np.int8)

    candidates = []
    
    # Structured
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 64, 128]:
        candidates.append(gen_block(bs))
    
    candidates.append(gen_thue_morse())
    candidates.append(gen_parity())
    
    # Random seeds
    for s in range(20):
        candidates.append(gen_random(s))
        
    # Also try some "low density" or "high density" biases
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        random.seed(100 + int(p*100))
        candidates.append(np.array([1 if random.random() < p else 0 for _ in range(_N)], dtype=np.int8))

    # Evaluate all candidates
    for cand in candidates:
        score = _score(cand)
        if score < best_score:
            best_score = score
            best_colours = cand.copy()
            
    # 2. Local Search (First Descent)
    current = best_colours.copy()
    current_score = best_score
    
    improved = True
    pass_count = 0
    while improved and get_time_left() > 15.0:
        improved = False
        indices = list(range(_N))
        random.shuffle(indices)
        
        for i in indices:
            current[i] = 1 - current[i]
            new_score = _score(current)
            
            if new_score < current_score:
                current_score = new_score
                improved = True
                if new_score < best_score:
                    best_score = new_score
                    best_colours = current.copy()
            else:
                current[i] = 1 - current[i]
        
        pass_count += 1
        
        # If stuck, try a small random perturbation (kick)
        if not improved:
            kicks = 10
            for _ in range(kicks):
                if get_time_left() < 15.0: break
                trial = current.copy()
                num_flips = random.randint(2, 5)
                flip_idx = random.sample(range(_N), num_flips)
                for idx in flip_idx:
                    trial[idx] = 1 - trial[idx]
                
                trial_score = _score(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    if trial_score < best_score:
                        best_score = trial_score
                        best_colours = current.copy()
                    improved = True
                    break

    # 3. Simulated Annealing
    if get_time_left() > 10.0:
        current = best_colours.copy()
        current_score = best_score
        T = 5.0
        cooling = 0.99995
        
        while get_time_left() > 5.0:
            i = random.randint(0, _N - 1)
            current[i] = 1 - current[i]
            new_score = _score(current)
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 0.01)):
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_colours = current.copy()
            else:
                current[i] = 1 - current[i]
                
            T *= cooling
            if T < 0.01:
                T = 0.01

    # Final Hill Climbing
    if get_time_left() > 1.0:
        arr = best_colours.copy()
        current_score = best_score
        improved = True
        while improved and get_time_left() > 0.5:
            improved = False
            indices = list(range(_N))
            random.shuffle(indices)
            for i in indices:
                arr[i] = 1 - arr[i]
                new_score = _score(arr)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_colours = arr.copy()
                else:
                    arr[i] = 1 - arr[i]

    return best_colours.tolist()
# EVOLVE-BLOCK-END
