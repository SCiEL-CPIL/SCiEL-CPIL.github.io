"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from itertools import product

# Precompute all 4-term arithmetic progressions
_N = 256
_AP_INDICES = []
for d in range(1, _N // 3 + 1):
    for a in range(0, _N - 3 * d):
        _AP_INDICES.append((a, a + d, a + 2 * d, a + 3 * d))

# Convert to numpy arrays for fast vectorized access
_AP_A = np.array([t[0] for t in _AP_INDICES], dtype=np.int32)
_AP_B = np.array([t[1] for t in _AP_INDICES], dtype=np.int32)
_AP_C = np.array([t[2] for t in _AP_INDICES], dtype=np.int32)
_AP_D = np.array([t[3] for t in _AP_INDICES], dtype=np.int32)

def _evaluate_fast(arr):
    """
    arr: numpy array of 0/1 of length 256.
    Returns the count of monochromatic 4-term APs.
    """
    # Get colors at all AP positions
    ca = arr[_AP_A]
    cb = arr[_AP_B]
    cc = arr[_AP_C]
    cd = arr[_AP_D]
    
    # Check if all four are equal
    # (ca == cb) & (cb == cc) & (cc == cd)
    mask = (ca == cb) & (cb == cc) & (cc == cd)
    return np.sum(mask)

def _evaluate(colours_list):
    """
    colours_list: list of 256 values in {0,1}
    Returns the count of monochromatic 4-term APs.
    """
    arr = np.array(colours_list, dtype=np.int8)
    return _evaluate_fast(arr)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def get_time_left():
        return deadline - time.time()

    # Helper to evaluate numpy array directly for speed in loops
    def eval_np(arr):
        return _evaluate_fast(arr)

    # 1. Initialization: Try various structured and random colorings
    best_colours = None
    best_score = float('inf')
    
    # Candidate generators
    def gen_random(seed):
        random.seed(seed)
        return np.array([random.randint(0, 1) for _ in range(256)], dtype=np.int8)

    def gen_block(block_size):
        return np.array([(i // block_size) % 2 for i in range(256)], dtype=np.int8)

    def gen_thue_morse():
        # Thue-Morse sequence: number of 1s in binary representation mod 2
        return np.array([bin(i).count('1') % 2 for i in range(256)], dtype=np.int8)

    def gen_parity():
        return np.array([i % 2 for i in range(256)], dtype=np.int8)

    candidates = []
    
    # Structured
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 64, 128]:
        candidates.append(gen_block(bs))
    
    candidates.append(gen_thue_morse())
    candidates.append(gen_parity())
    
    # Random seeds
    for s in range(40):
        candidates.append(gen_random(s))
        
    # Also try some "low density" or "high density" biases
    for p in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        random.seed(100 + int(p*100))
        candidates.append(np.array([1 if random.random() < p else 0 for _ in range(256)], dtype=np.int8))

    # Evaluate all candidates
    for cand in candidates:
        score = eval_np(cand)
        if score < best_score:
            best_score = score
            best_colours = cand.copy()
            
    # 2. Local Search (First Descent) with delta-based evaluation
    current = best_colours.copy()
    current_score = best_score
    
    # Precompute for each position, which APs contain it
    # ap_positions[i] = list of AP indices that include position i
    n_aps = len(_AP_A)
    ap_positions = [[] for _ in range(256)]
    for ap_idx in range(n_aps):
        for pos in (_AP_A[ap_idx], _AP_B[ap_idx], _AP_C[ap_idx], _AP_D[ap_idx]):
            ap_positions[pos].append(ap_idx)
    
    # Convert to numpy arrays for faster iteration
    ap_pos_arrays = [np.array(ap_positions[i], dtype=np.int32) for i in range(256)]
    
    # Count how many of each color in each AP: ap_color_count[ap_idx, color]
    # ap_color_count[ap_idx] = [count_0, count_1]
    ap_color_count = np.zeros((n_aps, 2), dtype=np.int32)
    
    def update_ap_counts(arr, ap_idx, pos, old_color):
        """Update AP counts when position pos changes from old_color to 1-old_color"""
        new_color = 1 - old_color
        ap_color_count[ap_idx, old_color] -= 1
        ap_color_count[ap_idx, new_color] += 1
    
    def init_ap_counts(arr):
        """Initialize AP color counts from a coloring array"""
        ap_color_count[:] = 0
        for ap_idx in range(n_aps):
            for pos in (_AP_A[ap_idx], _AP_B[ap_idx], _AP_C[ap_idx], _AP_D[ap_idx]):
                ap_color_count[ap_idx, arr[pos]] += 1
    
    def compute_score_from_counts():
        """Compute score from AP color counts: AP is monochromatic if all 4 same"""
        # Monochromatic if count_0==4 or count_1==4
        mono = (ap_color_count[:, 0] == 4) | (ap_color_count[:, 1] == 4)
        return int(np.sum(mono))
    
    # Initialize AP counts
    init_ap_counts(current)
    
    improved = True
    pass_count = 0
    while improved and get_time_left() > 8.0:
        improved = False
        # Randomize order to avoid cycles
        indices = list(range(256))
        random.shuffle(indices)
        
        for i in indices:
            old_color = current[i]
            new_color = 1 - old_color
            
            # Compute delta: how does flipping position i affect the score?
            # For each AP containing i, check if it was monochromatic before and after
            delta = 0
            for ap_idx in ap_pos_arrays[i]:
                # Before flip
                was_mono = (ap_color_count[ap_idx, 0] == 4) or (ap_color_count[ap_idx, 1] == 4)
                # After flip
                new_count_0 = ap_color_count[ap_idx, 0] - (1 if old_color == 0 else 0) + (1 if new_color == 0 else 0)
                new_count_1 = ap_color_count[ap_idx, 1] - (1 if old_color == 1 else 0) + (1 if new_color == 1 else 0)
                is_mono_after = (new_count_0 == 4) or (new_count_1 == 4)
                delta += (1 if is_mono_after else 0) - (1 if was_mono else 0)
            
            if delta < 0:
                # Accept: update state
                current[i] = new_color
                for ap_idx in ap_pos_arrays[i]:
                    update_ap_counts(current, ap_idx, i, old_color)
                current_score += delta
                improved = True
                if current_score < best_score:
                    best_score = current_score
                    best_colours = current.copy()
            # If delta >= 0, don't flip (strict descent)
        
        pass_count += 1
        
        # If stuck, try a small random perturbation (kick)
        if not improved:
            # Try flipping 2-5 random bits to escape local minimum
            kicks = 10
            for _ in range(kicks):
                if get_time_left() < 8.0: break
                trial = current.copy()
                num_flips = random.randint(2, 5)
                flip_idx = random.sample(range(256), num_flips)
                for idx in flip_idx:
                    trial[idx] = 1 - trial[idx]
                
                trial_score = eval_np(trial)
                if trial_score < current_score:
                    current = trial
                    current_score = trial_score
                    init_ap_counts(current)
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_colours = current.copy()
                    break

    # 3. Simulated Annealing (if time permits)
    if get_time_left() > 15.0:
        current = best_colours.copy()
        current_score = best_score
        init_ap_counts(current)
        temp = 8.0
        cooling = 0.997
        
        while get_time_left() > 8.0 and temp > 0.01:
            i = random.randint(0, 255)
            old_color = current[i]
            new_color = 1 - old_color
            
            # Compute delta
            delta = 0
            for ap_idx in ap_pos_arrays[i]:
                was_mono = (ap_color_count[ap_idx, 0] == 4) or (ap_color_count[ap_idx, 1] == 4)
                new_count_0 = ap_color_count[ap_idx, 0] - (1 if old_color == 0 else 0) + (1 if new_color == 0 else 0)
                new_count_1 = ap_color_count[ap_idx, 1] - (1 if old_color == 1 else 0) + (1 if new_color == 1 else 0)
                is_mono_after = (new_count_0 == 4) or (new_count_1 == 4)
                delta += (1 if is_mono_after else 0) - (1 if was_mono else 0)
            
            if delta < 0 or random.random() < np.exp(-delta / max(temp, 1e-6)):
                current[i] = new_color
                for ap_idx in ap_pos_arrays[i]:
                    update_ap_counts(current, ap_idx, i, old_color)
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colours = current.copy()
            else:
                # Don't flip
                pass
            
            temp *= cooling
            
            # Final polish after SA
            if get_time_left() < 10.0:
                 # Run a few more local search passes with delta-based eval
                 improved = True
                 while improved and get_time_left() > 2.0:
                     improved = False
                     indices = list(range(256))
                     random.shuffle(indices)
                     for i in indices:
                         old_color = current[i]
                         new_color = 1 - old_color
                         
                         delta = 0
                         for ap_idx in ap_pos_arrays[i]:
                             was_mono = (ap_color_count[ap_idx, 0] == 4) or (ap_color_count[ap_idx, 1] == 4)
                             new_count_0 = ap_color_count[ap_idx, 0] - (1 if old_color == 0 else 0) + (1 if new_color == 0 else 0)
                             new_count_1 = ap_color_count[ap_idx, 1] - (1 if old_color == 1 else 0) + (1 if new_color == 1 else 0)
                             is_mono_after = (new_count_0 == 4) or (new_count_1 == 4)
                             delta += (1 if is_mono_after else 0) - (1 if was_mono else 0)
                         
                         if delta < 0:
                             current[i] = new_color
                             for ap_idx in ap_pos_arrays[i]:
                                 update_ap_counts(current, ap_idx, i, old_color)
                             current_score += delta
                             improved = True
                             if current_score < best_score:
                                 best_score = current_score
                                 best_colours = current.copy()

    # Final verification
    final_score = eval_np(best_colours)
    if final_score < best_score:
        best_score = final_score

    return best_colours.tolist()
# EVOLVE-BLOCK-END
