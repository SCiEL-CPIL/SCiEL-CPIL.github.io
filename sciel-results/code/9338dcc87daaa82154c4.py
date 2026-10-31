"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START

def _score_of(U):
    """Compute the score for a set U using FFT."""
    if not U or 0 not in U:
        return 0.0
    U = sorted(U)
    n = len(U)
    max_u = max(U)
    
    # Compute |U+U| using FFT
    # Convolution of indicator vector with itself
    size = 2 * max_u + 1
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < 2 * max_u + 1:
        fft_size *= 2
    
    indicator = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        indicator[u] = 1.0
    
    # FFT convolution for sumset
    f = np.fft.rfft(indicator)
    conv = np.fft.irfft(f * f, n=fft_size)
    sumset_size = int(np.sum(conv >= 0.5))  # Count non-zero entries
    
    # Compute |U-U| using FFT
    # Difference set: convolve indicator with reversed indicator
    indicator_rev = np.zeros(fft_size, dtype=np.float64)
    for u in U:
        indicator_rev[max_u - u] = 1.0
    
    f_rev = np.fft.rfft(indicator_rev)
    diff_conv = np.fft.irfft(f * f_rev, n=fft_size)
    diffset_size = int(np.sum(diff_conv >= 0.5))
    
    if sumset_size == 0 or diffset_size == 0:
        return 0.0
    
    # Score: 1 + log(|U-U|/|U+U|)/log(2*max(U)+1)
    log_ratio = np.log(diffset_size / sumset_size)
    log_denom = np.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 1.0
    
    score = 1.0 + log_ratio / log_denom
    return score


def _score_of_fast(U):
    """Faster score computation using set operations for small sets."""
    if not U or 0 not in U:
        return 0.0
    U_set = set(U)
    n = len(U)
    max_u = max(U)
    
    # Sumset
    sumset = set()
    for a in U:
        for b in U:
            sumset.add(a + b)
    sumset_size = len(sumset)
    
    # Difference set
    diffset = set()
    for a in U:
        for b in U:
            diffset.add(a - b)
    diffset_size = len(diffset)
    
    if sumset_size == 0 or diffset_size == 0:
        return 0.0
    
    log_ratio = np.log(diffset_size / sumset_size)
    log_denom = np.log(2 * max_u + 1)
    
    if log_denom <= 0:
        return 1.0
    
    score = 1.0 + log_ratio / log_denom
    return score


def _construct_golomb_ruler(n):
    """Construct a Golomb ruler-like set of size n."""
    if n <= 0:
        return [0]
    if n == 1:
        return [0]
    
    # Start with a simple arithmetic progression-like construction
    # Use a greedy approach to create a set with good difference properties
    U = [0]
    current_max = 0
    
    for i in range(1, n):
        # Try to place next element to maximize difference set diversity
        # Use a heuristic: place at position that creates new differences
        best_pos = None
        best_diffs = -1
        
        # Try positions in a range
        max_search = current_max + i * 3 + 10
        for pos in range(1, max_search + 1):
            if pos in U:
                continue
            U_test = U + [pos]
            # Compute number of distinct differences
            diffs = set()
            for a in U_test:
                for b in U_test:
                    diffs.add(a - b)
            n_diffs = len(diffs)
            if n_diffs > best_diffs:
                best_diffs = n_diffs
                best_pos = pos
        
        if best_pos is not None:
            U.append(best_pos)
            current_max = max(current_max, best_pos)
    
    return sorted(U)


def _local_search(U, max_time=10.0):
    """Perform local search to improve the set."""
    start_time = time.time()
    U = list(U)
    current_score = _score_of_fast(U)
    
    n = len(U)
    max_u = max(U) if U else 0
    
    # Define neighborhood operations
    def try_add(U, pos):
        if pos in U:
            return None
        return U + [pos]
    
    def try_remove(U, idx):
        if len(U) <= 1:
            return None
        return U[:idx] + U[idx+1:]
    
    def try_move(U, idx, new_pos):
        if new_pos in U:
            return None
        U_new = U[:]
        U_new[idx] = new_pos
        return sorted(U_new)
    
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() - start_time < max_time:
        improved = False
        n = len(U)
        
        # Try adding a new element
        if n < 200:
            # Try adding near existing elements or at larger positions
            candidates = []
            for u in U:
                candidates.extend([u + 1, u + 2, u + 3, u + 5, u + 8])
            candidates.extend([max_u + 1, max_u + 2, max_u + 5, max_u + 10])
            random.shuffle(candidates)
            
            for pos in candidates[:20]:
                if pos < 0 or pos in U:
                    continue
                U_new = U + [pos]
                s = _score_of_fast(U_new)
                if s > current_score + 1e-10:
                    U = U_new
                    current_score = s
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try removing an element
        if len(U) > 3:
            indices = list(range(len(U)))
            random.shuffle(indices)
            for idx in indices[:10]:
                if idx == 0 and U[0] == 0:
                    continue  # Keep 0
                U_new = U[:idx] + U[idx+1:]
                s = _score_of_fast(U_new)
                if s > current_score + 1e-10:
                    U = U_new
                    current_score = s
                    improved = True
                    no_improve_count = 0
                    break
        
        # Try moving an element
        if len(U) > 2:
            indices = [i for i in range(len(U)) if U[i] != 0]
            random.shuffle(indices)
            for idx in indices[:10]:
                old_pos = U[idx]
                # Try nearby positions
                for delta in [-3, -2, -1, 1, 2, 3, 5, 8]:
                    new_pos = old_pos + delta
                    if new_pos < 0 or new_pos in U:
                        continue
                    U_new = U[:]
                    U_new[idx] = new_pos
                    U_new = sorted(U_new)
                    s = _score_of_fast(U_new)
                    if s > current_score + 1e-10:
                        U = U_new
                        current_score = s
                        improved = True
                        no_improve_count = 0
                        break
                if improved:
                    break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    return sorted(U), current_score


def _construct_large_set(n, time_budget=30.0):
    """Construct a large set using FFT-based evaluation."""
    start_time = time.time()
    
    # Start with a Golomb ruler-like construction
    U = _construct_golomb_ruler(n)
    
    # Refine with local search using FFT for larger sets
    if len(U) > 30:
        # Use FFT-based scoring
        current_score = _score_of(U)
        
        # Simple hill climbing with FFT
        no_improve = 0
        max_no_improve = 30
        
        while time.time() - start_time < time_budget:
            improved = False
            n = len(U)
            max_u = max(U)
            
            # Try adding elements
            if n < 500:
                candidates = []
                for u in U:
                    candidates.extend([u + 1, u + 2, u + 3])
                candidates.extend([max_u + 1, max_u + 2, max_u + 5])
                random.shuffle(candidates)
                
                for pos in candidates[:15]:
                    if pos < 0 or pos in U:
                        continue
                    U_new = U + [pos]
                    s = _score_of(U_new)
                    if s > current_score + 1e-10:
                        U = U_new
                        current_score = s
                        improved = True
                        no_improve = 0
                        break
            
            # Try removing elements
            if len(U) > 10:
                indices = [i for i in range(len(U)) if U[i] != 0]
                random.shuffle(indices)
                for idx in indices[:10]:
                    U_new = U[:idx] + U[idx+1:]
                    s = _score_of(U_new)
                    if s > current_score + 1e-10:
                        U = U_new
                        current_score = s
                        improved = True
                        no_improve = 0
                        break
            
            if not improved:
                no_improve += 1
                if no_improve >= max_no_improve:
                    break
        
        return sorted(U), current_score
    
    return sorted(U), _score_of_fast(U)


def construct():
    """Main construct function with deadline guard."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    random.seed(42)
    
    best_U = [0, 1, 3]
    best_score = _score_of_fast(best_U)
    
    # Try different set sizes
    sizes_to_try = [5, 10, 15, 20, 30, 40, 50]
    
    for n in sizes_to_try:
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            
            # Construct initial set
            if n <= 30:
                U = _construct_golomb_ruler(n)
            else:
                U = _construct_golomb_ruler(min(n, 30))
                # Extend with random elements
                while len(U) < n:
                    pos = random.randint(1, max(U) + 20)
                    if pos not in U:
                        U.append(pos)
                U = sorted(U)
            
            # Local search
            remaining_time = deadline - time.time() - 5.0  # 5s safety
            if remaining_time > 5.0:
                U, score = _local_search(U, max_time=remaining_time)
            else:
                score = _score_of_fast(U)
            
            if score > best_score:
                best_score = score
                best_U = U
        
        # Try large set construction with FFT
        if n >= 30 and time.time() < deadline - 35.0:
            time_budget = min(30.0, deadline - time.time() - 10.0)
            if time_budget > 10.0:
                U_large, score_large = _construct_large_set(n, time_budget=time_budget)
                if score_large > best_score:
                    best_score = score_large
                    best_U = U_large
    
    # Final polish: try to improve the best solution
    if time.time() < deadline - 10.0:
        remaining = deadline - time.time() - 5.0
        if remaining > 5.0:
            if len(best_U) > 30:
                best_U, best_score = _construct_large_set(len(best_U), time_budget=remaining)
            else:
                best_U, best_score = _local_search(best_U, max_time=remaining)
    
    # Ensure 0 is in the set and all elements are nonnegative
    best_U = sorted(set(u for u in best_U if u >= 0))
    if 0 not in best_U:
        best_U.insert(0, 0)
    
    return best_U

# EVOLVE-BLOCK-END
