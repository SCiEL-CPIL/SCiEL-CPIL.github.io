import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0  # safety margin vs 150 s kill

def _compute_sets_fft(u, max_val):
    """
    Compute |U+U| and |U-U| using FFT.
    u: list of nonnegative integers
    max_val: max(u)
    
    Returns: (size_sum, size_diff)
    """
    if not u:
        return 0, 0
    if len(u) == 1:
        return 1, 1
    
    M = max_val
    # N needs to be at least 2*M + 1 to avoid circular convolution aliasing for sums [0, 2M]
    # and differences [-M, M]
    N = 1
    while N < 2 * M + 1:
        N *= 2
    
    f = np.zeros(N, dtype=np.float64)
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
    
    F = np.fft.fft(f)
    
    # Sum set
    conv_sum = np.fft.ifft(F * F).real
    # Valid range for sums is [0, 2*M]
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set
    # Autocorrelation r[k] = sum_x f[x] f[x+k]
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # r[k] for k in [1, M] corresponds to positive differences
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff)

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 1.0
    
    size_sum, size_diff = _compute_sets_fft(u, m)
    
    if size_sum == 0:
        return 0.0
        
    ratio = size_diff / float(size_sum)
    
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * m + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _random_initial_set(n, max_val):
    """
    Generate a random set of size n containing 0, within [0, max_val].
    """
    if n <= 0:
        return [0]
    if n == 1:
        return [0]
    
    others = random.sample(range(1, max_val + 1), n - 1)
    u = [0] + others
    return sorted(u)

def _local_search(u, max_val, iterations, temperature_start=1.0, cooling=0.999):
    """
    Simulated Annealing local search.
    """
    current_u = list(u)
    current_score = _score(current_u)
    best_u = list(current_u)
    best_score = current_score
    
    temperature = temperature_start
    
    for i in range(iterations):
        if time.time() >= _DEADLINE:
            break
            
        neighbor = list(current_u)
        move_type = random.random()
        
        if move_type < 0.7:
            # Replace a random element (not 0, to keep 0)
            if len(neighbor) > 1:
                idx = random.randint(1, len(neighbor) - 1)
                new_val = random.randint(0, max_val)
                neighbor[idx] = new_val
                neighbor = sorted(set(neighbor))
                if len(neighbor) != len(current_u):
                    # Try to restore size if it changed
                    if len(neighbor) < len(current_u):
                        while len(neighbor) < len(current_u):
                            candidate = random.randint(0, max_val)
                            if candidate not in neighbor:
                                neighbor.append(candidate)
                                break
                    else:
                        non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                        if non_zero_indices:
                            rm_idx = random.choice(non_zero_indices)
                            neighbor.pop(rm_idx)
                            neighbor = sorted(neighbor)
                    if len(neighbor) != len(current_u):
                        continue
            else:
                continue
        else:
            # Perturb: add or remove an element
            if move_type < 0.85:
                # Add
                candidate = random.randint(0, max_val)
                if candidate not in neighbor:
                    neighbor.append(candidate)
                    neighbor = sorted(neighbor)
            else:
                # Remove
                if len(neighbor) > 1:
                    non_zero_indices = [j for j, v in enumerate(neighbor) if v != 0]
                    if non_zero_indices:
                        rm_idx = random.choice(non_zero_indices)
                        neighbor.pop(rm_idx)
                        neighbor = sorted(neighbor)
                else:
                    continue
        
        if not neighbor:
            continue
            
        neighbor_score = _score(neighbor)
        
        # Acceptance probability
        delta = neighbor_score - current_score
        if delta > 0 or random.random() < math.exp(delta / max(temperature, 1e-9)):
            current_u = neighbor
            current_score = neighbor_score
            
            if current_score > best_score:
                best_u = list(current_u)
                best_score = current_score
        
        temperature *= cooling
        
    return best_u, best_score

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    random.seed(42)
    
    best_u = [0, 1, 2]
    best_s = _score(best_u)
    
    # Strategy:
    # 1. Explore structured sets (geometric, random) for various (n, M).
    # 2. Use SA to refine.
    # 3. Incorporate insights: bias towards small M? Or large ratio?
    # Score = 1 + log(ratio) / log(2M+1).
    # To maximize, we want high log(ratio) and small log(2M+1).
    # Small M implies small n.
    # Let's try a wider range of M, including very small M with small n.
    
    # Phase 1: Small M, small n
    # M in [2, 50], n in [3, M+1]
    
    # Phase 2: Medium M, medium n
    
    # We will run a loop of SA phases.
    # Each phase picks a random n and M.
    
    num_phases = 200 # Increased phases
    
    for phase in range(num_phases):
        if time.time() >= _DEADLINE - 10.0:
            break
            
        # Choose n and M
        # Bias towards smaller M to keep denominator small?
        # Let's sample M from a distribution that favors smaller values but allows large.
        # M = random.randint(2, 200)
        
        # Let's try specific ranges
        if phase % 2 == 0:
            M = random.randint(3, 50)
            n = random.randint(3, min(M+1, 20))
        else:
            M = random.randint(20, 200)
            n = random.randint(3, min(M+1, 50))
        
        # Ensure n <= M+1
        if n > M + 1:
            n = M + 1
            
        # Initial candidate
        u_init = _random_initial_set(n, M)
        
        # Run SA
        # Shorter SA per phase, more phases
        u_best, s_best = _local_search(u_init, M, 300, temperature_start=1.0, cooling=0.99)
        
        if s_best > best_s:
            best_s = s_best
            best_u = u_best
            
    # Final polish on the best found
    if time.time() < _DEADLINE - 5.0:
        m_best = max(best_u)
        # Allow M to grow slightly? No, keep M fixed to avoid denominator increase unless ratio jumps.
        # But if we can add elements to increase ratio significantly, it might be worth it.
        # Let's try to polish with M slightly larger to allow growth.
        M_polish = int(m_best * 1.5) + 10
        u_polished, s_polished = _local_search(best_u, M_polish, 1000, temperature_start=0.5, cooling=0.995)
        if s_polished > best_s:
            best_s = s_polished
            best_u = u_polished
            
    # Ensure 0 is in best_u and all are non-negative integers
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
