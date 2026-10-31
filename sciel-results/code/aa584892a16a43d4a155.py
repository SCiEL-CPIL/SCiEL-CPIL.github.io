import time
import math
import numpy as np

_START = time.time()
# Safety margin: The evaluator kills at 150s. We aim to finish by 130s to be safe.
_DEADLINE = _START + 130.0

def _compute_scores(u):
    """
    Computes |U+U| and |U-U| efficiently using FFT.
    Returns (size_sum, size_diff, max_u)
    """
    if not u:
        return 0, 0, 0
    if len(u) == 1:
        return 1, 1, max(u)
        
    M = max(u)
    if M == 0:
        return 1, 1, 0
        
    # Next power of 2 for FFT size
    # We need size at least 2*M + 1 to capture all sums/diffs without wrap-around aliasing
    # for the convolution of length M+1 with itself.
    N = 1
    while N < 2 * M + 1:
        N *= 2
        
    f = np.zeros(N, dtype=np.float64)
    # Use the list of elements directly
    for x in u:
        if 0 <= x < N:
            f[x] = 1.0
            
    F = np.fft.fft(f)
    
    # Sum set: Convolution of f with f
    conv_sum = np.fft.ifft(F * F).real
    # Range [0, 2*M]
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set: Autocorrelation
    # The autocorrelation R[k] = sum f[i] f[i+k] gives counts of differences of size k.
    # We want the number of distinct differences.
    # The difference set U-U contains {u1 - u2 | u1, u2 in U}.
    # Since U contains 0, U-U is symmetric around 0.
    # |U-U| = 1 + 2 * (number of positive differences).
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # auto_diff[k] for k=0..M-1 corresponds to differences of magnitude k.
    # Note: auto_diff[0] is |U|.
    # We check k=1 to M.
    pos_diffs = np.sum(auto_diff[1:M + 1] > 0.5)
    size_diff = 1 + 2 * pos_diffs
    
    return int(size_sum), int(size_diff), M

def _score(u):
    """
    Compute the score: 1 + log(|U-U|/|U+U|) / log(2*max U + 1)
    """
    if not u:
        return 0.0
    size_sum, size_diff, M = _compute_scores(u)
    if size_sum == 0 or M == 0:
        return 1.0 if M == 0 else 0.0
        
    ratio = size_diff / float(size_sum)
    if ratio <= 0:
        return 0.0
        
    denom = math.log(2 * M + 1)
    if denom == 0:
        return 1.0
        
    return 1.0 + math.log(ratio) / denom

def _generate_golomb_seed(max_val, target_size):
    """
    Generates a set resembling a Golomb ruler or Sidon set.
    These sets have distinct differences, maximizing |U-U| for a given |U| and M.
    """
    u = {0}
    # Greedy construction: pick next element that creates the most new differences
    # and minimizes sum set growth (heuristic).
    # For simplicity and speed, we use a "sparse random" approach with collision checking.
    
    # Start with a base spacing that grows
    # A simple heuristic: pick random numbers, ensure no difference repeats (Golomb property)
    # This is hard to enforce strictly while maximizing score, so we use a "soft" Golomb:
    # Prefer elements that are far from existing ones.
    
    candidates = np.random.choice(max_val + 1, size=max_val + 1, replace=False)
    # Sort candidates by value to allow incremental construction? 
    # Actually, random order is fine if we just check distinctness of diffs.
    
    # Let's try a simpler structural seed: Geometric progression with noise
    # Or: "Sidon-like" set: {0, 1, 3, 7, 15, ...} is 2^k - 1.
    # Let's generate a few structural seeds and return the best.
    
    seeds = []
    
    # Seed 1: Powers of 2 (maximizes differences relative to sums? No, sums are unique too)
    # For U = {0, 1, 2, 4, ...}, |U+U| is small (each sum unique), |U-U| is large (each diff unique).
    # Actually for Sidon sets, |U+U| is large (O(n^2)) and |U-U| is large (O(n^2)).
    # We want ratio |U-U|/|U+U| to be high.
    # If U is an arithmetic progression {0, 1, ..., n-1}:
    # |U+U| = 2n-1, |U-U| = 2n-1. Ratio = 1. Score = 1.
    # If U is a geometric progression {0, 1, 3, 9, ...}:
    # Sums are unique (binary representation like), so |U+U| ~ n^2/2.
    # Diffs are unique, so |U-U| ~ n^2.
    # Ratio ~ 2. Score > 1.
    
    # So Geometric Progressions are good seeds!
    
    # Seed A: Geometric base 2
    u = [0]
    x = 1
    while x < max_val:
        u.append(x)
        x *= 2
    if len(u) < target_size:
        # Fill remaining with random large numbers
        while len(u) < target_size:
            cand = np.random.randint(max(u) + 1, max_val + 1)
            if cand not in u:
                u.append(int(cand))
    elif len(u) > target_size:
        u = u[:target_size]
    seeds.append(sorted(u))
    
    # Seed B: Geometric base 3
    u = [0]
    x = 1
    while x < max_val:
        u.append(x)
        x *= 3
    if len(u) < target_size:
        while len(u) < target_size:
            cand = np.random.randint(max(u) + 1, max_val + 1)
            if cand not in u:
                u.append(int(cand))
    elif len(u) > target_size:
        u = u[:target_size]
    seeds.append(sorted(u))
    
    # Seed C: "Sparse Random" - Randomly sample from [0, max_val]
    # This tends to have |U+U| ~ min(n^2, M) and |U-U| ~ min(n^2, M).
    # But random sets often have "collisions" in sums more than diffs?
    # Actually, for random sets, sums are more likely to collide than diffs?
    # Let's just generate one.
    u = list(np.random.choice(max_val + 1, size=min(target_size, max_val + 1), replace=False))
    if 0 not in u:
        u[0] = 0
    seeds.append(sorted(u))
    
    # Evaluate seeds and return the best
    best_seed = seeds[0]
    best_s = _score(best_seed)
    for s in seeds[1:]:
        sc = _score(s)
        if sc > best_s:
            best_s = sc
            best_seed = s
            
    return best_seed, best_s

def _local_search(u, max_iter=1000):
    """
    Local search with specific move types:
    1. Replace an element with a random one.
    2. Add an element.
    3. Remove an element.
    4. "Shift" an element (move it slightly).
    
    We use a greedy acceptance for improvements, and occasional random jumps.
    """
    current_u = set(u)
    current_score = _score(list(current_u))
    best_u = set(current_u)
    best_score = current_score
    
    T = 0.05
    cooling = 0.99
    
    max_val = max(current_u)
    # Allow growth up to 2 * max_val
    search_max = int(1.5 * max_val) + 50
    
    for i in range(max_iter):
        if time.time() > _DEADLINE - 5.0:
            break
            
        # Choose move type
        move_type = np.random.randint(4)
        
        if move_type == 0: # Replace
            if len(current_u) <= 1: continue
            x_remove = np.random.choice([x for x in current_u if x != 0])
            x_add = np.random.randint(0, search_max + 1)
            if x_add in current_u and x_add != x_remove:
                continue
            new_u = current_u - {int(x_remove)} | {int(x_add)}
            
        elif move_type == 1: # Add
            if len(current_u) > 500: continue # Cap size
            x_add = np.random.randint(0, search_max + 1)
            if x_add in current_u:
                continue
            new_u = current_u | {int(x_add)}
            
        elif move_type == 2: # Remove
            if len(current_u) <= 1: continue
            x_remove = np.random.choice([x for x in current_u if x != 0])
            new_u = current_u - {int(x_remove)}
            
        else: # Shift
            if len(current_u) <= 1: continue
            x_move = np.random.choice([x for x in current_u if x != 0])
            shift = np.random.randint(-10, 11)
            x_new = int(x_move) + shift
            if x_new < 0 or x_new in current_u:
                continue
            new_u = current_u - {int(x_move)} | {x_new}
            
        new_score = _score(list(new_u))
        delta = new_score - current_score
        
        if delta > 0 or np.random.rand() < math.exp(delta / T):
            current_u = new_u
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_u = set(current_u)
            
        T = max(1e-4, T * cooling)
        
    return list(best_u), best_score

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 130.0)
    
    best_u = [0]
    best_s = 0.0
    
    # Strategy:
    # 1. Generate diverse structural seeds (Geometric, Random) at different scales.
    # 2. Run local search on each.
    # 3. Keep the best.
    
    # We explore different "scales" of M.
    # The score depends on log(2M+1) in the denominator.
    # If M is small, the penalty is small, but the potential ratio is limited.
    # If M is large, the potential ratio can be higher, but the penalty grows.
    
    scales = [100, 500, 2000, 10000]
    
    for scale in scales:
        if time.time() > _DEADLINE - 20.0:
            break
            
        # Generate a seed for this scale
        # Target size: roughly sqrt(scale) or log(scale)?
        # Geometric sets have size ~ log(scale).
        # Let's try size ~ 20-50 for these scales.
        target_size = max(10, int(math.log(scale, 2)) + 5)
        
        seed_u, seed_s = _generate_golomb_seed(scale, target_size)
        
        if seed_s > best_s:
            best_s = seed_s
            best_u = seed_u
            
        # Run local search
        u_ls, s_ls = _local_search(seed_u, max_iter=500)
        
        if s_ls > best_s:
            best_s = s_ls
            best_u = u_ls
            
    # Final polish: Try to improve the best found set with a tighter local search
    if best_u:
        # Use a smaller search space around the best elements
        u_final, s_final = _local_search(best_u, max_iter=2000)
        if s_final > best_s:
            best_s = s_final
            best_u = u_final
            
    # Ensure 0 is in best_u
    if 0 not in best_u:
        best_u.append(0)
    best_u = sorted(set(int(x) for x in best_u))
        
    return best_u

if __name__ == "__main__":
    res = construct()
    print(res)
    print(_score(res))
