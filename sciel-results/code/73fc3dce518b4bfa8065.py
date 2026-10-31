import time
import math
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0

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
    # We only care about the range [0, 2*M]
    size_sum = np.sum(conv_sum[:2 * M + 1] > 0.5)
    
    # Difference set
    # Autocorrelation
    auto_diff = np.fft.ifft(F * np.conj(F)).real
    # Count positive differences
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

def _construct_two_block(b1, b2, n1, n2, shift2):
    """
    Construct U = A union B
    A = { i * b1 for i in [0, n1) }
    B = { shift2 + j * b2 for j in [0, n2) }
    Ensures 0 is in U (A contains 0).
    """
    A = [i * b1 for i in range(n1)]
    B = [shift2 + j * b2 for j in range(n2)]
    U = set(A)
    U.update(B)
    # Filter non-negative (should be true by construction if shift2 >= 0)
    U = {x for x in U if x >= 0}
    return sorted(U)

def _greedy_refine(u, max_val, max_steps=200):
    """
    Greedily add elements to u to maximize score.
    """
    current_u = sorted(list(u))
    current_score = _score(current_u)
    current_set = set(current_u)
    
    # Precompute current sizes for potential speedup? 
    # For now, just recompute. FFT is fast enough for small sets.
    
    for _ in range(max_steps):
        if time.time() > _DEADLINE - 5.0:
            break
            
        best_candidate = None
        best_candidate_score = current_score
        
        # We can only add elements if they are <= max_val
        # And we want to try candidates that might be "good"
        # Random sampling of candidates is safer than iterating all if max_val is large.
        # But if max_val is small, we can iterate.
        
        # Let's try a mix: if max_val < 500, iterate all. Else, sample.
        if max_val < 500:
            candidates = [x for x in range(max_val + 1) if x not in current_set]
        else:
            # Sample 200 candidates
            candidates = []
            attempts = 0
            while len(candidates) < 200 and attempts < 1000:
                x = np.random.randint(1, max_val + 1)
                if x not in current_set:
                    candidates.append(x)
                attempts += 1
            
        for x in candidates:
            new_u = current_u + [x]
            new_u.sort()
            s = _score(new_u)
            if s > best_candidate_score:
                best_candidate_score = s
                best_candidate = x
                
        if best_candidate is not None:
            current_u.append(best_candidate)
            current_u.sort()
            current_set.add(best_candidate)
            current_score = best_candidate_score
        else:
            break
            
    return current_u, current_score

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    best_u = [0]
    best_s = 0.0
    
    # Strategy:
    # 1. Explore a grid of "two-block" parameters (b1, b2, n1, n2, shift2).
    #    This provides strong structural candidates with high ratio potential.
    # 2. Refine the best candidates using greedy addition.
    
    # Parameters for two-block structure
    # b1, b2: bases. Multiplicatively independent is good.
    # Let's try small primes or coprime integers.
    # n1, n2: lengths.
    # shift2: offset for the second block.
    
    # We want to maximize score = 1 + log(ratio)/log(2M+1).
    # Small M is good for the denominator, but limits n.
    # Large ratio is good.
    
    # Let's sweep over a few configurations.
    
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # To keep M reasonable, we limit n1*b1 and shift2+n2*b2.
    # Let's target M up to ~2000 or so.
    
    for b1 in bases:
        if time.time() > _DEADLINE - 10.0:
            break
        for b2 in bases:
            if b1 == b2:
                continue
            # Ensure some independence
            if b1 % b2 == 0 or b2 % b1 == 0:
                continue
                
            # Try different sizes
            # n1 * b1 ~ M1, shift2 + n2 * b2 ~ M2
            # Let's fix a target M range, say [50, 500]
            
            max_n1 = 50
            max_n2 = 50
            
            # We don't want to iterate all n1, n2, shift2.
            # Let's pick a few representative sizes.
            for n1 in [5, 10, 20, 30, 50]:
                for n2 in [5, 10, 20, 30, 50]:
                    # shift2 should be such that the blocks don't overlap too much
                    # and M is controlled.
                    # Let shift2 be roughly n1*b1/2 or something that separates them.
                    # Actually, let's try a few shifts.
                    for shift2 in [0, n1*b1//2, n1*b1, n1*b1 + b2//2]:
                        if time.time() > _DEADLINE - 5.0:
                            break
                            
                        u = _construct_two_block(b1, b2, n1, n2, shift2)
                        if not u:
                            continue
                        M = max(u)
                        if M > 2000: # Keep M bounded to keep FFT fast and denominator small
                            continue
                            
                        s = _score(u)
                        if s > best_s:
                            best_s = s
                            best_u = u
                            
    # Refine the best found so far
    if best_u:
        M_best = max(best_u)
        # Allow greedy to add elements up to maybe 1.5 * M_best or a fixed cap
        # If M is small, we can expand. If M is large, be careful.
        max_val = max(500, int(1.5 * M_best))
        
        # Run greedy refinement
        u_refined, s_refined = _greedy_refine(best_u, max_val, max_steps=100)
        
        if s_refined > best_s:
            best_s = s_refined
            best_u = u_refined
            
    # Ensure 0 is in best_u
    if 0 not in best_u:
        best_u.append(0)
    best_u = sorted(set(int(x) for x in best_u))
        
    return best_u

if __name__ == "__main__":
    res = construct()
    print(res)
    print(_score(res))
