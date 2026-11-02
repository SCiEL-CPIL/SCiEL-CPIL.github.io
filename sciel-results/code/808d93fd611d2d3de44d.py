"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap_binary(h):
    """Compute max overlap of h with shifted (1-h) for binary h in {0,1}."""
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    # overlap at shift k: sum_i h[i] * (1 - h[(i+k) % n])
    # This equals sum_i h[i] - sum_i h[i]*h[(i+k)%n]
    # = count_ones - autocorr(k)
    ones = h.sum()
    max_ov = 0.0
    for k in range(n):
        shifted = np.roll(h, -k)
        prod = (h * shifted).sum()
        ov = ones - prod
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _build_periodic(pattern, N):
    """Tile a base pattern to length N, ensuring sum = N/2."""
    p = np.array(pattern, dtype=np.float64)
    reps = N // len(p)
    rem = N % len(p)
    h = np.tile(p, reps)
    if rem > 0:
        h = np.concatenate([h, p[:rem]])
    # Adjust to make sum exactly N/2
    target = N / 2.0
    current_sum = h.sum()
    diff = target - current_sum
    # diff should be 0 if pattern has exact half ones and N divisible by len(p)
    # If not, we need to fix. For our patterns, ensure len(p) divides N or adjust.
    return h

def _score_binary(h):
    """Score = max overlap of h with shifted 1-h. Lower is better."""
    return _max_overlap_binary(h)

def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    
    best_h = None
    best_score = float('inf')
    
    # Candidate periodic binary patterns with duty cycle 1/2
    # Pattern must have even length with half ones
    candidates = []
    
    # Alternating: 1010...
    for p in range(2, 50, 2):
        # Try all binary patterns of length p with exactly p/2 ones? Too many for large p.
        # Instead, use known low-autocorrelation structures.
        pass
    
    # Specific structured patterns:
    # 1. Alternating 10
    candidates.append([1, 0])
    # 2. 1100
    candidates.append([1, 1, 0, 0])
    # 3. 111000
    candidates.append([1, 1, 1, 0, 0, 0])
    # 4. 11110000
    candidates.append([1, 1, 1, 1, 0, 0, 0, 0])
    # 5. 111111000000
    candidates.append([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # 6. Balanced m-sequence-like: 110100 (period 6, 3 ones)
    candidates.append([1, 1, 0, 1, 0, 0])
    # 7. 100101 (another period 6)
    candidates.append([1, 0, 0, 1, 0, 1])
    # 8. 110010 (period 6)
    candidates.append([1, 1, 0, 0, 1, 0])
    # 9. 101100 (period 6)
    candidates.append([1, 0, 1, 1, 0, 0])
    # 10. 110100 already above
    # 11. Period 8: 11101000 (4 ones)
    candidates.append([1, 1, 1, 0, 1, 0, 0, 0])
    # 12. Period 8: 11011000
    candidates.append([1, 1, 0, 1, 1, 0, 0, 0])
    # 13. Period 8: 10110100
    candidates.append([1, 0, 1, 1, 0, 1, 0, 0])
    # 14. Period 8: 11001100
    candidates.append([1, 1, 0, 0, 1, 1, 0, 0])
    # 15. Period 8: 10010010
    candidates.append([1, 0, 0, 1, 0, 0, 1, 0])
    # 16. Period 10: 1111100000
    candidates.append([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    # 17. Period 10: 1110110000
    candidates.append([1, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    # 18. Period 10: 1101101000
    candidates.append([1, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    # 19. Period 12: 111111000000
    candidates.append([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    # 20. Period 12: 111011100000
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    # 21. Period 12: 110110110000
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    # 22. Period 12: 101101101000
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    # 23. Period 14: 11111110000000
    candidates.append([1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    # 24. Period 14: 11101110100000
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    # 25. Period 16: 1111111100000000
    candidates.append([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # 26. Period 16: 1110111011100000
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    # 27. Period 16: 1101101101100000
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    # 28. Period 16: 1011011010110000
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    # 29. Period 20: 11111111110000000000
    candidates.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 30. Period 20: 11101110111010000000
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    # 31. Period 20: 11011011011011000000
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    # 32. Period 20: 10110110101101100000
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0])
    # 33. Period 25: 1111111111100000000000000
    candidates.append([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 34. Period 25: 1110111011101110000000000
    candidates.append([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 35. Period 25: 1101101101101101000000000
    candidates.append([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    # 36. Period 25: 1011011010110110100000000
    candidates.append([1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # 37. Period 50: 111111111111111111111111110000000000000000000000000000
    candidates.append([1]*25 + [0]*25)
    # 38. Period 50: 111011101110111011101110110000000000000000000000000000
    candidates.append([1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # 39. Period 50: 110110110110110110110110110110000000000000000000000000
    candidates.append([1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    # 40. Period 50: 101101101011011010110110101101000000000000000000000000
    candidates.append([1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    
    # Also try the simple alternating for reference
    candidates.append([1, 0])
    
    for pat in candidates:
        if time.time() > deadline:
            break
        
        p = len(pat)
        # Check if pattern has exactly p/2 ones
        if sum(pat) != p / 2:
            continue
        
        # Build h of length N by tiling
        reps = N // p
        rem = N % p
        h = np.tile(pat, reps)
        if rem > 0:
            h = np.concatenate([h, np.array(pat[:rem])])
        
        # Ensure sum is exactly N/2
        target = N / 2.0
        current_sum = h.sum()
        if abs(current_sum - target) > 1e-9:
            # Adjust: flip some bits to fix sum
            diff = target - current_sum
            if diff > 0:
                # Need to add diff ones; flip 0s to 1s
                zeros = np.where(h == 0)[0]
                for i in range(int(round(diff))):
                    if i < len(zeros):
                        h[zeros[i]] = 1
            else:
                # Need to remove -diff ones; flip 1s to 0s
                ones = np.where(h == 1)[0]
                for i in range(int(round(-diff))):
                    if i < len(ones):
                        h[ones[i]] = 0
        
        # Verify sum
        if abs(h.sum() - target) > 1e-9:
            continue
        
        # Ensure values are in [0,1]
        h = np.clip(h, 0, 1)
        
        score = _score_binary(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    if best_h is None:
        # Fallback: alternating
        best_h = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    
    return best_h
# EVOLVE-BLOCK-END
