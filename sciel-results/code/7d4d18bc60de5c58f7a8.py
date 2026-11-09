import numpy as np
import time

_N = 17
_MIN_D = 6
_SEED = 32
_START = time.time()
_DEADLINE = 85.0  # safety margin before the 90s kill

def _qr_code():
    """[18,9,6] extended quadratic-residue code (256 words) shortened to length 17."""
    p = 17
    # quadratic residues mod 17
    QR = set((i * i) % p for i in range(1, p))
    # generator matrix of the [17,8,5] QR code
    G = np.zeros((8, p), dtype=np.uint8)
    for k in range(8):
        for i in range(p):
            if i == 0:
                G[k, i] = 1
            elif (i % p) in QR:
                G[k, i] = (k + 1) % 2
    # parity-check column (extend to length 18)
    par = np.array([G[:, i].sum() % 2 for i in range(p)], dtype=np.uint8)
    # build all 256 codewords of the [18,9,6] code
    full = np.zeros((256, p + 1), dtype=np.uint8)
    for m in range(256):
        bits = np.array([(m >> j) & 1 for j in range(8)], dtype=np.uint8)
        row = (bits @ G) % 2
        full[m, :p] = row
        full[m, p] = row.sum() % 2
    # shorten by one coordinate: drop the last column, keep words with that bit = 0
    mask = full[:, p] == 0
    short = full[mask, :p]
    # convert to integers
    powers = (1 << np.arange(p, dtype=np.int64))[::-1]
    ints = (short @ powers).astype(np.int64)
    return ints

def _popcount_vec(diffs):
    """Vectorized popcount for an array of int64 values (17-bit)."""
    d = diffs.astype(np.int64)
    cnt = np.zeros(len(d), dtype=np.int32)
    for s in range(_N):
        cnt += (d >> s) & 1
    return cnt

def _is_valid(cand, code_arr):
    """Check if cand can be added to code_arr."""
    if len(code_arr) == 0:
        return True
    diffs = code_arr ^ cand
    cnt = _popcount_vec(diffs)
    return np.min(cnt) >= _MIN_D

def _is_valid_swap(cand_in, cand_out, code_arr):
    """Check if swapping cand_out for cand_in in code_arr results in a valid code."""
    # Remove cand_out, add cand_in
    # We need to check distances between cand_in and all other words in code_arr (excluding cand_out)
    # And we don't need to check cand_out against others (already valid)
    # So, check cand_in against code_arr \ {cand_out}
    
    # Find index of cand_out
    # Since code_arr is a numpy array, finding index is O(N).
    # We can assume cand_out is in code_arr.
    # To speed up, we can just check cand_in against all code_arr, 
    # but ignore the distance to cand_out itself (which will be 0).
    # Wait, if cand_in == cand_out, distance is 0.
    # If cand_in != cand_out, we check all.
    # If the min distance to any word in code_arr is >= MIN_D, 
    # AND cand_in != cand_out, then it's valid?
    # No, if cand_in is close to cand_out, but far from others, it's valid.
    # The only constraint is that cand_in must be far from ALL words in the NEW code.
    # New code = (Old Code - {cand_out}) + {cand_in}.
    # So cand_in must be far from (Old Code - {cand_out}).
    
    # We can compute distances to all code_arr.
    # If min dist >= MIN_D, it's valid.
    # But if cand_in == cand_out, dist is 0.
    # If cand_in != cand_out, and min dist to code_arr is >= MIN_D, 
    # then min dist to (code_arr - {cand_out}) is also >= MIN_D?
    # Yes, because removing an element can only increase the min distance.
    # So, simply checking _is_valid(cand_in, code_arr) is sufficient?
    # No! If cand_in is close to cand_out, _is_valid fails.
    # But we are REMOVING cand_out.
    # So we need to check cand_in against code_arr EXCLUDING cand_out.
    
    # Efficient check:
    # Compute distances to all code_arr.
    # If cand_in is in code_arr (it shouldn't be if we pick random), skip.
    # We want min(distances) >= MIN_D, ignoring the distance to cand_out.
    
    diffs = code_arr ^ cand_in
    cnt = _popcount_vec(diffs)
    
    # If cand_in is not in code_arr, then all distances are > 0.
    # We just need min(cnt) >= MIN_D.
    # BUT, if cand_in is very close to cand_out, that distance is small.
    # Since we remove cand_out, that small distance doesn't matter.
    # So we need to check: Is there any word w in code_arr \ {cand_out} such that dist(cand_in, w) < MIN_D?
    
    # Let's find the index of cand_out in code_arr.
    # This is slow in a loop.
    # Alternative: 
    # If min(cnt) >= MIN_D, return True.
    # If min(cnt) < MIN_D, find the indices where cnt < MIN_D.
    # If the ONLY word causing violation is cand_out, return True.
    # Otherwise, return False.
    
    violations = np.where(cnt < _MIN_D)[0]
    if len(violations) == 0:
        return True
    if len(violations) == 1:
        # Check if the violating word is cand_out
        return code_arr[violations[0]] == cand_out
    return False

def construct():
    t0 = time.time()
    deadline = t0 + (_DEADLINE - (time.time() - _START))
    if deadline - t0 < 1.0:
        deadline = t0 + 1.0

    # Initialize with QR code
    try:
        ints = _qr_code()
        ints = np.unique(ints)
    except Exception:
        ints = np.array([0], dtype=np.int64)

    code_arr = ints.astype(np.int64)
    code_set = set(code_arr.tolist())
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Greedy Add
    # Try to add random words
    for _ in range(200000):
        if time.time() > deadline - 10.0:
            break
        cand = int(rng.integers(0, 1 << _N))
        if cand in code_set:
            continue
        if _is_valid(cand, code_arr):
            code_arr = np.append(code_arr, cand)
            code_set.add(cand)
            
    # Phase 2: Swap-based Local Search
    # Try to swap words to find a better configuration
    # We track the best size found
    best_size = len(code_arr)
    
    # Pre-generate a pool of candidates? 
    # Or just random.
    # To make swaps more effective, we should pick cand_in that is "close" to the code 
    # but not too close.
    
    for i in range(500000):
        if time.time() > deadline - 2.0:
            break
            
        # Try Add first (cheap)
        if i % 10 == 0:
            cand = int(rng.integers(0, 1 << _N))
            if cand not in code_set:
                if _is_valid(cand, code_arr):
                    code_arr = np.append(code_arr, cand)
                    code_set.add(cand)
                    if len(code_arr) > best_size:
                        best_size = len(code_arr)
                    continue
        
        # Try Swap
        # Pick a random word to remove
        idx_out = int(rng.integers(0, len(code_arr)))
        cand_out = code_arr[idx_out]
        
        # Pick a random word to add
        cand_in = int(rng.integers(0, 1 << _N))
        if cand_in == cand_out or cand_in in code_set:
            continue
            
        if _is_valid_swap(cand_in, cand_out, code_arr):
            # Perform swap
            code_arr[idx_out] = cand_in
            code_set.remove(cand_out)
            code_set.add(cand_in)
            
    return code_arr.tolist()

def main():
    return construct()
