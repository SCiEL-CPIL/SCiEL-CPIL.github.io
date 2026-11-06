import time
import numpy as np

N = 18
MIN_DIST = 6
_DEADLINE = time.time() + 85  # 5s safety margin before 90s limit

_CODEWORDS = None

def _compute_codewords():
    """Compute the 512 codewords using the QR code construction."""
    p = 17
    # Quadratic residues mod 17
    qr = set((i * i) % p for i in range(p))
    
    # Generator matrix: 9 x 17, G[i][j] = 1 if (j-i) mod 17 is in QR
    G = [[0] * p for _ in range((p + 1) // 2)]
    for i in range((p + 1) // 2):
        for j in range(p):
            if (j - i) % p in qr:
                G[i][j] = 1
    
    # Extend to length 18 with overall parity bit
    G18 = []
    for row in G:
        parity = sum(row) % 2
        G18.append(row + [parity])
    
    # Generate all 512 codewords
    codewords = []
    for msg in range(512):
        word = 0
        for i in range(9):
            if (msg >> i) & 1:
                for j in range(18):
                    if G18[i][j]:
                        word ^= (1 << j)
        codewords.append(word)
    
    return codewords

def _verify_fast(words):
    """Verify all pairs have Hamming distance >= 6 using numpy."""
    n = len(words)
    if n <= 1:
        return True
    arr = np.array(words, dtype=np.uint32)
    # Compute all pairwise XORs
    for i in range(n):
        if time.time() > _DEADLINE:
            return False
        diffs = arr[i+1:] ^ arr[i]
        # Count bits using lookup-free method: sum of bits
        # Use np.unpackbits for efficiency
        unpacked = np.unpackbits(diffs.astype(np.uint8).reshape(-1, 1), axis=1)
        counts = unpacked.sum(axis=1)
        if np.any(counts < MIN_DIST):
            return False
    return True

def _verify_slow(words):
    """Verify all pairs have Hamming distance >= 6 using pure Python."""
    n = len(words)
    for i in range(n):
        if time.time() > _DEADLINE:
            return False
        for j in range(i + 1, n):
            if (words[i] ^ words[j]).bit_count() < MIN_DIST:
                return False
    return True

def construct():
    global _CODEWORDS, _DEADLINE
    _DEADLINE = time.time() + 85  # Reset deadline each call
    
    if _CODEWORDS is not None:
        return _CODEWORDS
    
    # Compute the 512 codewords
    codewords = _compute_codewords()
    
    if codewords is not None and len(codewords) == 512:
        # Verify using fast numpy method
        valid = _verify_fast(codewords)
        
        if not valid:
            # Fallback to slow verification (in case numpy had issues)
            valid = _verify_slow(codewords)
        
        if valid:
            _CODEWORDS = codewords
            return codewords
    
    # Fallback: return a small valid code
    # The zero word plus a few carefully chosen words
    fallback = [0]
    # Add words that are known to be at distance >= 6 from each other
    # Use the first few codewords from the QR construction if available
    if codewords is not None and len(codewords) > 0:
        # Try to build a valid subset
        subset = [codewords[0]]
        for w in codewords[1:]:
            ok = True
            for s in subset:
                if (w ^ s).bit_count() < MIN_DIST:
                    ok = False
                    break
            if ok:
                subset.append(w)
            if time.time() > _DEADLINE:
                break
        if len(subset) > 1:
            return subset
    
    return [0]
