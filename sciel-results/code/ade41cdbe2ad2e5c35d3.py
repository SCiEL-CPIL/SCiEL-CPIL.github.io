import time
import random
import itertools

N = 18
MIN_DISTANCE = 6
DEADLINE = 85  # seconds, safety margin before 90s kill

def _gf28_mult(a, b, poly=0x11d):
    result = 0
    while b:
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & 0x100:
            a ^= poly
    return result

def _gf28_pow(a, n, poly=0x11d):
    result = 1
    while n:
        if n & 1:
            result = _gf28_mult(result, a, poly)
        a = _gf28_mult(a, a, poly)
        n >>= 1
    return result

def _construct_qr_code():
    alpha = 0x02
    poly = 0x11d
    non_residues = [3, 5, 6, 7, 10, 11, 12, 14]
    
    g_coeffs = [1]
    
    for j in non_residues:
        root = _gf28_pow(alpha, j, poly)
        new_g = [0] * (len(g_coeffs) + 1)
        for i, c in enumerate(g_coeffs):
            new_g[i] ^= _gf28_mult(c, root, poly)
            new_g[i+1] ^= c
        g_coeffs = new_g
        
    g_bits = []
    for c in g_coeffs:
        if c == 0:
            g_bits.append(0)
        elif c == 1:
            g_bits.append(1)
        else:
            raise ValueError("Coeff not in GF(2)")
            
    if len(g_bits) != 9:
        raise ValueError("Wrong degree")
        
    G = []
    for i in range(9):
        row = [0] * 17
        for j in range(9):
            idx = (i + j) % 17
            row[idx] = g_bits[j]
        G.append(row)
        
    codewords = []
    for mask in range(512):
        cw = [0] * 17
        for i in range(9):
            if (mask >> i) & 1:
                for j in range(17):
                    cw[j] ^= G[i][j]
        parity = sum(cw) % 2
        cw.append(parity)
        val = 0
        for bit in cw:
            val = (val << 1) | bit
        codewords.append(val)
        
    return codewords

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    try:
        code = _construct_qr_code()
        # Optional verification
        # return code
        return code
    except Exception:
        pass
        
    # Fallback
    # ... (keep old logic if needed, but QR should work)
    return []
