import numpy as np
import time

_N = 17
_MIN_D = 6
_SEED = 32
_START = time.time()
_DEADLINE = 55.0  # Safety margin

def _qr_code_generator():
    """
    Returns the generator matrix for the [17,8,5] binary quadratic residue code.
    """
    p = 17
    # Quadratic residues mod 17
    QR = set((i * i) % p for i in range(1, p))
    
    v = np.zeros(p, dtype=np.uint8)
    for q in QR:
        v[q] = 1
        
    G = np.zeros((8, p), dtype=np.uint8)
    for k in range(8):
        G[k] = np.roll(v, k)
        
    return G

def _construct_extended_qr():
    """
    Constructs the [18,9,6] extended QR code.
    """
    G17 = _qr_code_generator()
    
    # Step 1: Create [18,8,6] code by adding parity bit
    # Parity bit for row k is sum(G17[k]) % 2
    parities = G17.sum(axis=1) % 2
    G18_8 = np.column_stack((G17, parities))
    
    # Step 2: Add all-ones vector to get [18,9,6]
    ones = np.ones(18, dtype=np.uint8)
    G18_9 = np.vstack((G18_8, ones))
    
    # Generate all 512 codewords
    code = np.zeros((512, 18), dtype=np.uint8)
    for m in range(512):
        bits = np.array([(m >> j) & 1 for j in range(9)], dtype=np.uint8)
        code[m] = (bits @ G18_9) % 2
        
    return code

def _shorten_code(code, coord_to_fix=17):
    """
    Shortens the code by fixing 'coord_to_fix' to 0.
    """
    # Keep words where the bit at coord_to_fix is 0
    mask = code[:, coord_to_fix] == 0
    short_code = code[mask, :17]
    return short_code

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    
    # 1. Get the linear [17,8,6] code (256 words)
    ext_code = _construct_extended_qr()
    lin_code = _shorten_code(ext_code)
    
    # Convert to list of ints
    # We want to map the numpy array of bits to integers.
    # The columns are bits. We need to decide on bit ordering.
    # The task says "integer encodes one N-bit binary codeword".
    # Usually, bit 0 is LSB or MSB?
    # "including leading zeros".
    # Let's assume standard integer representation where index 0 in the array corresponds to LSB or MSB?
    # The previous code used: powers = (1 << np.arange(_N, dtype=np.int64))[::-1]
    # This maps column 0 to 2^16, column 16 to 2^0.
    # So column 0 is MSB.
    
    powers = (1 << np.arange(_N, dtype=np.int64))[::-1]
    code_list = (lin_code @ powers).astype(np.int64).tolist()
    
    current_code = code_list
    current_set = set(current_code)
    
    rng = np.random.default_rng(42)
    
    # Helper to check validity
    # Using bit_count is very fast.
    def is_valid(cand):
        if cand in current_set:
            return False
        for w in current_code:
            if (cand ^ w).bit_count() < _MIN_D:
                return False
        return True

    # Greedy addition
    # We will try to add words.
    # Candidates:
    # 1. Random
    # 2. Mutations of existing words (flip 6+ bits)
    
    while time.time() < deadline - 1.0:
        # Generate candidate
        r = rng.random()
        if r < 0.5:
            cand = int(rng.integers(0, 1 << _N))
        else:
            # Pick a random existing word and flip 6 to 10 bits
            base = int(current_code[int(rng.integers(0, len(current_code)))])
            num_flips = int(rng.integers(6, 11)) # 6 to 10
            positions = list(range(_N))
            rng.shuffle(positions)
            flip_mask = 0
            for p in positions[:num_flips]:
                flip_mask |= (1 << p)
            cand = base ^ flip_mask
        
        if is_valid(cand):
            current_code.append(cand)
            current_set.add(cand)
            
    return current_code

def main():
    return construct()
