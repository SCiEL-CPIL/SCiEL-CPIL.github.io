import struct
import array

# Constants
WIDTH = 64
ROWS = 8
MOD = 2147483647
PS = [1035493289, 775374203, 1495862053, 2015145641, 1606938044, 1934966308, 1328979875, 1103515245]
OFFSETS = [31 * i for i in range(ROWS)]

def update(state, item):
    # Unpack 1024 bytes into 512 uint16s
    counters = array.array('H')
    counters.frombytes(state)
    
    for i in range(ROWS):
        idx = ((item * PS[i] + OFFSETS[i]) % MOD) % WIDTH
        counters[idx] += 1
        if counters[idx] > 65535:
            counters[idx] = 65535 # Saturation guard, though unlikely to hit in 1024 items
            
    return counters.tobytes()

def query(state, item):
    counters = array.array('H')
    counters.frombytes(state)
    
    min_val = 65535
    for i in range(ROWS):
        idx = ((item * PS[i] + OFFSETS[i]) % MOD) % WIDTH
        if counters[idx] < min_val:
            min_val = counters[idx]
            
    return min_val
