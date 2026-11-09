import struct
import math
import random

# Strategy: Offline Optimized Hybrid Allocation
# We simulate the three known streams to find the best allocation of 1024 bytes.
# Allocation: 
#   - First N items (0..N-1) get exact counters. We test N in {0, 32, 64, 128, 256}.
#   - Remaining items (N..399) go into a Count-Min Sketch.
#   - We vary the number of rows in the CM sketch (depth) and counter width (8 or 16 bits).
#   - Bytes used by exact: N * w_exact (w_exact is 2 or 4 bytes).
#   - Remaining bytes for sketch: R = 1024 - N * w_exact.
#   - Sketch width W = R // (depth * w_sketch).
#
# We pre-compute the best configuration by simulating the streams.
# To keep update/query fast, we hardcode the winning configuration.
# However, since we don't know the "winning" config until we run the search,
# we will implement a lightweight search that runs ONCE at import time (or first call)
# to pick the best config, then cache it.
# Given the 15s limit per chunk and 120s total, a quick search at import is safe.
# Actually, import happens once. The search takes < 1s.

# Simulation of the three streams
def gen_uniform(n, low, high):
    return [random.randint(low, high) for _ in range(n)]

def gen_zipf(n, s, low, high):
    # Zipf distribution: P(k) ~ 1/k^s
    # We need to map ranks to items.
    # Items are 0..399.
    # Let's generate ranks 1..n and map to items.
    # Simpler: generate random numbers, apply Zipf inverse CDF?
    # Or just use the fact that Zipf(1.6) on 0..399 is heavily skewed to 0.
    # Let's use a simple rejection sampling or just assume the standard Zipf rank mapping.
    # For simulation, we can approximate:
    # Rank 1 -> Item 0, Rank 2 -> Item 1, etc.
    # But the stream is "Zipf(1.6) modulo 400". This usually means items are drawn from a Zipf distribution over the support 0..399.
    # Let's assume item i has weight 1/(i+1)^1.6.
    items = list(range(400))
    weights = [1.0 / (i + 1) ** 1.6 for i in range(400)]
    total_w = sum(weights)
    cum_weights = []
    c = 0
    for w in weights:
        c += w
        cum_weights.append(c)
    cum_weights = [c / total_w for c in cum_weights]
    
    stream = []
    for _ in range(n):
        r = random.random()
        # Binary search in cum_weights
        lo, hi = 0, 399
        while lo < hi:
            mid = (lo + hi) // 2
            if cum_weights[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        stream.append(lo)
    return stream

def gen_bimodal(n):
    # 512 uniform 0..19, 512 uniform 150..399
    stream1 = [random.randint(0, 19) for _ in range(512)]
    stream2 = [random.randint(150, 399) for _ in range(512)]
    return stream1 + stream2

# Pre-generate streams for simulation (fixed seed for reproducibility)
random.seed(42)
STREAMS = [
    gen_uniform(1024, 0, 399),
    gen_zipf(1024, 1.6, 0, 399),
    gen_bimodal(1024)
]

# Configuration search
def evaluate_config(N, w_exact, depth, w_sketch):
    # N: number of items in exact table
    # w_exact: bytes per exact counter (2 or 4)
    # depth: number of rows in CM sketch
    # w_sketch: bytes per sketch counter (1 or 2)
    
    bytes_exact = N * w_exact
    bytes_sketch = 1024 - bytes_exact
    if bytes_sketch <= 0:
        return float('inf')
    
    W = bytes_sketch // (depth * w_sketch)
    if W < 1:
        return float('inf')
    
    total_error = 0.0
    for stream in STREAMS:
        # Simulate
        counts = [0] * 400
        sketch = [[0] * W for _ in range(depth)]
        
        for item in stream:
            if item < N:
                counts[item] += 1
            else:
                for r in range(depth):
                    # Simple hash for simulation
                    idx = (item * (r + 1) * 2654435761) % W
                    if w_sketch == 1:
                        if sketch[r][idx] < 255:
                            sketch[r][idx] += 1
                    else:
                        if sketch[r][idx] < 65535:
                            sketch[r][idx] += 1
        
        # Query all 400 items
        err = 0.0
        for item in range(400):
            if item < N:
                est = counts[item]
            else:
                min_val = float('inf')
                for r in range(depth):
                    idx = (item * (r + 1) * 2654435761) % W
                    if sketch[r][idx] < min_val:
                        min_val = sketch[r][idx]
                est = min_val
            err += abs(est - counts[item])
        
        total_error += err / 1024.0
    
    return total_error / 3.0

# Run search
best_config = None
best_error = float('inf')

# Candidates:
# N in [0, 16, 32, 64, 128, 256]
# w_exact in [2, 4]
# depth in [2, 3, 4, 5, 8]
# w_sketch in [1, 2]

for N in [0, 16, 32, 64, 128, 256]:
    for w_exact in [2, 4]:
        for depth in [2, 3, 4, 5, 8]:
            for w_sketch in [1, 2]:
                err = evaluate_config(N, w_exact, depth, w_sketch)
                if err < best_error:
                    best_error = err
                    best_config = (N, w_exact, depth, w_sketch)

N, w_exact, depth, w_sketch = best_config
bytes_exact = N * w_exact
bytes_sketch = 1024 - bytes_exact
W = bytes_sketch // (depth * w_sketch)

# Hardcode the winning config for speed
# We need to implement update and query using this config.

# Hash function for sketch
# We need 4 different hashes for the rows.
# Use simple multiplicative hashing.
# h_r(x) = (x * a_r) % W
# We need distinct a_r.
A = [2654435761, 40503, 2246822519, 3266489917]

def _hash(item, r, W):
    return (item * A[r]) % W

def update(state, item):
    buf = bytearray(state)
    
    if item < N:
        # Exact counter
        off = item * w_exact
        if w_exact == 2:
            v = buf[off] | (buf[off+1] << 8)
            if v < 0xFFFF:
                v += 1
                buf[off] = v & 0xFF
                buf[off+1] = (v >> 8) & 0xFF
        else: # w_exact == 4
            v = buf[off] | (buf[off+1] << 8) | (buf[off+2] << 16) | (buf[off+3] << 24)
            if v < 0xFFFFFFFF:
                v += 1
                buf[off] = v & 0xFF
                buf[off+1] = (v >> 8) & 0xFF
                buf[off+2] = (v >> 16) & 0xFF
                buf[off+3] = (v >> 24) & 0xFF
    else:
        # Sketch
        for r in range(depth):
            idx = _hash(item, r, W)
            off = bytes_exact + (r * W + idx) * w_sketch
            if w_sketch == 1:
                v = buf[off]
                if v < 255:
                    buf[off] = v + 1
            else:
                v = buf[off] | (buf[off+1] << 8)
                if v < 0xFFFF:
                    v += 1
                    buf[off] = v & 0xFF
                    buf[off+1] = (v >> 8) & 0xFF
    
    return bytes(buf)

def query(state, item):
    if item < N:
        off = item * w_exact
        if w_exact == 2:
            return state[off] | (state[off+1] << 8)
        else:
            return state[off] | (state[off+1] << 8) | (state[off+2] << 16) | (state[off+3] << 24)
    else:
        mn = float('inf')
        for r in range(depth):
            idx = _hash(item, r, W)
            off = bytes_exact + (r * W + idx) * w_sketch
            if w_sketch == 1:
                v = state[off]
            else:
                v = state[off] | (state[off+1] << 8)
            if v < mn:
                mn = v
        return mn
