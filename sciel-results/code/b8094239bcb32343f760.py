"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random

def construct():
    deadline = time.time() + 85
    N = 18
    MIN_DIST = 6

    # Start with first-order Reed-Muller RM(1,4) on 16 bits, extended with 2 zero bits
    base = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            base.append(word)

    code = list(base)
    code_set = set(code)
    code_arr = np.array(code, dtype=np.uint32)

    rng = random.Random(42)

    # Greedy insertion with numpy-accelerated distance checks
    while time.time() < deadline - 3:
        # Generate candidates in a batch
        batch = []
        for _ in range(10000):
            w = rng.randint(4, 14)
            pos = rng.sample(range(N), w)
            word = 0
            for p in pos:
                word |= (1 << p)
            if word not in code_set:
                batch.append(word)

        if not batch:
            continue

        # Check each candidate against the code
        added = []
        for c in batch:
            xors = code_arr ^ c
            dists = np.array([x.bit_count() for x in xors], dtype=np.int16)
            if dists.min() >= MIN_DIST:
                added.append(c)

        for a in added:
            code.append(a)
            code_set.add(a)

        if added:
            code_arr = np.array(code, dtype=np.uint32)

    return code
# EVOLVE-BLOCK-END
