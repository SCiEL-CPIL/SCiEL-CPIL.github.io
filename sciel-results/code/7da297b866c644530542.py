import time
import random

N = 17
MIN_D = 6

def hamming(a, b):
    return (a ^ b).bit_count()

def all_ok(code):
    n = len(code)
    for i in range(n):
        ci = code[i]
        for j in range(i + 1, n):
            if hamming(ci, code[j]) < MIN_D:
                return False
    return True

def greedy_insert(code, cand):
    # returns True if cand is at distance >= MIN_D from all in code
    for c in code:
        if hamming(c, cand) < MIN_D:
            return False
    return True

def construct():
    deadline = time.time() + 80  # 80s safety margin before 90s kill
    rng = random.Random(12345)

    # Start with a simple valid base: the 17-bit shortening of the first-order
    # Reed-Muller code on 16 coords plus a parity-like coordinate.
    # Actually, let's start with the 32-word affine code extended to 17 bits
    # by adding a 17th bit set to 0 (or computed). To keep distance >= 6,
    # we need to be careful. The 16-bit affine code has d=8. Adding a 0 bit
    # keeps d=8. So we have 32 words.

    code = []
    # Build the 16-bit affine code, append a 0 at bit 16
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            # bit 16 is 0
            code.append(word)

    base_size = len(code)

    # Phase 1: Greedy addition of random words
    while time.time() < deadline - 10:
        progress = False
        for _ in range(2000):
            if time.time() > deadline - 10:
                break
            cand = rng.getrandbits(17)
            if greedy_insert(code, cand):
                code.append(cand)
                progress = True

    # Phase 2: Local search / hill climbing
    # Try to add more words by searching in neighborhoods of existing words
    # or by using a greedy approach with backtracking on small conflicts.
    
    # A more effective strategy: try to find a word that minimizes the
    # minimum distance violation, and if it's close, try to modify the code.
    # For simplicity, let's do a simple local search:
    # Pick a random word from the code, mutate it, and see if it can replace
    # a conflicting word to allow a new addition. This is complex.
    
    # Instead, let's try a "greedy with restarts" approach on the remaining candidates.
    # But first, let's try to add words from a structured set.
    
    # Let's try adding the complements of existing words?
    # If d(a,b) >= 6, then d(~a, ~b) = 17 - d(a,b). If d(a,b)=6, d(~a,~b)=11.
    # If d(a,b)=8, d(~a,~b)=9. So complements are safe if original distances are >=6.
    # Let's add complements of all current words.
    for w in code[:]:  # iterate over a copy
        comp = w ^ ((1 << 17) - 1)
        if greedy_insert(code, comp):
            code.append(comp)
    
    # Phase 3: Further random greedy
    while time.time() < deadline - 5:
        for _ in range(5000):
            if time.time() > deadline - 5:
                break
            cand = rng.getrandbits(17)
            if greedy_insert(code, cand):
                code.append(cand)

    return code

# EVOLVE-BLOCK-END
