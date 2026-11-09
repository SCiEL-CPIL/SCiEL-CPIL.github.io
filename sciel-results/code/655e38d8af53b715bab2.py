import random
import time

def construct():
    N = 512
    deadline = time.time() + 85.0  # Safety margin
    
    # Bitmask representation: list of 8 integers, each 64 bits
    # Index i in S corresponds to bit (i % 64) in S[i // 64]
    
    def base4_seed():
        S = [0] * 8
        for i in range(1, N + 1):
            x = i
            valid = True
            while x > 0:
                if x % 4 > 1:
                    valid = False
                    break
                x //= 4
            if valid:
                S[i >> 6] |= (1 << (i & 63))
        return S

    def count_bits(S):
        return sum(x.bit_count() for x in S)

    def can_add(x, S):
        # Inline bit checks for speed
        # Case 1: x, x+d, x+2d, x+3d
        max_d = (N - x) // 3
        for d in range(1, max_d + 1):
            i1 = x + d
            i2 = x + 2 * d
            i3 = x + 3 * d
            if (S[i1 >> 6] >> (i1 & 63) & 1) and (S[i2 >> 6] >> (i2 & 63) & 1) and (S[i3 >> 6] >> (i3 & 63) & 1):
                return False

        # Case 2: x-d, x, x+d, x+2d
        max_d = min(x - 1, (N - x) // 2)
        for d in range(1, max_d + 1):
            i1 = x - d
            i2 = x + d
            i3 = x + 2 * d
            if (S[i1 >> 6] >> (i1 & 63) & 1) and (S[i2 >> 6] >> (i2 & 63) & 1) and (S[i3 >> 6] >> (i3 & 63) & 1):
                return False

        # Case 3: x-2d, x-d, x, x+d
        max_d = min(x // 2, N - x)
        for d in range(1, max_d + 1):
            i1 = x - 2 * d
            i2 = x - d
            i3 = x + d
            if (S[i1 >> 6] >> (i1 & 63) & 1) and (S[i2 >> 6] >> (i2 & 63) & 1) and (S[i3 >> 6] >> (i3 & 63) & 1):
                return False

        # Case 4: x-3d, x-2d, x-d, x
        max_d = x // 3
        for d in range(1, max_d + 1):
            i1 = x - 3 * d
            i2 = x - 2 * d
            i3 = x - d
            if (S[i1 >> 6] >> (i1 & 63) & 1) and (S[i2 >> 6] >> (i2 & 63) & 1) and (S[i3 >> 6] >> (i3 & 63) & 1):
                return False

        return True

    def greedy_random():
        S = [0] * 8
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, S):
                S[x >> 6] |= (1 << (x & 63))
        return S

    # Start with base-4 seed
    best_S = base4_seed()
    best_size = count_bits(best_S)
    
    # Phase 1: Randomized Greedy
    num_greedy = 2000
    for i in range(num_greedy):
        if time.time() > deadline:
            break
        S = greedy_random()
        size = count_bits(S)
        if size > best_size:
            best_size = size
            best_S = S[:]
            
    # Phase 2: Local Search (Remove-Add)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S = best_S[:]
        elements = [i for i in range(1, N + 1) if (S[i >> 6] >> (i & 63) & 1)]
        random.shuffle(elements)
        
        max_removals_to_try = 50
        for u in elements[:max_removals_to_try]:
            if time.time() > deadline:
                break
            S[u >> 6] &= ~(1 << (u & 63))
            
            # Try to add any element not in S
            for v in range(1, N + 1):
                if not (S[v >> 6] >> (v & 63) & 1):
                    if can_add(v, S):
                        S[v >> 6] |= (1 << (v & 63))
                        
            new_size = count_bits(S)
            if new_size > best_size:
                best_S = S[:]
                best_size = new_size
                improved = True
                break
                
    # Phase 3: Final Polish
    S = best_S
    for v in range(1, N + 1):
        if not (S[v >> 6] >> (v & 63) & 1) and can_add(v, S):
            S[v >> 6] |= (1 << (v & 63))
            
    return [i for i in range(1, N + 1) if (S[i >> 6] >> (i & 63) & 1)]
