import time
import math
import random
import sys

N = 128

# Precompute exclusion masks
# exclusion_masks[u][v] is a bitmask of w such that {u+1, v+1, w+1} is a forbidden triple
exclusion_masks = [[0] * N for _ in range(N)]

def precompute_masks():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    ai, bi, ci = a - 1, b - 1, c - 1
                    exclusion_masks[ai][bi] |= (1 << ci)
                    exclusion_masks[bi][ai] |= (1 << ci)
                    exclusion_masks[ai][ci] |= (1 << bi)
                    exclusion_masks[ci][ai] |= (1 << bi)
                    exclusion_masks[bi][ci] |= (1 << ai)
                    exclusion_masks[ci][bi] |= (1 << ai)

precompute_masks()

def popcount(x):
    return bin(x).count('1')

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    best_size = 0
    best_set = 0  # Bitmask
    
    # Greedy to get a good initial solution
    def greedy(order):
        S = 0
        excl = 0
        S_bits = []
        for x in order:
            if (S >> x) & 1:
                continue
            if (excl >> x) & 1:
                continue
            S |= (1 << x)
            S_bits.append(x)
            for u in S_bits[:-1]:
                excl |= exclusion_masks[x][u]
        return S, excl

    best_greedy = 0
    S, _ = greedy(range(N))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
    
    S, _ = greedy(range(N - 1, -1, -1))
    if popcount(S) > popcount(best_greedy):
        best_greedy = S
        
    for _ in range(200):
        if time.time() > deadline - 15:
            break
        order = list(range(N))
        random.shuffle(order)
        S, _ = greedy(order)
        if popcount(S) > popcount(best_greedy):
            best_greedy = S
            
    best_size = popcount(best_greedy)
    best_set = best_greedy

    # Local search: try to flip bits to improve
    def local_search(S):
        nonlocal best_size, best_set
        S_bits = []
        for i in range(N):
            if (S >> i) & 1:
                S_bits.append(i)
        
        # Try to add elements
        improved = True
        while improved:
            if time.time() > deadline - 2:
                return
            improved = False
            excl = 0
            for i in range(len(S_bits)):
                for j in range(i + 1, len(S_bits)):
                    excl |= exclusion_masks[S_bits[i]][S_bits[j]]
            
            for x in range(N):
                if (S >> x) & 1:
                    continue
                if (excl >> x) & 1:
                    continue
                # Try adding x
                new_S = S | (1 << x)
                new_S_bits = S_bits + [x]
                # Check if valid (should be since we checked excl)
                sz = len(new_S_bits)
                if sz > best_size:
                    best_size = sz
                    best_set = new_S
                S = new_S
                S_bits = new_S_bits
                excl |= 0  # will recompute
                improved = True
                break
        
        # Try to swap: remove one, add another
        for _ in range(50):
            if time.time() > deadline - 2:
                return
            if not S_bits:
                break
            # Recompute excl for current S
            excl = 0
            for i in range(len(S_bits)):
                for j in range(i + 1, len(S_bits)):
                    excl |= exclusion_masks[S_bits[i]][S_bits[j]]
            
            swapped = False
            for idx, u in enumerate(S_bits):
                # Remove u
                new_S_bits = [b for i, b in enumerate(S_bits) if i != idx]
                # Recompute excl without u
                new_excl = 0
                for i in range(len(new_S_bits)):
                    for j in range(i + 1, len(new_S_bits)):
                        new_excl |= exclusion_masks[new_S_bits[i]][new_S_bits[j]]
                
                # Try to add a new element
                for x in range(N):
                    if x == u:
                        continue
                    if any(x == b for b in new_S_bits):
                        continue
                    if (new_excl >> x) & 1:
                        continue
                    # Valid swap
                    final_S_bits = new_S_bits + [x]
                    sz = len(final_S_bits)
                    if sz > best_size:
                        best_size = sz
                        S_bits = final_S_bits
                        final_S = 0
                        for b in final_S_bits:
                            final_S |= (1 << b)
                        best_set = final_S
                    swapped = True
                    break
                if swapped:
                    break
            if not swapped:
                break
    
    local_search(best_greedy)

    # B&B with propagation
    def bnb(S, R, excl, S_bits):
        nonlocal best_size, best_set
        
        if time.time() > deadline:
            return
            
        R &= ~excl
        
        if R == 0:
            sz = len(S_bits)
            if sz > best_size:
                best_size = sz
                best_set = S
            return
            
        if len(S_bits) + popcount(R) <= best_size:
            return
            
        x = (R & -R).bit_length() - 1
        
        new_S = S | (1 << x)
        new_excl = excl
        new_S_bits = S_bits + [x]
        for u in S_bits:
            new_excl |= exclusion_masks[x][u]
        new_R = R & ~(1 << x)
        bnb(new_S, new_R, new_excl, new_S_bits)
        
        new_R = R & ~(1 << x)
        bnb(S, new_R, excl, S_bits)

    initial_R = (1 << N) - 1
    bnb(0, initial_R, 0, [])
    
    result = []
    for i in range(N):
        if (best_set >> i) & 1:
            result.append(i + 1)
            
    if not result:
        result = [1]
        
    return result
