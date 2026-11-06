import time
import random
from math import gcd

def construct():
    N = 128
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute LCMs
    lcm = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, N+1):
            lcm[i][j] = i * j // gcd(i, j)
            
    # Precompute bad_c[a][b] bitmask
    bad_c = [[0]*(N+1) for _ in range(N+1)]
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            L_ab = lcm[a][b]
            mask = 0
            for c in range(1, N+1):
                if c == a or c == b: continue
                if lcm[a][c] == L_ab and lcm[b][c] == L_ab:
                    mask |= (1 << (c-1))
            bad_c[a][b] = mask
            bad_c[b][a] = mask
            
    def calc_forbidden(S_list):
        mask = 0
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                mask |= bad_c[a][b]
        return mask

    # Greedy initialization
    best_S = []
    best_mask = 0
    orders = [list(range(1, N+1)), list(range(N, 0, -1))]
    random.seed(42)
    for _ in range(20):
        order = list(range(1, N+1))
        random.shuffle(order)
        orders.append(order)
        
    for order in orders:
        S_list = []
        S_mask = 0
        Forbidden_mask = 0
        
        for x in order:
            if (S_mask >> (x-1)) & 1:
                continue
            if (Forbidden_mask >> (x-1)) & 1:
                continue
            
            S_list.append(x)
            S_mask |= (1 << (x-1))
            
            for z in S_list[:-1]:
                Forbidden_mask |= bad_c[x][z]
                
        if len(S_list) > len(best_S):
            best_S = S_list
            best_mask = S_mask
            
    current_S = list(best_S)
    current_mask = best_mask
    current_Forbidden = calc_forbidden(current_S)
    best_size = len(current_S)
    stall_counter = 0
    kick_size = 1
    
    while time.time() < deadline:
        added = False
        for x in range(1, N+1):
            if (current_mask >> (x-1)) & 1:
                continue
            if (current_Forbidden >> (x-1)) & 1:
                continue
            
            current_S.append(x)
            current_mask |= (1 << (x-1))
            
            for z in current_S[:-1]:
                current_Forbidden |= bad_c[x][z]
                
            added = True
            stall_counter = 0
            if len(current_S) > best_size:
                best_size = len(current_S)
            break
            
        if added:
            continue
            
        # Stuck
        if len(current_S) == 0:
            break
            
        stall_counter += 1
        if stall_counter > 10:
            kick_size = min(3, kick_size + 1)
            stall_counter = 0
        else:
            kick_size = 1
            
        # Kick
        for _ in range(kick_size):
            if not current_S:
                break
            idx = random.randint(0, len(current_S)-1)
            removed = current_S.pop(idx)
            current_mask &= ~(1 << (removed-1))
            
        current_Forbidden = calc_forbidden(current_S)

    return current_S
