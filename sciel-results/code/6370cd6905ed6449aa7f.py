import time
import random
import math

N = 17
M = 1 << N

_CACHE = {}

def _get_cache():
    if 'W' not in _CACHE:
        # Compute W
        W = [0] * N
        for w in range(N):
            x = 0
            step = 1 << (w + 1)
            half = 1 << w
            block_mask = (1 << half) - 1
            for i in range(0, M, step):
                x |= block_mask << (i + half)
            W[w] = x
        _CACHE['W'] = W
        
        # Compute Expected Popcounts
        E = [0] * N
        # Precompute binomial coefficients up to N
        # C(N, k)
        # Using math.comb
        for i in range(N):
            count = 0
            for j in range(i + 1, N + 1):
                count += math.comb(N, j)
            E[i] = count
        _CACHE['E'] = E
    return _CACHE['W'], _CACHE['E']

INIT_W, EXP_POP = _get_cache()

def _popcount(x):
    return bin(x).count('1')

def _is_valid(W):
    # Check nestedness
    for i in range(N - 1):
        if (W[i] & W[i+1]) != W[i+1]:
            return False
    # Check popcounts
    for i in range(N):
        if _popcount(W[i]) != EXP_POP[i]:
            return False
    return True

def _apply_network(W, network):
    W = W[:]
    for i, j in network:
        A = W[i]
        B = W[j]
        W[i] = A & B
        W[j] = A | B
    return W

def _verify_network(network):
    W_final = _apply_network(INIT_W, network)
    return _is_valid(W_final)

def _reduce_network(net):
    changed = True
    while changed:
        changed = False
        for i in range(len(net)):
            new_net = net[:i] + net[i+1:]
            if _verify_network(new_net):
                net = new_net
                changed = True
                break
    return net

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Seed
    seed_network = []
    for p in range(N):
        for i in range(p % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    current_network = seed_network
    best_network = current_network[:]
    best_len = len(current_network)
    
    current_network = _reduce_network(current_network)
    best_network = current_network[:]
    best_len = len(current_network)
    
    random.seed(42)
    
    while time.time() < deadline:
        cand = best_network[:]
        
        num_mutations = random.randint(1, 3)
        for _ in range(num_mutations):
            if not cand:
                break
            op = random.randint(0, 2)
            if op == 0: # Swap
                if len(cand) > 1:
                    i, j = random.sample(range(len(cand)), 2)
                    cand[i], cand[j] = cand[j], cand[i]
            elif op == 1: # Move
                i = random.randint(0, len(cand) - 1)
                j = random.randint(0, len(cand))
                comp = cand.pop(i)
                cand.insert(j, comp)
            else: # Insert
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                pos = random.randint(0, len(cand))
                cand.insert(pos, (i, j))
        
        cand = _reduce_network(cand)
        
        if len(cand) < best_len:
            best_len = len(cand)
            best_network = cand[:]
            
    return best_network
