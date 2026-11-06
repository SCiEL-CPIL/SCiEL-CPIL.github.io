import time
import random
from collections import deque

N = 12
FULL = (1 << N) - 1

def _reset_length_bitmask(a, b):
    """Exact shortest reset word length via BFS on subset automaton (bitmasks).
    Returns -1 if not synchronizing."""
    dest_a = [1 << a[i] for i in range(N)]
    dest_b = [1 << b[i] for i in range(N)]

    # Precompute mapping for all 2^N subsets for both letters
    map_a = [0] * (1 << N)
    map_b = [0] * (1 << N)
    
    for mask in range(1 << N):
        res_a = 0
        res_b = 0
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            res_a |= dest_a[idx]
            res_b |= dest_b[idx]
            m ^= lsb
        map_a[mask] = res_a
        map_b[mask] = res_b

    visited = [False] * (1 << N)
    visited[FULL] = True
    queue = deque([(FULL, 0)])
    while queue:
        mask, dist = queue.popleft()
        if mask & (mask - 1) == 0:
            return dist
        nm = map_a[mask]
        if not visited[nm]:
            visited[nm] = True
            queue.append((nm, dist + 1))
        nm = map_b[mask]
        if not visited[nm]:
            visited[nm] = True
            queue.append((nm, dist + 1))
    return -1

def construct():
    deadline = time.time() + 85.0

    def eval_ab(a, b):
        if time.time() > deadline:
            return -1
        return _reset_length_bitmask(a, b)

    # Generate candidate families
    candidates = []
    
    # Family 1: Standard Cycle a, various b
    a_cycle = list(range(1, N)) + [0]
    
    # b = Cycle
    candidates.append((a_cycle, list(range(1, N)) + [0]))
    
    # b = Identity
    candidates.append((a_cycle, list(range(N))))
    
    # b = Almost Identity (last maps to 0)
    b_almost_id = list(range(N))
    b_almost_id[N-1] = 0
    candidates.append((a_cycle, b_almost_id))
    
    # b = Shift 1, last maps to 0 (Pinwheel variant?)
    b_shift_break = list(range(1, N)) + [0]
    b_shift_break[N-1] = 0
    candidates.append((a_cycle, b_shift_break))
    
    # b = Shift 1, second last maps to 0
    b_shift_break2 = list(range(1, N)) + [0]
    b_shift_break2[N-2] = 0
    candidates.append((a_cycle, b_shift_break2))
    
    # b = Reverse Cycle
    candidates.append((a_cycle, list(range(N-1, -1, -1))))
    
    # b = Reverse Cycle with break
    b_rev_break = list(range(N-1, -1, -1))
    b_rev_break[0] = 0
    candidates.append((a_cycle, b_rev_break))
    
    # b = i -> i+1 mod N, except i=0 -> 1, i=1 -> 0 (Swap 0,1 in cycle)
    b_swap = list(range(1, N)) + [0]
    b_swap[0], b_swap[1] = b_swap[1], b_swap[0]
    candidates.append((a_cycle, b_swap))

    # Add some random perturbations of the cycle
    rng_init = random.Random(42)
    for _ in range(50):
        b_rand = list(range(1, N)) + [0]
        # Swap two random elements
        i, j = rng_init.randrange(N), rng_init.randrange(N)
        if i != j:
            b_rand[i], b_rand[j] = b_rand[j], b_rand[i]
        candidates.append((a_cycle, b_rand))
        
    # Also try a = Identity, b = Cycle? (Symmetric)
    a_id = list(range(N))
    candidates.append((a_id, list(range(1, N)) + [0]))
    
    best_a, best_b, best_len = None, None, -1

    # Evaluate all candidates
    for a, b in candidates:
        if time.time() > deadline:
            break
        L = eval_ab(a, b)
        if L > best_len:
            best_len = L
            best_a = a
            best_b = b

    if best_a is None:
        # Fallback
        best_a = a_cycle
        best_b = list(range(1, N)) + [0]
        best_len = eval_ab(best_a, best_b)

    # Local Search / Hill Climbing
    def improve(a, b, budget):
        nonlocal best_a, best_b, best_len
        rng = random.Random(12345)
        cur_len = best_len
        no_improve_count = 0
        max_no_improve = 50
        
        while time.time() < budget:
            improved = False
            # Try several random moves
            for _ in range(20):
                if time.time() > budget:
                    break
                r = rng.random()
                if r < 0.5:
                    # Modify b
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j: continue
                    nb = list(best_b)
                    nb[i], nb[j] = nb[j], nb[i]
                    L = eval_ab(best_a, nb)
                    if L > cur_len:
                        best_b, cur_len, improved = nb, L, True
                        no_improve_count = 0
                elif r < 0.9:
                    # Modify a
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j: continue
                    na = list(best_a)
                    na[i], na[j] = na[j], na[i]
                    L = eval_ab(na, best_b)
                    if L > cur_len:
                        best_a, cur_len, improved = na, L, True
                        no_improve_count = 0
                else:
                    # Swap letters
                    na, nb = list(best_b), list(best_a)
                    L = eval_ab(na, nb)
                    if L > cur_len:
                        best_a, best_b, cur_len, improved = na, nb, L, True
                        no_improve_count = 0
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > max_no_improve:
                    break
            else:
                no_improve_count = 0
                
        best_len = cur_len

    improve(best_a, best_b, deadline)

    return [best_a, best_b]
