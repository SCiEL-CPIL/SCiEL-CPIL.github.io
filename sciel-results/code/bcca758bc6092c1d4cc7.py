"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 85
    N = 256

    # Precompute conflict as bitmasks for fast checking
    conflict_mask = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if (i * j) % (i + j) == 0:
                conflict_mask[i] |= (1 << j)
                conflict_mask[j] |= (1 << i)

    def can_add(x, S_mask):
        return (conflict_mask[x] & S_mask) == 0

    def greedy(order, S_mask=0, S_list=None):
        if S_list is None:
            S_list = []
        for x in order:
            if (1 << x) & S_mask:
                continue
            if can_add(x, S_mask):
                S_mask |= (1 << x)
                S_list.append(x)
        return S_mask, S_list

    def local_improve(S_mask, S_list, remaining_mask):
        S_list = list(S_list)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if time.time() > deadline:
                    return S_mask, S_list
                if (1 << x) & S_mask:
                    continue
                if (1 << x) & remaining_mask:
                    if can_add(x, S_mask):
                        S_mask |= (1 << x)
                        remaining_mask &= ~(1 << x)
                        S_list.append(x)
                        changed = True
        return S_mask, S_list

    best_S_mask = 0
    best_S_list = []

    # Strategy 1: Odds first, then try to add evens
    S_mask, S_list = greedy(list(range(1, N + 1, 2)))
    for e in range(2, N + 1, 2):
        if can_add(e, S_mask):
            S_mask |= (1 << e)
            S_list.append(e)
    if len(S_list) > len(best_S_list):
        best_S_mask = S_mask
        best_S_list = S_list[:]

    # Strategy 2: Various greedy orders
    orders = [list(range(1, N + 1)), list(range(N, 0, -1))]
    for _ in range(15):
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        orders.append(rand_order)

    for order in orders:
        if time.time() > deadline:
            break
        S_mask, S_list = greedy(order)
        if len(S_list) > len(best_S_list):
            best_S_mask = S_mask
            best_S_list = S_list[:]

    # Local improvement on best
    rem = 0
    for i in range(1, N + 1):
        if not (1 << i) & best_S_mask:
            rem |= (1 << i)
    best_S_mask, best_S_list = local_improve(best_S_mask, best_S_list, rem)

    # More random restarts with local improvement
    for _ in range(10):
        if time.time() > deadline:
            break
        rand_order = list(range(1, N + 1))
        random.shuffle(rand_order)
        S_mask, S_list = greedy(rand_order)
        r = 0
        for i in range(1, N + 1):
            if not (1 << i) & S_mask:
                r |= (1 << i)
        S_mask, S_list = local_improve(S_mask, S_list, r)
        if len(S_list) > len(best_S_list):
            best_S_mask = S_mask
            best_S_list = S_list[:]

    return best_S_list
# EVOLVE-BLOCK-END
