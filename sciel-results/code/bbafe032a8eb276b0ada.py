import time
import random
import math
from math import gcd
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 110

    N = 512

    def get_primes(n):
        sieve = [True] * n
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    primes = get_primes(N)
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    def get_factor_mask(n):
        mask = 0
        tmp = n
        for i, p in enumerate(primes):
            if p * p > tmp:
                break
            if tmp % p == 0:
                mask |= (1 << i)
                while tmp % p == 0:
                    tmp //= p
        if tmp > 1:
            mask |= (1 << prime_to_idx[tmp])
        return mask

    factor_masks = [0] * N
    for a in range(1, N):
        factor_masks[a] = get_factor_mask(a)

    scores = [0.0] * N
    for a in range(1, N):
        scores[a] = 1.0 / (N - a)

    # --- Heuristic Phase ---
    
    def greedy_construct(ordering):
        chosen = []
        chosen_set = set()
        chosen_mask = 0
        for a in ordering:
            if factor_masks[a] & chosen_mask == 0:
                chosen.append(a)
                chosen_set.add(a)
                chosen_mask |= factor_masks[a]
        return chosen, chosen_set, chosen_mask

    best_chosen = []
    best_score = -1.0
    best_chosen_set = set()
    best_chosen_mask = 0

    def update_best(chosen, chosen_set, chosen_mask, score):
        nonlocal best_chosen, best_score, best_chosen_set, best_chosen_mask
        if score > best_score:
            best_score = score
            best_chosen = chosen
            best_chosen_set = chosen_set
            best_chosen_mask = chosen_mask

    # Strategy 1: Score descending
    ordering1 = list(range(511, 0, -1))
    c1, s1, m1 = greedy_construct(ordering1)
    update_best(c1, s1, m1, sum(scores[a] for a in c1))

    # Strategy 2: Score / npf
    def npf(a):
        return bin(factor_masks[a]).count('1')
    ordering2 = sorted(range(1, 512), key=lambda x: scores[x] / max(1, npf(x)), reverse=True)
    c2, s2, m2 = greedy_construct(ordering2)
    update_best(c2, s2, m2, sum(scores[a] for a in c2))

    # Randomized Greedy
    for _ in range(20):
        if time.time() > deadline - 30: # Reserve time for B&B
            break
        ordering = list(range(1, 512))
        random.shuffle(ordering)
        weighted = [(random.random() * 0.2 + scores[x] / max(1, npf(x)), x) for x in ordering]
        weighted.sort(reverse=True)
        ordering_w = [x for _, x in weighted]
        cw, sw, mw = greedy_construct(ordering_w)
        update_best(cw, sw, mw, sum(scores[a] for a in cw))

    # Local Search on best
    def local_search_1swap(chosen, chosen_set, chosen_mask, current_score, max_rounds=10):
        for _ in range(max_rounds):
            if time.time() > deadline - 20:
                break
            improved = False
            for a in list(chosen):
                best_b = None
                best_gain = 0.0
                temp_mask = chosen_mask & ~factor_masks[a]
                for b in range(511, 0, -1):
                    if b in chosen_set:
                        continue
                    if factor_masks[b] & temp_mask:
                        continue
                    gain = scores[b] - scores[a]
                    if gain > best_gain + 1e-12:
                        best_gain = gain
                        best_b = b
                if best_b is not None:
                    chosen.remove(a)
                    chosen.append(best_b)
                    chosen_set.remove(a)
                    chosen_set.add(best_b)
                    chosen_mask = (chosen_mask & ~factor_masks[a]) | factor_masks[best_b]
                    current_score += best_gain
                    improved = True
                    break
            if not improved:
                break
        return chosen, chosen_set, chosen_mask, current_score

    best_chosen, best_chosen_set, best_chosen_mask, best_score = local_search_1swap(
        best_chosen, best_chosen_set, best_chosen_mask, best_score
    )

    # --- Branch and Bound Phase ---
    
    # Sort nodes by score descending for B&B
    sorted_nodes = sorted(range(1, 512), key=lambda x: scores[x], reverse=True)
    node_weights = [scores[a] for a in sorted_nodes]
    node_masks = [factor_masks[a] for a in sorted_nodes]
    
    # Suffix sum of weights
    suffix_sum = [0.0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + node_weights[i]

    best_bb_score = best_score
    best_bb_set = set(best_chosen)
    
    def bb(idx, current_mask, current_score, current_set):
        nonlocal best_bb_score, best_bb_set
        
        if time.time() > deadline:
            return
        
        if idx == N:
            if current_score > best_bb_score:
                best_bb_score = current_score
                best_bb_set = set(current_set)
            return

        # Loose bound
        if current_score + suffix_sum[idx] <= best_bb_score:
            return

        # Tight bound: sum of weights of remaining nodes coprime to current_mask
        # Optimization: Stop summing if we already know we won't prune
        tight_bound = current_score
        will_prune = True
        for j in range(idx, N):
            if node_masks[j] & current_mask == 0:
                tight_bound += node_weights[j]
                if tight_bound > best_bb_score:
                    will_prune = False
                    break
        
        if will_prune:
            return

        # Branch 1: Include sorted_nodes[idx]
        a = sorted_nodes[idx]
        if node_masks[idx] & current_mask == 0:
            current_set.append(a)
            bb(idx + 1, current_mask | node_masks[idx], current_score + node_weights[idx], current_set)
            current_set.pop()

        # Branch 2: Exclude sorted_nodes[idx]
        bb(idx + 1, current_mask, current_score, current_set)

    # Run B&B
    bb(0, 0, 0.0, [])

    if best_bb_score > best_score:
        best_score = best_bb_score
        best_chosen = list(best_bb_set)

    return best_chosen
