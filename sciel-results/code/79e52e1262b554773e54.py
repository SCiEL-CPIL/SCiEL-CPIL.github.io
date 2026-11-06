import time
import random
from itertools import combinations


def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s limit

    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)

    # Map triple to its index
    triple_index = {t: i for i, t in enumerate(all_triples)}

    # Precompute for each triple, the conflict groups
    # triple_conflicts[i] = list of 7-tuples of indices of other triples
    # that together with triple i form a complete K(2,2,2)
    triple_conflicts = [[] for _ in range(num_triples)]

    for i, t in enumerate(all_triples):
        a, b, c = t
        remaining = [x for x in range(n) if x not in (a, b, c)]
        for a_prime in remaining:
            for b_prime in remaining:
                if b_prime == a_prime:
                    continue
                for c_prime in remaining:
                    if c_prime == a_prime or c_prime == b_prime:
                        continue
                    trans = [
                        (a, b, c),
                        (a, b, c_prime),
                        (a, b_prime, c),
                        (a, b_prime, c_prime),
                        (a_prime, b, c),
                        (a_prime, b, c_prime),
                        (a_prime, b_prime, c),
                        (a_prime, b_prime, c_prime),
                    ]
                    others = []
                    for tr in trans:
                        if tr != t:
                            others.append(triple_index[tr])
                    triple_conflicts[i].append(tuple(others))

    best_solution = []
    best_len = 0

    def greedy_add(order):
        added = set()
        result = []
        for idx in order:
            if idx in added:
                continue
            valid = True
            for others in triple_conflicts[idx]:
                all_present = True
                for o in others:
                    if o not in added:
                        all_present = False
                        break
                if all_present:
                    valid = False
                    break
            if valid:
                added.add(idx)
                result.append(all_triples[idx])
        return result

    while time.time() < deadline:
        order = list(range(num_triples))
        random.shuffle(order)
        result = greedy_add(order)

        if len(result) > best_len:
            best_len = len(result)
            best_solution = result

        if time.time() > deadline - 3:
            break

    return best_solution
