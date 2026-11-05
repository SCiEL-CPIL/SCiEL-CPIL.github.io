"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        s = set(S)
        for a in S:
            for b in S:
                if a + b in s:
                    return False
        return True

    def count_violations(S):
        s = set(S)
        cnt = 0
        for a in S:
            for b in S:
                if a + b in s:
                    cnt += 1
        return cnt

    def greedy_expand(S, remaining, max_iter=100000):
        """Greedily add elements from remaining to S if they don't create violations."""
        S_set = set(S)
        S_list = list(S)
        added = 0
        for x in remaining:
            if x in S_set:
                continue
            # Check if adding x creates any violation
            ok = True
            for a in S_list:
                if a + x in S_set or x + a in S_set:
                    ok = False
                    break
                if 2 * x in S_set or x + x in S_set:
                    ok = False
                    break
            if ok:
                S_list.append(x)
                S_set.add(x)
                added += 1
        return S_list

    def greedy_expand_fast(S, remaining, all_vals_set):
        """Faster greedy: precompute conflicts."""
        S_set = set(S)
        S_list = list(S)
        for x in remaining:
            if x in S_set:
                continue
            # x conflicts if exists a in S such that a+x in S, or 2x in S
            conflict = False
            for a in S_list:
                if a + x in S_set:
                    conflict = True
                    break
            if not conflict and 2 * x in S_set:
                conflict = True
            if not conflict:
                S_list.append(x)
                S_set.add(x)
        return S_list

    def local_search_2opt(S, remaining, all_set, max_rounds=5000):
        """2-opt: try swapping out one or two elements for better ones."""
        S_set = set(S)
        S_list = list(S)
        remaining_set = set(remaining) - S_set
        best_size = len(S_list)

        for rnd in range(max_rounds):
            if time.time() > deadline - 5:
                break
            improved = False
            # Try 1-opt: remove one, add one
            for i, a in enumerate(S_list):
                new_S_list = S_list[:i] + S_list[i+1:]
                new_S_set = set(new_S_list)
                # Try adding each remaining element
                for x in list(remaining_set):
                    if x in new_S_set:
                        continue
                    ok = True
                    for b in new_S_list:
                        if b + x in new_S_set:
                            ok = False
                            break
                    if ok and 2 * x not in new_S_set:
                        new_S_list.append(x)
                        new_S_set.add(x)
                        # Check full sum-free
                        valid = True
                        for p in new_S_list:
                            for q in new_S_list:
                                if p + q in new_S_set:
                                    valid = False
                                    break
                            if not valid:
                                break
                        if valid:
                            S_list = new_S_list
                            S_set = new_S_set
                            remaining_set = set(remaining) - S_set
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                # Try 2-opt: remove two, add two
                found = False
                n = len(S_list)
                if n >= 2:
                    for i in range(min(n, 30)):
                        for j in range(i + 1, min(n, 30)):
                            new_S_list = [S_list[k] for k in range(n) if k != i and k != j]
                            new_S_set = set(new_S_list)
                            # Try pairs from remaining
                            rem_list = list(remaining_set)
                            for x in rem_list[:50]:
                                if x in new_S_set:
                                    continue
                                for y in rem_list[:50]:
                                    if y in new_S_set or y == x:
                                        continue
                                    cand = new_S_list + [x, y]
                                    cand_set = set(cand)
                                    valid = True
                                    for p in cand:
                                        for q in cand:
                                            if p + q in cand_set:
                                                valid = False
                                                break
                                        if not valid:
                                            break
                                    if valid:
                                        S_list = cand
                                        S_set = cand_set
                                        remaining_set = set(remaining) - S_set
                                        found = True
                                        improved = True
                                        break
                                if found:
                                    break
                        if found:
                            break
                if not improved:
                    break
        return S_list

    def solve_instance(size, seed):
        vals = instance(size, seed)
        vals_set = set(vals)
        best = None
        best_size = 0

        # Multiple initialization strategies
        inits = []

        # 1. Parity: odds
        inits.append([x for x in vals if x % 2 != 0])
        # 2. Parity: evens
        inits.append([x for x in vals if x % 2 == 0])
        # 3. Mod 3: residue 1
        inits.append([x for x in vals if x % 3 == 1])
        # 4. Mod 3: residue 2
        inits.append([x for x in vals if x % 3 == 2])
        # 5. Mod 4: residue 1
        inits.append([x for x in vals if x % 4 == 1])
        # 6. Mod 4: residue 3
        inits.append([x for x in vals if x % 4 == 3])
        # 7. Largest positives
        pos_sorted = sorted([x for x in vals if x > 0], reverse=True)
        inits.append(pos_sorted[:max(1, size // 3)])
        # 8. Smallest negatives
        neg_sorted = sorted([x for x in vals if x < 0])
        inits.append(neg_sorted[:max(1, size // 3)])
        # 9. Random sample
        rng = random.Random(seed + 12345)
        inits.append(rng.sample(vals, max(1, size // 4)))

        for init in inits:
            if time.time() > deadline - 10:
                break
            if not init:
                continue
            # Verify sum-free
            if not is_sum_free(init):
                continue
            remaining = [x for x in vals if x not in set(init)]
            # Greedy expand
            expanded = greedy_expand_fast(init, remaining, vals_set)
            # Local search
            result = local_search_2opt(expanded, remaining, vals_set, max_rounds=3000)
            if len(result) > best_size:
                best_size = len(result)
                best = result

        if best is None:
            # Fallback: single largest element
            best = [max(vals)]
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            results.append([0])
            continue
        res = solve_instance(size, seed)
        results.append(res)

    return results
# EVOLVE-BLOCK-END
