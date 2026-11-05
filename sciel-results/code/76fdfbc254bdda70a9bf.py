import time
import random
from math import gcd

def construct():
    N = 128
    # Precompute gcd table
    gt = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gt[i][j] = g
            gt[j][i] = g

    def is_valid_set(sel):
        """Check if the set is valid (no bad triples)."""
        lst = list(sel)
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(i + 1, n):
                b = lst[j]
                gab = gt[a][b]
                for k in range(j + 1, n):
                    c = lst[k]
                    if gt[a][c] == gab and gt[b][c] == gab:
                        return False
        return True

    def can_add(x, sel):
        """True if x can be added to sel without creating a bad triple."""
        groups = {}
        for a in sel:
            g = gt[x][a]
            if g not in groups:
                groups[g] = []
            groups[g].append(a)
            
        for g, elems in groups.items():
            if len(elems) < 2:
                continue
            for i in range(len(elems)):
                a = elems[i]
                for j in range(i + 1, len(elems)):
                    b = elems[j]
                    if gt[a][b] == g:
                        return False
        return True

    def polish(sel):
        """Greedy completion: add any element that fits."""
        sel = set(sel)
        changed = True
        while changed:
            changed = False
            for x in range(1, N + 1):
                if x not in sel and can_add(x, sel):
                    sel.add(x)
                    changed = True
        return sel

    def local_search(sel, max_iters=20):
        sel = set(sel)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline - 5:
            improved = False
            iters += 1

            # 1) Add any excluded element that fits
            for x in range(1, N + 1):
                if x not in sel and can_add(x, sel):
                    sel.add(x)
                    improved = True
                    break

            # 2) Random 1-1 swap
            if not improved:
                excluded = [x for x in range(1, N + 1) if x not in sel]
                if excluded and sel:
                    random.shuffle(excluded)
                    sel_list = list(sel)
                    random.shuffle(sel_list)
                    for _ in range(10):
                        r = random.choice(sel_list)
                        y = random.choice(excluded)
                        base = sel - {r}
                        if can_add(y, base):
                            sel = base | {y}
                            improved = True
                            break

            # 3) Random 2-1 exchange
            if not improved:
                excluded = [x for x in range(1, N + 1) if x not in sel]
                if len(excluded) >= 2 and len(sel) >= 1:
                    random.shuffle(excluded)
                    sel_list = list(sel)
                    random.shuffle(sel_list)
                    for _ in range(10):
                        r = random.choice(sel_list)
                        base = sel - {r}
                        added = []
                        random.shuffle(excluded)
                        for y in excluded:
                            if len(added) >= 2:
                                break
                            if can_add(y, base | set(added)):
                                added.append(y)
                        if len(added) >= 2:
                            sel = base | set(added)
                            improved = True
                            break

            # 4) Random 2-2 exchange
            if not improved:
                excluded = [x for x in range(1, N + 1) if x not in sel]
                if len(excluded) >= 2 and len(sel) >= 2:
                    random.shuffle(excluded)
                    sel_list = list(sel)
                    random.shuffle(sel_list)
                    for _ in range(10):
                        r1 = random.choice(sel_list)
                        r2 = random.choice([x for x in sel_list if x != r1])
                        base = sel - {r1, r2}
                        added = []
                        random.shuffle(excluded)
                        for y in excluded:
                            if len(added) >= 2:
                                break
                            if can_add(y, base | set(added)):
                                added.append(y)
                        if len(added) >= 2:
                            sel = base | set(added)
                            improved = True
                            break

        return polish(sel)

    deadline = time.time() + 55.0
    best = set()
    best_size = 0

    def consider(s):
        nonlocal best, best_size
        if len(s) > best_size:
            best_size = len(s)
            best = set(s)

    random.seed(12345)

    # Initial seeds
    seeds = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: x.bit_count()),
    ]
    for _ in range(10):
        o = list(range(1, N + 1))
        random.shuffle(o)
        seeds.append(o)

    for o in seeds:
        if time.time() > deadline:
            break
        s = polish(set(o))
        consider(s)
        if time.time() > deadline:
            break
        s = local_search(s)
        consider(s)

    # ILS
    current_best = best
    while time.time() < deadline - 10:
        # Perturb
        perturbed = set(current_best)
        comp = [x for x in range(1, N + 1) if x not in perturbed]
        
        # Try to find a valid perturbation
        valid_perturb = False
        for _ in range(10):
            if time.time() > deadline - 10:
                break
            k = random.choice([1, 2, 3])
            if len(perturbed) < k or len(comp) < k:
                continue
            
            to_remove = random.sample(list(perturbed), k)
            new_set = perturbed - set(to_remove)
            candidates = random.sample(comp, k)
            trial_set = new_set | set(candidates)
            
            # Quick validity check
            if is_valid_set(trial_set):
                perturbed = trial_set
                valid_perturb = True
                break
        
        if valid_perturb:
            s = local_search(perturbed)
            consider(s)
            if len(s) > len(current_best):
                current_best = s

    # Final polish
    final_best = polish(best)
    consider(final_best)

    if not best:
        best = {1}
    return sorted(best)
