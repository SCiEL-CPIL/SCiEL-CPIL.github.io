import random
import time
import math

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118  # safety margin before 120s kill

    total_diffs = k * (k + 1) // 2  # per row
    counting_lb = n_rows * total_diffs

    def row_diffs(row):
        s = set()
        L = len(row)
        for i in range(L):
            ri = row[i]
            for j in range(i + 1, L):
                s.add(row[j] - ri)
        return s

    def is_valid(rows):
        all_d = set()
        for row in rows:
            if row[0] != 0:
                return False
            for i in range(len(row) - 1):
                if row[i + 1] <= row[i]:
                    return False
            d = row_diffs(row)
            if d & all_d:
                return False
            all_d |= d
        return True

    def score_of(rows):
        return max(max(r) for r in rows)

    # 1. Initial Greedy Construction
    def initial_construct(max_hint):
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                found = False
                for v in range(base + 1, max_hint + 1):
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        row.append(v)
                        for a in row[:-1]:
                            local.add(v - a)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
            used |= row_diffs(row)
        return rows

    hint = max(counting_lb + 10, 130)
    init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 60
        init_rows = initial_construct(hint)
    if init_rows is None:
        hint = counting_lb + 200
        init_rows = initial_construct(hint)
    if init_rows is None:
        # Fallback: Unbounded greedy
        rows = []
        used = set()
        for i in range(n_rows):
            row = [0]
            local = set()
            for _ in range(k):
                base = row[-1]
                v = base + 1
                while True:
                    ok = True
                    for a in row:
                        d = v - a
                        if d in used or d in local:
                            ok = False
                            break
                    if ok:
                        break
                    v += 1
                row.append(v)
                for a in row[:-1]:
                    local.add(v - a)
            rows.append(row)
            used |= row_diffs(row)
        init_rows = rows

    best_rows = [r[:] for r in init_rows]
    best_score = score_of(best_rows)

    # 2. Candidate Pool + Independent Set Search
    def gen_rand_row(max_val, seed):
        local_rand = random.Random(seed)
        row = [0]
        local_diffs = set()
        for _ in range(k):
            base = row[-1]
            valid_cands = []
            for v in range(base + 1, max_val + 1):
                ok = True
                for a in row:
                    d = v - a
                    if d in local_diffs:
                        ok = False
                        break
                if ok:
                    valid_cands.append(v)
                    if len(valid_cands) > 20:
                        break
            if not valid_cands:
                return None
            choice_idx = local_rand.randint(0, min(4, len(valid_cands) - 1))
            v = valid_cands[choice_idx]
            row.append(v)
            for a in row[:-1]:
                local_diffs.add(v - a)
        if row[-1] <= max_val:
            return row
        return None

    def find_independent_set(candidates, target_size):
        n_cand = len(candidates)
        if n_cand < target_size:
            return None
        
        conflicts = [set() for _ in range(n_cand)]
        for i in range(n_cand):
            di = candidates[i][1]
            for j in range(i + 1, n_cand):
                dj = candidates[j][1]
                if di & dj:
                    conflicts[i].add(j)
                    conflicts[j].add(i)
        
        selected = []
        available = set(range(n_cand))
        
        def backtrack(remaining):
            if time.time() > deadline:
                return False
            if remaining == 0:
                return True
            if not available:
                return False
            if len(available) < remaining:
                return False
            
            best_node = -1
            max_conflicts = -1
            avail_list = list(available)
            random.shuffle(avail_list)
            for node in avail_list[:50]:
                deg = len(conflicts[node] & available)
                if deg > max_conflicts:
                    max_conflicts = deg
                    best_node = node
            
            if best_node == -1:
                best_node = avail_list[0]
            
            selected.append(best_node)
            neighbors = conflicts[best_node] & available
            available.remove(best_node)
            available.difference_update(neighbors)
            
            if backtrack(remaining - 1):
                return True
            
            available.add(best_node)
            available.update(neighbors)
            selected.pop()
            
            return False

        if backtrack(target_size):
            return [candidates[i][0] for i in selected]
        return None

    # Iterative Deepening
    for target in range(best_score - 1, counting_lb - 1, -1):
        if time.time() > deadline - 10:
            break
            
        candidates = []
        for i in range(300):
            if time.time() > deadline - 10:
                break
            r = gen_rand_row(target, i)
            if r:
                d = row_diffs(r)
                candidates.append((r, d))
                if len(candidates) >= 200:
                    break
        
        if len(candidates) < n_rows:
            continue
            
        sol = find_independent_set(candidates, n_rows)
        if sol:
            if is_valid(sol):
                new_score = score_of(sol)
                if new_score < best_score:
                    best_score = new_score
                    best_rows = [r[:] for r in sol]

    if not is_valid(best_rows):
        return init_rows

    return best_rows

# EVOLVE-BLOCK-END
