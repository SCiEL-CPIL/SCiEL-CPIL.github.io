import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 145.0
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    # Lower bound for scope
    lower_bound = total_diffs + 1

    def is_valid(rows):
        if len(rows) != n_rows:
            return False
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_construct(n_rows, k_val, rng, max_scope=None):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for pos in range(k_val):
                found = False
                if max_scope is not None:
                    lo = row[-1] + 1
                    hi = max_scope
                    if lo > hi:
                        return None
                    for _ in range(20):
                        x = lo + int((rng.random() ** 2) * (hi - lo + 1))
                        if x > hi: x = hi
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
                    if not found:
                        x = lo
                        while x <= hi:
                            ok = True
                            for a in row:
                                if (x - a) in used:
                                    ok = False
                                    break
                            if ok:
                                for a in row:
                                    used.add(x - a)
                                row.append(x)
                                found = True
                                break
                            x += 1
                        if not found:
                            return None
                else:
                    x = row[-1]
                    while True:
                        x += 1
                        ok = True
                        for a in row:
                            if (x - a) in used:
                                ok = False
                                break
                        if ok:
                            for a in row:
                                used.add(x - a)
                            row.append(x)
                            found = True
                            break
            rows.append(row)
        return rows

    def backtracking_search(n_rows, k_val, max_scope, rng, time_limit):
        """
        Backtracking search with symmetry breaking.
        Rows are built in order, with constraint that rows[i][1] > rows[i-1][1] for i > 0.
        """
        used_diffs = set()
        rows = []
        
        def build_row(used_diffs, max_scope, rng, min_first=None):
            """
            Build a single row of length m, starting with 0, max entry <= max_scope.
            If min_first is not None, the second element must be > min_first.
            """
            row = [0]
            
            def backtrack(pos):
                if time.time() > time_limit:
                    return False
                
                if pos == m:
                    return True
                
                lo = row[-1] + 1
                if min_first is not None and pos == 1:
                    lo = max(lo, min_first + 1)
                
                hi = max_scope
                if lo > hi:
                    return False
                
                # Try candidates in random order, biased towards smaller values
                candidates = list(range(lo, hi + 1))
                rng.shuffle(candidates)
                
                for x in candidates:
                    new_diffs = []
                    valid = True
                    for i in range(len(row)):
                        d = x - row[i]
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if not valid:
                        continue
                    
                    row.append(x)
                    for d in new_diffs:
                        used_diffs.add(d)
                    
                    if backtrack(pos + 1):
                        return True
                    
                    row.pop()
                    for d in new_diffs:
                        used_diffs.remove(d)
                
                return False
            
            if backtrack(1):
                return row
            return None
        
        # Build rows one by one with symmetry breaking
        for r in range(n_rows):
            if time.time() > time_limit:
                return None
            
            min_first = rows[r-1][1] if r > 0 else None
            row = build_row(used_diffs, max_scope, rng, min_first)
            if row is None:
                return None
            
            rows.append(row)
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used_diffs.add(row[j] - row[i])
        
        return rows

    def improve_scope(current_rows, current_scope, rng, deadline):
        """
        Try to reduce scope by 1 using backtracking search.
        """
        if time.time() > deadline - 5:
            return current_rows, current_scope
        
        target_scope = current_scope - 1
        if target_scope < lower_bound:
            return current_rows, current_scope
        
        time_limit = time.time() + min(15, deadline - time.time() - 3)
        rng_attempt = random.Random(rng.randint(0, 1000000))
        new_rows = backtracking_search(n_rows, k, target_scope, rng_attempt, time_limit)
        
        if new_rows is not None:
            return new_rows, target_scope
        
        return current_rows, current_scope

    # Main execution
    rng = random.Random(42)
    
    # 1. Get a baseline solution using greedy
    baseline_rows = greedy_construct(n_rows, k, rng)
    if baseline_rows is None:
        # Fallback: simple sequential
        used = set()
        baseline_rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < m:
                x += 1
                ok = True
                for a in row:
                    if (x - a) in used:
                        ok = False
                        break
                if ok:
                    for a in row:
                        used.add(x - a)
                    row.append(x)
            baseline_rows.append(row)
    
    best_rows = baseline_rows
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Try to improve using backtracking search with symmetry breaking
    if time.time() < deadline - 20:
        # Try to find a solution with scope close to lower bound
        for target_scope in range(lower_bound, min(best_scope, lower_bound + 50)):
            if time.time() > deadline - 10:
                break
            
            time_limit = time.time() + min(10, deadline - time.time() - 5)
            rng_attempt = random.Random(rng.randint(0, 1000000))
            rows = backtracking_search(n_rows, k, target_scope, rng_attempt, time_limit)
            
            if rows is not None:
                best_rows = rows
                best_scope = target_scope
                break
    
    # 3. Try to polish the best solution
    if time.time() < deadline - 5:
        best_rows, best_scope = improve_scope(best_rows, best_scope, rng, deadline)
    
    return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
