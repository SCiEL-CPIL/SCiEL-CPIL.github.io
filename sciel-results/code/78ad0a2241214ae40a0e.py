# EVOLVE-BLOCK-START
import time
import random
import sys

def construct(n_rows, k):
    start = time.time()
    deadline = start + 135  # safety margin before 150s limit

    def now():
        return time.time()

    def time_left():
        return deadline - now()

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

    sys.setrecursionlimit(100000)

    def diff_set_of_row(row):
        s = set()
        n = len(row)
        for i in range(n):
            ai = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ai)
        return s

    def validate(rows):
        used = set()
        for r in rows:
            if not r or r[0] != 0:
                return False
            for a, b in zip(r, r[1:]):
                if b <= a:
                    return False
            diffs_list = []
            n = len(r)
            for i in range(n):
                ai = r[i]
                for j in range(i + 1, n):
                    diffs_list.append(r[j] - ai)
            if len(diffs_list) != len(set(diffs_list)):
                return False
            for d in diffs_list:
                if d in used:
                    return False
                used.add(d)
        return True

    def max_val(rows):
        return max(max(r) for r in rows)

    # ---------- Bitset backtracking row builder ----------
    # used_bits: an integer whose bit d is 1 if difference d is already used.
    # We need differences up to scope, so we mask with (1 << (scope+1)) - 1.

    def build_row_backtrack(used_bits, scope, time_limit=None, rng=None, max_nodes=None, prefer_random=False):
        """Find a strictly increasing row of length row_len starting at 0,
        all <= scope, with no difference colliding with used_bits (and none
        internal to the row). Returns row or None."""
        if rng is None:
            rng = random
        mask = (1 << (scope + 1)) - 1
        nodes = [0]
        target = row_len
        bit_of = [1 << d for d in range(scope + 1)]

        def conflict(row, x, cur_used):
            seen = 0
            for a in row:
                d = x - a
                if d > scope:
                    return False
                bit = bit_of[d]
                if (used_bits & bit) or (seen & bit) or (cur_used & bit):
                    return True
                seen |= bit
            return False

        def dfs(row, used, depth):
            if max_nodes is not None:
                nodes[0] += 1
                if nodes[0] > max_nodes:
                    return None
            if time_limit is not None and now() > time_limit:
                return None
            if len(row) == target:
                return list(row)

            last = row[-1]
            rem = target - len(row)
            min_next = last + 1
            max_next = scope - (rem - 1)
            if min_next > max_next:
                return None

            cands = list(range(min_next, max_next + 1))
            if prefer_random:
                rng.shuffle(cands)
            else:
                r = rng.random()
                if r < 0.3:
                    if len(cands) > 4:
                        split = min(3, len(cands))
                        head = cands[:split]
                        tail = cands[split:]
                        rng.shuffle(tail)
                        cands = head + tail
                elif r < 0.5:
                    rng.shuffle(cands)

            for x in cands:
                if conflict(row, x, used):
                    continue
                new_used = used
                for a in row:
                    d = x - a
                    new_used |= bit_of[d]
                row.append(x)
                res = dfs(row, new_used, depth + 1)
                row.pop()
                if res is not None:
                    return res
            return None

        res = dfs([0], used_bits & mask, 0)
        return res

    def build_row_greedy(used_bits, scope=None, temp=0.0, rng=None):
        """Fast greedy row builder. Returns row or None."""
        if rng is None:
            rng = random
        row = [0]
        x = 0
        while len(row) < row_len:
            x += 1
            if scope is not None and x > scope:
                return None
            new_diffs = [x - a for a in row]
            conflict = False
            seen = 0
            for d in new_diffs:
                if d > 5000:
                    continue
                bit = 1 << d
                if (used_bits & bit) or (seen & bit):
                    conflict = True
                    break
                seen |= bit
            if conflict:
                continue
            if temp > 0 and rng.random() < temp and x < 500:
                continue
            row.append(x)
            for d in new_diffs:
                used_bits |= (1 << d)
        return row

    # ---------- New Strategy: Difference-Packing Heuristic ----------
    # Instead of building rows one by one, we attempt to assign the largest
    # differences first to minimize the "scope" (max value).
    # For each row, we define a set of "structural differences" that MUST exist.
    # Specifically, if a row is [0, x1, ..., x_{k-1}], the differences
    # x_{k-1}-0, x_{k-1}-x_{k-2}, ..., x_1-0 are the "critical" ones.
    # By packing these large differences into the available space (1..M),
    # we constrain the rows to be as "compact" as possible.

    def pack_rows_by_diffs(n_rows, scope, rng):
        """
        Attempt to construct rows by pre-assigning the largest differences.
        1. Select a set of 'large' differences for each row.
        2. Ensure these differences are distinct globally.
        3. Solve for the row values that realize these differences.
        """
        # We know we need diffs_per_row differences per row.
        # The largest difference in a row of length L ending at M is at most M.
        # Let's pick diffs_per_row * n_rows distinct integers from [1, scope].
        # But we can't just pick any; they must form a valid set of differences
        # for a strictly increasing sequence starting at 0.
        
        # A valid set of differences for a row [0, v1, ..., v_{k}] is a set D
        # such that there exists a permutation of [v1, ..., vk] where the
        # differences vi - vj are in D.
        
        # Simpler approach:
        # For each row, we define the "gaps" g1, g2, ..., gk where vi = sum_{j=1..i} gj.
        # The differences are all sums of contiguous gaps.
        # To minimize max value, we want to use large gaps.
        
        # Let's try to assign a "signature" large difference to each row.
        # Actually, let's just use a greedy assignment of the largest available
        # differences to the "end" of the rows.
        
        # 1. Initialize used_bits for differences
        used_bits = 0
        rows = []
        
        # We will pick rows one by one.
        # For each row, we want to find a set of gaps that uses large differences.
        # Let's pick the 'k' largest available differences that can form a valid row.
        # This is still hard.
        
        # Alternative:
        # Start with empty rows.
        # For scope in [1..M], try to fill the "most constrained" differences.
        
        # Let's stick to a modified greedy:
        # When building a row, prioritize placing elements that create large differences.
        # i.e., instead of picking the smallest valid next element, pick the largest valid next element?
        # No, that would make the row exceed scope.
        
        # Let's try: "Maximal Gap" heuristic.
        # When choosing the next element x in a row, among all valid x,
        # prefer x that maximizes x - last_element (the gap).
        # This forces the row to "jump" early, leaving less room for later jumps,
        # potentially keeping the total sum (max value) lower?
        # Actually, if we jump early, we use up the "distance budget" quickly.
        # If we jump late, we might be forced to have a huge final jump.
        # So, "uniform" or "balanced" gaps might be better?
        # Or "front-loaded" gaps?
        
        # Let's implement "Front-loaded Gaps":
        # At each step, pick the LARGEST valid x such that we can still complete the row
        # within scope.
        
        def build_row_frontloaded(used_bits, scope, rng):
            row = [0]
            # We need row_len - 1 more elements.
            # Remaining "slots" = row_len - len(row)
            
            # To ensure we can complete, the minimum possible sum of remaining gaps
            # is 1 * (remaining slots).
            # So max possible value for current step is scope - (remaining slots - 1).
            
            # We want to pick x such that x is as large as possible (to use up scope early?)
            # Wait, if we pick x large, we use up the "distance" to scope.
            # If we pick x small, we save distance for later.
            # If we save distance for later, we might be forced to have a very large gap at the end.
            # A very large gap at the end means the last difference is large.
            # But the largest difference is always the last element (x_k - 0).
            # So the max value is the max difference.
            # To minimize max value, we just need to minimize the last element.
            # This is equivalent to minimizing the sum of gaps.
            # The sum of gaps is fixed by the set of differences? No.
            
            # Let's try: Minimize the last element by picking the SMALLEST valid x?
            # That's the standard greedy.
            
            # Let's try: Pick x to balance the remaining capacity.
            # Target remaining capacity = scope - current_max.
            # Ideal gap = (scope - current_max) / (remaining_slots)
            
            def build_row_balanced(used_bits, scope, rng):
                row = [0]
                cur_max = 0
                slots_left = row_len - 1
                
                while slots_left > 0:
                    # Ideal next value
                    ideal_gap = (scope - cur_max) / slots_left
                    target_x = cur_max + int(ideal_gap)
                    
                    # Search around target_x for a valid x
                    # We want x to be valid (no diff conflicts)
                    # And we must be able to complete the row.
                    # Min next x = cur_max + 1
                    # Max next x = scope - (slots_left - 1)
                    
                    min_x = cur_max + 1
                    max_x = scope - (slots_left - 1)
                    
                    if min_x > max_x:
                        return None
                    
                    # Search range: from target_x downwards and upwards
                    # Prioritize x close to target
                    cands = []
                    for d in range(0, max(max_x - min_x, 1)):
                        if target_x - d >= min_x:
                            cands.append(target_x - d)
                        if target_x + d <= max_x and target_x + d != target_x:
                            cands.append(target_x + d)
                    
                    # Shuffle to add randomness
                    rng.shuffle(cands)
                    
                    found = False
                    for x in cands:
                        if x < min_x or x > max_x:
                            continue
                        new_diffs = [x - a for a in row]
                        conflict = False
                        seen = 0
                        for d in new_diffs:
                            if d > scope:
                                continue
                            bit = 1 << d
                            if (used_bits & bit) or (seen & bit):
                                conflict = True
                                break
                            seen |= bit
                        if conflict:
                            continue
                        
                        # Check if we can complete the row with this x
                        # This is expensive, so we skip deep check and rely on greedy completion
                        # But we can do a quick check:
                        # Is there space for remaining slots?
                        # Yes, by construction of max_x.
                        
                        # Accept x
                        row.append(x)
                        cur_max = x
                        for d in new_diffs:
                            used_bits |= (1 << d)
                        slots_left -= 1
                        found = True
                        break
                    
                    if not found:
                        return None
                
                return row

            return build_row_balanced(used_bits, scope, rng)
        
        # Try to construct rows with this heuristic
        # We will try multiple scopes, starting from a lower bound.
        
        # Lower bound on scope:
        # We need n_rows * diffs_per_row distinct differences in [1, scope].
        # So scope >= n_rows * diffs_per_row.
        # Also, for a single row of length k+1, the max difference is at least k (if 0..k).
        # But we want to minimize the GLOBAL max.
        
        lb = n_rows * diffs_per_row
        # Add a small buffer
        start_scope = lb + 5
        
        best_rows = None
        best_max = float('inf')
        
        # Try scopes from start_scope upwards
        # But we have a time limit.
        # We can run multiple attempts with different random seeds and scopes.
        
        attempts = 0
        max_attempts = 50
        
        while time_left() > 30 and attempts < max_attempts:
            attempts += 1
            # Increase scope slightly each attempt to explore larger spaces
            # Or restart with same scope but different seed
            scope = start_scope + (attempts // 10) * 5
            
            used_bits = 0
            trial_rows = []
            ok = True
            
            for _ in range(n_rows):
                if time_left() < 30:
                    ok = False
                    break
                row = build_row_frontloaded(used_bits, scope, rng)
                if row is None:
                    ok = False
                    break
                trial_rows.append(row)
                for d in diff_set_of_row(row):
                    used_bits |= (1 << d)
            
            if not ok:
                continue
            
            mv = max_val(trial_rows)
            if mv < best_max:
                best_max = mv
                best_rows = [list(r) for r in trial_rows]
                
                # If we found a solution close to the lower bound, we can stop or refine
                if mv <= lb + 10:
                    break

        return best_rows if best_rows is not None else None

    # ---------- Initial construction ----------
    rng = random
    rng.seed(42)

    # Greedy initial solution with a generous scope
    used_bits = 0
    rows = []
    for _ in range(n_rows):
        row = build_row_greedy(used_bits, scope=10000, temp=0.0, rng=rng)
        if row is None:
            row = list(range(row_len))
        rows.append(row)
        for d in diff_set_of_row(row):
            used_bits |= (1 << d)

    best_rows = [list(r) for r in rows]
    best_max = max_val(best_rows)

    # Try the new heuristic
    heuristic_rows = pack_rows_by_diffs(n_rows, 10000, rng)
    if heuristic_rows is not None:
        h_max = max_val(heuristic_rows)
        if h_max < best_max:
            best_rows = heuristic_rows
            best_max = h_max

    # ---------- Multi-temperature greedy restarts ----------
    temps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]
    attempts = 0
    max_attempts = 150
    while time_left() > 40 and attempts < max_attempts:
        attempts += 1
        temp = temps[attempts % len(temps)]
        used_bits = 0
        trial_rows = []
        ok = True
        for _ in range(n_rows):
            if time_left() < 40:
                ok = False
                break
            row = build_row_greedy(used_bits, scope=best_max + 20, temp=temp, rng=rng)
            if row is None:
                ok = False
                break
            trial_rows.append(row)
            for d in diff_set_of_row(row):
                used_bits |= (1 << d)
        if not ok:
            continue
        mv = max_val(trial_rows)
        if mv < best_max:
            best_max = mv
            best_rows = [list(r) for r in trial_rows]

    # ---------- Compute used bits for all rows except index i ----------
    def used_bits_except(i):
        ub = 0
        for j, r in enumerate(best_rows):
            if j == i:
                continue
            for d in diff_set_of_row(r):
                ub |= (1 << d)
        return ub

    # ---------- Row replacement to reduce scope ----------
    def try_replace_row(i, scope_target, time_limit=None, rng=None, max_nodes=None):
        """Find a replacement for row i with max <= scope_target, disjoint from
        the differences of all other rows."""
        if rng is None:
            rng = random
        fixed_used = used_bits_except(i)
        for scope in range(scope_target, min(best_max, scope_target + 4) + 1):
            if time_limit is not None and now() > time_limit:
                return None
            res = build_row_backtrack(fixed_used, scope, time_limit=time_limit,
                                      rng=rng, max_nodes=max_nodes)
            if res is not None:
                new_rows = [r if j != i else res for j, r in enumerate(best_rows)]
                if validate(new_rows):
                    return res
        return None

    # Iterative row replacement to reduce scope
    improved = True
    while improved and time_left() > 30:
        improved = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if time_left() < 30:
                break
            cur_max_row = best_rows[i][-1]
            target = cur_max_row - 1
            if target < k:
                continue
            new_row = try_replace_row(i, target, time_limit=deadline - 30, rng=rng,
                                      max_nodes=200000)
            if new_row is not None:
                new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                mv = max_val(new_rows)
                if mv < best_max:
                    best_max = mv
                    best_rows = new_rows
                    improved = True

    # ---------- Pair-swap refinement ----------
    if time_left() > 40:
        pairs = [(i, j) for i in range(n_rows) for j in range(i + 1, n_rows)]
        rng.shuffle(pairs)
        for (i, j) in pairs:
            if time_left() < 30:
                break
            fixed_used = 0
            for x, r in enumerate(best_rows):
                if x == i or x == j:
                    continue
                for d in diff_set_of_row(r):
                    fixed_used |= (1 << d)

            for scope in range(best_max - 1, max(best_max - 8, k) - 1, -1):
                if time_left() < 30:
                    break
                r1 = build_row_backtrack(fixed_used, scope, time_limit=deadline - 30,
                                         rng=rng, max_nodes=100000)
                if r1 is None:
                    continue
                ub1 = fixed_used
                for d in diff_set_of_row(r1):
                    ub1 |= (1 << d)
                r2 = build_row_backtrack(ub1, scope, time_limit=deadline - 30,
                                         rng=rng, max_nodes=100000)
                if r2 is None:
                    continue
                new_rows = [r if x not in (i, j) else (r1 if x == i else r2)
                            for x, r in enumerate(best_rows)]
                if validate(new_rows):
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                    break

    # ---------- Final polish: aggressive single-row backtracking ----------
    while time_left() > 12:
        any_improve = False
        order = sorted(range(n_rows), key=lambda i: best_rows[i][-1], reverse=True)
        for i in order:
            if time_left() < 12:
                break
            cur_max_row = best_rows[i][-1]
            for target in range(min(cur_max_row - 1, best_max - 1),
                                max(cur_max_row - 6, k) - 1, -1):
                if time_left() < 12:
                    break
                new_row = try_replace_row(i, target, time_limit=deadline - 12, rng=rng,
                                          max_nodes=150000)
                if new_row is not None:
                    new_rows = [r if j != i else new_row for j, r in enumerate(best_rows)]
                    mv = max_val(new_rows)
                    if mv < best_max:
                        best_max = mv
                        best_rows = new_rows
                        any_improve = True
                        break
        if not any_improve:
            break

    # ---------- Final validation and cleanup ----------
    for i in range(n_rows):
        r = best_rows[i]
        if not r or r[0] != 0:
            best_rows[i] = list(range(row_len))
        else:
            for a, b in zip(r, r[1:]):
                if b <= a:
                    best_rows[i] = list(range(row_len))
                    break

    return best_rows
# EVOLVE-BLOCK-END
