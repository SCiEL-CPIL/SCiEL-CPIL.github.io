import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85  # seconds, safety margin before 90s kill
    max_elements = 48
    n = 192

    def is_safe_incremental(a, S_list, S_set_local):
        """Check if adding a to S_list keeps validity. Only checks new constraints involving a."""
        L = len(S_list)
        if L >= 3:
            rem_count = defaultdict(int)
            for v in S_list:
                rem_count[v % a] += 1

            for i in range(L):
                u = S_list[i]
                for j in range(i + 1, L):
                    v = S_list[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False

        for b in S_list:
            others = [v for v in S_list if v != b]
            if len(others) < 2:
                continue

            target_rem = (-a) % b
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % b] += 1

            L_others = len(others)
            for i in range(L_others):
                u = others[i]
                needed_rem = (target_rem - u % b) % b
                cnt = rem_count.get(needed_rem, 0)
                if u % b == needed_rem:
                    cnt -= 1
                if cnt > 0:
                    return False

        return True

    def is_safe_set(S_list):
        """Check if S_list is valid: no element divides sum of 3 distinct others."""
        for idx, a in enumerate(S_list):
            others = [v for i, v in enumerate(S_list) if i != idx]
            L_len = len(others)
            if L_len < 3:
                continue

            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % a] += 1

            for i in range(L_len):
                u = others[i]
                for j in range(i + 1, L_len):
                    v = others[j]
                    target_rem = (-u - v) % a
                    cnt = rem_count.get(target_rem, 0)
                    if u % a == target_rem:
                        cnt -= 1
                    if v % a == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True

    def compute_conflict_degree(x, S_list):
        """Compute how many conflicts x would have with existing set S_list."""
        conflicts = 0
        L = len(S_list)

        # Count pairs (u,v) in S_list where u+v+w ≡ 0 (mod x) for some w in S_list
        if L >= 3:
            rem_count = defaultdict(int)
            for v in S_list:
                rem_count[v % x] += 1
            for i in range(L):
                u = S_list[i]
                for j in range(i + 1, L):
                    v = S_list[j]
                    target_rem = (-u - v) % x
                    cnt = rem_count.get(target_rem, 0)
                    if u % x == target_rem:
                        cnt -= 1
                    if v % x == target_rem:
                        cnt -= 1
                    if cnt > 0:
                        conflicts += cnt

        # Count for each b in S_list, pairs (u,v) in S_list\{b} where a+u+v ≡ 0 (mod b)
        for b in S_list:
            others = [v for v in S_list if v != b]
            if len(others) < 2:
                continue
            target_rem = (-x) % b
            rem_count = defaultdict(int)
            for v in others:
                rem_count[v % b] += 1
            L_others = len(others)
            for i in range(L_others):
                u = others[i]
                needed_rem = (target_rem - u % b) % b
                cnt = rem_count.get(needed_rem, 0)
                if u % b == needed_rem:
                    cnt -= 1
                if cnt > 0:
                    conflicts += cnt

        return conflicts

    # Precompute static conflict potential: for each x, count how many triples (a,b,c)
    # from [1,n]\{x} satisfy a+b+c ≡ 0 (mod x). Lower = less constraining.
    # Also count for each b, how many pairs (a,c) satisfy x+a+c ≡ 0 (mod b).
    # Total static degree = sum over all hyperedges containing x.
    static_degree = [0] * (n + 1)

    # For each potential "divisor" d in 1..n, count triples (a,b,c) distinct from d
    # where a+b+c ≡ 0 mod d. This tells us how "bad" d is as a selected element.
    # We approximate: for each d, count = number of (a,b,c) with a+b+c ≡ 0 mod d, a,b,c in [1,n]\{d}
    # This is expensive to compute exactly, so use a sampling/heuristic approach.
    # Heuristic: d divides more sums if d is small. Use d as proxy, but refine with
    # actual modular counting for a subset.
    for d in range(1, n + 1):
        # Count pairs (a,b) in [1,n]\{d} such that a+b ≡ r mod d for some r,
        # then for each such pair, count c in [1,n]\{d} with c ≡ -a-b mod d.
        # Total = sum over (a,b) of count_c
        # Simplify: for each residue r mod d, let cnt[r] = #{v in [1,n]\{d} : v ≡ r mod d}
        # Then # of (a,b,c) with a+b+c ≡ 0 mod d = sum_{r1+r2+r3 ≡ 0 mod d} cnt[r1]*cnt[r2]*cnt[r3]
        # minus corrections for a=d, b=d, c=d cases.
        # For now, use a simpler proxy: d itself (smaller d = more constraining)
        static_degree[d] = d

    # Better: compute actual degree by counting for each d, the number of valid
    # (a,b,c) triples from the full universe [1,n] (excluding d) with sum ≡ 0 mod d.
    # Do this efficiently using residue counting.
    for d in range(1, n + 1):
        cnt_res = [0] * d
        for v in range(1, n + 1):
            if v != d:
                cnt_res[v % d] += 1
        # Count triples (a,b,c) with a+b+c ≡ 0 mod d
        total = 0
        for r1 in range(d):
            for r2 in range(d):
                r3 = (-r1 - r2) % d
                total += cnt_res[r1] * cnt_res[r2] * cnt_res[r3]
        # Subtract cases where one of a,b,c equals d (but we already excluded d from cnt_res)
        # So total is already for a,b,c in [1,n]\{d}
        static_degree[d] = total

    # For the "other direction": for each b, how many pairs (a,c) with a+c ≡ -x mod b
    # This contributes to x's degree when b is selected.
    # We'll handle this dynamically during greedy.

    # Dynamic greedy: at each step, pick the candidate x not in S that minimizes
    # compute_conflict_degree(x, S), breaking ties by static_degree (prefer lower static degree).
    def dynamic_greedy(initial_order=None, seed=None):
        if seed is not None:
            random.seed(seed)

        S = []
        S_set = set()
        candidates = list(range(1, n + 1))

        if initial_order is not None:
            candidates = initial_order[:]

        while len(S) < max_elements:
            if time.time() - start_time > deadline - 5:
                break

            best_x = None
            best_conf = None
            best_static = None

            for x in candidates:
                if x in S_set:
                    continue
                conf = compute_conflict_degree(x, S)
                if conf == 0:
                    # No conflicts - pick the one with lowest static degree
                    if best_x is None or static_degree[x] < static_degree[best_x]:
                        best_x = x
                        best_conf = conf
                        best_static = static_degree[x]
                else:
                    if best_x is None or conf < best_conf or (conf == best_conf and static_degree[x] < static_degree[best_x]):
                        best_x = x
                        best_conf = conf
                        best_static = static_degree[x]

            if best_x is None:
                break

            # Verify safety (compute_conflict_degree should be 0 if safe)
            if best_conf == 0:
                S.append(best_x)
                S_set.add(best_x)
            else:
                # Should not happen if logic is correct, but break to be safe
                break

        return S

    # Local search: try to swap out one element and swap in another to gain net +1
    def local_search_swap(S):
        """Try to improve S by removing one element and adding one or two others."""
        S = S[:]
        S_set = set(S)
        improved = True
        while improved and len(S) < max_elements:
            if time.time() - start_time > deadline - 3:
                break
            improved = False

            for i in range(len(S)):
                if time.time() - start_time > deadline - 3:
                    break
                removed = S[i]
                S_temp = [v for j, v in enumerate(S) if j != i]
                S_set_temp = S_set - {removed}

                # Try adding one element
                for x in range(n, 0, -1):
                    if x in S_set_temp:
                        continue
                    if is_safe_incremental(x, S_temp, S_set_temp):
                        S_new = S_temp + [x]
                        if len(S_new) > len(S):
                            S = S_new
                            S_set = set(S)
                            improved = True
                            break

                if improved:
                    break

                # Try adding two elements
                if not improved and len(S_temp) + 2 <= max_elements:
                    for x1 in range(n, 0, -1):
                        if time.time() - start_time > deadline - 3:
                            break
                        if x1 in S_set_temp:
                            continue
                        if is_safe_incremental(x1, S_temp, S_set_temp):
                            S_temp2 = S_temp + [x1]
                            S_set_temp2 = S_set_temp | {x1}
                            for x2 in range(n, 0, -1):
                                if x2 in S_set_temp2:
                                    continue
                                if is_safe_incremental(x2, S_temp2, S_set_temp2):
                                    S_new = S_temp2 + [x2]
                                    if len(S_new) > len(S):
                                        S = S_new
                                        S_set = set(S)
                                        improved = True
                                        break
                            if improved:
                                break

        return S

    # Final polish: try to add more elements after swaps
    def final_polish(S):
        S = S[:]
        S_set = set(S)
        for x in range(n, 0, -1):
            if len(S) >= max_elements:
                break
            if time.time() - start_time > deadline - 1:
                break
            if x in S_set:
                continue
            if is_safe_incremental(x, S, S_set):
                S.append(x)
                S_set.add(x)
        return S

    best_S = []
    best_size = 0

    # Run dynamic greedy with different random seeds
    for seed in range(20):
        if time.time() - start_time > deadline - 10:
            break
        S_result = dynamic_greedy(seed=seed)
        if len(S_result) > best_size:
            best_size = len(S_result)
            best_S = S_result[:]

    # Also try the original static orders
    for order in [list(range(n, 0, -1)), list(range(1, n + 1))]:
        if time.time() - start_time > deadline - 10:
            break
        S_result = []
        S_set = set()
        for x in order:
            if len(S_result) >= max_elements:
                break
            if x in S_set:
                continue
            if is_safe_incremental(x, S_result, S_set):
                S_result.append(x)
                S_set.add(x)
        if len(S_result) > best_size:
            best_size = len(S_result)
            best_S = S_result[:]

    # Local search on best
    if len(best_S) < max_elements and time.time() - start_time < deadline - 3:
        improved_S = local_search_swap(best_S)
        if len(improved_S) > best_size:
            best_size = len(improved_S)
            best_S = improved_S[:]

    # Final polish
    if len(best_S) < max_elements and time.time() - start_time < deadline - 1:
        polished = final_polish(best_S)
        if len(polished) > best_size:
            best_size = len(polished)
            best_S = polished[:]

    return best_S
