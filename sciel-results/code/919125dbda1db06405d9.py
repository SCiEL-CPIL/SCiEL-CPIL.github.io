import time
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit

    # 1. Generate all 2,3,5-smooth numbers up to 128
    smooth = []
    for x in range(1, 129):
        y = x
        while y % 2 == 0:
            y //= 2
        while y % 3 == 0:
            y //= 3
        while y % 5 == 0:
            y //= 5
        if y == 1:
            smooth.append(x)

    smooth_set = set(smooth)
    m = len(smooth)

    # Precompute valid antichain pairs and store in dictionary by sum
    # This is the key optimization: instead of searching for each target,
    # we generate all valid antichains once and look up by sum.
    pair_solutions = {}
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            s = a + b
            if s > 128:
                break
            # Check antichain: a does not divide b and b does not divide a
            # Since a < b, we only need to check b % a != 0
            if b % a != 0:
                if s not in pair_solutions:
                    pair_solutions[s] = [a, b]

    # Precompute valid antichain triples
    triple_solutions = {}
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            for k in range(j + 1, m):
                c = smooth[k]
                s = a + b + c
                if s > 128:
                    break
                # Check antichain for (a, b, c)
                # a < b < c
                # Check pairs: (a,b), (a,c), (b,c)
                if b % a != 0 and c % a != 0 and c % b != 0:
                    if s not in triple_solutions:
                        triple_solutions[s] = [a, b, c]

    # Precompute valid antichain quadruples (for safety, in case some targets need 4 terms)
    quad_solutions = {}
    # Optimization: only go up to k=4 if needed, but precomputing all is fast enough
    # for m ~ 45-50 smooth numbers. C(50,4) is ~230k, which is fast.
    for i in range(m):
        a = smooth[i]
        for j in range(i + 1, m):
            b = smooth[j]
            for k in range(j + 1, m):
                c = smooth[k]
                base_sum = a + b + c
                if base_sum > 128:
                    break
                # Check antichain for (a, b, c) first
                if b % a == 0 or c % a == 0 or c % b == 0:
                    continue
                for l in range(k + 1, m):
                    d = smooth[l]
                    s = base_sum + d
                    if s > 128:
                        break
                    # Check antichain for new pairs involving d: (a,d), (b,d), (c,d)
                    # We already know (a,b), (a,c), (b,c) are ok.
                    if d % a != 0 and d % b != 0 and d % c != 0:
                        if s not in quad_solutions:
                            quad_solutions[s] = [a, b, c, d]

    # Precompute valid antichain quintuples (just in case, though unlikely)
    # C(50,5) is ~2M, might be a bit slow in Python if not optimized, 
    # but let's see if we need it. Actually, let's just do a targeted search for 5+ if needed.
    
    answers = []

    for n in range(1, 129):
        if time.time() > deadline:
            # Emergency fallback: if we run out of time, we must return something valid.
            # But given the precomputation, this loop should be very fast.
            pass

        found = None

        # Try k=1
        if n in smooth_set:
            found = [n]
        
        # Try k=2
        if found is None:
            if n in pair_solutions:
                found = pair_solutions[n]
        
        # Try k=3
        if found is None:
            if n in triple_solutions:
                found = triple_solutions[n]
        
        # Try k=4
        if found is None:
            if n in quad_solutions:
                found = quad_solutions[n]
        
        # Try k=5 (Targeted search if precomputation didn't cover it)
        if found is None:
            # Brute force for k=5 for this specific n
            # a < b < c < d < e
            # This is O(m^5) worst case, but we can prune heavily.
            # Since we only do this for remaining targets, and likely none remain, it's safe.
            # But if some do, we need it to be fast.
            # Let's use a simple backtracking.
            def search_k(n, k, start_idx, current_sum, current_list):
                if k == 0:
                    return current_list if current_sum == n else None
                if k == 1:
                    rem = n - current_sum
                    if rem > smooth[start_idx - 1] if start_idx > 0 else True:
                        if rem in smooth_set and (start_idx == 0 or rem > smooth[start_idx-1]):
                            # Check antichain with existing
                            is_ok = True
                            for x in current_list:
                                if rem % x == 0 or x % rem == 0:
                                    is_ok = False
                                    break
                            if is_ok:
                                return current_list + [rem]
                    return None
                
                # Pruning: min possible sum with remaining k items
                # min_next = smooth[start_idx]
                # if current_sum + sum(smooth[start_idx:start_idx+k]) > n: return None
                # if current_sum + sum(smooth[-k:]) < n: return None (approx)
                
                for i in range(start_idx, m - k + 1):
                    a = smooth[i]
                    new_sum = current_sum + a
                    if new_sum > n:
                        break
                    # Check antichain with existing
                    is_ok = True
                    for x in current_list:
                        if a % x == 0 or x % a == 0:
                            is_ok = False
                            break
                    if not is_ok:
                        continue
                    res = search_k(n, k - 1, i + 1, new_sum, current_list + [a])
                    if res is not None:
                        return res
                return None

            found = search_k(n, 5, 0, 0, [])

        if found is None:
            # Fallback for k=6+ (extremely unlikely for n<=128)
            # Try a generic search up to 10 terms
            def search_k_generic(n, max_k, start_idx, current_sum, current_list):
                if len(current_list) > max_k:
                    return None
                if current_sum > n:
                    return None
                if len(current_list) >= 1:
                    # Check if we can finish with 1 more term
                    rem = n - current_sum
                    if rem > 0:
                        if rem in smooth_set:
                            # Check distinct
                            if rem not in current_list:
                                # Check antichain
                                is_ok = True
                                for x in current_list:
                                    if rem % x == 0 or x % rem == 0:
                                        is_ok = False
                                        break
                                if is_ok:
                                    return current_list + [rem]
                
                # Try adding more terms
                if len(current_list) < max_k:
                    for i in range(start_idx, m):
                        a = smooth[i]
                        if current_sum + a > n:
                            break
                        if a in current_list:
                            continue
                        # Check antichain
                        is_ok = True
                        for x in current_list:
                            if a % x == 0 or x % a == 0:
                                is_ok = False
                                break
                        if not is_ok:
                            continue
                        res = search_k_generic(n, max_k, i + 1, current_sum + a, current_list + [a])
                        if res is not None:
                            return res
                return None

            found = search_k_generic(n, 10, 0, 0, [])

        if found is None:
            # Should not happen
            found = []

        answers.append(found)

    return answers
