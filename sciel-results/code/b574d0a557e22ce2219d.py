import time
import itertools
import numpy as np
import random

_DEADLINE = 115.0  # Safety margin vs 120s wall-clock kill

def _all_words(q, n):
    return np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int64)

def construct(q, n, radius):
    t0 = time.time()
    total = q ** n
    
    # Fallback for very large spaces (though task is fixed 3^7=2187)
    if total > 10_000_000:
        if q == 3 and n >= 4:
            tet = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tet for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(range(n))]

    # Precompute balls
    base = _all_words(q, n)
    powers = q ** np.arange(n - 1, -1, -1, dtype=np.int64)
    idx_of = (base * powers).sum(axis=1)
    
    if radius == 1:
        ball = np.zeros((total, 1 + n * (q - 1)), dtype=np.int64)
        ball[:, 0] = idx_of
        k = 1
        for pos in range(n):
            for delta in range(1, q):
                flipped = base.copy()
                flipped[:, pos] = (base[:, pos] + delta) % q
                ball[:, k] = (flipped * powers).sum(axis=1)
                k += 1
        balls = ball
    else:
        # Generic BFS for small spaces
        balls_list = []
        for i in range(total):
            seen = {i}
            frontier = [i]
            for _ in range(radius):
                nxt = []
                for widx in frontier:
                    w = base[widx]
                    for pos in range(n):
                        for v in range(q):
                            if v == w[pos]:
                                continue
                            nw = w.copy()
                            nw[pos] = v
                            j = int((nw * powers).sum())
                            if j not in seen:
                                seen.add(j)
                                nxt.append(j)
                frontier = nxt
            balls_list.append(list(seen))
        maxb = max(len(b) for b in balls_list)
        balls = np.array([[x for x in b] + [0] * (maxb - len(b)) for b in balls_list], dtype=np.int64)

    # 1. Greedy Set Cover
    cov = np.zeros(total, dtype=np.int32)
    chosen = []
    in_code = np.zeros(total, dtype=bool)
    
    def time_left():
        return _DEADLINE - (time.time() - t0)

    # Greedy initialization
    while time_left() > 10.0:
        # Calculate gain for each candidate
        # gain[i] = number of words in ball[i] that are currently uncovered
        gains = np.zeros(total, dtype=np.int64)
        for i in range(total):
            if in_code[i]:
                continue
            b = balls[i]
            gains[i] = (cov[b] == 0).sum()
        
        best = int(np.argmax(gains))
        if gains[best] == 0:
            break
        
        chosen.append(best)
        in_code[best] = True
        cov[balls[best]] += 1

    # Update covcnt for local search
    covcnt = np.zeros(total, dtype=np.int32)
    for i in chosen:
        covcnt[balls[i]] += 1

    # Helper functions for local search
    def try_remove(i):
        b = balls[i]
        return (covcnt[b] >= 2).all()

    def do_remove(i):
        b = balls[i]
        covcnt[b] -= 1
        in_code[i] = False
        chosen.remove(i)

    def do_add(j):
        b = balls[j]
        covcnt[b] += 1
        in_code[j] = True
        chosen.append(j)

    def try_swap(i, j):
        b_i = balls[i]
        b_j = balls[j]
        # Words in b_i not in b_j must have covcnt >= 2
        # To avoid O(N) check, we can just check all words in b_i
        # if w in b_i:
        #   if w in b_j: no change in coverage count for w relative to removal of i (j adds it back)
        #   else: covcnt[w] must be >= 2
        # Optimization: create a boolean mask for b_j
        in_j = np.zeros(total, dtype=bool)
        in_j[b_j] = True
        critical = b_i[~in_j[b_i]]
        if critical.size == 0:
            return True
        return (covcnt[critical] >= 2).all()

    # 2. Local Search
    # We perform multiple passes of removal, swap, and 2-for-1 exchange.
    # We limit the number of iterations to stay within time.
    
    improved = True
    while improved and time_left() > 5.0:
        improved = False
        
        # Pass A: Redundancy Removal
        for idx in range(len(chosen)):
            if time_left() < 5.0:
                break
            i = chosen[idx]
            if try_remove(i):
                do_remove(i)
                improved = True
                break # Restart scan after modification
        
        if improved:
            continue

        # Pass B: 1-for-1 Swap
        if time_left() > 10.0:
            n_chosen = len(chosen)
            # Shuffle chosen to avoid bias
            chosen_indices = list(range(n_chosen))
            random.shuffle(chosen_indices)
            
            for idx in chosen_indices:
                if time_left() < 10.0:
                    break
                i = chosen[idx]
                b_i = balls[i]
                
                # Try random candidates first
                found = False
                for _ in range(50):
                    j = random.randint(0, total - 1)
                    if in_code[j]:
                        continue
                    if try_swap(i, j):
                        do_remove(i)
                        do_add(j)
                        improved = True
                        found = True
                        break
                
                if found:
                    break
                
                # If random failed, do a full scan if list is small enough
                if not found and n_chosen < 300:
                    for j in range(total):
                        if time_left() < 10.0:
                            break
                        if in_code[j]:
                            continue
                        if try_swap(i, j):
                            do_remove(i)
                            do_add(j)
                            improved = True
                            found = True
                            break
                    if found:
                        break

        if improved:
            continue

        # Pass C: 2-for-1 Exchange
        # Try to remove two codewords i, k and add one j.
        # This is expensive, so we only do it if we have time and haven't improved recently.
        if time_left() > 30.0:
            n_chosen = len(chosen)
            # Sample pairs of codewords to test
            # We iterate through a subset of pairs to save time
            pairs_to_check = []
            if n_chosen < 200:
                for a in range(n_chosen):
                    for b in range(a+1, n_chosen):
                        pairs_to_check.append((chosen[a], chosen[b]))
            else:
                # Sample 1000 random pairs
                for _ in range(1000):
                    a = random.randint(0, n_chosen - 1)
                    b = random.randint(0, n_chosen - 1)
                    if a != b:
                        pairs_to_check.append((chosen[a], chosen[b]))
            
            random.shuffle(pairs_to_check)
            
            for i, k in pairs_to_check:
                if time_left() < 30.0:
                    break
                if i not in chosen or k not in chosen:
                    continue
                
                # Calculate the set of words that would be uncovered if i and k were removed
                # A word w is uncovered if:
                # w in B_i U B_k AND covcnt[w] - I[w in B_i] - I[w in B_k] < 1
                # i.e., covcnt[w] <= I[w in B_i] + I[w in B_k]
                
                b_i = balls[i]
                b_k = balls[k]
                
                # Union of balls
                union_mask = np.zeros(total, dtype=bool)
                union_mask[b_i] = True
                union_mask[b_k] = True
                
                # Candidates for uncovered words: words in union with low coverage
                # We need to check if there exists a j such that all such words are in B_j
                # First, identify the "critical" words for this pair
                # w is critical if w in Union and covcnt[w] <= 2 (since max removal is 2)
                # Specifically:
                # if w in B_i and w in B_k: covcnt[w] must be >= 2 to survive? No.
                #   If we remove i and k, new covcnt = covcnt[w] - 2. We need new covcnt >= 1 => covcnt[w] >= 3.
                #   If covcnt[w] == 2, it becomes 0. Uncovered.
                #   If covcnt[w] == 1, impossible since w in B_i and B_k implies covcnt >= 2.
                # if w in B_i xor B_k: 
                #   new covcnt = covcnt[w] - 1. We need new covcnt >= 1 => covcnt[w] >= 2.
                #   If covcnt[w] == 1, it becomes 0. Uncovered.
                
                # So, a word w is "lost" if:
                # 1. w in B_i and w in B_k and covcnt[w] == 2
                # 2. w in B_i xor B_k and covcnt[w] == 1
                
                # Let's compute the set of lost words L
                in_i = np.zeros(total, dtype=bool)
                in_i[b_i] = True
                in_k = np.zeros(total, dtype=bool)
                in_k[b_k] = True
                
                # Case 1: in both, covcnt == 2
                lost1 = in_i & in_k & (covcnt == 2)
                # Case 2: in exactly one, covcnt == 1
                lost2 = (in_i ^ in_k) & (covcnt == 1)
                
                lost = lost1 | lost2
                
                if not lost.any():
                    # We can remove i and k without adding anything? 
                    # This would reduce size by 2.
                    # But is it possible? If so, why didn't single removal catch it?
                    # Because single removal requires ALL words in B_i to have covcnt >= 2.
                    # Here, words in B_i \ B_k might have covcnt 1, but if they are covered by k? No, k is removed too.
                    # If lost is empty, it means no word becomes uncovered.
                    # So we can just remove i and k.
                    do_remove(i)
                    do_remove(k)
                    improved = True
                    break
                
                # If lost is not empty, we need to find a j that covers ALL words in lost.
                # To make this efficient, we can check candidates j that cover many lost words.
                # A necessary condition: j must cover the "hardest" word?
                # Let's just check if any j covers all lost words.
                # Since lost is likely small, we can iterate through j's that are "good".
                # Which j's are good? Those that intersect lost significantly.
                
                # Optimization: Precompute for each j, the number of lost words it covers?
                # That's O(N * |Lost|). If |Lost| is small, this is fast.
                
                found_j = -1
                # We only need to check j's that are not in code
                # And ideally, j's that are "close" to the lost words?
                # Let's just scan all j's not in code, but break early if we find one.
                # To speed up, we can first check if the union of balls of any single j can cover lost.
                # This is equivalent to: Is there a j such that lost is a subset of B_j?
                
                # We can represent lost as a set of indices.
                lost_indices = np.where(lost)[0]
                
                # If lost is very large, it's unlikely a single j covers it all.
                if len(lost_indices) > 50:
                    continue

                for j in range(total):
                    if time_left() < 30.0:
                        break
                    if in_code[j]:
                        continue
                    b_j = balls[j]
                    # Check if all lost_indices are in b_j
                    # We can do this by checking if any lost index is NOT in b_j
                    # Efficient way: use a boolean mask for b_j
                    in_j = np.zeros(total, dtype=bool)
                    in_j[b_j] = True
                    if in_j[lost_indices].all():
                        found_j = j
                        break
                
                if found_j != -1:
                    do_remove(i)
                    do_remove(k)
                    do_add(found_j)
                    improved = True
                    break

    # Final verification and repair
    if time_left() > 1.0:
        covered = np.zeros(total, dtype=bool)
        for i in chosen:
            covered[balls[i]] = True
        if not covered.all():
            # Greedily fill gaps
            while time_left() > 0.5:
                g = (~covered)
                if not g.any():
                    break
                best = -1
                bestg = 0
                for i in range(total):
                    if in_code[i]:
                        continue
                    ng = g[balls[i]].sum()
                    if ng > bestg:
                        bestg = ng
                        best = i
                if best < 0 or bestg == 0:
                    break
                do_add(best)
                covered[balls[best]] = True

    return [base[i].tolist() for i in chosen]
