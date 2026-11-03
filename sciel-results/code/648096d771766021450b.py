import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Precomputation
    space_size = q ** n
    idx_to_digits = []
    for idx in range(space_size):
        digits = []
        x = idx
        for i in range(n):
            digits.append(x % q)
            x //= q
        digits.reverse()
        idx_to_digits.append(tuple(digits))

    def index_of_digits(digits):
        idx = 0
        for d in digits:
            idx = idx * q + d
        return idx

    def hamming_dist(d1, d2):
        return sum(1 for i in range(n) if d1[i] != d2[i])

    # Precompute balls (list of indices and set of indices)
    all_balls_list = []
    all_balls_set = []
    for idx in range(space_size):
        center_digits = idx_to_digits[idx]
        ball = [idx]
        if radius == 1:
            for pos in range(n):
                for d in range(q):
                    if d == center_digits[pos]:
                        continue
                    new_digits = list(center_digits)
                    new_digits[pos] = d
                    ball.append(index_of_digits(tuple(new_digits)))
        elif radius == 0:
            pass
        else:
            # Generic radius (not expected for this task but safe)
            for delta_tuple in itertools.product(range(q), repeat=n):
                dist = sum(1 for i in range(n) if delta_tuple[i] != 0)
                if 0 < dist <= radius:
                    new_digits = [(center_digits[i] + delta_tuple[i]) % q for i in range(n)]
                    ball.append(index_of_digits(tuple(new_digits)))
        
        # Remove duplicates just in case (though logic above should be unique)
        ball = list(set(ball))
        all_balls_list.append(ball)
        all_balls_set.append(set(ball))

    # Phase 1: Initialization
    # Build distance-2 graph adjacency for independent set
    dist2_adj = [[] for _ in range(space_size)]
    for i in range(space_size):
        di = idx_to_digits[i]
        for j in range(i + 1, space_size):
            dj = idx_to_digits[j]
            if hamming_dist(di, dj) == 2:
                dist2_adj[i].append(j)
                dist2_adj[j].append(i)

    degree = [len(dist2_adj[i]) for i in range(space_size)]
    order = sorted(range(space_size), key=lambda x: -degree[x])
    
    in_set = [False] * space_size
    independent_set = []
    for v in order:
        conflict = False
        for u in dist2_adj[v]:
            if in_set[u]:
                conflict = True
                break
        if not conflict:
            in_set[v] = True
            independent_set.append(v)

    selected = list(independent_set)
    
    # Initialize coverage counts
    cover_count = [0] * space_size
    for idx in selected:
        for w in all_balls_list[idx]:
            cover_count[w] += 1

    # Fill gaps with Greedy Set Cover
    uncovered_count = sum(1 for c in cover_count if c == 0)
    while uncovered_count > 0 and time.time() < deadline:
        best_gain = 0
        best_candidate = -1
        for ci in range(space_size):
            if in_set[ci]:
                continue
            gain = 0
            for w in all_balls_list[ci]:
                if cover_count[w] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_candidate = ci
        if best_candidate == -1 or best_gain == 0:
            break
        in_set[best_candidate] = True
        selected.append(best_candidate)
        for w in all_balls_list[best_candidate]:
            cover_count[w] += 1
            if cover_count[w] == 1:
                uncovered_count -= 1

    # Phase 2: Local Search
    def is_redundant(idx, selected_list, cover_counts):
        """
        Check if idx is redundant given selected_list and cover_counts.
        Assumes cover_counts reflects the state WITH idx included.
        """
        for w in all_balls_list[idx]:
            if cover_counts[w] == 1:
                return False
        return True

    def remove_idx(idx, selected_list, cover_counts, in_set_arr):
        for w in all_balls_list[idx]:
            cover_counts[w] -= 1
        selected_list.remove(idx)
        in_set_arr[idx] = False

    def add_idx(idx, selected_list, cover_counts, in_set_arr):
        for w in all_balls_list[idx]:
            cover_counts[w] += 1
        selected_list.append(idx)
        in_set_arr[idx] = True

    improved = True
    stuck_count = 0
    
    while improved and time.time() < deadline:
        improved = False
        stuck_count = 0
        
        # Pass 1: Redundancy Removal
        i = 0
        while i < len(selected):
            if time.time() > deadline:
                break
            idx = selected[i]
            if is_redundant(idx, selected, cover_count):
                remove_idx(idx, selected, cover_count, in_set)
                improved = True
                # Do not increment i, check next element which shifted
            else:
                i += 1
        
        if improved:
            continue # Restart passes as state changed

        # Pass 2: 1-Swap (Replace i with j)
        swapped = False
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            idx_i = selected[i]
            # Find unique words covered by i
            unique_words = []
            for w in all_balls_list[idx_i]:
                if cover_count[w] == 1:
                    unique_words.append(w)
            
            if not unique_words:
                # Should have been caught in Pass 1, but just in case
                remove_idx(idx_i, selected, cover_count, in_set)
                improved = True
                swapped = True
                break
            
            # Try to find j that covers all unique_words
            for j in range(space_size):
                if in_set[j]:
                    continue
                # Check if ball[j] covers all unique_words
                # Optimization: if len(unique_words) > len(ball[j]), skip (though ball size is fixed)
                if all(w in all_balls_set[j] for w in unique_words):
                    # Replace i with j
                    remove_idx(idx_i, selected, cover_count, in_set)
                    add_idx(j, selected, cover_count, in_set)
                    improved = True
                    swapped = True
                    break
            if swapped:
                break
        
        if swapped:
            continue # Restart passes

        # Pass 3: 2-Swap (Replace i, k with j)
        swapped = False
        n_sel = len(selected)
        for i in range(n_sel):
            if time.time() > deadline:
                break
            idx_i = selected[i]
            for k in range(i + 1, n_sel):
                if time.time() > deadline:
                    break
                idx_k = selected[k]
                
                # Find words covered by i or k but not by others
                # unique_ik = (ball[i] U ball[k]) \ (Union of balls of others)
                # Efficient way: 
                # A word w is in unique_ik if:
                # 1. w in ball[i] or w in ball[k]
                # 2. cover_count[w] <= 2 (since only i and k might cover it, plus others? No, cover_count includes i and k)
                #    If cover_count[w] == 1, it's covered by exactly one of {i, k, others}.
                #    If cover_count[w] == 2, it's covered by exactly two.
                #    We need words where the ONLY coverers are from {i, k}.
                #    So, if w is in ball[i] and cover_count[w] == 1, it's unique to i (and thus unique to pair).
                #    If w is in ball[i] and cover_count[w] == 2, it is covered by i and exactly one other.
                #       If that other is k, then it's covered by i and k.
                #       If that other is not k, then removing i and k leaves it uncovered?
                #       No, if w is covered by i and x (x!=k), then removing i and k leaves it covered by x.
                #       So we only care about words where the set of coverers is a subset of {i, k}.
                
                unique_ik = []
                for w in all_balls_list[idx_i]:
                    if cover_count[w] == 1:
                        unique_ik.append(w)
                    elif cover_count[w] == 2:
                        # Check if the other coverer is k
                        # We don't have a direct list of coverers.
                        # But we can check: if w is in ball[k], then coverers are subset of {i, k, others}.
                        # If cover_count is 2, and w in ball[i] and w in ball[k], then the two coverers are i and k?
                        # Not necessarily. Could be i and some x, and k also covers it? No, cover_count is 2.
                        # So if w in ball[i] and w in ball[k] and cover_count[w] == 2, then the coverers are exactly {i, k}.
                        if w in all_balls_set[idx_k]:
                            unique_ik.append(w)
                
                # Also need to check words in ball[k] that are not in ball[i]
                # If w in ball[k] and cover_count[w] == 1, it's unique to k.
                # If w in ball[k] and cover_count[w] == 2 and w in ball[i], already handled above.
                # If w in ball[k] and cover_count[w] == 2 and w NOT in ball[i], then it's covered by k and some x.
                # Removing i and k leaves it covered by x. So not unique to the pair.
                
                # So the logic above for ball[i] covers:
                # - Words unique to i
                # - Words covered by both i and k (and no one else)
                # We missed words unique to k.
                for w in all_balls_list[idx_k]:
                    if cover_count[w] == 1:
                        unique_ik.append(w)
                    # If cover_count[w] == 2 and w in ball[i], it was added in the first loop.
                    # We don't want duplicates.
                
                if not unique_ik:
                    # Both are redundant together?
                    # This implies they are individually redundant?
                    # If unique_ik is empty, it means every word in ball[i] U ball[k] is covered by someone else.
                    # So we can remove both.
                    remove_idx(idx_i, selected, cover_count, in_set)
                    # Note: selected changed, indices shifted.
                    # We need to be careful. Let's just break and restart.
                    # Actually, if we remove i, k's index might change.
                    # Better to just mark for removal and restart.
                    improved = True
                    swapped = True
                    break
                    
                if swapped:
                    break

                # Try to find j that covers all unique_ik
                # Optimization: unique_ik is small.
                for j in range(space_size):
                    if in_set[j]:
                        continue
                    if all(w in all_balls_set[j] for w in unique_ik):
                        # Replace i, k with j
                        remove_idx(idx_i, selected, cover_count, in_set)
                        # k might have shifted if i < k? 
                        # In remove_idx, we use list.remove(idx).
                        # So idx_k is still valid as a value, but its position in list changed.
                        # list.remove(idx_k) will find it.
                        remove_idx(idx_k, selected, cover_count, in_set)
                        add_idx(j, selected, cover_count, in_set)
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
            
            if swapped:
                break
        
        if swapped:
            continue # Restart passes

        # If no improvement in this full cycle, we are stuck in local optimum
        # Try a "kick": random swap to escape
        if time.time() < deadline - 10.0:
            if len(selected) > 1:
                i = random.randint(0, len(selected) - 1)
                idx_i = selected[i]
                # Find a random j not in selected
                candidates = [j for j in range(space_size) if not in_set[j]]
                if candidates:
                    j = random.choice(candidates)
                    # Perform swap: remove i, add j
                    remove_idx(idx_i, selected, cover_count, in_set)
                    add_idx(j, selected, cover_count, in_set)
                    improved = True # Force restart of optimization
                    stuck_count += 1
                    if stuck_count > 50:
                        # Give up on kicking if it doesn't help
                        break

    return [list(idx_to_digits[idx]) for idx in selected]
