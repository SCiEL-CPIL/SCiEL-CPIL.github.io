import numpy as np
import time
import random
import sys

def construct(q, n, radius):
    start_time = time.time()
    # Safety margin: 120s limit, keep 10s buffer
    deadline = start_time + 110.0

    total_words = q ** n
    if total_words == 1:
        return [[0] * n]

    # Precompute neighbors: for each word, list all words at distance <= radius
    # For radius=1, neighbors are the word itself and words differing in 1 position
    n_neighbors = 1 + n * (q - 1)
    
    # Efficient neighbor generation
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)
    q_powers = [q ** i for i in range(n)]
    
    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // q_powers[n - 1 - pos]) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * q_powers[n - 1 - pos]
                neighbors[w, idx] = w + diff
                idx += 1

    # Precompute neighbor sets for fast intersection and membership
    # This is memory heavy for large spaces but q=3, n=8 -> 6561 words is small
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]

    def greedy_init(seed_offset=0, randomized=False):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words

        while remaining > 0:
            if time.time() > deadline - 30.0:
                break
            
            if randomized:
                # Randomized greedy: pick from top K gains
                # This helps escape deterministic traps
                top_k = min(10, remaining)
                # Find indices of top K gains
                # Using argsort on a slice or partial sort logic
                # For speed, just pick a few random candidates and check their gain
                candidates = []
                for _ in range(min(20, remaining)):
                    w = rng.randint(0, total_words - 1)
                    if gain[w] > 0:
                        candidates.append(w)
                
                if candidates:
                    # Pick the one with max gain among candidates
                    best_word = max(candidates, key=lambda x: gain[x])
                else:
                    best_word = int(np.argmax(gain))
            else:
                best_word = int(np.argmax(gain))
            
            best_gain = gain[best_word]
            if best_gain == 0:
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])

            codewords.append(best_word)

            # Update coverage and gains
            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1

        return codewords

    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)

        # Count how many codewords cover each word
        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)

        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1

        # Temperature for simulated annealing (if we use it)
        # We will primarily use deterministic improvements + random 2-for-2 swaps
        
        while time.time() < time_limit:
            improved = False

            # Pass 1: Remove redundant codewords (count[u] > 1 for all u in N(ci))
            # We can do this in a single pass if we shuffle
            indices = list(range(len(cw)))
            random.shuffle(indices)
            
            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                # If min count in neighborhood is > 1, we can remove ci
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]

            # Pass 2: 1-for-1 Swaps
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)
                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]
                    
                    # Words that are critically covered (count == 1) by ci
                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]
                    
                    if len(critical) == 0:
                        # Can remove, but Pass 1 should have caught this. 
                        # Might be due to order of operations.
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        continue

                    # Find candidates that cover all critical words
                    # Intersection of neighbor sets of critical words
                    candidate_set = neighbor_sets[critical[0]].copy()
                    for u in critical[1:]:
                        candidate_set &= neighbor_sets[u]
                        if not candidate_set:
                            break
                    
                    if candidate_set:
                        # Exclude current and existing codewords
                        valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                        if valid_cands:
                            cand = random.choice(valid_cands)
                            # Apply swap
                            for u in neighbors[ci]:
                                count[u] -= 1
                            for u in neighbors[cand]:
                                count[u] += 1
                            in_code[ci] = False
                            in_code[cand] = True
                            cw[i] = cand
                            improved = True

            # Pass 3: 2-for-2 Swaps (Escaping local minima)
            if not improved and len(cw) > 2:
                # Sample pairs of codewords
                num_samples = min(50, len(cw) * 2)
                for _ in range(num_samples):
                    if time.time() > time_limit:
                        break
                    idx1, idx2 = random.sample(range(len(cw)), 2)
                    c1, c2 = cw[idx1], cw[idx2]
                    
                    # If we remove c1 and c2, what is the deficit?
                    # Deficit words: u where count[u] - (1 if c1 covers u) - (1 if c2 covers u) < 1
                    # i.e., count[u] == 1 and (c1 covers u XOR c2 covers u) is false? No.
                    # count[u] is total.
                    # If we remove c1, count[u] becomes count[u]-1.
                    # If we remove c2, count[u] becomes count[u]-1.
                    # If both, count[u] becomes count[u]-2.
                    # We need count[u] - (1 if c1~u) - (1 if c2~u) >= 1 for all u?
                    # No, we need the NEW count to be >= 1.
                    # New count[u] = count[u] - I(c1 covers u) - I(c2 covers u).
                    # We need New count[u] >= 1.
                    # So we need count[u] >= 1 + I(c1 covers u) + I(c2 covers u).
                    
                    # Let's identify words that become uncovered if we remove c1 and c2.
                    # These are words u such that count[u] < 1 + I(c1~u) + I(c2~u).
                    # Since count[u] >= 1 always in a valid code:
                    # Case 1: c1 covers u, c2 covers u. Need count[u] >= 3. If count[u] < 3, deficit.
                    # Case 2: c1 covers u, c2 doesn't. Need count[u] >= 2. If count[u] < 2, deficit.
                    # Case 3: c2 covers u, c1 doesn't. Need count[u] >= 2. If count[u] < 2, deficit.
                    # Case 4: Neither. Need count[u] >= 1. Always true.
                    
                    deficit_set = set()
                    n1 = neighbors[c1]
                    n2 = neighbors[c2]
                    
                    # Check intersection of n1 and n2
                    # We can iterate through n1 and check if in n2
                    # n1 and n2 are numpy arrays.
                    # Faster: create a boolean mask for n2
                    mask_n2 = np.zeros(total_words, dtype=bool)
                    mask_n2[n2] = True
                    
                    for u in n1:
                        if mask_n2[u]:
                            # Both cover u
                            if count[u] < 3:
                                deficit_set.add(u)
                        else:
                            # Only c1 covers u
                            if count[u] < 2:
                                deficit_set.add(u)
                    
                    for u in n2:
                        if not mask_n2[u]: # This is always true in the loop above? No, we iterated n2 here.
                            # Only c2 covers u
                            if count[u] < 2:
                                deficit_set.add(u)

                    if not deficit_set:
                        # 2-for-0: Remove both
                        # This is a size reduction!
                        for u in n1:
                            count[u] -= 1
                        for u in n2:
                            count[u] -= 1
                        in_code[c1] = False
                        in_code[c2] = False
                        if idx1 < idx2:
                            cw.pop(idx2)
                            cw.pop(idx1)
                        else:
                            cw.pop(idx1)
                            cw.pop(idx2)
                        improved = True
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
                        break
                    else:
                        # 2-for-2: Find two new codewords that cover all deficit_set
                        # This is a 2-set cover problem on deficit_set.
                        # Heuristic: Pick one random candidate from neighbor_sets[deficit[0]]
                        # Then find another that covers the rest.
                        
                        # To keep it fast, we only try if deficit_set is small
                        if len(deficit_set) > 10:
                            continue
                            
                        deficit_list = list(deficit_set)
                        
                        # Try a few random candidates for the first new codeword
                        for _ in range(10):
                            if time.time() > time_limit:
                                break
                            # Candidate 1 must cover at least one deficit word
                            # Pick a random deficit word and a random neighbor
                            d_word = random.choice(deficit_list)
                            cands_1 = list(neighbor_sets[d_word])
                            cands_1 = [c for c in cands_1 if c not in (c1, c2) and not in_code[c]]
                            if not cands_1:
                                continue
                            cand1 = random.choice(cands_1)
                            
                            # Update counts temporarily? No, just check coverage
                            # Words covered by cand1
                            covered_by_c1 = set(neighbors[cand1].tolist())
                            
                            # Remaining deficit
                            rem_deficit = [u for u in deficit_list if u not in covered_by_c1]
                            
                            if not rem_deficit:
                                # cand1 covers all deficit. We can pick cand2 to be anything?
                                # No, we need to replace c1, c2 with cand1, cand2.
                                # If cand1 covers all deficit, we just need cand2 to be valid?
                                # Actually, if we remove c1, c2 and add cand1, the code might still be valid
                                # if cand1 covers the deficit. But we are doing 2-for-2.
                                # If we only need 1 new codeword, that's a 2-for-1 swap (size reduction).
                                # Let's check if 2-for-1 is possible.
                                # If cand1 covers all deficit, we can remove c1, c2 and add cand1.
                                # Size reduces by 1.
                                for u in n1:
                                    count[u] -= 1
                                for u in n2:
                                    count[u] -= 1
                                for u in neighbors[cand1]:
                                    count[u] += 1
                                in_code[c1] = False
                                in_code[c2] = False
                                in_code[cand1] = True
                                if idx1 < idx2:
                                    cw.pop(idx2)
                                    cw.pop(idx1)
                                else:
                                    cw.pop(idx1)
                                    cw.pop(idx2)
                                cw.append(cand1)
                                improved = True
                                if len(cw) < best_size:
                                    best_size = len(cw)
                                    best_cw = cw[:]
                                break
                            
                            # Need cand2 to cover rem_deficit
                            # Intersection of neighbor sets of rem_deficit
                            cand2_set = neighbor_sets[rem_deficit[0]].copy()
                            for u in rem_deficit[1:]:
                                cand2_set &= neighbor_sets[u]
                                if not cand2_set:
                                    break
                            
                            if cand2_set:
                                cands_2 = [c for c in cand2_set if c not in (c1, c2, cand1) and not in_code[c]]
                                if cands_2:
                                    cand2 = random.choice(cands_2)
                                    
                                    # Apply 2-for-2 swap
                                    for u in n1:
                                        count[u] -= 1
                                    for u in n2:
                                        count[u] -= 1
                                    for u in neighbors[cand1]:
                                        count[u] += 1
                                    for u in neighbors[cand2]:
                                        count[u] += 1
                                    
                                    in_code[c1] = False
                                    in_code[c2] = False
                                    in_code[cand1] = True
                                    in_code[cand2] = True
                                    
                                    # Update cw list
                                    # Remove c1, c2
                                    if idx1 < idx2:
                                        cw.pop(idx2)
                                        cw.pop(idx1)
                                    else:
                                        cw.pop(idx1)
                                        cw.pop(idx2)
                                    cw.extend([cand1, cand2])
                                    
                                    improved = True
                                    # No size change, but we moved. 
                                    # We don't update best_size here as it's equal.
                                    break
                            
                        if improved:
                            break

            # Perturbation: Random 2-for-2 swap even if not improving (Annealing)
            if not improved and len(cw) > 2 and random.random() < 0.3:
                idx1, idx2 = random.sample(range(len(cw)), 2)
                c1, c2 = cw[idx1], cw[idx2]
                
                # Pick two random new codewords
                cand1 = random.randint(0, total_words - 1)
                cand2 = random.randint(0, total_words - 1)
                
                if cand1 == cand2 or in_code[cand1] or in_code[cand2]:
                    continue
                if cand1 in (c1, c2) or cand2 in (c1, c2):
                    continue

                # Check if new code is valid (all counts >= 1)
                temp_count = count.copy()
                for u in neighbors[c1]:
                    temp_count[u] -= 1
                for u in neighbors[c2]:
                    temp_count[u] -= 1
                for u in neighbors[cand1]:
                    temp_count[u] += 1
                for u in neighbors[cand2]:
                    temp_count[u] += 1
                
                if np.min(temp_count) >= 1:
                    # Accept move
                    count = temp_count
                    in_code[c1] = False
                    in_code[c2] = False
                    in_code[cand1] = True
                    in_code[cand2] = True
                    if idx1 < idx2:
                        cw.pop(idx2)
                        cw.pop(idx1)
                    else:
                        cw.pop(idx1)
                        cw.pop(idx2)
                    cw.extend([cand1, cand2])
                    improved = True

        return best_cw

    def finish(cw):
        result = []
        for ci in cw:
            word = []
            val = ci
            for pos in range(n - 1, -1, -1):
                word.append(val % q)
                val //= q
            word.reverse()
            result.append(word)
        return result

    best_cw = None
    best_size = float('inf')

    # Multiple restarts
    seeds = [42, 1337, 2023, 1234, 5678]
    
    for seed in seeds:
        if time.time() > deadline - 40.0:
            break
            
        # Try deterministic greedy
        cw = greedy_init(seed, randomized=False)
        if len(cw) < best_size:
            polished = local_search(cw, deadline - 20.0)
            if len(polished) < best_size:
                best_size = len(polished)
                best_cw = polished

        # Try randomized greedy
        if time.time() > deadline - 40.0:
            break
        cw_r = greedy_init(seed, randomized=True)
        if len(cw_r) < best_size:
            polished_r = local_search(cw_r, deadline - 20.0)
            if len(polished_r) < best_size:
                best_size = len(polished_r)
                best_cw = polished_r

    # Final polish
    if best_cw is not None and time.time() < deadline - 10.0:
        final_polish = local_search(best_cw, deadline - 5.0)
        if len(final_polish) < best_size:
            best_cw = final_polish
            best_size = len(final_polish)

    if best_cw is None:
        # Fallback: just greedy
        best_cw = greedy_init(42, randomized=False)

    return finish(best_cw)
