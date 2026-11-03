import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # For the specific task: q=3, n=6, radius=1
    # We'll use a specialized approach for this case, with a generic fallback
    
    if q == 3 and n == 6 and radius == 1:
        return _construct_361(start_time, deadline)
    else:
        return _generic_construct(q, n, radius, start_time, deadline)

def _construct_361(start_time, deadline):
    """Specialized solver for K_3(6,1) - ternary covering code length 6 radius 1"""
    
    q, n, radius = 3, 6, 1
    total = q ** n  # 729
    
    # Generate all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: covers[c] = set of word indices within distance <= radius of word c
    # For radius 1: the word itself + all words differing in exactly 1 position
    covers = [None] * total
    for i, w in enumerate(all_words):
        cov = set()
        cov.add(i)
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                cov.add(word_to_idx[tuple(new_w)])
        covers[i] = cov
    
    # Precompute distance-2 neighbors for the conflict graph
    # Two codewords are "conflicting" if they are at distance 2 (their balls overlap in a way
    # that makes one potentially redundant given the other, but more precisely, for the
    # independent set formulation: we want no two selected codewords at distance <= 2? No,
    # actually for covering codes, the constraint is just coverage. The independent set
    # idea is: if two codewords are at distance 2, they share some covered words.
    # Actually, the key insight: a set of codewords is a valid covering iff every word is
    # within distance 1 of at least one codeword. The "conflict" for reduction is:
    # if codeword c's ball is entirely covered by other codewords, c is redundant.
    
    # Let's use a different structural approach:
    # 1. Start with a good initial cover
    # 2. Use greedy removal + simulated annealing with smart moves
    
    # Initial construction: Tetracode-based
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    init_indices = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            cw = tuple(h + list(tail))
            init_indices.append(word_to_idx[cw])
    
    active = set(init_indices)
    
    # Compute cover counts
    cover_count = np.zeros(total, dtype=np.int32)
    for idx in active:
        for w in covers[idx]:
            cover_count[w] += 1
    
    best_set = set(active)
    best_size = len(active)
    
    # Phase 1: Greedy redundancy removal
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        items = list(active)
        random.shuffle(items)
        for idx in items:
            if idx not in active:
                continue
            # Check if redundant: all words in its ball have cover_count > 1
            redundant = True
            for w in covers[idx]:
                if cover_count[w] <= 1:
                    redundant = False
                    break
            if redundant:
                for w in covers[idx]:
                    cover_count[w] -= 1
                active.remove(idx)
                changed = True
                if len(active) < best_size:
                    best_size = len(active)
                    best_set = set(active)
    
    # Phase 2: Simulated Annealing
    # Key improvements:
    # - Use numpy for fast ball operations
    # - Smart move selection: prioritize swaps that maintain coverage
    # - Periodic greedy removal after SA steps
    
    # Convert covers to numpy arrays for speed
    covers_arr = [np.array(list(cov), dtype=np.int32) for cov in covers]
    
    T = 2.0
    cooling_rate = 0.9995
    min_T = 0.05
    
    def get_energy(size, uncov):
        return uncov * 100000 + size
    
    current_energy = get_energy(len(active), int(np.sum(cover_count == 0)))
    
    all_indices = np.arange(total)
    uncovered_mask = (cover_count == 0)
    uncovered_count = int(np.sum(uncovered_mask))
    
    step = 0
    max_steps = 500000
    
    while time.time() < deadline - 3 and step < max_steps:
        step += 1
        
        # Periodic greedy removal
        if step % 200 == 0:
            changed = True
            while changed:
                changed = False
                items = list(active)
                random.shuffle(items)
                for idx in items:
                    if idx not in active:
                        continue
                    redundant = True
                    for w in covers_arr[idx]:
                        if cover_count[w] <= 1:
                            redundant = False
                            break
                    if redundant:
                        cover_count[covers_arr[idx]] -= 1
                        active.remove(idx)
                        changed = True
                        if len(active) < best_size:
                            best_size = len(active)
                            best_set = set(active)
                        uncovered_count = int(np.sum(cover_count == 0))
                        break
        
        # If we have uncovered words, prioritize fixing them
        if uncovered_count > 0:
            # Find an uncovered word and add a codeword that covers it
            uncov_words = np.where(cover_count == 0)[0]
            if len(uncov_words) > 0:
                target = random.choice(uncov_words)
                # Find codewords that cover target
                # A codeword c covers target if target is in covers[c]
                # Precompute reverse: covered_by[w] = set of codewords covering w
                # For speed, just try random ones
                found = False
                for _ in range(30):
                    cand = random.randint(0, total - 1)
                    if cand not in active and target in covers[cand]:
                        found = True
                        break
                if not found:
                    # Try all codewords that could cover target
                    # target is at distance 1 from words that differ in exactly 1 position
                    # So codewords covering target are: target itself + words at distance 1 from target
                    # Wait, no. covers[c] contains words within distance 1 of c.
                    # So c covers target iff target is within distance 1 of c, i.e., c is within distance 1 of target.
                    # So candidates are target and its distance-1 neighbors.
                    cands = list(covers[target])  # words within distance 1 of target = codewords that cover target
                    cands = [c for c in cands if c not in active]
                    if cands:
                        cand = random.choice(cands)
                        found = True
                    else:
                        cand = random.randint(0, total - 1)
                        while cand in active:
                            cand = random.randint(0, total - 1)
                
                ball = covers_arr[cand]
                delta_uncov = -int(np.sum(cover_count[ball] == 0))
                delta_energy = get_energy(len(active) + 1, uncovered_count + delta_uncov) - current_energy
                
                if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                    cover_count[ball] += 1
                    active.add(cand)
                    uncovered_count += delta_uncov
                    current_energy += delta_energy
        
        else:
            # All covered, try to reduce size
            move_type = random.randint(0, 2)
            
            if move_type == 0:  # Remove
                if len(active) <= 1:
                    T *= cooling_rate
                    continue
                idx = random.choice(list(active))
                ball = covers_arr[idx]
                delta_uncov = int(np.sum(cover_count[ball] == 1))
                delta_energy = get_energy(len(active) - 1, uncovered_count + delta_uncov) - current_energy
                
                if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                    cover_count[ball] -= 1
                    active.remove(idx)
                    uncovered_count += delta_uncov
                    current_energy += delta_energy
                    if uncovered_count == 0 and len(active) < best_size:
                        best_size = len(active)
                        best_set = set(active)
            
            elif move_type == 1:  # Swap
                if len(active) <= 1:
                    T *= cooling_rate
                    continue
                idx_remove = random.choice(list(active))
                
                # Choose idx_add: prefer one that covers words uniquely covered by idx_remove
                ball_remove = covers_arr[idx_remove]
                # Words uniquely covered by idx_remove (cover_count == 1 and in ball_remove)
                unique_words = ball_remove[cover_count[ball_remove] == 1]
                
                if len(unique_words) > 0:
                    # Find a codeword that covers some of these unique words
                    target_w = random.choice(unique_words)
                    # Candidates: codewords within distance 1 of target_w that are not in active
                    cands = [c for c in covers[target_w] if c not in active and c != idx_remove]
                    if cands:
                        idx_add = random.choice(cands)
                    else:
                        idx_add = random.randint(0, total - 1)
                        while idx_add in active or idx_add == idx_remove:
                            idx_add = random.randint(0, total - 1)
                else:
                    idx_add = random.randint(0, total - 1)
                    while idx_add in active or idx_add == idx_remove:
                        idx_add = random.randint(0, total - 1)
                
                ball_add = covers_arr[idx_add]
                
                # Compute delta for swap
                # Words that become uncovered: in ball_remove, not in ball_add, cover_count == 1
                # Words that become covered: in ball_add, not in ball_remove, cover_count == 0
                ball_remove_set = set(ball_remove)
                ball_add_set = set(ball_add)
                
                become_uncov = 0
                for w in ball_remove:
                    if cover_count[w] == 1 and w not in ball_add_set:
                        become_uncov += 1
                
                become_cov = 0
                for w in ball_add:
                    if cover_count[w] == 0 and w not in ball_remove_set:
                        become_cov += 1
                
                delta_uncov = become_uncov - become_cov
                delta_energy = get_energy(len(active), uncovered_count + delta_uncov) - current_energy
                
                if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                    cover_count[ball_remove] -= 1
                    cover_count[ball_add] += 1
                    active.remove(idx_remove)
                    active.add(idx_add)
                    uncovered_count += delta_uncov
                    current_energy += delta_energy
                    if uncovered_count == 0 and len(active) < best_size:
                        best_size = len(active)
                        best_set = set(active)
            
            else:  # 2-for-1: try to replace two codewords with one
                if len(active) <= 2:
                    T *= cooling_rate
                    continue
                c1, c2 = random.sample(list(active), 2)
                union = covers[c1] | covers[c2]
                # Find a codeword c3 that covers all of union
                # c3 must be within distance 1 of every word in union
                # Sample a word from union and check codewords that cover it
                sample_w = random.choice(list(union))
                candidates = [c for c in covers[sample_w] if c not in active and c != c1 and c != c2]
                found_c3 = -1
                for c3 in candidates:
                    if union.issubset(covers[c3]):
                        found_c3 = c3
                        break
                
                if found_c3 >= 0:
                    ball1 = covers_arr[c1]
                    ball2 = covers_arr[c2]
                    ball3 = covers_arr[found_c3]
                    
                    # Simulate: remove c1, c2, add c3
                    # cover_count_new = cover_count - 1 on ball1 - 1 on ball2 + 1 on ball3
                    # Check if any word becomes uncovered
                    
                    # Words that could become uncovered: those in (ball1 ∪ ball2) \ ball3 with cover_count <= 2
                    # More precisely:
                    # For w in ball1 ∪ ball2: new_count = cover_count[w] - 1*(w in ball1) - 1*(w in ball2) + 1*(w in ball3)
                    # For w not in ball1 ∪ ball2: new_count = cover_count[w] + 1*(w in ball3)
                    
                    become_uncov = 0
                    for w in union:
                        in_b1 = 1 if w in ball1 else 0
                        in_b2 = 1 if w in ball2 else 0
                        in_b3 = 1 if w in ball3 else 0
                        new_count = cover_count[w] - in_b1 - in_b2 + in_b3
                        if new_count == 0:
                            become_uncov += 1
                    
                    # Words that become covered (were uncovered, now in ball3 but not in ball1∪ball2)
                    become_cov = 0
                    for w in ball3:
                        if cover_count[w] == 0 and w not in union:
                            become_cov += 1
                    
                    delta_uncov = become_uncov - become_cov
                    delta_energy = get_energy(len(active) - 1, uncovered_count + delta_uncov) - current_energy
                    
                    if delta_energy < 0 or random.random() < np.exp(-delta_energy / T):
                        cover_count[ball1] -= 1
                        cover_count[ball2] -= 1
                        cover_count[ball3] += 1
                        active.remove(c1)
                        active.remove(c2)
                        active.add(found_c3)
                        uncovered_count += delta_uncov
                        current_energy += delta_energy
                        if uncovered_count == 0 and len(active) < best_size:
                            best_size = len(active)
                            best_set = set(active)
        
        T *= cooling_rate
        if T < min_T:
            # Restart with perturbation
            T = 2.0
            # Add some random codewords
            for _ in range(random.randint(1, 3)):
                cand = random.randint(0, total - 1)
                if cand not in active:
                    cover_count[covers_arr[cand]] += 1
                    active.add(cand)
                    uncovered_count = int(np.sum(cover_count == 0))
                    current_energy = get_energy(len(active), uncovered_count)
            
            # Quick greedy removal
            for _ in range(5):
                items = list(active)
                random.shuffle(items)
                for idx in items:
                    if idx not in active:
                        continue
                    redundant = True
                    for w in covers_arr[idx]:
                        if cover_count[w] <= 1:
                            redundant = False
                            break
                    if redundant:
                        cover_count[covers_arr[idx]] -= 1
                        active.remove(idx)
                        uncovered_count = int(np.sum(cover_count == 0))
                        current_energy = get_energy(len(active), uncovered_count)
                        break

    # Final verification and cleanup
    # Ensure best_set is valid
    if best_size < len(active):
        active = set(best_set)
    
    # Final greedy removal pass
    changed = True
    while changed and time.time() < deadline - 1:
        changed = False
        items = list(active)
        random.shuffle(items)
        # Recompute cover count for active
        cc = np.zeros(total, dtype=np.int32)
        for idx in active:
            cc[covers_arr[idx]] += 1
        for idx in items:
            if idx not in active:
                continue
            redundant = True
            for w in covers_arr[idx]:
                if cc[w] <= 1:
                    redundant = False
                    break
            if redundant:
                active.remove(idx)
                changed = True
                if len(active) < best_size:
                    best_size = len(active)
                break
    
    # Convert to result
    result = [list(all_words[idx]) for idx in active]
    return result

def _generic_construct(q, n, radius, start_time, deadline):
    """Generic fallback for other parameter values"""
    total = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = [None] * total
    digits_array = np.array(all_words, dtype=np.int32)
    
    for i in range(total):
        diff = digits_array != digits_array[i]
        dists = np.sum(diff, axis=1)
        balls[i] = np.where(dists <= radius)[0]
    
    # Greedy set cover
    active = set()
    covered = np.zeros(total, dtype=bool)
    
    while not covered.all() and time.time() < deadline - 5:
        # Find codeword that covers most uncovered words
        best_c = -1
        best_count = -1
        for c in range(total):
            if c in active:
                continue
            count = np.sum(~covered[balls[c]])
            if count > best_count:
                best_count = count
                best_c = c
        if best_count <= 0:
            break
        active.add(best_c)
        covered[balls[best_c]] = True
    
    # Greedy removal
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        cover_count = np.zeros(total, dtype=np.int32)
        for idx in active:
            cover_count[balls[idx]] += 1
        for idx in list(active):
            if np.all(cover_count[balls[idx]] > 1):
                cover_count[balls[idx]] -= 1
                active.remove(idx)
                changed = True
    
    return [list(all_words[idx]) for idx in active]
