import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill
    
    N = 1000
    max_sum = 2 * N  # 2000
    
    def greedy_forward(counts, chosen):
        """Continue greedy from current state. Returns (chosen, counts)."""
        for x in range(chosen[-1] + 1, N + 1) if chosen else range(1, N + 1):
            # Check if adding x keeps all sums <= 2
            ok = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    ok = False
            if ok:
                # Update counts
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        return chosen, counts
    
    def greedy_with_skip(counts, chosen, skip_prob=0.1):
        """Greedy with occasional skips to explore different paths."""
        for x in range(chosen[-1] + 1, N + 1) if chosen else range(1, N + 1):
            # Check if adding x keeps all sums <= 2
            ok = True
            for y in chosen:
                s = x + y
                if counts.get(s, 0) >= 2:
                    ok = False
                    break
            if ok:
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    ok = False
            if ok and random.random() > skip_prob:
                for y in chosen:
                    s = x + y
                    counts[s] = counts.get(s, 0) + 1
                s = 2 * x
                counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
        return chosen, counts
    
    def local_search(chosen, counts, max_steps=200):
        """Try to improve by swapping elements."""
        chosen_set = set(chosen)
        best_size = len(chosen)
        
        for _ in range(max_steps):
            if time.time() > deadline:
                break
            improved = False
            # Try to add a random element not in chosen
            candidates = [x for x in range(1, N + 1) if x not in chosen_set]
            if not candidates:
                break
            
            random.shuffle(candidates)
            for x in candidates[:50]:  # Try up to 50 random candidates
                # Check if we can add x by removing some other element
                # First check if x alone is valid
                ok = True
                for y in chosen:
                    if y != x:
                        s = x + y
                        if counts.get(s, 0) >= 2:
                            ok = False
                            break
                if not ok:
                    continue
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    continue
                
                # Find which elements conflict with x
                conflicts = []
                for y in chosen:
                    if y != x:
                        s = x + y
                        if counts.get(s, 0) >= 2:
                            conflicts.append(y)
                s = 2 * x
                if counts.get(s, 0) >= 2:
                    # 2x is tight, need to remove an element that contributes to 2x
                    # Only x itself contributes to 2x (since we're adding x)
                    # So this can't be fixed by removing others
                    continue
                
                if not conflicts:
                    # Can add x directly
                    for y in chosen:
                        s = x + y
                        counts[s] = counts.get(s, 0) + 1
                    s = 2 * x
                    counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    chosen_set.add(x)
                    best_size = len(chosen)
                    improved = True
                    break
                else:
                    # Try removing one conflicting element
                    for rem in conflicts:
                        if rem == x:
                            continue
                        # Check if removing rem and adding x improves
                        new_counts = dict(counts)
                        # Remove rem's contributions
                        for y in chosen:
                            if y != rem:
                                s = rem + y
                                new_counts[s] = new_counts.get(s, 0) - 1
                        s = 2 * rem
                        new_counts[s] = new_counts.get(s, 0) - 1
                        # Add x's contributions
                        for y in chosen:
                            if y != rem:
                                s = x + y
                                new_counts[s] = new_counts.get(s, 0) + 1
                        s = 2 * x
                        new_counts[s] = new_counts.get(s, 0) + 1
                        # Check all sums <= 2
                        valid = True
                        for s, c in new_counts.items():
                            if c > 2:
                                valid = False
                                break
                        if valid:
                            # Apply swap
                            for y in chosen:
                                if y != rem:
                                    s = rem + y
                                    counts[s] = counts.get(s, 0) - 1
                            s = 2 * rem
                            counts[s] = counts.get(s, 0) - 1
                            for y in chosen:
                                if y != rem:
                                    s = x + y
                                    counts[s] = counts.get(s, 0) + 1
                            s = 2 * x
                            counts[s] = counts.get(s, 0) + 1
                            chosen.remove(rem)
                            chosen_set.discard(rem)
                            chosen.append(x)
                            chosen_set.add(x)
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        return chosen, counts, best_size
    
    # Phase 1: Basic greedy
    chosen, counts = greedy_forward({}, [])
    best_chosen = list(chosen)
    best_size = len(chosen)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 2: Multiple greedy restarts with different skip probabilities
    for trial in range(50):
        if time.time() > deadline:
            break
        random.seed(trial * 7 + 13)
        chosen2, counts2 = greedy_with_skip({}, [], skip_prob=0.05 + 0.01 * trial)
        if len(chosen2) > best_size:
            best_size = len(chosen2)
            best_chosen = list(chosen2)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 3: Local search on best solution
    chosen_ls = list(best_chosen)
    # Rebuild counts from scratch for safety
    counts_ls = {}
    for i, a in enumerate(chosen_ls):
        for j in range(i, len(chosen_ls)):
            s = chosen_ls[i] + chosen_ls[j]
            counts_ls[s] = counts_ls.get(s, 0) + 1
    
    chosen_ls, counts_ls, best_size_ls = local_search(chosen_ls, counts_ls, max_steps=500)
    if best_size_ls > best_size:
        best_size = best_size_ls
        best_chosen = list(chosen_ls)
    
    if time.time() > deadline:
        return best_chosen
    
    # Phase 4: More greedy restarts with varying parameters
    for trial in range(30):
        if time.time() > deadline:
            break
        random.seed(1000 + trial * 31)
        skip_p = random.uniform(0.0, 0.15)
        chosen3, counts3 = greedy_with_skip({}, [], skip_prob=skip_p)
        if len(chosen3) > best_size:
            best_size = len(chosen3)
            best_chosen = list(chosen3)
            # Local search on new best
            chosen_ls2 = list(best_chosen)
            counts_ls2 = {}
            for i, a in enumerate(chosen_ls2):
                for j in range(i, len(chosen_ls2)):
                    s = chosen_ls2[i] + chosen_ls2[j]
                    counts_ls2[s] = counts_ls2.get(s, 0) + 1
            chosen_ls2, counts_ls2, best_size_ls2 = local_search(chosen_ls2, counts_ls2, max_steps=300)
            if best_size_ls2 > best_size:
                best_size = best_size_ls2
                best_chosen = list(chosen_ls2)
    
    return best_chosen
# EVOLVE-BLOCK-END
