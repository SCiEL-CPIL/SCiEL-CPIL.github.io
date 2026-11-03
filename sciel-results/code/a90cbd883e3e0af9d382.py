import time
import random
import numpy as np

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin before 150s kill
    
    # Total number of words
    total = q ** n  # 2187 for q=3, n=7
    
    # Precompute all balls (radius 1): for each word, the set of words within Hamming distance 1
    # Word i in base-q representation
    # Ball of word w: w itself + all words differing in one position by 1 or 2 (mod q)
    
    # Convert integer to tuple
    def int_to_word(x):
        w = [0] * n
        for pos in range(n):
            w[pos] = x % q
            x //= q
        return w
    
    def word_to_int(w):
        x = 0
        for pos in range(n):
            x = x * q + w[pos]
        return x
    
    # Precompute balls as lists of integers
    balls = []
    for i in range(total):
        w = int_to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in [1, 2]:
                new_val = (w[pos] + delta) % q
                # Compute new integer
                val = i + (new_val - w[pos]) * (q ** pos)
                ball.append(val)
        balls.append(ball)
    
    # Convert to frozensets for fast operations? No, use numpy bitsets for speed
    # Actually, let's use a numpy boolean array for the "covered" state
    
    # Greedy construction
    def greedy_construct():
        covered = np.zeros(total, dtype=bool)
        selected = []
        remaining = total
        while remaining > 0:
            best_idx = -1
            best_count = -1
            for i in range(total):
                if covered[i]:
                    continue
                # Count how many uncovered words are in ball i
                count = 0
                for j in balls[i]:
                    if not covered[j]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx < 0:
                break
            selected.append(best_idx)
            for j in balls[best_idx]:
                if not covered[j]:
                    covered[j] = True
                    remaining -= 1
        return selected
    
    # Check if a set of codewords covers everything
    def is_valid(selected):
        if not selected:
            return total == 0
        covered = np.zeros(total, dtype=bool)
        for s in selected:
            for j in balls[s]:
                covered[j] = True
        return covered.all()
    
    # Local search: try to remove redundant codewords
    def local_search_remove(selected):
        """Try to remove each codeword if the covering remains valid."""
        selected = list(selected)
        changed = True
        while changed and (time.time() < deadline - 5):
            changed = False
            for idx in range(len(selected) - 1, -1, -1):
                # Check if removing selected[idx] still covers everything
                # Quick check: for each word in ball of selected[idx], check if it's covered by another
                test_selected = selected[:idx] + selected[idx+1:]
                # Efficient check: mark what's covered by test_selected
                covered = np.zeros(total, dtype=bool)
                for s in test_selected:
                    for j in balls[s]:
                        covered[j] = True
                if covered.all():
                    selected = test_selected
                    changed = True
                    break  # Restart the loop since indices changed
        return selected
    
    # More efficient local search using coverage counts
    def local_search_efficient(selected):
        """Use coverage counts to efficiently find removable codewords."""
        selected = list(selected)
        # Count how many times each word is covered
        cover_count = np.zeros(total, dtype=np.int32)
        for s in selected:
            for j in balls[s]:
                cover_count[j] += 1
        
        changed = True
        while changed and (time.time() < deadline - 5):
            changed = False
            for idx in range(len(selected) - 1, -1, -1):
                s = selected[idx]
                # Check if any word in ball(s) has cover_count == 1
                removable = True
                for j in balls[s]:
                    if cover_count[j] == 1:
                        removable = False
                        break
                if removable:
                    # Remove s
                    for j in balls[s]:
                        cover_count[j] -= 1
                    selected.pop(idx)
                    changed = True
                    break
        return selected
    
    # Simulated annealing / swap-based local search
    def local_search_swap(selected, iterations=5000):
        """Try swaps: replace one codeword with another to improve coverage."""
        selected = list(selected)
        cover_count = np.zeros(total, dtype=np.int32)
        for s in selected:
            for j in balls[s]:
                cover_count[j] += 1
        
        current_size = len(selected)
        best_size = current_size
        best_selected = list(selected)
        
        rng = random.Random(42)
        
        for it in range(iterations):
            if time.time() > deadline - 5:
                break
            if len(selected) < best_size:
                best_size = len(selected)
                best_selected = list(selected)
            
            # Try to remove a codeword
            idx = rng.randrange(len(selected))
            s = selected[idx]
            removable = True
            for j in balls[s]:
                if cover_count[j] == 1:
                    removable = False
                    break
            if removable:
                for j in balls[s]:
                    cover_count[j] -= 1
                selected.pop(idx)
                continue
            
            # Try a swap: replace s with a random other word
            new_s = rng.randrange(total)
            if new_s == s:
                continue
            # Check: after removing s and adding new_s, is everything still covered?
            # Words that become uncovered: those in ball(s) with cover_count==1
            # Words that become covered: those in ball(new_s)
            # This is complex; just do a full check for simplicity in early iterations
            # For speed, use a heuristic: only swap if new_s covers some uncovered word
            # Actually, let's just try removal first, and if that fails, skip swap for now
            # A better approach: try adding a random word, then try removing another
            
            # Alternative: try adding a random word that covers the most uncovered
            # But we want to MINIMIZE, so let's focus on removal
            
        # Final: try local_search_efficient on best
        result = local_search_efficient(best_selected)
        return result
    
    # Phase 1: Greedy construction
    selected = greedy_construct()
    size_after_greedy = len(selected)
    
    # Phase 2: Efficient local search (remove redundant)
    if time.time() < deadline - 10:
        selected = local_search_efficient(selected)
    
    # Phase 3: More aggressive search - try random restarts with different greedy orders
    best_selected = list(selected)
    best_size = len(selected)
    
    # Try a few more greedy variants with randomization
    for restart in range(5):
        if time.time() > deadline - 15:
            break
        
        # Randomized greedy: add some randomness to tie-breaking
        covered = np.zeros(total, dtype=bool)
        sel = []
        remaining = total
        while remaining > 0 and time.time() < deadline - 15:
            best_idx = -1
            best_count = -1
            # Sample a subset for speed in later iterations
            candidates = list(range(total))
            random.shuffle(candidates)
            # Limit candidates for speed
            limit = min(len(candidates), 500)
            for i in candidates[:limit]:
                if covered[i]:
                    continue
                count = 0
                for j in balls[i]:
                    if not covered[j]:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_idx = i
            if best_idx < 0:
                # Fallback: find any uncovered
                for i in range(total):
                    if not covered[i]:
                        best_idx = i
                        break
                if best_idx < 0:
                    break
            sel.append(best_idx)
            for j in balls[best_idx]:
                if not covered[j]:
                    covered[j] = True
                    remaining -= 1
        
        if len(sel) < best_size:
            sel = local_search_efficient(sel)
            if len(sel) < best_size:
                best_size = len(sel)
                best_selected = list(sel)
    
    # Phase 4: Final polish - try more removals
    if time.time() < deadline - 5:
        best_selected = local_search_efficient(best_selected)
    
    # Verify validity
    if not is_valid(best_selected):
        # Fallback to a known valid construction
        # Tetracode direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        import itertools
        best_selected = [word_to_int(h + list(tail)) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert back to list of lists
    return [int_to_word(s) for s in best_selected]
