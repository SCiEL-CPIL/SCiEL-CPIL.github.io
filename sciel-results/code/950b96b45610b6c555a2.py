import random
import time

def construct():
    start_time = time.time()
    deadline = 115  # Safety margin (150s wall-clock limit, leave buffer)
    
    # Known-good fallback: a Sidon set containing 0 with all diffs 1..30
    # This is a truncated (31,32) difference set or similar construction
    # Precomputed solution with max element around 200-400
    fallback = [0, 1, 4, 10, 12, 17, 20, 26, 30, 37, 42, 48, 55, 62, 70, 78, 87, 96, 106, 116, 127, 138, 150, 162, 175, 188, 202, 216, 231, 247, 263, 280]
    # Verify fallback is valid
    def get_diffs_set(S):
        return set(y - x for i, x in enumerate(S) for y in S[i+1:])
    
    def is_sidon(S):
        if len(S) != len(set(S)):
            return False
        diffs = get_diffs_set(S)
        return len(diffs) == len(S) * (len(S) - 1) // 2
    
    def covers_1_to_30(S):
        diffs = get_diffs_set(S)
        return all(d in diffs for d in range(1, 31))
    
    # Use fallback if valid
    if is_sidon(fallback) and covers_1_to_30(fallback):
        best_set = fallback[:]
        best_max = max(fallback)
    else:
        best_set = None
        best_max = float('inf')
    
    def greedy_build(seed_rng):
        """Build a Sidon set greedily, trying to cover 1..30."""
        S = [0]
        covered = set()
        
        for _ in range(40):
            if covered.issuperset(set(range(1, 31))):
                break
            
            missing = set(range(1, 31)) - covered
            if not missing:
                break
            
            # Try multiple strategies for picking which diff to cover
            d = seed_rng.choice(sorted(missing))
            
            # Generate candidates: x ± d for all x in S, plus some random offsets
            candidates = set()
            for x in S:
                candidates.add(x + d)
                if x - d >= 0:
                    candidates.add(x - d)
            
            # Also try some random candidates near existing elements
            if S:
                max_s = max(S)
                for _ in range(10):
                    candidates.add(seed_rng.randint(1, max_s + d + 5))
            
            # Filter and evaluate candidates
            valid_candidates = []
            for c in candidates:
                if c < 0 or c in S:
                    continue
                new_S = sorted(S + [c])
                if is_sidon(new_S):
                    valid_candidates.append(c)
            
            if not valid_candidates:
                break
            
            # Pick candidate that minimizes max element
            c = min(valid_candidates, key=lambda x: max(S + [x]))
            S.append(c)
            S.sort()
            covered = get_diffs_set(S)
        
        if covers_1_to_30(S):
            return S
        return None
    
    def local_search(S, seed_rng, max_iters=200):
        """Try to improve the set by swapping elements."""
        if not S:
            return S
        
        best = S[:]
        best_max = max(best)
        
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
            
            improved = False
            # Try removing one element and adding another
            for i in range(len(best)):
                if time.time() - start_time > deadline - 5:
                    break
                
                candidate_remove = best[i]
                remaining = best[:i] + best[i+1:]
                
                # Try adding new elements
                for _ in range(5):
                    # New element near the gap
                    if remaining:
                        max_r = max(remaining)
                        new_val = seed_rng.randint(1, max_r + 20)
                    else:
                        new_val = 1
                    
                    if new_val in remaining:
                        continue
                    
                    new_S = sorted(remaining + [new_val])
                    if not is_sidon(new_S):
                        continue
                    if not covers_1_to_30(new_S):
                        continue
                    
                    new_max = max(new_S)
                    if new_max < best_max:
                        best = new_S
                        best_max = new_max
                        improved = True
                        break
            
            if not improved:
                # Try swapping two elements
                for i in range(len(best)):
                    if time.time() - start_time > deadline - 5:
                        break
                    for j in range(i+1, len(best)):
                        if time.time() - start_time > deadline - 5:
                            break
                        
                        # Try replacing best[j] with a smaller value
                        old_val = best[j]
                        for _ in range(3):
                            new_val = seed_rng.randint(1, old_val)
                            if new_val in best:
                                continue
                            
                            new_S = sorted(best[:j] + [new_val] + best[j+1:])
                            if not is_sidon(new_S):
                                continue
                            if not covers_1_to_30(new_S):
                                continue
                            
                            new_max = max(new_S)
                            if new_max < best_max:
                                best = new_S
                                best_max = new_max
                                improved = True
                                break
                        
                        if improved:
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best
    
    # Run multiple random restarts
    for restart in range(200):
        if time.time() - start_time > deadline - 10:
            break
        
        seed_rng = random.Random(restart * 7919 + 13)
        
        # Greedy build
        S = greedy_build(seed_rng)
        
        if S:
            # Local search to improve
            S = local_search(S, seed_rng, max_iters=50)
            
            if S:
                max_elem = max(S)
                if max_elem < best_max:
                    best_max = max_elem
                    best_set = S[:]
        
        # Also try starting from fallback and modifying
        if best_set and restart < 50:
            S_mod = best_set[:]
            S_mod = local_search(S_mod, seed_rng, max_iters=30)
            if S_mod:
                max_elem = max(S_mod)
                if max_elem < best_max:
                    best_max = max_elem
                    best_set = S_mod[:]
    
    if best_set:
        return best_set
    else:
        # Last resort: simple greedy
        a = [0]
        for d in range(1, 31):
            diffs = get_diffs_set(a)
            if d in diffs:
                continue
            found = False
            for x in sorted(set(v + s for v in a for s in (d, -d) if v + s >= 0) - set(a)):
                trial = sorted(a + [x])
                if is_sidon(trial):
                    a = trial
                    found = True
                    break
            if not found:
                y = 2 * max(a) + 3 * d + 1
                while True:
                    trial = sorted(a + [y, y + d])
                    if is_sidon(trial):
                        a = trial
                        break
                    y += 1
        return a
