"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 80.0  # Safety margin before 90s limit
    best_set = None
    best_size = 13  # worse than any possible answer
    
    def max_sidon_size(s):
        """Compute the largest Sidon subset of s (all pair sums i<=j unique)."""
        n = len(s)
        if n <= 1:
            return n
        # Backtracking: try to build largest Sidon subset
        # Sort to help pruning
        s_sorted = sorted(s)
        best = [0]
        
        def backtrack(idx, chosen, sums_set):
            if time.time() > deadline:
                return
            # Update best
            if len(chosen) > best[0]:
                best[0] = len(chosen)
            # Pruning: even if we take all remaining, can't beat best
            if len(chosen) + (n - idx) <= best[0]:
                return
            if idx >= n:
                return
            # Option 1: skip s[idx]
            backtrack(idx + 1, chosen, sums_set)
            # Option 2: take s[idx] if no collision
            v = s_sorted[idx]
            new_sums = set()
            conflict = False
            for u in chosen:
                new_sums.add(v + u)
            new_sums.add(2 * v)
            for sm in new_sums:
                if sm in sums_set:
                    conflict = True
                    break
            if not conflict:
                backtrack(idx + 1, chosen + [v], sums_set | new_sums)
        
        backtrack(0, [], set())
        return best[0]
    
    def evaluate(s):
        return max_sidon_size(s)
    
    # Initial random sets
    random.seed(42)
    candidates = []
    
    # Try several initial configurations
    for trial in range(50):
        if time.time() > deadline - 5:
            break
        # Different initialization strategies
        if trial < 10:
            # Clustered: small range
            base = random.randint(0, 20)
            s = set()
            while len(s) < 12:
                s.add(random.randint(base, base + 30))
        elif trial < 20:
            # Arithmetic progression-like
            start = random.randint(0, 10)
            step = random.randint(2, 8)
            s = set()
            i = 0
            while len(s) < 12:
                s.add(start + i * step + random.randint(-1, 1))
                i += 1
            s = {x for x in s if 0 <= x <= 128}
            while len(s) < 12:
                s.add(random.randint(0, 128))
        elif trial < 30:
            # Two clusters
            c1 = random.randint(0, 40)
            c2 = random.randint(60, 100)
            s = set()
            for _ in range(6):
                s.add(c1 + random.randint(0, 10))
            for _ in range(6):
                s.add(c2 + random.randint(0, 10))
            s = {x for x in s if 0 <= x <= 128}
            while len(s) < 12:
                s.add(random.randint(0, 128))
        else:
            # Random
            s = set()
            while len(s) < 12:
                s.add(random.randint(0, 128))
        
        s_list = sorted(s)
        sz = evaluate(s_list)
        if sz < best_size:
            best_size = sz
            best_set = s_list[:]
        
        candidates.append((sz, s_list[:]))
    
    # Local search on the best candidate
    if best_set is not None:
        current = best_set[:]
        current_size = best_size
        no_improve = 0
        max_iters = 2000
        
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            if no_improve > 200:
                break
            
            # Try random swap
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            
            old_val = current[i]
            current[i] = new_val
            current_sorted = sorted(current)
            
            sz = evaluate(current_sorted)
            if sz < current_size:
                current_size = sz
                no_improve = 0
                if sz < best_size:
                    best_size = sz
                    best_set = current_sorted[:]
            else:
                # Accept with some probability (simulated annealing-like)
                if random.random() < 0.1:
                    no_improve += 1
                else:
                    current[i] = old_val
                    no_improve += 1
        
        # Also try single-element replacements
        for _ in range(500):
            if time.time() > deadline - 3:
                break
            i = random.randint(0, 11)
            new_val = random.randint(0, 128)
            if new_val in current:
                continue
            old_val = current[i]
            current[i] = new_val
            current_sorted = sorted(current)
            sz = evaluate(current_sorted)
            if sz < current_size:
                current_size = sz
                if sz < best_size:
                    best_size = sz
                    best_set = current_sorted[:]
            else:
                current[i] = old_val
    
    if best_set is None:
        return list(range(12))
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
