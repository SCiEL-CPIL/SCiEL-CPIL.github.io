"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 9

# EVOLVE-BLOCK-START
import random

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    rng = random.Random(42)
    
    def get_candidates(cur, used, path):
        """Get valid next vertices from cur."""
        candidates = []
        for b in range(N):
            v = cur ^ (1 << b)
            if v in used:
                continue
            # Check if v conflicts with any non-adjacent vertex (all except last)
            conflict = False
            for p in path[:-1]:
                if (v ^ p).bit_count() == 1:
                    conflict = True
                    break
            if not conflict:
                candidates.append(v)
        return candidates
    
    def count_future_options(v, used, path):
        """Count how many options v would leave for the next step (fail-first heuristic)."""
        new_used = used | {v}
        count = 0
        for b in range(N):
            w = v ^ (1 << b)
            if w in new_used:
                continue
            # Check if w conflicts with any vertex in path (all of them, since w is 2 steps from v)
            conflict = False
            for p in path:
                if (w ^ p).bit_count() == 1:
                    conflict = True
                    break
            if not conflict:
                count += 1
        return count
    
    def backtrack(path, used, depth_limit):
        """Constraint-based backtracking with fail-first heuristic."""
        if time.time() > deadline:
            return False
        if len(path) >= depth_limit:
            return True
        
        cur = path[-1]
        candidates = get_candidates(cur, used, path)
        
        if not candidates:
            return False
        
        # Fail-first: sort candidates by future options (ascending) to find dead-ends early
        scored = []
        for v in candidates:
            opts = count_future_options(v, used, path)
            scored.append((opts, v))
        scored.sort()
        
        for opts, v in scored:
            path.append(v)
            used.add(v)
            if backtrack(path, used, depth_limit):
                return True
            path.pop()
            used.remove(v)
            if time.time() > deadline:
                return False
        return False
    
    def greedy_extend(path, used):
        """Greedy extension using fail-first heuristic."""
        while True:
            cur = path[-1]
            candidates = get_candidates(cur, used, path)
            if not candidates:
                break
            # Pick candidate with fewest future options (fail-first)
            best_v = min(candidates, key=lambda v: count_future_options(v, used, path))
            path.append(best_v)
            used.add(best_v)
    
    # Phase 1: Greedy extension with fail-first to get a good starting point
    path = [0]
    used = {0}
    greedy_extend(path, used)
    best_path = list(path)
    best_len = len(path)
    
    # Phase 2: Try to extend further with backtracking
    # Use iterative deepening: try to extend by 1, then 2, etc.
    target_len = best_len + 1
    while target_len <= best_len + 50 and time.time() < deadline:
        if backtrack(path, used, target_len):
            best_path = list(path)
            best_len = len(path)
            target_len = best_len + 1
        else:
            # Try a different random restart
            break
    
    # Phase 3: Multiple random restarts with different greedy strategies
    for restart in range(20):
        if time.time() > deadline:
            break
        
        # Randomize the greedy choice
        path = [0]
        used = {0}
        
        while True:
            if time.time() > deadline:
                break
            cur = path[-1]
            candidates = get_candidates(cur, used, path)
            if not candidates:
                break
            # With some probability, pick randomly among top candidates
            if rng.random() < 0.3:
                # Pick randomly
                v = rng.choice(candidates)
            else:
                # Fail-first
                v = min(candidates, key=lambda v: count_future_options(v, used, path))
            path.append(v)
            used.add(v)
        
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        
        # Try backtracking extension from this point
        if time.time() < deadline - 10:
            target = best_len + 1
            if backtrack(path, used, target):
                if len(path) > best_len:
                    best_path = list(path)
                    best_len = len(path)
    
    # Final validation
    def validate(path):
        if len(path) < 2:
            return True
        for i in range(len(path)):
            for j in range(i+1, len(path)):
                if abs(i - j) == 1:
                    if (path[i] ^ path[j]).bit_count() != 1:
                        return False
                else:
                    if (path[i] ^ path[j]).bit_count() == 1:
                        return False
        return True
    
    if not validate(best_path):
        # Fallback: simple greedy
        path = [0]
        used = {0}
        while True:
            cur = path[-1]
            candidates = get_candidates(cur, used, path)
            if not candidates:
                break
            path.append(candidates[0])
            used.add(candidates[0])
        best_path = path
    
    return best_path
# EVOLVE-BLOCK-END
