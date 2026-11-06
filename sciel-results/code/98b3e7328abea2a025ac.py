"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _score_bitmask(mask):
    """Fast score using bitmask. mask is an int with bits 1..512 set."""
    diffs = set()
    for d in range(1, 257):
        found = False
        max_a = 512 - 2*d
        a = 1
        while a <= max_a:
            if (mask >> a) & 1 and (mask >> (a+d)) & 1 and (mask >> (a+2*d)) & 1:
                found = True
                break
            a += 1
        if found:
            diffs.add(d)
    return len(diffs)

def _score_set(S):
    """Count distinct d>0 such that a, a+d, a+2d all in S for some a."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for d in range(1, 257):
            if a + d in S_set and a + 2*d in S_set:
                diffs.add(d)
    return len(diffs)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    best_score = 0
    best_set = list(range(1, 33))
    
    N = 32
    MAX_VAL = 512
    
    def greedy_build():
        """Build a set greedily, adding element that maximizes new differences."""
        best_local = 0
        best_local_set = set()
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            init_count = 5
            init_elements = random.sample(range(1, 513), init_count)
            current = set(init_elements)
            
            for _ in range(N - init_count):
                best_gain = -1
                best_elem = None
                for cand in range(1, 513):
                    if cand in current:
                        continue
                    new_diffs = set()
                    for d in range(1, 257):
                        if cand + 2*d > 512:
                            break
                        if (cand + d) in current and (cand + 2*d) in current:
                            new_diffs.add(d)
                    for d in range(1, 257):
                        if cand - d < 1 or cand + d > 512:
                            continue
                        if (cand - d) in current and (cand + d) in current:
                            new_diffs.add(d)
                    for d in range(1, 257):
                        if cand - 2*d < 1:
                            break
                        if (cand - d) in current and (cand - 2*d) in current:
                            new_diffs.add(d)
                    
                    gain = len(new_diffs)
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = cand
                
                if best_elem is not None:
                    current.add(best_elem)
            
            s = _score_set(list(current))
            if s > best_local:
                best_local = s
                best_local_set = set(current)
        
        return list(best_local_set), best_local
    
    # Run greedy construction
    greedy_set, greedy_score = greedy_build()
    if greedy_score > best_score:
        best_score = greedy_score
        best_set = greedy_set
    
    # Local search: try swaps
    if time.time() < deadline:
        current_set = set(best_set)
        current_score = best_score
        improved = True
        max_iters = 2000
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            for _ in range(100):
                if time.time() > deadline:
                    break
                if len(current_set) != N:
                    break
                to_remove = random.choice(list(current_set))
                candidates = [x for x in range(1, 513) if x not in current_set]
                if not candidates:
                    break
                to_add = random.choice(candidates)
                
                current_set.remove(to_remove)
                current_set.add(to_add)
                new_score = _score_set(list(current_set))
                
                if new_score > current_score:
                    current_score = new_score
                    improved = True
                    if new_score > best_score:
                        best_score = new_score
                        best_set = list(current_set)
                else:
                    current_set.remove(to_add)
                    current_set.add(to_remove)
    
    # Ensure we return exactly 32 distinct integers in 1..512
    result = list(set(best_set))
    if len(result) != N:
        if len(result) > N:
            result = result[:N]
        else:
            for x in range(1, 513):
                if x not in result:
                    result.append(x)
                if len(result) == N:
                    break
    
    return result
# EVOLVE-BLOCK-END
