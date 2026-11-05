import time
import itertools
import numpy as np
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Handle the specific case K_3(8,1)
    if q == 3 and n == 8 and radius == 1:
        return _solve_ternary_8_1(deadline)
    
    # Generic fallback for other cases
    return _generic_greedy(q, n, radius, deadline)

def _solve_ternary_8_1(deadline):
    q = 3
    n = 8
    total = q ** n  # 6561
    
    # Precompute all words
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    
    # Precompute balls (indices of words within distance 1)
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            original = w_list[i]
            for d in range(q):
                if d != original:
                    w_list[i] = d
                    neighbors.append(word_to_idx[tuple(w_list)])
            w_list[i] = original
        balls.append(np.array(neighbors, dtype=np.int32))
    
    # Precompute ball membership: for each word index, which codeword indices cover it
    # ball_members[j] = list of codeword indices c such that j in balls[c]
    ball_members = [[] for _ in range(total)]
    for c in range(total):
        for j in balls[c]:
            ball_members[j].append(c)
    
    # 1. Greedy Set Cover
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    remaining = total
    
    while remaining > 0 and time.time() < deadline:
        gains = np.array([uncovered[b].sum() for b in balls], dtype=np.int32)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        chosen.append(best_idx)
        uncovered[balls[best_idx]] = False
        remaining -= best_gain
        
    # 2. Prune
    chosen = _prune(chosen, balls, total)
    
    # 3. Local Search with Neighbor-Restricted Swaps and Restarts
    best_chosen = list(chosen)
    best_size = len(best_chosen)
    
    # Budget allocation: 60% for local search iterations, 40% for restarts
    remaining_time = deadline - time.time()
    num_restarts = max(3, int(remaining_time / 10.0))  # ~10s per restart
    time_per_restart = remaining_time / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        current_chosen = list(best_chosen)
        
        # Perturb with neighbor-restricted swaps
        for _ in range(20):
            if not current_chosen:
                break
            idx_to_remove = random.randint(0, len(current_chosen) - 1)
            c_to_remove = current_chosen[idx_to_remove]
            
            # Compute coverage count
            coverage_count = np.zeros(total, dtype=np.int32)
            for c in current_chosen:
                coverage_count[balls[c]] += 1
            
            coverage_count[balls[c_to_remove]] -= 1
            if np.any(coverage_count == 0):
                # Find critical words
                critical = [j for j in balls[c_to_remove] if coverage_count[j] == 0]
                if critical:
                    # Neighbor-restricted: only consider codewords within distance 2 of c_to_remove
                    # (i.e., codewords that share at least one neighbor with c_to_remove)
                    candidate_set = set()
                    for j in critical:
                        for c_prime in ball_members[j]:
                            if c_prime != c_to_remove and c_prime not in current_chosen:
                                candidate_set.add(c_prime)
                    
                    # Filter to neighbors (distance <= 2 from c_to_remove)
                    c_to_remove_set = set(balls[c_to_remove])
                    filtered_candidates = []
                    for c_prime in candidate_set:
                        # Check if c_prime is within distance 2 of c_to_remove
                        if len(set(balls[c_prime]) & c_to_remove_set) > 0:
                            filtered_candidates.append(c_prime)
                    
                    if not filtered_candidates:
                        # Fall back to all candidates if no neighbor-restricted candidates
                        filtered_candidates = list(candidate_set)
                    
                    random.shuffle(filtered_candidates)
                    swapped = False
                    for c_prime in filtered_candidates:
                        valid = True
                        for j in critical:
                            if c_prime not in ball_members[j]:
                                valid = False
                                break
                        if valid:
                            current_chosen[idx_to_remove] = c_prime
                            swapped = True
                            break
            else:
                current_chosen.pop(idx_to_remove)
        
        # Run local search with neighbor-restricted swaps
        refined_chosen = _local_search_neighbor_restricted(
            current_chosen, balls, ball_members, total, deadline, time_per_restart
        )
        
        if len(refined_chosen) < best_size:
            best_chosen = refined_chosen
            best_size = len(refined_chosen)
        
        # Final polish: run a tight local search on the best solution found
        if time.time() < deadline - 5.0:
            polished = _local_search_neighbor_restricted(
                best_chosen, balls, ball_members, total, deadline, deadline - 2.0
            )
            if len(polished) < best_size:
                best_chosen = polished
                best_size = len(best_chosen)
    
    return [all_words[i] for i in best_chosen]

def _prune(chosen, balls, total):
    if not chosen:
        return []
    
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        coverage_count[balls[c]] += 1
        
    keep = [True] * len(chosen)
    any_removed = True
    while any_removed:
        any_removed = False
        for i, c in enumerate(chosen):
            if not keep[i]:
                continue
            redundant = True
            for j in balls[c]:
                if coverage_count[j] <= 1:
                    redundant = False
                    break
            if redundant:
                keep[i] = False
                any_removed = True
                coverage_count[balls[c]] -= 1
                
    return [chosen[i] for i in range(len(chosen)) if keep[i]]

def _local_search_neighbor_restricted(chosen, balls, ball_members, total, deadline, time_limit):
    if not chosen:
        return []
        
    chosen = list(chosen)
    
    coverage_count = np.zeros(total, dtype=np.int32)
    for c in chosen:
        coverage_count[balls[c]] += 1
        
    chosen_set = set(chosen)
    
    improved = True
    while improved and time.time() < deadline and time.time() < time_limit:
        improved = False
        
        for i, c in enumerate(list(chosen)):
            if time.time() > deadline or time.time() > time_limit:
                break
                
            critical = [j for j in balls[c] if coverage_count[j] == 1]
            
            if not critical:
                chosen.pop(i)
                chosen_set.remove(c)
                coverage_count[balls[c]] -= 1
                improved = True
                break
            else:
                # Neighbor-restricted swap: only consider codewords that share a neighbor with c
                c_neighbors = set(balls[c])
                candidate_set = set()
                for j in critical:
                    for c_prime in ball_members[j]:
                        if c_prime != c and c_prime not in chosen_set:
                            candidate_set.add(c_prime)
                
                # Filter to neighbors
                filtered = [cp for cp in candidate_set if len(set(balls[cp]) & c_neighbors) > 0]
                if not filtered:
                    filtered = list(candidate_set)
                
                random.shuffle(filtered)
                found_swap = False
                for c_prime in filtered:
                    valid = True
                    for j in critical[1:]:
                        if c_prime not in ball_members[j]:
                            valid = False
                            break
                    if valid:
                        chosen[i] = c_prime
                        chosen_set.remove(c)
                        chosen_set.add(c_prime)
                        coverage_count[balls[c]] -= 1
                        coverage_count[balls[c_prime]] += 1
                        improved = True
                        found_swap = True
                        break
                
                if found_swap:
                    break
    return chosen

def _generic_greedy(q, n, radius, deadline):
    total = q ** n
    if total > 100000:
        return [list(w) for w in itertools.product(range(q), repeat=n)]
        
    all_words = [list(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {tuple(w): i for i, w in enumerate(all_words)}
    
    balls = []
    for w in all_words:
        neighbors = [word_to_idx[tuple(w)]]
        w_list = list(w)
        for i in range(n):
            original = w_list[i]
            for d in range(q):
                if d != original:
                    w_list[i] = d
                    neighbors.append(word_to_idx[tuple(w_list)])
            w_list[i] = original
        balls.append(np.array(neighbors, dtype=np.int32))
        
    uncovered = np.ones(total, dtype=bool)
    chosen = []
    remaining = total
    
    while remaining > 0 and time.time() < deadline:
        gains = np.array([uncovered[b].sum() for b in balls], dtype=np.int32)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        chosen.append(best_idx)
        uncovered[balls[best_idx]] = False
        remaining -= gains[best_idx]
        
    return [all_words[i] for i in chosen]
