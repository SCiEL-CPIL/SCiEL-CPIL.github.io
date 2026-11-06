import time
import random
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin for 60s limit
    
    n = 256
    inv = [0.0] + [1.0/i for i in range(1, n+1)]
    
    # Precompute 3-APs
    conflict_pairs = {i: [] for i in range(1, n+1)}
    for b in range(2, n):
        for a in range(1, b):
            c = 2*b - a
            if c <= n:
                conflict_pairs[a].append((b, c))
                conflict_pairs[b].append((a, c))
                conflict_pairs[c].append((a, b))
    
    # Helper to check if x can be added to mask M
    def can_add(x, M):
        for y, z in conflict_pairs[x]:
            if (M & (1 << (y-1))) and (M & (1 << (z-1))):
                return False
        return True

    def sum_recip(M):
        s = 0.0
        for i in range(1, n+1):
            if M & (1 << (i-1)):
                s += inv[i]
        return s

    # Branch and Bound with feasibility-based upper bound
    # Key insight: For a partial solution with chosen set C and remaining candidates R,
    # the upper bound is sum(1/x for x in C) + sum(1/x for x in R that can be added to C)
    # But this is too optimistic. Better: use the fact that 3-AP-free sets have density constraints.
    
    # Actually, let's use a different approach: iterative improvement with smart swaps
    # Start with greedy, then try to improve by:
    # 1. For each element x in S, try to remove it and add a smaller element y not in S
    # 2. For pairs of elements, try to replace them with better combinations
    
    # First, get a good initial solution via greedy
    def greedy_init():
        M = 0
        for x in range(1, n+1):
            if can_add(x, M):
                M |= (1 << (x-1))
        return M

    # Local search: try to improve by swaps
    def local_search(M, time_limit):
        current_score = sum_recip(M)
        best_score = current_score
        best_mask = M
        improved = True
        
        # Precompute list of elements in M and not in M for faster access
        while improved and time.time() < time_limit:
            improved = False
            
            # Get elements in M and not in M
            in_M = [i for i in range(1, n+1) if M & (1 << (i-1))]
            not_in_M = [i for i in range(1, n+1) if not (M & (1 << (i-1)))]
            
            # Try single swaps: remove x from M, add y to M\x
            for x in in_M:
                M_without_x = M & ~(1 << (x-1))
                for y in not_in_M:
                    if y >= x:
                        break  # Only consider y < x for potential improvement
                    if can_add(y, M_without_x):
                        new_M = M_without_x | (1 << (y-1))
                        new_score = current_score - inv[x] + inv[y]
                        if new_score > best_score + 1e-10:
                            best_score = new_score
                            best_mask = new_M
                            M = new_M
                            current_score = new_score
                            improved = True
            
            # Try 2-for-1 swaps: remove x1, x2 from M, add y to M\x1\x2
            for i, x1 in enumerate(in_M):
                for x2 in in_M[i+1:]:
                    M_without = M & ~(1 << (x1-1)) & ~(1 << (x2-1))
                    for y in not_in_M:
                        if inv[y] <= inv[x1] + inv[x2]:
                            break
                        if can_add(y, M_without):
                            new_M = M_without | (1 << (y-1))
                            new_score = current_score - inv[x1] - inv[x2] + inv[y]
                            if new_score > best_score + 1e-10:
                                best_score = new_score
                                best_mask = new_M
                                M = new_M
                                current_score = new_score
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            
            # Try 1-for-2 swaps: remove x from M, add y1, y2 to M\x
            for x in in_M:
                M_without_x = M & ~(1 << (x-1))
                found = False
                for y1 in not_in_M:
                    if found:
                        break
                    for y2 in not_in_M:
                        if y1 == y2:
                            continue
                        if inv[y1] + inv[y2] <= inv[x]:
                            continue
                        if can_add(y1, M_without_x) and can_add(y2, M_without_x | (1 << (y1-1))):
                            new_M = M_without_x | (1 << (y1-1)) | (1 << (y2-1))
                            new_score = current_score - inv[x] + inv[y1] + inv[y2]
                            if new_score > best_score + 1e-10:
                                best_score = new_score
                                best_mask = new_M
                                M = new_M
                                current_score = new_score
                                improved = True
                                found = True
                                break
                    if found:
                        break
                if improved:
                    break
        
        return best_mask, best_score

    # Run local search from greedy start
    M0 = greedy_init()
    best_mask, best_score = local_search(M0, deadline - 5.0)
    
    # Try a few random restarts with local search if time permits
    remaining = deadline - time.time()
    if remaining > 3.0:
        for _ in range(min(3, int(remaining // 2.0))):
            # Random valid initialization
            M_rand = 0
            candidates = list(range(1, n+1))
            random.shuffle(candidates)
            for x in candidates:
                if can_add(x, M_rand):
                    M_rand |= (1 << (x-1))
            
            m, s = local_search(M_rand, min(1.5, deadline - time.time() - 1.0))
            if s > best_score:
                best_score = s
                best_mask = m
            if time.time() > deadline - 2.0:
                break

    # Convert best_mask to list
    result = []
    for i in range(1, n+1):
        if best_mask & (1 << (i-1)):
            result.append(i)
            
    return result

# EVOLVE-BLOCK-END
