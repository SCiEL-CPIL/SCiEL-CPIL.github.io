import time
import sys
from itertools import combinations
from math import gcd

_START_TIME = time.time()
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def _compute_lcm():
    result = 1
    for i in range(1, 23):
        result = result * i // gcd(result, i)
    return result

_LCM = _compute_lcm()
_W = [0] + [_LCM // i for i in range(1, 23)]

def construct():
    # 1. Find minimal dependent sets (bad sets)
    bad_sets = []
    
    max_check_size = 6
    
    for k in range(3, max_check_size + 1):
        if _time_left() < 5.0:
            break
            
        for combo in combinations(range(1, 23), k):
            total_sum = sum(_W[i] for i in combo)
            
            if total_sum % 2 != 0:
                continue
            
            target = total_sum // 2
            
            found = False
            for mask in range(1, (1 << k) // 2):
                s = 0
                for j in range(k):
                    if mask & (1 << j):
                        s += _W[combo[j]]
                if s == target:
                    is_minimal = True
                    for bs in bad_sets:
                        if bs.issubset(set(combo)):
                            is_minimal = False
                            break
                    
                    if is_minimal:
                        bad_sets.append(frozenset(combo))
                    found = True
                    break
            
    # Optimization for Backtracking:
    # Map bad_sets to indices
    bs_indices = {}
    for i, bs in enumerate(bad_sets):
        bs_indices[bs] = i
        
    # For each vertex, list of bad_set indices it belongs to
    vertex_to_bs = {v: [] for v in range(1, 23)}
    for i, bs in enumerate(bad_sets):
        for v in bs:
            vertex_to_bs[v].append(i)
            
    # Size of each bad set
    bs_sizes = [len(bs) for bs in bad_sets]
    
    best_size = 0
    best_set = []
    
    # Greedy lower bound
    greedy_set = set()
    greedy_counts = [0] * len(bad_sets)
    for v in range(1, 23):
        conflict = False
        for bs_idx in vertex_to_bs[v]:
            if greedy_counts[bs_idx] == bs_sizes[bs_idx] - 1:
                conflict = True
                break
        if not conflict:
            greedy_set.add(v)
            for bs_idx in vertex_to_bs[v]:
                greedy_counts[bs_idx] += 1
                
    best_size = len(greedy_set)
    best_set = list(greedy_set)
    
    # Backtracking
    # Order vertices by degree (number of bad sets)
    all_vertices = list(range(1, 23))
    def degree(v):
        return len(vertex_to_bs[v])
    all_vertices.sort(key=degree, reverse=True)
    
    # counts array for current path
    current_counts = [0] * len(bad_sets)
    
    def backtrack(available_idx, current_set, current_size):
        nonlocal best_size, best_set
        
        if _time_left() < 1.0:
            return
        
        if current_size + (len(all_vertices) - available_idx) <= best_size:
            return
            
        if available_idx == len(all_vertices):
            if current_size > best_size:
                best_size = current_size
                best_set = list(current_set)
            return
            
        v = all_vertices[available_idx]
        rest_idx = available_idx + 1
        
        # Option 1: Exclude v
        backtrack(rest_idx, current_set, current_size)
        
        # Option 2: Include v
        # Check conflict
        conflict = False
        for bs_idx in vertex_to_bs[v]:
            if current_counts[bs_idx] == bs_sizes[bs_idx] - 1:
                conflict = True
                break
                
        if not conflict:
            # Update counts
            for bs_idx in vertex_to_bs[v]:
                current_counts[bs_idx] += 1
                
            new_set = current_set | {v}
            backtrack(rest_idx, new_set, current_size + 1)
            
            # Rollback
            for bs_idx in vertex_to_bs[v]:
                current_counts[bs_idx] -= 1

    backtrack(0, set(), 0)
    
    return best_set

if __name__ == "__main__":
    s = construct()
    print(f"Found set: {s}")
    print(f"Size: {len(s)}")
