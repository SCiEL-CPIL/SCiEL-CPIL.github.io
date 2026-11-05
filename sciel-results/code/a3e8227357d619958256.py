import time
import math
import random
import sys

N = 128

# Precompute conflicts and shadow
conflicts = [set() for _ in range(N + 1)]
shadow = [set() for _ in range(N + 1)]
all_triples = []

def precompute_conflicts():
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            L_ab = math.lcm(a, b)
            for c in range(b + 1, N + 1):
                if L_ab % c != 0:
                    continue
                if math.lcm(a, c) == L_ab and math.lcm(b, c) == L_ab:
                    conflicts[a].add((b, c))
                    conflicts[b].add((a, c))
                    conflicts[c].add((a, b))
                    shadow[a].add(b)
                    shadow[a].add(c)
                    shadow[b].add(a)
                    shadow[b].add(c)
                    shadow[c].add(a)
                    shadow[c].add(b)
                    all_triples.append((a, b, c))

precompute_conflicts()

def can_add(x, current_set):
    for u, v in conflicts[x]:
        if u in current_set and v in current_set:
            return False
    return True

def is_valid(s):
    s_list = sorted(s)
    for i in range(len(s_list)):
        for j in range(i + 1, len(s_list)):
            for k in range(j + 1, len(s_list)):
                a, b, c = s_list[i], s_list[j], s_list[k]
                if math.lcm(a, b) == math.lcm(a, c) == math.lcm(b, c):
                    return False
    return True

def greedy_color_bound(remaining):
    if not remaining:
        return 0
    color_assignment = {}
    max_color = 0
    rem_list = sorted(remaining, key=lambda x: -len(shadow[x] & remaining))
    for v in rem_list:
        used_colors = set()
        for u in shadow[v]:
            if u in color_assignment:
                used_colors.add(color_assignment[u])
        c = 0
        while c in used_colors:
            c += 1
        color_assignment[v] = c
        max_color = max(max_color, c + 1)
    return max_color

def greedy_construct():
    best = set()
    for _ in range(300):
        order = list(range(1, N + 1))
        random.shuffle(order)
        cur = set()
        for x in order:
            if can_add(x, cur):
                cur.add(x)
        if len(cur) > len(best):
            best = cur
    return best

def local_search(s, time_limit=5.0):
    s = set(s)
    start = time.time()
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try adding elements
        for x in range(1, N + 1):
            if x not in s and can_add(x, s):
                s.add(x)
                improved = True
        # Try removing and re-adding different elements (2-opt style)
        for x in list(s):
            s.remove(x)
            best_add = None
            best_gain = 0
            for y in range(1, N + 1):
                if y not in s and can_add(y, s):
                    # Check if adding y is beneficial
                    if best_add is None or y > best_add:
                        best_add = y
            if best_add is not None:
                s.add(best_add)
                if len(s) > len(set(s)):
                    improved = True
        # Try swap: remove x, add y
        for x in list(s):
            s.remove(x)
            candidates = [y for y in range(1, N + 1) if y not in s and can_add(y, s)]
            if candidates:
                # Pick the candidate that allows most future additions
                best_y = max(candidates, key=lambda y: sum(1 for z in range(1, N+1) if z not in s and z != y and can_add(z, s | {y})))
                s.add(best_y)
                improved = True
            else:
                s.add(x)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Polish with local search
    polished = local_search(best_set, time_limit=10.0)
    if len(polished) > best_size:
        best_set = polished
        best_size = best_size
    
    # B&B
    def bnb_rec(current_set, remaining, depth):
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
        
        if not remaining:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set.copy()
            return True
        
        ub = len(current_set) + greedy_color_bound(remaining)
        if ub <= best_size:
            return True
        
        # Pick branching variable
        best_x = None
        best_deg = -1
        for x in remaining:
            deg = len(shadow[x] & remaining)
            if deg > best_deg:
                best_deg = deg
                best_x = x
        
        rest = [x for x in remaining if x != best_x]
        
        # Branch 1: Include best_x
        new_set = current_set | {best_x}
        valid_rest_include = [y for y in rest if can_add(y, new_set)]
        
        bnb_rec(new_set, valid_rest_include, depth + 1)
        
        if time.time() > deadline:
            return False
        
        # Branch 2: Exclude best_x
        bnb_rec(current_set, rest, depth + 1)
        
        return True
    
    # Run B&B with time budget
    bnb_budget = deadline - time.time() - 15.0  # Leave 15s for local search
    if bnb_budget > 5.0:
        bnb_rec(set(), list(range(1, N + 1)), 0)
    
    # Final local search polish
    if time.time() < deadline - 2.0:
        polished = local_search(best_set, time_limit=min(10.0, deadline - time.time() - 1.0))
        if len(polished) > len(best_set):
            best_set = polished
    
    result = list(best_set)
    if not result:
        result = [1]
    
    return result
