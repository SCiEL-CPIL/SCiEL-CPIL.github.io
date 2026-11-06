import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 85
    
    N = 256
    
    squares = set()
    k = 1
    while k * k <= 512:
        squares.add(k * k)
        k += 1
    
    diagonal_forbidden = set()
    m = 1
    while 2 * m * m <= N:
        diagonal_forbidden.add(2 * m * m)
        m += 1
    
    conflicts = [set() for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if i + j in squares:
                conflicts[i].add(j)
                conflicts[j].add(i)
    
    def greedy_build(order):
        result = set()
        for v in order:
            if v in diagonal_forbidden:
                continue
            if not (conflicts[v] & result):
                result.add(v)
        return result
    
    best_set = set()
    best_size = 0
    
    for trial in range(300):
        if time.time() - start_time > DEADLINE - 15:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        s = greedy_build(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    def local_search(s, time_limit):
        s = set(s)
        best = set(s)
        best_sz = len(s)
        
        progress = True
        while progress and time.time() - start_time < time_limit:
            progress = False
            candidates = [v for v in range(1, N + 1) if v not in s and v not in diagonal_forbidden]
            random.shuffle(candidates)
            for v in candidates:
                if not (conflicts[v] & s):
                    s.add(v)
                    progress = True
                    if len(s) > best_sz:
                        best_sz = len(s)
                        best = set(s)
                    break
        
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            improved = False
            s_list = list(s)
            random.shuffle(s_list)
            for rv in s_list:
                if time.time() - start_time > time_limit:
                    break
                s.remove(rv)
                cands = [v for v in range(1, N + 1) if v not in s and v not in diagonal_forbidden]
                random.shuffle(cands)
                for av in cands:
                    if not (conflicts[av] & s):
                        s.add(av)
                        if len(s) > best_sz:
                            best_sz = len(s)
                            best = set(s)
                            improved = True
                        break
                if improved:
                    break
                else:
                    s.add(rv)
            if not improved:
                break
        
        return best
    
    remaining = DEADLINE - (time.time() - start_time)
    if remaining > 5:
        best_set = local_search(best_set, DEADLINE - 5)
        best_size = len(best_set)
    
    result = sorted(best_set)
    
    # Verify
    valid = True
    for a in result:
        if 2 * a in squares:
            valid = False
            break
    if valid:
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                if result[i] + result[j] in squares:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid or not result:
        result = [1]
    
    return result
